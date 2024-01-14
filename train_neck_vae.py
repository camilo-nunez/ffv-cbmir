import os
import argparse

from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime

import albumentations as A
from albumentations.pytorch import ToTensorV2

from cabifpn.utils.getter import IntermediateLayerGetter
from cabifpn.utils.datasets import CocoDetectionV2, LVISDetection

from model.neck_vae import NeckVAE
from utils import _create_model, _create_config

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from lion_pytorch import Lion

## Customs configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_option():
    parser = argparse.ArgumentParser(
        'Thesis cnunezf training Neck VAE model.', add_help=True)
    
    parser.add_argument('--checkpoint_extractor',
                        type=str,
                        required=True,
                        metavar="FILE",
                        default = None,
                        help="Checkpoint extractor model filename.")
    parser.add_argument('--checkpoint',
                        type=str,
                        metavar="FILE",
                        default = None,
                        help="Checkpoint model filename.")
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='/thesis/checkpoint', 
                        help='Path to complete DATASET.')
    
    parser.add_argument('--latent_dim',
                        type=int,
                        default=128)
    
    parser.add_argument('--in_channels',
                        type=int,
                        default=256)
    
    parser.add_argument('--neck_indices',
                        type=int,
                        nargs="+",
                        required=True,
                        help='Out indices from neck, like 0,1,2 .'
                       )
    
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/thesis/classical', 
                        help='Path to complete DATASET.')
    
    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Default 5 epochs.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2)
    
    parser.add_argument('--scheduler',
                        action='store_true',
                        help="Use scheduler")
    
    parser.add_argument('--summary',
                        action='store_true',
                        help="Display the summary of the model.")
    
    parser.add_argument('--amp',
                        action='store_true',
                        help="Enable Automatic Mixed Precision train.")
    
    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-3,
                        help='Learning rate used by the \'adamw\' optimizer. Default is 1e-3. For \'lion\' its recommend 2e-4.'
                       )
    parser.add_argument('--wd', 
                        type=float, 
                        default=1e-5,
                        help='Weight decay used by the \'adamw\' optimizer. Default is 1e-5. For \'lion\' its recommend 1e-2.'
                       )
    parser.add_argument('--optimizer', 
                        type=str, 
                        default='adamw',
                        help='The optimizer to use. The available opts are: \'adamw\' or \'lion\'. By default its \'adamw\' .'
                       )
    
    args, unparsed = parser.parse_known_args()

    return args

if __name__ == '__main__':
    
    # Load configs for the model and the dataset
    args = parse_option()
    
    # Check the principal exceptions
    if not torch.cuda.is_available(): raise Exception('This script is only available to run in GPU.')
    
    # === GLOBAL VARIABLES ===
    ## Create the dict with layer names neck
    set_neck_indices = set(args.neck_indices)
    _RETURN_NECK_NODES = dict([(f'backbone.neck.neck.neck_layer_{idx}.proj_p4_2', f'p4_2_l{idx}') for idx in set_neck_indices])
    
    ## Define principal vars
    _LATENT_DIM = args.latent_dim
    _IN_CHANNELS = args.in_channels
    
    # === Create and load extractor model ===
    print(f'[+] Loading extractor model ...')
    
    ## Load extractor model
    base_config, checkpoint = _create_config(os.path.join(args.checkpoint_path, args.checkpoint_extractor))
    model_extractor = _create_model(base_config, checkpoint).to(device).eval()

    ## freeze the extractor model
    for param in model_extractor.parameters():
        param.requires_grad = False
    
    ## Display the summary of the net
    if args.summary: summary(model_extractor)
    
    ## Define the hooker neck's layers fuction
    mid_extractor_getter = IntermediateLayerGetter(model_extractor,
                                                   return_layers=_RETURN_NECK_NODES,
                                                   keep_output=False)
    
    ## Get original shapes
    tmp_input = torch.rand(1,3,base_config.DATASET.IMAGE_SIZE, base_config.DATASET.IMAGE_SIZE, device=device)
    tmp_mid_outputs,_ = mid_extractor_getter(tmp_input)
    _IN_SHAPE = tuple(tmp_mid_outputs['p4_2_l0'].shape[2:])
    
    print('[+] Ready !')

    # === HERE GO THE ASSERT DEF !!!!!! DONT FORGET IT !!!! ===
#     if len(set_neck_indices) NUM_LAYERS
#     print(base_config)
#     print(_RETURN_NECK_NODES, _LATENT_DIM, _IN_CHANNELS)
#     print(args)

    # === Create NECK VAE base model ===
    print('[+] Building the NECK VAE base model ...')
    print(f'[++] Using VAE configs : total VAEs->{len(set_neck_indices)} | in_channels->{_IN_CHANNELS} | in_shape->{_IN_SHAPE} | latent_dim->{_LATENT_DIM}.')
    base_model = NeckVAE(len(set_neck_indices), _IN_CHANNELS, _IN_SHAPE, _LATENT_DIM).to(device).train()
    
    ## Display the summary of the net
    if args.summary: summary(base_model)
    print('[+] Ready !')

    # === Load the dataset ===
    print(f'[+] Loading {base_config.DATASET.NAME} dataset...')
    print(f'[++] Using batch_size: {args.batch_size}')
    
    ## Albumentations to use
    train_transform = A.Compose([A.Resize(base_config.DATASET.IMAGE_SIZE, base_config.DATASET.IMAGE_SIZE),
                                 A.Normalize(mean=base_config.DATASET.MEAN,
                                             std=base_config.DATASET.STD,
                                             max_pixel_value=255.0),
                                 ToTensorV2()
                                ],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
                               )

    ## Training dataset
    print('[++] Loading training dataset...')
    training_params = {'batch_size': args.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': lambda batch: tuple(zip(*batch)),
                       'num_workers': 4,
                       'pin_memory':True,
                      }
    
    if base_config.DATASET.NAME == 'coco2017':
        train_dataset = CocoDetectionV2(root=os.path.join(args.dataset_path,'coco2017/train2017'),
                                        annFile=os.path.join(args.dataset_path,'coco2017/annotations/instances_train2017.json'),
                                        transform = train_transform)
    elif base_config.DATASET.NAME == 'lvisv1':
        train_dataset = LVISDetection(root=os.path.join(args.dataset_path,'lvisdataset/train2017'),
                                      annFile=os.path.join(args.dataset_path,'lvisdataset/lvis_v1_train.json'),
                                      transform = train_transform)

    training_loader = torch.utils.data.DataLoader(train_dataset, **training_params)
    print('[++] Ready !')
    print('[+] Ready !')
    
    # === General train variables ===
    ## Cofig the optimizer
    params = [p for p in base_model.parameters() if p.requires_grad]

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                      lr=args.lr,
                                      weight_decay=args.wd)
        print(f'[+] Using AdamW optimizer. Configs: lr->{args.lr}, weight_decay->{args.wd}')
    elif args.optimizer == 'lion':
        optimizer = Lion(params,
                         lr=args.lr,
                         weight_decay=args.wd)
        print(f'[+] Using Lion optimizer. Configs:{args.optimizer}')
    else:
        raise Exception("The optimizer selected doesn't exist. The available optis are: \'adamw\' or \'lion\'.")  

    start_epoch = 1
    end_epoch = args.num_epochs
    best_loss = 1e5
    global_steps = 0
    
    ## Scheduler
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=1, eta_min= 5e-3)
        print("[+] Using \'CosineAnnealingWarmRestarts\' ")
    
    ## Prepare Automatic Mixed Precision
    if args.amp:
        print("[+] Using Automatic Mixed Precision")
        use_amp = True
    else:
        use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    ## Load the checkpoint if is need it
    if args.checkpoint:
        print('[+] Loading checkpoint...')
        checkpoint = torch.load(os.path.join(args.checkpoint))
        
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if args.scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if use_amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f'[+] Ready. start_epoch: {start_epoch} - best_loss: {best_loss}')

    ## Define the loss fuction
    def calculate_losses(x, vae_dict):
        x_hat = vae_dict['x_proj']

        log_sigma = vae_dict['log_sigma']
        mu = vae_dict['mu']
        sigma = torch.exp(log_sigma)

        kld_loss = torch.mean(torch.sum(-0.5* (1 + torch.log(torch.pow(sigma, 2)) - torch.pow(mu, 2) - torch.pow(sigma, 2)), dim = 1), dim = 0)
        reconstruction_loss = F.mse_loss(x, x_hat)

        loss = kld_loss + reconstruction_loss

        return OrderedDict([('loss', loss),
                            ('kld_loss', kld_loss),
                            ('reconstruction_loss', reconstruction_loss),
                            ])

    # === Train the model ===
    print('[+] Starting training ...')
    start_t = datetime.now()
    
    for e, epoch in enumerate(range(start_epoch, end_epoch + 1)):
        loss_l = []
        with tqdm(training_loader, unit=" batch") as tepoch:
            for i, data in enumerate(tepoch):

                images, targets = data

                if None in images and None in targets: continue
                if not all(('boxes' in d.keys() and 'labels' in d.keys() and 'masks' in d.keys()) for d in targets): continue
                
                images = [image.to(device) for image in images]
                
                ## Get original shapes
                layers_vector, _ = mid_extractor_getter(images)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    ## Get reconstruction
                    layers_vector_proj = base_model(layers_vector)

                    ## Get all losses from layers
                    layer_losses = [calculate_losses(x, vae_dict) for i, (x, vae_dict) in enumerate(zip(layers_vector.values(), layers_vector_proj.values()))]
                    
#                     losses = torch.mean(torch.stack([d['loss'] for d in layer_losses], dim=0))
                    losses = torch.sum(torch.stack([d['loss'] for d in layer_losses], dim=0))

                scaler.scale(losses).backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad(set_to_none=True)

                current_lr = optimizer.param_groups[0]['lr']
                
                str_losses = " ".join(["|Losees Layer-{}: kld_l:{:1.8}, recons_l:{:1.8} |".format(i, vae_dict['kld_loss'],vae_dict['kld_loss']) for i, vae_dict in enumerate(layer_losses)])
                
                description_s = 'Epoch: {}/{}. lr: {:1.8f} loss {:1.4f}'\
                                   .format(epoch,end_epoch,current_lr,losses)\
#                                 + str_losses

                tepoch.set_description(description_s)
                
                global_steps+=1

                if args.scheduler:
                    scheduler.step(e + i/len(tepoch))

        if losses < best_loss:
            best_loss = losses

            torch.save({'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict() if args.scheduler else None,
                        'scaler_state_dict': scaler.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                       },
                       os.path.join(args.checkpoint_path, f'{datetime.utcnow().strftime("%Y%m%d_%H%M")}_VAE_{base_config.MODEL.BACKBONE.MODEL_NAME}_{base_config.MODEL.NECK.MODEL_NAME}_{epoch}.pth'))

    end_t = datetime.now()
    print('[+] Ready, the train phase took:', (end_t - start_t))            
