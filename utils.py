import os

import torch
import torch.nn as nn
import torchvision

from omegaconf import OmegaConf

from cabifpn.config.init import default_config
from cabifpn.model.builder import BackboneNeck

#################### AUX Fuctions ####################
def _create_model(base_config, checkpoint):
        # Create the backbone and neck model
        print(f'[i+] Configuring backbone and neck models with variables: {base_config.MODEL}')
        backbone_neck = BackboneNeck(base_config.MODEL)
        ## freeze the backbone
        for param in backbone_neck.backbone.parameters():
            param.requires_grad = False
        backbone_neck.out_channels = base_config.MODEL.NECK.NUM_CHANNELS
        print('[i+] Ready !')

        # MaskRCNN's head config
        print('[i+] Building the base model with MaskRCNN head ...')
        anchor_sizes = ((32),(64),(128),(256)) 
        aspect_ratios = ((0.5,1.0,1.5,2.0,)) * len(anchor_sizes)
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['P0','P1','P2','P3'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['P0','P1','P2','P3'],
                                                             output_size=14,
                                                             sampling_ratio=2)

        # Create de base model with the FasterRCNN's head
        _num_classes = len(base_config.DATASET.OBJ_LIST)
        print(f'[++] Numbers of classes: {_num_classes}')
        base_model = torchvision.models.detection.MaskRCNN(backbone_neck,
                                                           num_classes=_num_classes + 1, # +1 = background
                                                           rpn_anchor_generator=anchor_generator,
                                                           box_roi_pool=roi_pooler,
                                                           mask_roi_pool=mask_roi_pooler)
        print('[+] Loading checkpoint...')
        out_n = base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if len(out_n.unexpected_keys)!=0: 
            print(f'[++] The unexpected keys was: {out_n.unexpected_keys}')
        else:
            print('[++] All keys matched successfully')
        print(f"[+] Ready. last_epoch: {checkpoint['epoch']} - last_loss: {checkpoint['best_loss']}")
        
        print('[i+] Ready !')
        
        return base_model

def _create_config(checkpoint_path: str, path_lib:str = 'cabifpn'):
    print('[+] Loading checkpoint...')
    checkpoint = torch.load(os.path.join(checkpoint_path))
    print('[+] Ready !')
    
    print('[+] Preparing base configs...')

    model_backbone_conf = OmegaConf.load(os.path.join(path_lib, checkpoint['fn_cfg_model_backbone']))
    model_neck_conf = OmegaConf.load(os.path.join(path_lib, checkpoint['fn_cfg_model_neck']))

    dataset_conf = OmegaConf.load(os.path.join(path_lib, checkpoint['fn_cfg_dataset']))

    base_config = default_config()
    base_config.MODEL = OmegaConf.merge(base_config.MODEL, model_backbone_conf, model_neck_conf)
    base_config.DATASET = OmegaConf.merge(base_config.DATASET, dataset_conf)

    print('[+] Ready !')
    
    return base_config, checkpoint