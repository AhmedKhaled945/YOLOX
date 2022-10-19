#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp
import torch.nn as nn
import os
import random
import torch

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/home/ec2-user/SageMaker/Ahmed_Yolox_Trials/YOLOX/big_coco_dataset"
        self.train_ann = "train.json"
        self.val_ann = "validation.json"

        self.num_classes = 9

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 5
        self.no_aug = False
        self.no_aug_epochs = 1
        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0
        self.save_history_ckpt = True
        self.nmsthre = 0.45
        
    def get_model(self, load_pretrain=True):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        
        
        if load_pretrain:
            """Load COCO pretrained model"""
        
            #filename = "pretrained/yolox_x.pth"
            ckpt_path = "pretrained/yolox_x.pth"
            print("Loading COCO pretrained weights from %s" % ckpt_path)
            state_dict = torch.load(ckpt_path, map_location='cpu')["model"]
            # Deal with SOT and MOT head pretrained parameters
            new_state_dict = {}

            for k, v in state_dict.items():
                if not k.startswith("head."):
                    new_state_dict[k] = v
                else:
                    if k in ["head.cls_preds.0.weight", "head.cls_preds.0.bias", "head.cls_preds.1.weight", "head.cls_preds.1.bias",
                    "head.cls_preds.2.weight", "head.cls_preds.2.bias"]:
                        if self.num_classes == 8:
                            new_state_dict[k] = v[[0,0,2,7,5,6,3,1]] # [80] or [80, 256, 1, 1]
                        elif self.num_classes == 1:
                            new_state_dict[k] = v[0:1]
                        
                        elif self.num_classes == 8:
                            new_state_dict[k] = v[[0,3,2,5,1,7,7,7]]
                        else:
                            raise ValueError("Invalid num_classes")
                    
                    else:
                        new_state_dict[k] = v
                        
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            del state_dict
            torch.cuda.empty_cache()
        
        return self.model
