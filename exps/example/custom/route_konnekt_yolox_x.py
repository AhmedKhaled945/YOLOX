#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1
        self.width = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/home/ec2-user/SageMaker/Ahmed_Yolox_Trials/coco_format_dataset"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.num_classes = 11

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 5
