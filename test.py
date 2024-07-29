import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from options import Option
from utils.valid import valid_cls
from utils.util import setup_seed, load_checkpoint, count_parameters
from models.model import load_clip_to_cpu, CustomCLIP


def main(args):
    assert args.PREC in ['fp16', 'fp32', 'amp']

    if args.TEST_CLASS == 'test_class_sketchy25':
        with open(args.DATA_PATH + '/Sketchy/zeroshot1/cname_cid.txt', 'r') as f:
            file_content = f.readlines()
        classnames = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        from data_utils.Sketchy import SketchyDatasetTest
        test_sketch = SketchyDatasetTest(args=args, type_skim='sk')
        test_image = SketchyDatasetTest(args=args, type_skim='im')
    elif args.TEST_CLASS == 'test_class_sketchy21':
        with open(args.DATA_PATH + '/Sketchy/zeroshot0/cname_cid.txt', 'r') as f:
            file_content = f.readlines()
        classnames = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        from data_utils.Sketchy import SketchyDatasetTest
        test_sketch = SketchyDatasetTest(args=args, type_skim='sk')
        test_image = SketchyDatasetTest(args=args, type_skim='im')
    elif args.TEST_CLASS == 'test_class_tuberlin30':
        with open(args.DATA_PATH + '/TUBerlin/zeroshot/cname_cid.txt', 'r') as f:
            file_content = f.readlines()
        classnames = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        from data_utils.TUBerlin import TUBerlinDatasetTest
        test_sketch = TUBerlinDatasetTest(args=args, type_skim='sk')
        test_image = TUBerlinDatasetTest(args=args, type_skim='im')
    elif args.TEST_CLASS == 'test_class_quickdraw30':
        with open(args.DATA_PATH + '/QuickDraw/zeroshot/cname_cid.txt', 'r') as f:
            file_content = f.readlines()
        classnames = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        from data_utils.QuickDraw import QuickDrawDatasetTest
        test_sketch = QuickDrawDatasetTest(args=args, type_skim='sk')
        test_image = QuickDrawDatasetTest(args=args, type_skim='im')

    print(f'Loading CLIP (backbone: {args.BACKBONE_NAME}).')
    clip_model = load_clip_to_cpu(args)

    if args.PREC == 'fp32' or args.PREC == 'amp':
        clip_model.float()

    print('Building custom CLIP.')
    model = CustomCLIP(args, classnames, clip_model)
    print(f"all_Parameters of model: {count_parameters(model)}.")

    if args.LOAD is not None:
        checkpoint = load_checkpoint(args.LOAD)
        cur = model.state_dict()

        assert len(cur) == len(checkpoint['model'])

        new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
        cur.update(new)
        model.load_state_dict(cur)
        print('Weights loaded successfully.')
    else:
        print('Use Original-CLIP Encoder.')

    model = model.cuda()

    test_sketch_loader = DataLoader(dataset=test_sketch, batch_size=args.BATCH,
                                    num_workers=2, drop_last=False)
    test_image_loader = DataLoader(dataset=test_image, batch_size=args.BATCH,
                                   num_workers=2, drop_last=False)

    # test
    print('------------------------test------------------------')
    valid_cls(model, test_sketch_loader, test_image_loader, split='test', test_class=args.TEST_CLASS)


if __name__ == '__main__':
    cfg = Option().parse()
    cfg.BATCH = 128
    cfg.TEST = True
    setup_seed(cfg.SEED)
    main(cfg)
