import os
import time
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from options import Option
from utils.valid import valid_cls
from data_utils.Dataset import load_train_data
from models.model import load_clip_to_cpu, CustomCLIP
from utils.util import save_checkpoint, setup_seed, count_parameters, adjust_learning_rate

from thop import profile


def main(args):
    assert args.PREC in ['fp16', 'fp32', 'amp']

    if args.TEST_CLASS == 'test_class_sketchy25':
        train_data, train_class_label = load_train_data(args=args)
        from data_utils.Sketchy import SketchyDatasetTest
        val_sketch = SketchyDatasetTest(args=args, type_skim='sk')
        val_image = SketchyDatasetTest(args=args, type_skim='im')
        train_data_loader = DataLoader(dataset=train_data, batch_size=args.BATCH,
                                       shuffle=True,
                                       num_workers=2, drop_last=True)
        val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=args.BATCH // 2,
                                       num_workers=2, drop_last=False)
        val_image_loader = DataLoader(dataset=val_image, batch_size=args.BATCH // 2,
                                      num_workers=2, drop_last=False)
    elif args.TEST_CLASS == 'test_class_sketchy21':
        train_data, train_class_label = load_train_data(args=args)
        from data_utils.Sketchy import SketchyDatasetTest
        val_sketch = SketchyDatasetTest(args=args, type_skim='sk')
        val_image = SketchyDatasetTest(args=args, type_skim='im')
        train_data_loader = DataLoader(dataset=train_data, batch_size=args.BATCH,
                                       shuffle=True,
                                       num_workers=2, drop_last=True)
        val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=args.BATCH // 2,
                                       num_workers=2, drop_last=False)
        val_image_loader = DataLoader(dataset=val_image, batch_size=args.BATCH // 2,
                                      num_workers=2, drop_last=False)
    elif args.TEST_CLASS == 'test_class_tuberlin30':
        train_data, train_class_label = load_train_data(args=args)
        from data_utils.TUBerlin import TUBerlinDatasetTest
        val_sketch = TUBerlinDatasetTest(args=args, type_skim='sk')
        val_image = TUBerlinDatasetTest(args=args, type_skim='im')
        train_data_loader = DataLoader(dataset=train_data, batch_size=args.BATCH,
                                       shuffle=True,
                                       num_workers=2, drop_last=True)
        val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=args.BATCH // 2,
                                       num_workers=2, drop_last=False)
        val_image_loader = DataLoader(dataset=val_image, batch_size=args.BATCH // 2,
                                      num_workers=2, drop_last=False)
    elif args.TEST_CLASS == 'test_class_quickdraw30':
        train_data, train_class_label = load_train_data(args=args)
        from data_utils.QuickDraw import QuickDrawDatasetTest
        val_sketch = QuickDrawDatasetTest(args=args, type_skim='sk')
        val_image = QuickDrawDatasetTest(args=args, type_skim='im')
        train_data_loader = DataLoader(dataset=train_data, batch_size=args.BATCH,
                                       shuffle=True,
                                       num_workers=2, drop_last=True)
        val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=args.BATCH // 2,
                                       num_workers=2, drop_last=False)
        val_image_loader = DataLoader(dataset=val_image, batch_size=args.BATCH // 2,
                                      num_workers=2, drop_last=False)

    print(f'Loading CLIP (backbone: {args.BACKBONE_NAME}).')
    clip_model = load_clip_to_cpu(args)

    if args.PREC == 'fp32' or args.PREC == 'amp':
        clip_model.float()
        print("CLIP Use Float32")

    print('Building custom CLIP.')
    model = CustomCLIP(args, train_class_label, clip_model)
    # print(f"all_Parameters of model: {count_parameters(model)}.")

    print('Turning off gradients in both the image and the text encoder.')
    for name, param in model.named_parameters():
        if 'VPT' in name:
            param.requires_grad_(True)
        elif 'prompt_learner' in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f'Parameters to be updated: {len(enabled)}.')

    count_parameters(model)

    model = model.cuda()

    optimizer = Adam(params=model.parameters(),
                     lr=args.LR,
                     weight_decay=args.WEIGHT_DECAY
                     )

    scaler = GradScaler() if args.PREC == 'amp' else None

    accuracy = 0
    for i in range(args.MAX_EPOCH):
        # ========================================== Train ==================================================
        print('------------------------train------------------------')
        adjust_learning_rate(args=args, optimizer=optimizer, epoch=i)
        model.train()
        epoch = i + 1
        start_time = time.time()
        num_total_steps = len(train_data_loader)
        for batch_idx, (sketch, image, image_neg, sk_label, im_label, im_label_neg) in enumerate(
                train_data_loader):
            # prepare data
            all_image = torch.cat([image, image_neg], dim=0)

            sk_label, im_label, im_label_neg = torch.cat([sk_label]), torch.cat([im_label]), torch.cat([im_label_neg])
            all_label = torch.cat([sk_label, im_label, im_label_neg], dim=0)

            all_sketch, all_image, all_label = sketch.cuda(), all_image.cuda(), all_label.cuda()

            # calculate feature

            prec = args.PREC
            if prec == 'amp':
                with autocast():
                    loss_itc, loss_tri, loss_sdm = model(sketch=all_sketch, image=all_image, label=all_label,
                                                         split='train')
                loss = loss_itc + loss_tri + loss_sdm
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_itc, loss_tri, loss_sdm = model(sketch=all_sketch, image=all_image, label=all_label,
                                                     split='train')
                loss = loss_itc + loss_tri + loss_sdm
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # log
            step = batch_idx + 1
            if step % 100 == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(
                    f'epoch_{epoch} step_{step} eta {remaining_time}: loss:{loss.item():.3f} loss_itc:{loss_itc.item():.3f} loss_tri:{loss_tri.item():.3f} loss_sdm:{loss_sdm.item():.3f}')

        # ========================================== Valid ==================================================
        print('------------------------valid------------------------')
        map_all, map_200, p_100, p_200 = valid_cls(model, val_sketch_loader, val_image_loader, split='val',
                                                   test_class=args.TEST_CLASS)

        if map_all > accuracy:
            accuracy = map_all
            print('Save the BEST {}th model......'.format(epoch))
            save_checkpoint(
                state={'model': model.state_dict(),
                       'epoch': epoch,
                       'map_all': accuracy},
                directory=args.SAVE,
                file_name=args.FILE_NAME)


if __name__ == '__main__':
    cfg = Option().parse()
    setup_seed(cfg.SEED)
    main(cfg)
