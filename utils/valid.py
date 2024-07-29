import time
import numpy as np
import os
import pickle

import torch
import torch.nn.functional as F
from utils.metric import compute_metric


def valid_cls(model, val_sketch_loader, val_image_loader, split, test_class):
    model.eval()

    save_dir = '/home/zgy/Code/DRCPL/Feat'
    feature_file = os.path.join(save_dir, test_class, 'features_zero.pickle')
    if os.path.isfile(feature_file):
        print('load saved ZS-SBIR features')
        with open(feature_file, 'rb') as fh:
            sketch_vectors, sketch_labels, image_vectors, image_labels = pickle.load(fh)
    else:
        sketch_vectors, sketch_labels, image_vectors, image_labels = [], [], [], []
        with torch.no_grad():
            for batch_idx, (file, file_label) in enumerate(val_sketch_loader):
                if batch_idx % 50 == 0:
                    print(batch_idx, end=' ', flush=True)

                file = file.cuda()
                sketch_emb = model(sketch=file, split=split)
                sketch_vectors.append(sketch_emb.cpu())
                sketch_labels.append(file_label)

            sketch_vectors = torch.cat(sketch_vectors, dim=0)
            sketch_labels = torch.cat(sketch_labels, dim=0)
            print('')

            for batch_idx, (file, file_label) in enumerate(val_image_loader):
                if batch_idx % 50 == 0:
                    print(batch_idx, end=' ', flush=True)

                file = file.cuda()
                image_emb = model(image=file, split=split)
                image_vectors.append(image_emb.cpu())
                image_labels.append(file_label)

            image_vectors = torch.cat(image_vectors, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            print('')

        print('save ZS-SBIR features')
        if not os.path.exists(os.path.join(save_dir, test_class)):
            print('mkdir folder')
            os.makedirs(os.path.join(save_dir, test_class), exist_ok=True)

        with open(os.path.join(save_dir, test_class, 'features_zero.pickle'), 'wb') as fh:
            pickle.dump([sketch_vectors, sketch_labels, image_vectors, image_labels], fh)

    map_all, map_200, p_100, p_200 = compute_metric(sketch_vectors, sketch_labels, image_vectors, image_labels)
    print(
        f'map_all:{map_all:.3f} map_200:{map_200:.3f} p_100:{p_100:.3f} p_200:{p_200:.3f}')

    if split != 'test':
        return map_all, map_200, p_100, p_200
