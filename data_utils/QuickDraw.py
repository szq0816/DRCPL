import os
import numpy as np
from torch.utils import data
from collections import defaultdict
from .utils import preprocess


class QuickDrawDatasetTrain(data.Dataset):
    def __init__(self, args):
        root_dir = args.DATA_PATH + '/QuickDraw'
        # ——————————————————————————————————— 类别名和标签到类别名字典 ———————————————————————————————————
        with open(args.DATA_PATH + "/QuickDraw/zeroshot/cname_cid.txt", 'r') as f:
            file_content = f.readlines()
        self.all_class_label = [int(ff.strip().split()[-1]) for ff in file_content]
        self.label2name = {int(ff.strip().split()[-1]): ' '.join(ff.strip().split()[:-1]) for ff in file_content}

        # ————————————————————————————— 按顺序保存所有的草图路径和对应的标签 —————————————————————————————
        sketch_file = args.DATA_PATH + f'/QuickDraw/zeroshot/sketch_train.txt'
        with open(sketch_file, 'r') as fh:
            file_content = fh.readlines()
        all_sketch = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
        all_sketch_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        assert all_sketch.shape[0] == all_sketch_label.shape[0]

        self.half_sketch = np.array([all_sketch[i] for i in range(0, all_sketch.shape[0], 4)])
        self.half_sketch_label = np.array([all_sketch_label[i] for i in range(0, all_sketch_label.shape[0], 4)])

        # ————————————————————————————— 创建标签-图像字典  label:[image_path] —————————————————————————————
        self.all_label2image = defaultdict(list)
        image_file = args.DATA_PATH + f'/QuickDraw/zeroshot/all_photo_train.txt'
        with open(image_file, 'r') as fh:
            file_content = fh.readlines()
        for ff in file_content:
            self.all_label2image[int(ff.strip().split()[-1])].append(
                os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])))

    def __len__(self):
        return len(self.half_sketch_label)

    def __getitem__(self, index):
        sketch = self.half_sketch[index]
        sketch_image_label = self.half_sketch_label[index]
        pos_image = np.random.choice(self.all_label2image[sketch_image_label])

        neg_class_label = self.all_class_label.copy()
        neg_class_label.remove(sketch_image_label)
        neg_image_label = np.random.choice(neg_class_label)
        neg_image = np.random.choice(self.all_label2image[neg_image_label])

        sketch = preprocess(sketch, 'sk')
        pos_image = preprocess(pos_image, 'im')
        neg_image = preprocess(neg_image, 'im')

        return sketch, pos_image, neg_image, sketch_image_label, neg_image_label


class QuickDrawDatasetTest(data.Dataset):
    def __init__(self, args, type_skim='sk'):
        root_dir = args.DATA_PATH + '/QuickDraw'
        # ————————————————————————————— 按顺序保存所有的=草图-图像=路径和对应的标签 —————————————————————————————
        if type_skim == 'sk':
            sketch_file = args.DATA_PATH + f'/QuickDraw/zeroshot/sketch_zero.txt'
            with open(sketch_file, 'r') as fh:
                file_content = fh.readlines()
            self.all_file = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
            self.all_file_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])
            # 对验证的样本数量进行缩减
            if args.TEST:
                self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 1)]  # 92291
            else:
                self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 120)]  # sketch 92291->770
        else:
            image_file = args.DATA_PATH + f'/QuickDraw/zeroshot/all_photo_zero.txt'
            with open(image_file, 'r') as fh:
                file_content = fh.readlines()
            self.all_file = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
            self.all_file_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])
            # 对验证的样本数量进行缩减
            if args.TEST:
                self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 1)]  # 54151
            else:
                self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 30)]  # image 54151->1806

        self.type_skim = type_skim

    def __len__(self):
        return len(self.all_label_index)

    def __getitem__(self, index):
        file = self.all_file[self.all_label_index[index]]
        file_label = self.all_file_label[self.all_label_index[index]]

        if self.type_skim == 'sk':
            file = preprocess(file, 'sk')
        else:
            file = preprocess(file, 'im')

        return file, file_label
