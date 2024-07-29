import os
import numpy as np
from torch.utils import data
from collections import defaultdict
from .utils import preprocess


class SketchyDatasetTrain(data.Dataset):
    def __init__(self, args):
        root_dir = args.DATA_PATH + '/Sketchy'
        if args.TEST_CLASS == 'test_class_sketchy25':
            # ——————————————————————————————————— 类别名和标签到类别名字典 ———————————————————————————————————
            with open(args.DATA_PATH + "/Sketchy/zeroshot1/cname_cid.txt", 'r') as f:
                file_content = f.readlines()
            self.all_class_label = [int(ff.strip().split()[-1]) for ff in file_content]
            self.label2name = {int(ff.strip().split()[-1]): ' '.join(ff.strip().split()[:-1]) for ff in file_content}

            # ————————————————————————————— 按顺序保存所有的草图路径和对应的标签 —————————————————————————————
            sketch_file = args.DATA_PATH + f'/Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_train.txt'
            with open(sketch_file, 'r') as fh:
                file_content = fh.readlines()
            self.all_sketch = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
            self.all_sketch_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])

            # ————————————————————————————— 创建标签-图像字典  label:[image_path] —————————————————————————————
            self.all_label2image = defaultdict(list)
            image_file = args.DATA_PATH + f'/Sketchy/zeroshot1/all_photo_filelist_train.txt'
            with open(image_file, 'r') as fh:
                file_content = fh.readlines()
            for ff in file_content:
                self.all_label2image[int(ff.strip().split()[-1])].append(
                    os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])))

        elif args.TEST_CLASS == "test_class_sketchy21":
            # ——————————————————————————————————— 类别名和标签到类别名字典 ———————————————————————————————————
            with open(args.DATA_PATH + "/Sketchy/zeroshot0/cname_cid.txt", 'r') as f:
                file_content = f.readlines()
            self.all_class_label = [int(ff.strip().split()[-1]) for ff in file_content]
            self.label2name = {int(ff.strip().split()[-1]): ' '.join(ff.strip().split()[:-1]) for ff in file_content}

            # ————————————————————————————— 按顺序保存所有的草图路径和对应的标签 —————————————————————————————
            sketch_file = args.DATA_PATH + f'/Sketchy/zeroshot0/sketch_tx_000000000000_ready_filelist_train.txt'
            with open(sketch_file, 'r') as fh:
                file_content = fh.readlines()
            self.all_sketch = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
            self.all_sketch_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])

            # ————————————————————————————— 创建标签-图像字典  label:[image_path] —————————————————————————————
            self.all_label2image = defaultdict(list)
            image_file = args.DATA_PATH + f'/Sketchy/zeroshot0/all_photo_filelist_train.txt'
            with open(image_file, 'r') as fh:
                file_content = fh.readlines()
            for ff in file_content:
                self.all_label2image[int(ff.strip().split()[-1])].append(
                    os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])))

    def __len__(self):
        return len(self.all_sketch_label)

    def __getitem__(self, index):
        sketch = self.all_sketch[index]
        sketch_image_label = self.all_sketch_label[index]
        pos_image = np.random.choice(self.all_label2image[sketch_image_label])

        neg_class_label = self.all_class_label.copy()
        neg_class_label.remove(sketch_image_label)
        neg_image_label = np.random.choice(neg_class_label)
        neg_image = np.random.choice(self.all_label2image[neg_image_label])

        sketch = preprocess(sketch, 'sk')
        pos_image = preprocess(pos_image, 'im')
        neg_image = preprocess(neg_image, 'im')

        return sketch, pos_image, neg_image, sketch_image_label, neg_image_label


class SketchyDatasetTest(data.Dataset):
    def __init__(self, args, type_skim='sk'):
        root_dir = args.DATA_PATH + '/Sketchy'
        if args.TEST_CLASS == 'test_class_sketchy25':
            # ————————————————————————————— 按顺序保存所有的=草图-图像=路径和对应的标签 —————————————————————————————
            if type_skim == 'sk':
                sketch_file = args.DATA_PATH + f'/Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_zero.txt'
                with open(sketch_file, 'r') as fh:
                    file_content = fh.readlines()
                self.all_file = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
                self.all_file_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])
                # 对验证的样本数量进行缩减
                if args.TEST:
                    self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 1)]  # 15229
                else:
                    self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 20)]  # sketch 15229->762
            else:
                image_file = args.DATA_PATH + f'/Sketchy/zeroshot1/all_photo_filelist_zero.txt'
                with open(image_file, 'r') as fh:
                    file_content = fh.readlines()
                self.all_file = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
                self.all_file_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])
                # 对验证的样本数量进行缩减
                if args.TEST:
                    self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 1)]  # 17101
                else:
                    self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 10)]  # image 17101->1711

        elif args.TEST_CLASS == "test_class_sketchy21":
            # ————————————————————————————— 按顺序保存所有的草图路径和对应的标签 —————————————————————————————
            if type_skim == 'sk':
                sketch_file = args.DATA_PATH + f'/Sketchy/zeroshot0/sketch_tx_000000000000_ready_filelist_zero.txt'
                with open(sketch_file, 'r') as fh:
                    file_content = fh.readlines()
                self.all_file = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
                self.all_file_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])
                # 对验证的样本数量进行缩减
                if args.TEST:
                    self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 1)]  # 15229
                else:
                    self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 20)]  # sketch 15229->762
            else:
                image_file = args.DATA_PATH + f'/Sketchy/zeroshot0/all_photo_filelist_zero.txt'
                with open(image_file, 'r') as fh:
                    file_content = fh.readlines()
                self.all_file = np.array([os.path.join(root_dir, ' '.join(ff.strip().split()[:-1])) for ff in file_content])
                self.all_file_label = np.array([int(ff.strip().split()[-1]) for ff in file_content])
                # 对验证的样本数量进行缩减
                if args.TEST:
                    self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 1)]  # 17101
                else:
                    self.all_label_index = [i for i in range(0, self.all_file_label.shape[0], 10)]  # image 17101->1711

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
