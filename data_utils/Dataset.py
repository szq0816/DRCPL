import os
import numpy as np
from torch.utils import data
from .utils import get_all_train_file, get_file_iccv, preprocess, create_dict_texts


def load_para(args):
    if args.TEST_CLASS == 'test_class_sketchy25':
        with open(args.DATA_PATH + "/Sketchy/zeroshot1/cname_cid.txt", 'r') as f:
            file_content = f.readlines()
            train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    elif args.TEST_CLASS == "test_class_sketchy21":
        with open(args.DATA_PATH + "/Sketchy/zeroshot0/cname_cid.txt", 'r') as f:
            file_content = f.readlines()
            train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    elif args.TEST_CLASS == 'test_class_tuberlin30':
        with open(args.DATA_PATH + "/TUBerlin/zeroshot/cname_cid.txt", 'r') as f:
            file_content = f.readlines()
            train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    elif args.TEST_CLASS == 'test_class_quickdraw30':
        with open(args.DATA_PATH + "/QuickDraw/zeroshot/cname_cid.txt", 'r') as f:
            file_content = f.readlines()
            train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    print('training classes: ', train_class_label.shape)

    return train_class_label


class PreLoad:
    def __init__(self, args):
        self.all_train_sketch = []
        self.all_train_sketch_label = []
        self.all_train_sketch_cls_name = []

        self.all_train_image = []
        self.all_train_image_label = []
        self.all_train_image_cls_name = []

        self.init_train(args)

    def init_train(self, args):
        self.all_train_sketch, self.all_train_sketch_label, self.all_train_sketch_cls_name = \
            get_all_train_file(args, "sketch")

        self.all_train_image, self.all_train_image_label, self.all_train_image_cls_name = \
            get_all_train_file(args, "image")

        print("used for train sketch / image:")
        print(self.all_train_sketch.shape, self.all_train_image.shape)


class TrainSet(data.Dataset):
    def __init__(self, args, train_class_label, pre_load):
        self.args = args
        self.train_class_label = train_class_label
        self.pre_load = pre_load

        self.class_dict = create_dict_texts(train_class_label)

        if args.TEST_CLASS == 'test_class_sketchy25' or args.TEST_CLASS == "test_class_sketchy21":
            self.root_dir = args.DATA_PATH + '/Sketchy'
        elif args.TEST_CLASS == 'test_class_tuberlin30':
            self.root_dir = args.DATA_PATH + '/TUBerlin'
        elif args.TEST_CLASS == 'test_class_quickdraw30':
            self.root_dir = args.DATA_PATH + '/QuickDraw'

    def __getitem__(self, index):
        # choose 3 label
        self.choose_label_name = np.random.choice(self.train_class_label, 3, replace=False)

        sk_label = self.class_dict.get(self.choose_label_name[0])
        im_label = self.class_dict.get(self.choose_label_name[0])
        im_label_neg = self.class_dict.get(self.choose_label_name[-1])

        sketch = get_file_iccv(self.pre_load.all_train_sketch_label, self.root_dir, self.choose_label_name[0],
                               self.pre_load.all_train_sketch_cls_name, 1, self.pre_load.all_train_sketch)
        image = get_file_iccv(self.pre_load.all_train_image_label, self.root_dir, self.choose_label_name[0],
                              self.pre_load.all_train_image_cls_name, 1, self.pre_load.all_train_image)
        image_neg = get_file_iccv(self.pre_load.all_train_image_label, self.root_dir, self.choose_label_name[-1],
                                  self.pre_load.all_train_image_cls_name, 1, self.pre_load.all_train_image)

        sketch = preprocess(sketch, 'sk')
        image = preprocess(image)
        image_neg = preprocess(image_neg)

        return sketch, image, image_neg, sk_label, im_label, im_label_neg

    def __len__(self):
        return self.args.DATASET_LEN


def load_train_data(args):
    # 类名
    train_class_label = load_para(args)
    pre_load = PreLoad(args)
    train_data = TrainSet(args, train_class_label, pre_load)

    return train_data, train_class_label
