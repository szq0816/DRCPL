import argparse


class Option:

    def __init__(self):
        parser = argparse.ArgumentParser(description='args for models')

        # MODEL
        parser.add_argument('--RES_MLP', default=True)
        parser.add_argument('--SIZE', type=tuple, default=(224, 224))
        parser.add_argument('--CTX_INIT', type=str, default='a photo of a')
        parser.add_argument('--BACKBONE_NAME', type=str, default='ViT-B/16')

        # TRAIN
        parser.add_argument('--BATCH', type=int, default=16)
        parser.add_argument('--MAX_EPOCH', type=int, default=10)
        parser.add_argument('--LR', type=float, default=5e-6)
        parser.add_argument('--MIN_LR', type=float, default=3e-6)
        parser.add_argument('--WEIGHT_DECAY', type=float, default=5e-4)

        # Prompt Config
        parser.add_argument('--N_CTX', type=int, default=2)
        parser.add_argument('--PROMPT_DEPTH', type=int, default=9)

        # DATASET
        parser.add_argument('--DATASET_LEN', type=int, default=10000)
        parser.add_argument('--DATA_PATH', type=str, default='/home/zgy/Data')
        parser.add_argument('--TEST_CLASS', type=str, default='test_class_sketchy25',
                            choices=['test_class_sketchy25',
                                     'test_class_sketchy21',
                                     'test_class_tuberlin30',
                                     'test_class_quickdraw30']
                            )

        # SAVE PATH
        parser.add_argument('--SAVE', type=str,
                            default='/home/zgy/Code/DRCPL/checkpoints/sketchy25')
        parser.add_argument('--FILE_NAME', type=str, default='ALL')

        # TEST
        parser.add_argument('--LOAD', type=str,
                            default='/home/zgy/Code/DRCPL/checkpoints/sketchy25/ALL.pth')

        # OTHER
        parser.add_argument('--SEED', type=int, default=2021)
        parser.add_argument('--PREC', type=str, default='fp32', choices=['fp16', 'fp32', 'amp'])
        parser.add_argument('--MARGIN', type=float, default=0.3)
        parser.add_argument('--TEMPERATURE', type=float, default=0.07)
        parser.add_argument('--TEST', default=False, action='store_true', help='train/test scale')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
