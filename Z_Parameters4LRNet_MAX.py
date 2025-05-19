# Parameters in TripleNet's main script
class Parameters:
    def __init__(self):
        # Fixed Parameters
        self.type_sub_model = "VGG16"   #"VGG16", "ResNet34", "ResNet50", "ResNet101", "ResNet152"
        self.N_EPOCHS = 200             # 200
        self.BATCH_SIZE = 16            # T-UNet-16, DESSN-16, DSIFN-8
        self.PATCH_SIDE = 256
        self.INIT_LR = 0.0001
        self.NORMALISE_IMGS = True
        self.DATA_AUG = False


        self.cuda_device_id = 3         # 3:huan.zhong
        self.TYPE_DATASET = 3           # 0:TEST | 1:DSIFN-Dataset | 2:WHU-Building-Dataset | 3:LEVIR-CD | 4:S2Looking | 5:WHU-BCD
        if self.TYPE_DATASET == 5:
            self.MODE_NORMALISE = 2
        else:
            self.MODE_NORMALISE = 1     # 正则化类型：1：单张影像的mean&std；2：整个数据集（训练集/测试集）的mean&std

        self.LOAD_TRAINED = False
        self.PATH_STATE_DICT = r"TripleNet-DATA3-20221114-best_epoch-52_fm-0.91629094.pth.tar"


        if self.TYPE_DATASET == 5:
            self.ValOrNot = False
        else:
            self.ValOrNot = True

