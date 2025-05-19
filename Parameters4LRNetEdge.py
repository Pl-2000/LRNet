# Parameters in TripleNet's main script
class Parameters:
    def __init__(self):
        # Fixed Parameters
        self.type_sub_model = "VGG16"   #"VGG16", "ResNet34", "ResNet50", "ResNet101", "ResNet152"
        self.N_EPOCHS = 200             # 200
        self.BATCH_SIZE = 16            # T-UNet-16, DESSN-16, DSIFN-8
        self.PATCH_SIDE = 256
        self.NORMALISE_IMGS = True
        self.DATA_AUG = False


        # Hyper Parameters
        self.cuda_device_id = 3  # 3:huan.zhong
        self.TYPE_DATASET = 5  # 0:TEST | 1:DSIFN-Dataset | 2:WHU-Building-Dataset | 3:LEVIR-CD | 4:S2Looking | 5:WHU-BCD

        self.INIT_LR = 0.0001
        self.lrnet_cos_sim_threshold = 0.5  #cos相似度阈值，最小值，大于阈值为相似           0.4
        self.lrnet_label_threshold = 0.5    #标签阈值，大于阈值为changed，反之为unchanged         0.5
        self.beta = 0.05                    #标签平滑系数/软标签软化程度label_smoothing_para_beta            0.05
        self.theta = 0                      #软硬标签loss分配中硬标签占比           0.5

        self.LOAD_TRAINED = False
        self.PATH_STATE_DICT = r""


        if self.TYPE_DATASET == 5:
            self.MODE_NORMALISE = 2
            self.ValOrNot = False
        else:
            self.MODE_NORMALISE = 1     # 正则化类型：1：单张影像的mean&std；2：整个数据集（训练集/测试集）的mean&std
            self.ValOrNot = True
        self.test_id={1:['2','29','69','72','78','81'], 3:['15','42','43','114','400','2040'], 5:['27','256','257','757','1353','1491']}