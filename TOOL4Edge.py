# PyTorch
import torch
from torch.utils.data import Dataset

# Other
import os
import numpy as np
import random
from skimage import io
from tqdm import tqdm as tqdm
import cv2 as cv
from pandas import read_csv

from Method.Utils.GetMeanStdForNorm import get_mean_std_for_norm


def transformEdge(input_img, dilate=True):
    input_img = input_img.detach().cpu().numpy()
    input_img = np.squeeze(input_img).astype(np.uint8)
    input_shape=list(input_img.shape)
    if len(input_shape)==2:
        length=1
    else:
        length=input_shape[0]

    output=[]
    kernel = np.ones((3, 3), np.uint8)

    for i in range(length):
        if length==1:
            edges = cv.Canny(input_img*255, 30, 150)    #cv.Canny(img, 100, 200)
            if dilate==True:
                edges = cv.dilate(edges, kernel, iterations=1)
                output.append(edges)
            else:
                output.append(edges)
        else:
            img = np.squeeze(input_img[i,:,:])
            edges = cv.Canny(img*255, 30, 150)     #cv.Canny(img, 100, 200)
            if dilate == True:
                edges = cv.dilate(edges, kernel, iterations=1)
                output.append(edges)
            else:
                output.append(edges)
    return output


# Functions
def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""

    # crop if necesary
    I = I[:s[0], :s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])

    return np.pad(I, ((0, p0), (0, p1)), 'edge')


def read_img(path, NORMALISE, mode_NORMALISE=2, MEAN_STD=None):
    """
    Read image: all bands.
    :param path: 图片的路径
    :param NORMALISE: 是否正则化：True？False？
    :param mode_NORMALISE: 正则化类型：1：单张影像的mean&std；2：整个数据集（训练集/测试集）的mean&std
    :param MEAN_STD: mean&std数组，用于“2”型正则化
    :return: 读取并处理后的影像
    """
    I = io.imread(path).astype('float')

    if NORMALISE:
        if mode_NORMALISE==1:
            I = (I - I.mean()) / I.std()
        elif mode_NORMALISE==2:
            I = (I - MEAN_STD[0]) / MEAN_STD[1]

    return I


def read_img_and_cm(TYPE_DATASET, path, img_name, mode_file, NORMALISE, mode_NORMALISE=1, MEAN_STD_T1=None, MEAN_STD_T2=None):
    """Read image pair and change map."""
    # read images
    if TYPE_DATASET == 1 or TYPE_DATASET == 3 or TYPE_DATASET == 5:
        if mode_file == 'train' or mode_file == 'val':
            I1 = read_img(path + 't1/' + img_name + '.tif', NORMALISE, mode_NORMALISE, MEAN_STD_T1)
            I2 = read_img(path + 't2/' + img_name + '.tif', NORMALISE, mode_NORMALISE, MEAN_STD_T2)
            cm = io.imread(path + 'mask/' + img_name + '.tif', as_gray=True) != 0
            cm32 = io.imread(path + 'mask_32/' + img_name + '.tif', as_gray=True) != 0
            edge = io.imread(path + 'edge/' + img_name + '.tif', as_gray=True) != 0
            edge32 = io.imread(path + 'edge_32/' + img_name + '.tif', as_gray=True) != 0
        else:
            I1 = read_img(path + 't1/' + img_name + '.tif', NORMALISE, mode_NORMALISE, MEAN_STD_T1)
            I2 = read_img(path + 't2/' + img_name + '.tif', NORMALISE, mode_NORMALISE, MEAN_STD_T2)
            cm = io.imread(path + 'mask/' + img_name + '.tif', as_gray=True) != 0
            cm32 = io.imread(path + 'mask_32/' + img_name + '.tif', as_gray=True) != 0
            edge = io.imread(path + 'edge/' + img_name + '.tif', as_gray=True) != 0
            edge32 = io.imread(path + 'edge_32/' + img_name + '.tif', as_gray=True) != 0

    return I1, I2, cm, cm32, edge, edge32


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)


def get_weights(TYPE_DATASET, path, fp_modifier=1):
    fname = 'train.txt'  # FILE_TRAIN: 'train.txt'
    names = list(map(str, np.array(read_csv(path + fname, sep="\t")['index'])))

    n_pix = 0
    true_pix = 0

    for im_name in tqdm(names, position=0, desc="GET WEIGHTS"):
        if TYPE_DATASET == 1:
            cm = io.imread(path + 'mask/' + im_name + '.tif', as_gray=True) != 0
        elif TYPE_DATASET == 3:
            cm = io.imread(path + 'mask/' + im_name + '.tif', as_gray=True) != 0

        s = cm.shape
        n_pix += np.prod(s)
        true_pix += cm.sum()

    return [fp_modifier * (2 * true_pix / n_pix), 2 * (n_pix - true_pix) / n_pix]


class ChangeDetectionDataset(Dataset):
    """Change Detection Dataset class, used for both training and test data."""

    def __init__(self, TYPE_DATASET, path, train_val_test='test', patch_side=256, NORMALISE=True, MODE_NORMALISE=1, transform=None):
        """
        变化检测数据集类-构造函数
        :param TYPE_DATASET: 数据集类型 1:DSIFN-Dataset | 2:WHU-Building-Dataset | 3:LEVIR-CD | 4:S2Looking | 5:WHU-BCD
        :param path: 数据路径(精确到train/test/val/) eg. '../../../Dataset/DSIFN-Dataset/train/'
        :param train_val_test: 训练?验证?测试? train/val/test
        :param patch_side: patch大小 32?512?
        :param NORMALISE: 是否标准化处理图像? True?Flase?
        :param mode_NORMALISE: 正则化类型：1：单张影像的mean&std；2：整个数据集（训练集/测试集）的mean&std
        :param transform: 是否进行图像转换?
        """

        # basics
        self.type_dataset = TYPE_DATASET
        self.path = path
        self.train_val_test = train_val_test
        self.normalise = NORMALISE
        self.transform = transform

        if train_val_test=='train':
            fname = 'train.txt'  #FILE_TRAIN: 'train.txt'
        elif train_val_test=='val':
            fname = 'val.txt'   #FILE_VAL: 'val.txt'
        else:
            fname = 'test.txt'   #FILE_TEST: 'test.txt'

        self.names = list(map(str, np.array(read_csv(path + fname, sep="\t")['index'])))
        self.n_imgs = len(self.names)

        self.MODE_NORMALISE = MODE_NORMALISE
        self.MEAN_STD_T1 = None
        self.MEAN_STD_T2 = None
        if self.MODE_NORMALISE == 2:
            self.MEAN_STD_T1 = get_mean_std_for_norm(os.path.join(self.path, "t1/"))
            self.MEAN_STD_T2 = get_mean_std_for_norm(os.path.join(self.path, "t2/"))


    def get_img(self, im_name):
        img1, img2, cm, cm32, edge, edge32 = read_img_and_cm(TYPE_DATASET=self.type_dataset, path=self.path, img_name=im_name, mode_file=self.train_val_test, NORMALISE=self.normalise, mode_NORMALISE=self.MODE_NORMALISE, MEAN_STD_T1=self.MEAN_STD_T1, MEAN_STD_T2=self.MEAN_STD_T2)
        img1 = reshape_for_torch(img1)
        img2 = reshape_for_torch(img2)
        return img1, img2, cm, cm32, edge, edge32

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, idx):
        img1, img2, cm, cm32, edge, edge32 = read_img_and_cm(TYPE_DATASET=self.type_dataset, path=self.path, img_name=self.names[idx], mode_file=self.train_val_test, NORMALISE=self.normalise, mode_NORMALISE=self.MODE_NORMALISE, MEAN_STD_T1=self.MEAN_STD_T1, MEAN_STD_T2=self.MEAN_STD_T2)

        I1 = reshape_for_torch(img1)
        I2 = reshape_for_torch(img2)
        label = torch.from_numpy(1 * np.array(cm)).float()
        label32 = torch.from_numpy(1 * np.array(cm32)).float()
        edge = torch.from_numpy(1 * np.array(edge)).float()
        edge32 = torch.from_numpy(1 * np.array(edge32)).float()



        sample = {'I1': I1, 'I2': I2, 'label': label, 'label32': label32, 'edge': edge, 'edge32': edge32}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip(object):
    """Flip randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        if random.random() > 0.5:
            I1 = I1.numpy()[:, :, ::-1].copy()
            I1 = torch.from_numpy(I1)
            I2 = I2.numpy()[:, :, ::-1].copy()
            I2 = torch.from_numpy(I2)
            label = label.numpy()[:, ::-1].copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        n = random.randint(0, 3)
        if n:
            I1 = sample['I1'].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            I2 = sample['I2'].numpy()
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I2 = torch.from_numpy(I2)
            label = sample['label'].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kappa(tp, tn, fp, fn):
    N = tp + tn + fp + fn
    p0 = torch.true_divide((tp + tn) , N)
    pe = torch.true_divide(((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) , (N * N))

    return torch.true_divide((p0 - pe) , (1 - pe))

