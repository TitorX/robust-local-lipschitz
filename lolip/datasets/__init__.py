import os

import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from autovar.base import RegisteringChoiceType, register_var, VariableClass


class ImgSet:
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.X)


def load_B():
    IMG_PATH = '/content/drive/My Drive/Adversarial Machine Learning/Data/B Dataset/images'
    LAB_PATH = '/content/drive/My Drive/Adversarial Machine Learning/Data/B Dataset/DatasetB.xlsx'

    labels = pd.read_excel(LAB_PATH, dtype={'Image': str})

    X = []
    for f in labels['Image']:
        f = '{}.png'.format(f)
        img = Image.open(os.path.join(IMG_PATH, f))
        img = img.convert('RGB')
        X.append(img)

    ind = np.ones(len(X), dtype=bool)
    ind[[39, 41, 161, 162]] = False
    X = np.array(X, dtype=object)[ind]
    y = (labels['Type'].values == 'Malignant').astype(int)[ind]
    return X, y


def load_BUS():
    IMG_PATH = '/content/drive/My Drive/Adversarial Machine Learning/Data/BUS Dataset/Images_256pix'
    LAB_PATH = '/content/drive/My Drive/Adversarial Machine Learning/Data/BUS Dataset/BUS562_info.xlsx'

    labels = pd.read_excel(LAB_PATH)
    y = labels['Tumor Type'].values

    X = []
    for f in labels['Image Name']:
        f = 'case{}.bmp'.format(int(f[4:]))
        img = Image.open(os.path.join(IMG_PATH, f))
        img = img.convert('RGB')
        X.append(img)

    X = np.array(X, dtype=object)
    y = (labels['Tumor Type'].values == 'M').astype(int)
    return X, y


def load_BUSI():
    IMG_PATH = '/content/drive/My Drive/Adversarial Machine Learning/Data/BUSI Dataset/images'
    LAB_PATH = '/content/drive/My Drive/Adversarial Machine Learning/Data/BUSI Dataset/BUSI_labels.xlsx'

    labels = pd.read_excel(LAB_PATH)

    X = []
    for f in range(1, len(labels)+1):
        f = 'case{}.png'.format(f)
        img = Image.open(os.path.join(IMG_PATH, f))
        img = img.convert('RGB')
        X.append(img)

    ind = np.ones(len(X), dtype=bool)
    ind[471] = False

    X = np.array(X, dtype=object)[ind]
    y = (labels['Type'].values == 'Malignant').astype(int)[ind]
    return X, y


def load_mixed(mixed=(load_B, load_BUS, load_BUSI)):
    all_X = []
    all_y = []
    for load in mixed:
        X, y = load()
        all_X.append(X)
        all_y.append(y)

    return np.concatenate(all_X), np.concatenate(all_y)


TRANS = T.Compose([T.Resize(224, 224), T.ToTensor()])

DEBUG = int(os.environ.get('DEBUG', 0))

class DatasetVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines the dataset to use"""
    var_name = 'dataset'

    @register_var(argument=r"b", shown_name="b")
    @staticmethod
    def b(auto_var, var_value, inter_var):
        X, y = load_B()
        loader = DataLoader(
            ImgSet(X, y, TRANS,
            batch_size=len(X)
        )
        X, y = next(iter(loader))
        X = X.permute(2, 0, 1).numpy()
        y = y.numpy()

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=11)
        return x_train, y_train, x_test, y_test

    @register_var(argument=r"bus", shown_name="bus")
    @staticmethod
    def bus(auto_var, var_value, inter_var):
        X, y = load_BUS()
        loader = DataLoader(
            ImgSet(X, y, TRANS,
            batch_size=len(X)
        )
        X, y = next(iter(loader))
        X = X.permute(2, 0, 1).numpy()
        y = y.numpy()

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=11)
        return x_train, y_train, x_test, y_test

    @register_var(argument=r"busi", shown_name="busi")
    @staticmethod
    def busi(auto_var, var_value, inter_var):
        X, y = load_BUSI()
        loader = DataLoader(
            ImgSet(X, y, TRANS,
            batch_size=len(X)
        )
        X, y = next(iter(loader))
        X = X.permute(2, 0, 1).numpy()
        y = y.numpy()

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=11)
        return x_train, y_train, x_test, y_test

    @register_var(argument=r"mixed", shown_name="mixed")
    @staticmethod
    def mixed(auto_var, var_value, inter_var):
        X, y = load_mixed()
        loader = DataLoader(
            ImgSet(X, y, TRANS,
            batch_size=len(X)
        )
        X, y = next(iter(loader))
        X = X.permute(2, 0, 1).numpy()
        y = y.numpy()

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=11)
        return x_train, y_train, x_test, y_test

    @register_var(argument=r"mnist", shown_name="mnist")
    @staticmethod
    def mnist(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"cifar10", shown_name="Cifar10")
    @staticmethod
    def cifar10(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"svhn", shown_name="SVHN")
    @staticmethod
    def svhn(auto_var, var_value, inter_var):
        from torchvision.datasets import SVHN

        trn_svhn = SVHN("./data/", split='train', download=True)
        tst_svhn = SVHN("./data/", split='test', download=True)

        x_train, y_train, x_test, y_test = [], [], [], []
        for x, y in trn_svhn:
            x_train.append(np.array(x).reshape(32, 32, 3))
            y_train.append(y)
        for x, y in tst_svhn:
            x_test.append(np.array(x).reshape(32, 32, 3))
            y_test.append(y)
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        x_test, y_test = np.asarray(x_test), np.asarray(y_test)

        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"resImgnet112v3", shown_name="Restricted ImageNet")
    @staticmethod
    def resImgnet112v3(auto_var, inter_var, eval_trn=False):
        if eval_trn:
            trn_ds = ImageFolder("./data/RestrictedImgNet/train",
                transform=transforms.Compose([
                    transforms.Resize(72),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                ]))
        else:
            trn_ds = ImageFolder("./data/RestrictedImgNet/train",
                transform=transforms.Compose([
                    transforms.Resize(72),
                    transforms.RandomCrop(64, padding=8),
                    transforms.ToTensor(),
                ]))
        tst_ds = ImageFolder("./data/RestrictedImgNet/val",
            transform=transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]))
        return trn_ds, tst_ds
