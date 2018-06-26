# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


PATH = "/home/user/.kaggle/competitions/dog-breed-identification/"
sz=224
import pandas as pd
folder = "train/"
csv_fname = "labels.csv"
#pd.read_csv(PATH+csv_fname)

#
from sklearn.model_selection import train_test_split
image_folder = "train"
files = os.listdir(f'{PATH}{folder}')

n = len(files)
train_idxs, val_idxs = train_test_split(np.arange(n))
arch=resnet34
data = ImageClassifierData.from_csv(PATH, folder = image_folder,
                                    csv_fname = PATH+csv_fname,
                                    val_idxs = val_idxs,
                                    suffix = ".jpg",
                                    tfms=tfms_from_model(arch, sz),
                                   test_name = "test")
learn = ConvLearner.pretrained(arch, data, precompute=True)
print(learn)