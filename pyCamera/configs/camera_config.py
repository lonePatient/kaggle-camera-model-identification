#encoding:utf-8
from os import path
import multiprocessing

#****************** PATH **********************************
# 主路径
BASE_DIR = 'pyCamera'

# 训练数据集
TRAIN_DIR = path.sep.join([BASE_DIR,'dataset/train'])
VAL_DIR = path.sep.join([BASE_DIR,'dataset/val'])
TEST_DIR = path.sep.join([BASE_DIR,'dataset/test'])

DATA_FILE_PATH = path.sep.join([BASE_DIR,'dataset/data_file.json'])
LABEL_PATH = path.sep.join([BASE_DIR,'dataset/class_ids.json'])

# 模型运行日志
LOG_DIR = path.sep.join([BASE_DIR,'output/log'])

# TSboard信息保存路径
WRITER_DIR = path.sep.join([BASE_DIR,'output/TSboard'])

# 图形保存路径
FIG_DIR = path.sep.join([BASE_DIR,'output/figure'])

# 模型保存路径
CHECKPOINT_DIR = path.sep.join([BASE_DIR,'output/checkpoints/{arch}'])

# pseudo_data
PSEUDO_DIR = path.sep.join([BASE_DIR,'dataset/pseudo'])
PSEUDO_PATH = path.sep.join([BASE_DIR,'dataset/pseudo.csv'])

# 提交文件信息
SUBMIT_FILE = path.sep.join([BASE_DIR,'output/result','submition.csv'])
#****************** model configs ******************
# 数据划分
TEST_SIZE = 0.2

# 类别个数
NUM_CLASSES = 10

#  GPU个数
#  如果只写一个数字，则表示gpu标号从0开始，
#  并且默认使用gpu:0作为controller
#  如果以列表形式表示，即[1,3,5],则
#  我们默认list[0]作为controller
N_GPU = [1,0]

# 线程个数
NUM_WORKERS = multiprocessing.cpu_count()

# 图像大小
IAMGE_SIZE = (224,224)

# 学习率
LR_PATIENCE = 3

# 动量
MOMENTUM = 0.9

# 权重衰减因子
WEIGHT_DECAY = 1e-4

# 标准化(imagenet)
MEAN = [0.485,0.456,0.406]
STD  = [0.229,0.224,0.225]

# TOP accuracy
TOPK = 1

# augmentation
CROP_SIZE = 224

#***************** callback *************

# 模式
WORLD_SIZE = 1 # number pf nodes for distributed training
MODE = 'min'
MONITOR = 'val_loss'

# early_stopping
EARLY_PATIENCE = 10

# checkpoint
SAVE_BEST_ONLY = True
BEST_NAME = '{arch}-best.pth'
EPOCH_NAME = '{arch}-{epoch}-{val_loss}.pth'
# 保存模型频率，当save_best_only为False时候，指定才有作用
SAVE_CHECKPOINTS_FREP = 10

# tta增强个数
TTA_COUNTS = 12

# make pseud and test label model list
MODELS = [
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext101',
    'dpn92',
    'dpn98',
    'se_resnet50',
    'se_resnet101',
    'se_resnext50',
]
