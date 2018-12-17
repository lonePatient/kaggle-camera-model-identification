#encoding:utf-8
import pandas as pd
import argparse
import torch
import random
import numpy as np
import warnings
from torch import optim
from torch.utils.data import DataLoader
from pyCamera.train.losses import CrossEntropy
from pyCamera.train.metrics import Accuracy
from pyCamera.train.trainer import Trainer
from pyCamera.utils.utils import json_read
from pyCamera.io.dataset import CreateDataset
from pyCamera.utils.logginger import init_logger
from pyCamera.configs import camera_config as config
from pyCamera.callback.lrscheduler import ReduceLROnPlateau
from pyCamera.model.cnn.makemodel import MakeModel
from pyCamera.preprocessing.augmentation import Augmentator
from pyCamera.callback.earlystopping import EarlyStopping
from pyCamera.callback.modelcheckpoint import ModelCheckpoint
from pyCamera.callback.trainingmonitor import TrainingMonitor
warnings.filterwarnings("ignore")

# 主函数
def main(arch):
    logger = init_logger(log_name=arch, log_dir=config.LOG_DIR)
    logger.info("seed is %d"%args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    # 加载数据集
    logger.info('starting load train data from disk')
    data_files = json_read(filename = config.DATA_FILE_PATH)
    # 加载pseudo数据
    if args['use_pseudo']:
        pseudo = pd.read_csv(config.PSEUDO_PATH)['fname'].tolist()
        data_files['train'].extend(pseudo)

    train_augmentator = Augmentator(is_train_mode=True,crop_size  = config.CROP_SIZE)
    val_augmentator   = Augmentator(is_train_mode=False,crop_size = config.CROP_SIZE)

    train_dataset = CreateDataset(data_path     = data_files['train'],
                                  augmentator   = train_augmentator,
                                  label_path    = config.LABEL_PATH)

    val_dataset   = CreateDataset(data_path     = data_files['val'],
                                  augmentator   = val_augmentator,
                                  tta_count     = config.TTA_COUNTS,
                                  expand_dataset= True,
                                  label_path    = config.LABEL_PATH)

    train_loader = DataLoader(dataset     = train_dataset,
                              batch_size  = args['batch_size'],
                              shuffle     = True,
                              drop_last   = True,
                              num_workers = config.NUM_WORKERS,
                              pin_memory  = False)

    val_loader = DataLoader(dataset     = val_dataset,
                            batch_size  = args['batch_size'],
                            num_workers = config.NUM_WORKERS,
                            pin_memory  = False)
    # 初始化模型和优化器
    logger.info("initializing model")
    model = MakeModel(num_classes = config.NUM_CLASSES,
                      arch        = arch,
                      pretrained  = args['pretrained'])

    optimizer = optim.Adam(params       = model.parameters(),
                           lr           = args['learning_rate'],
                           weight_decay = config.WEIGHT_DECAY
                           )
    logger.info("initializing callbacks")
    # 模型保存
    model_checkpoint = ModelCheckpoint(checkpoint_dir   = config.CHECKPOINT_DIR,
                                       mode             = config.MODE,
                                       monitor          = config.MONITOR,
                                       save_best_only   = config.SAVE_BEST_ONLY,
                                       best_model_name  = config.BEST_NAME,
                                       epoch_model_name = config.EPOCH_NAME,
                                       arch             = arch,
                                       logger           = logger)
    # eraly_stopping功能
    early_stop = EarlyStopping(mode     = config.MODE,
                               patience = config.EARLY_PATIENCE,
                               monitor  = config.MONITOR)
    # 监控训练过程
    train_monitor = TrainingMonitor(fig_dir  = config.FIG_DIR,
                                    json_dir = config.LOG_DIR,
                                    arch     = arch)
    #学习率模式
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                     factor   = 0.5,
                                     patience = config.LR_PATIENCE,
                                     min_lr   = 1e-9,
                                     epsilon  = 1e-5,
                                     verbose  =1,
                                     mode     = config.MODE)
    # 初始化模型训练器
    logger.info('training model....')
    trainer = Trainer(model            = model,
                      train_data       = train_loader,
                      val_data         = val_loader,
                      optimizer        = optimizer,
                      epochs           = args['epochs'],
                      criterion        = CrossEntropy(),
                      metric           = Accuracy(topK=config.TOPK),
                      logger           = logger,
                      model_checkpoint = model_checkpoint,
                      training_monitor = train_monitor,
                      early_stopping   = early_stop,
                      resume           = args['resume'],
                      lr_scheduler     = lr_scheduler,
                      n_gpu            = config.N_GPU)
    # 查看模型结构
    trainer.summary()
    # 拟合模型
    trainer.train()

    # 释放显存
    torch.cuda.empty_cache()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-s',
                    '--seed',
                    default=20180209,
                    type = int,
                    help = 'Seed for initializing training.')
    ap.add_argument('-b',
                    '--batch_size',
                    required=True,
                    type = int,
                    help = 'Batch size for dataset iterators')
    ap.add_argument('-p',
                    '--pretrained',
                    default=False,
                    type = bool,
                    help = 'Choose whether using pretrained weight to initializing weights of model')
    ap.add_argument('-l',
                    '--learning_rate',
                    default=True,
                    type = float,
                    help = 'Learning rate.')
    ap.add_argument('-u',
                    '--use_pseudo',
                    default=False,
                    type=bool,
                    help = 'Choose whether add pseudo labels data')
    ap.add_argument('-r',
                    '--resume',
                    default=False,
                    type = bool,
                    help = 'Choose whether resume checkpoint model')
    ap.add_argument('-e',
                    '--epochs',
                    required=True,
                    type = int,
                    default=100,
                    help = 'Number of epochs to train')
    args = vars(ap.parse_args())

    print('Training total of {} models'.format(len(config.MODELS)))
    for i, model_name in enumerate(config.MODELS):
        if args['use_pseudo']:
            model_name = model_name + '_pseudo'
        print('{}/{}: Training {} '.format(i + 1, len(config.MODELS), model_name))
        main(arch = model_name)
