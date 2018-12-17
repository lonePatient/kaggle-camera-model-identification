#encoding:utf-8
import torch
import argparse
import warnings
from torch.utils.data import DataLoader
from pyCamera.test.submit import TestSubmit
from pyCamera.utils.utils import json_read
from pyCamera.utils.logginger import init_logger
from pyCamera.configs import camera_config as config
from pyCamera.io.dataset import CreateDataset
from pyCamera.preprocessing.augmentation import Augmentator
warnings.filterwarnings("ignore")

def main():
    logger = init_logger(log_name='test',log_dir=config.LOG_DIR)
    logger.info('starting load test data from disk')
    data_files = json_read(filename = config.DATA_FILE_PATH)
    test_augmentator = Augmentator(is_train_mode = False,crop_size=config.CROP_SIZE)
    test_dataset = CreateDataset(data_path     = data_files['test'],
                                 label_path    = config.LABEL_PATH,
                                 augmentator   = test_augmentator,
                                 expand_dataset= True,
                                 tta_count     =config.TTA_COUNTS)
    test_loader = DataLoader(dataset    = test_dataset,
                             batch_size = args['batch_size'],
                             shuffle    = False,
                             drop_last  = False,
                             num_workers=config.NUM_WORKERS)
    logger.info('generater test label')
    submiter = TestSubmit(archs          = [x+"_pseudo" for x in config.MODELS],
                         logger          = logger,
                         files           = data_files['test'],
                         outfile         = config.SUBMIT_FILE,
                         test_data       = test_loader,
                         num_classes     = config.NUM_CLASSES,
                         tta_count       = config.TTA_COUNTS,
                         n_gpu           = config.N_GPU,
                         label_path      = config.LABEL_PATH,
                         checkpoint_dir  = config.CHECKPOINT_DIR
                    )
    submiter.submit()
    # 释放显存
    torch.cuda.empty_cache()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch model testing')
    ap.add_argument('-b',
                    '--batch_size',
                    required=True,
                    type = int,
                    help = 'Batch size for dataset iterators')
    args = vars(ap.parse_args())
    main()
