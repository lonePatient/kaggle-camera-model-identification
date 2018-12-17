#encoding:utf-8
import os
import shutil
from tqdm import tqdm
from os import path
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from .predicter import Predicter
from ..utils.predict_utils import geometric_mean
from ..utils.utils import json_read
from ..model.cnn.makemodel import MakeModel

class PseudoLabel(object):
    def __init__(self,
                 archs,
                 test_data,
                 logger,
                 files,
                 outfile,
                 label_path,
                 pseudo_dir,
                 checkpoint_dir,
                 num_models  = 1,
                 num_classes = 10,
                 tta_count   = 12,
                 n_gpu       = 0
                 ):

        self.archs          = archs             # 模型列表
        self.num_models     = num_models   # top n checkpoints
        self.test_loader    = test_data   # 数据集
        self.num_classes    = num_classes # 类别个数
        self.logger         = logger           # 日志
        self.tta_count      = tta_count     # tta总数
        self.files          = files
        self.outfile        = outfile
        self.checkpoint_dir = checkpoint_dir
        self.pseudo_dir     = pseudo_dir
        self.n_gpu          = n_gpu


        self.id_labels = {value:key for key,value in json_read(label_path).items()}

    # 获取单个模型的weight列表
    # def _checkpoints(self,arch):
    #     paths = defaultdict(list)
    #     all_checkpoints = os.listdir(self.checkpoint_path.format(arch = arch))
    #     val_losses = np.array([name[:-4].split('_')[-1] for name in all_checkpoints], dtype=float)
    #     indices = np.argpartition(val_losses, self.num_models)[:self.num_models]
    #     weight_names = [all_checkpoints[idx] for idx in indices]
    #     paths[arch] = [self.checkpoint_path.format(arch) + '/' + name for name in weight_names]
    #     return paths

    # 获取所有模型的权重
    # 如果保存的是best模型，则获取所有模型的best weight path
    # 如果保存的是epoch checkpont，则根据val_loss获取topK模型进行计算
    def _weights(self):
        weights = defaultdict(list)
        for i,arch in enumerate(self.archs):
            for filename in glob(path.sep.join([self.checkpoint_dir.format(arch = arch),"*.pth"])):
                weights[arch].append(filename)
        return weights

    # 转化为标签
    def _generator_pseudo(self,predicts):
        gm = geometric_mean(predicts, test_size=(len(self.files), len(self.id_labels)))
        labels = gm.argmax(axis=-1)
        labels = np.array([self.id_labels[x] for x in labels])
        df = pd.DataFrame()
        df['classes'] = labels
        df['fname']   = self.files
        return df

    # 每个模型的tta预测
    def _predict_one_moddel(self,arch,weight):
        model = MakeModel(arch = arch,num_classes=self.num_classes,pretrained=False)
        predicter = Predicter(model= model,
                              test_data = self.test_loader,
                              logger = self.logger,
                              checkpoint_path = weight,
                              n_gpu = self.n_gpu)
        predictions = predicter.predict()
        results = []
        # results保存的是每一种data augmentation的结果
        for i in range(self.tta_count):
            results.append(predictions[i::self.tta_count])
        predictions = geometric_mean(results,test_size = (len(self.files),len(self.id_labels)))
        return predictions

    # 将pseudo信息写入文件中
    def _move_pseudo_to_separate_folder(self,pseudo_proba_df):
        cats = pseudo_proba_df['classes']
        fnames =  pseudo_proba_df['fname']
        image_paths = []
        for c, n in tqdm(zip(cats,fnames)):
            os.makedirs(self.pseudo_dir + '/{}'.format(c), exist_ok=True)
            image_name = n.split("/")[-1]
            image_path = self.pseudo_dir + '/{}/{}'.format(c, image_name)
            shutil.copyfile(n, image_path)
            image_paths.append(image_path)
        pseudo_paths = pd.DataFrame({'fname': image_paths})
        return pseudo_paths

    # 产生pseudo label
    def save_pseudo(self):
        predicts = []
        weights = self._weights()
        for arch in self.archs:
            for weight in weights[arch]:
                predicts.append(self._predict_one_moddel(arch = arch,weight=weight))
        final_proba = self._generator_pseudo(predicts=predicts)
        pseudo_path = self._move_pseudo_to_separate_folder(final_proba)
        pseudo_path.to_csv(self.outfile,index=False)
