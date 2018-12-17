#encoding:utf-8
from os import path
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
import copy
from .predicter import Predicter
from ..utils.predict_utils import geometric_mean
from ..utils.utils import json_read
from ..model.cnn.makemodel import MakeModel

class TestSubmit(object):
    def __init__(self,
                 archs,
                 test_data,
                 logger,
                 files,
                 outfile,
                 label_path,
                 checkpoint_dir,
                 num_models=1,
                 num_classes = 10,
                 tta_count = 12,
                 n_gpu=0
                 ):

        self.archs          = archs  # 模型列表
        self.num_models     = num_models # top n checkpoints
        self.test_loader    = test_data #数据集
        self.num_classes    = num_classes #类别个数
        self.logger         = logger           # 日志
        self.tta_count      = tta_count     # tta总数
        self.files          = files
        self.outfile        = outfile
        self.checkpoint_dir = checkpoint_dir
        self.n_gpu          = n_gpu


        self.id_labels = {value:key for key,value in json_read(label_path).items()}

    def _get_weights(self):
        weights = []
        for i,arch in enumerate(self.archs):
            for filename in glob(path.sep.join([self.checkpoint_dir.format(arch = arch),"*.pth"])):
                weights.append((arch,filename))
        return weights

    # 转化为标签
    def _get_predicts(self,predicts, coefficients):
        predicts = copy.deepcopy(predicts)
        for i in range(len(coefficients)):
            predicts[:, i] *= coefficients[i]
        return predicts

    # 统计标签分布
    def _get_labels_distribution(self,predicts, coefficients):
        predicts = self._get_predicts(predicts, coefficients)
        labels = predicts.argmax(axis=-1)
        counter = Counter(labels)
        return labels, counter

    # 计算得分
    def _compute_score_with_coefficients(self,predicts, coefficients):
        _, counter = self._get_labels_distribution(predicts, coefficients)
        score = 0.
        for label in range(len(self.id_labels)):
            score += min(100. * counter[label] / len(predicts), len(self.id_labels))
        return score

    def _find_best_coefficients(self,predicts, alpha=0.001, iterations=10000):
        coefficients = [1] * len(self.id_labels)

        best_coefficients = coefficients[:]
        best_score = self._compute_score_with_coefficients(predicts, coefficients)

        for _ in range(iterations):
            _, counter = self._get_labels_distribution(predicts, coefficients)
            labels_distribution = map(lambda x: x[1], sorted(counter.items()))
            label = np.argmax(labels_distribution)
            coefficients[label] -= alpha
            score = self._compute_score_with_coefficients(predicts, coefficients)
            if score > best_score:
                best_score = score
                best_coefficients = coefficients[:]
        return best_coefficients

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

    def submit(self):
        predicts = []
        weights = self._get_weights()
        self.logger.info("ensemble %d models result"%len(weights))
        for arch,weight in weights:
            predicts.append(self._predict_one_moddel(arch = arch,weight=weight))
        final_proba = geometric_mean(predicts, test_size=(len(self.files), len(self.id_labels)))
        coefficients = self._find_best_coefficients(final_proba)
        labels, _ = self._get_labels_distribution(final_proba, coefficients)
        labels = list(map(lambda label: self.id_labels[label], labels))
        df = pd.DataFrame({
            "fname": [x.split("/")[-1] for x in self.files],
            "camera": labels
        })
        df.to_csv(self.outfile,index = False)
