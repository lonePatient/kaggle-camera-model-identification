#encoding:utf-8
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from ..utils.train_utils import restore_checkpoint,model_device

# 单个模型进行预测
class Predicter(object):
    def __init__(self,
                 model,
                 test_data,
                 logger,
                 checkpoint_path,
                 n_gpu = 0):

        self.model           = model
        self.test_data       = test_data
        self.logger          = logger
        self.checkpoint_path = checkpoint_path
        self.n_gpu           = n_gpu
        self._reset()

    # 重载模型
    def _reset(self):
        self.batch_num = len(self.test_data)
        self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model, logger=self.logger)
        if self.checkpoint_path:
            self.logger.info("\nLoading checkpoint: {} ...".format(self.checkpoint_path))
            self.model, _, _, _ = restore_checkpoint(resume_path=self.checkpoint_path,model=self.model)
            self.logger.info("\nCheckpoint '{}' loaded".format(self.checkpoint_path))

    # batch预测
    def _predict_batch(self,X):
        with torch.no_grad():
            X = X.to(self.device)
            y_pred = self.model(X)
            y_pred = F.softmax(y_pred, dim=-1)
            return y_pred.cpu().numpy()

    #预测test数据集
    def predict(self):
        self.model.eval()
        predictions = []
        for batch_idx,(X,y) in tqdm(enumerate(self.test_data),total=len(self.test_data),desc='test_data'):
            y_pred_batch = self._predict_batch(X)
            predictions.append(y_pred_batch)
        return np.concatenate(predictions)