#encoding:utf-8
import os
import time
import numpy as np
import torch
from ..callback.progressbar import ProgressBar
from ..utils.utils import AverageMeter
from ..utils.train_utils import restore_checkpoint,model_device

# 训练包装器
class Trainer(object):
    def __init__(self,model,
                 train_data,
                 val_data,
                 optimizer,
                 epochs,
                 logger,
                 metric,
                 criterion,
                 n_gpu            = None,
                 lr_scheduler     = None,
                 resume           = None,
                 model_checkpoint = None,
                 training_monitor = None,
                 early_stopping   = None,
                 writer           = None,
                 verbose = 1):
        self.model            = model
        self.train_data       = train_data
        self.val_data         = val_data
        self.epochs           = epochs
        self.optimizer        = optimizer
        self.logger           = logger
        self.verbose          = verbose
        self.writer           = writer
        self.training_monitor = training_monitor
        self.early_stopping   = early_stopping
        self.resume           = resume
        self.model_checkpoint = model_checkpoint
        self.lr_scheduler     = lr_scheduler
        self.criterion        = criterion
        self.metric           = metric
        self.n_gpu            = n_gpu

        self._reset()

    def _reset(self):
        self.batch_num         = len(self.train_data)
        self.progressbar       = ProgressBar(n_batch = self.batch_num)
        self.model,self.device = model_device(n_gpu=self.n_gpu,model = self.model,logger = self.logger)
        self.start_epoch = 1
        # 重载模型，进行训练
        if self.resume:
            if '_pseudo' in self.model_checkpoint.arch:
                arch = self.model_checkpoint.arch.split('_pseudo')[0]
            else:
                arch = self.model_checkpoint.arch
            resume_path = os.path.join(self.model_checkpoint.checkpoint_dir.format(arch = arch),
                                       self.model_checkpoint.best_model_name.format(arch = arch))
            self.logger.info("\nLoading checkpoint: {} ...".format(resume_path))
            self.model, self.optimizer, best, self.start_epoch = restore_checkpoint(resume_path = resume_path,
                                                                     model = self.model,
                                                                     optimizer = self.optimizer
                                                                     )
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info("\nCheckpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # for p in model_parameters:
        #     print(p.size())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # 总的模型参数量
        self.logger.info('Model {}: trainable parameters: {:4}M'.format(self.model.__get_name(),params / 1000 / 1000))
        # 模型结构
        self.logger.info(self.model)

    # 保存模型信息
    def _save_info(self,epoch,val_loss):
        state = {
            'arch': self.model_checkpoint.arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': round(val_loss,4)
        }
        return state

    # val数据集预测
    def _valid_epoch(self):
        self.model.eval()
        val_losses = AverageMeter()
        val_acc = AverageMeter()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                acc = self.metric(output = output,target=target)
                val_losses.update(loss.item(),data.size(0))
                val_acc.update(acc.item(),data.size(0))
        return {
            'val_loss': val_losses.avg,
            'val_acc': val_acc.avg
        }

    # epoch训练
    def _train_epoch(self):
        self.model.train()
        train_loss = AverageMeter()
        train_acc  = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.train_data):
            start  = time.time()
            data   = data.to(self.device)
            target = target.to(self.device)

            outputs = self.model(data)
            loss    = self.criterion(output = outputs,target=target)
            acc     = self.metric(output=outputs,target=target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item(),data.size(0))
            train_acc.update(acc.item(),data.size(0))
            if self.verbose >= 1:
                self.progressbar.step(batch_idx=batch_idx,
                                      loss     = loss.item(),
                                      acc      = acc.item(),
                                      use_time = time.time() - start)
        print("\ntraining result:")
        train_log = {
            'loss': train_loss.avg,
            'acc': train_acc.avg
        }
        return train_log

    def train(self):
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            print("----------------- training start -----------------------")
            print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch+self.epochs -1))
            train_log = self._train_epoch()
            val_log = self._valid_epoch()
            logs = dict(train_log,**val_log)
            self.logger.info('\nEpoch: %d - loss: %.4f acc: %.4f - val_loss: %.4f - val_acc: %.4f'%(
                            epoch,logs['loss'],logs['acc'],logs['val_loss'],logs['val_acc'])
                             )
            if self.lr_scheduler:
                self.lr_scheduler.step(logs['loss'],epoch)
            if self.training_monitor:
                self.training_monitor.step(logs)
            if self.model_checkpoint:
                state = self._save_info(epoch,val_loss = logs['val_loss'])
                self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor],state = state)
            if self.writer:
                self.writer.set_step(epoch,'train')
                self.writer.add_scalar('loss', logs['loss'])
                self.writer.add_scalar('acc', logs['acc'])
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', logs['val_loss'])
                self.writer.add_scalar('val_acc', logs['val_acc'])
            if self.early_stopping:
                self.early_stopping.step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break

