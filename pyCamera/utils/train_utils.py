#encoding:utf-8
import os
import torch
from .utils import prepare_device

def restore_checkpoint(resume_path,model = None,optimizer = None):
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    if model:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model,optimizer,best,start_epoch

def model_device(n_gpu,model,logger):
    device, device_ids = prepare_device(n_gpu,logger)
    if len(device_ids) > 1:
        logger.info("current {} GPUs".format(len(device_ids)))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model,device