#encoding:utf-8
import numpy as np

# 几何平均
def geometric_mean(list_preds,test_size):
    result = np.ones(test_size)
    for predict in list_preds:
        result *= predict
    result **= 1. / len(list_preds)
    return result