#encoding:utf-8
from glob import glob
from pyCamera.configs import camera_config as config
from pyCamera.utils.util import json_write
from sklearn.model_selection import train_test_split

# 数据划分
def main():
    trainPaths = glob('%s/*/*'%config.TRAIN_DIR)
    trainLabels = [p.split("/")[-2] for p in trainPaths]
    testPaths = glob('%s/*'%config.TEST_DIR)

    label_ids = {key:idx for idx,key in enumerate(list(set(trainLabels)))}
    split = train_test_split(trainPaths,trainLabels,test_size=config.TEST_SIZE,stratify=trainLabels,random_state=2018)
    (trainPaths,valPaths,_,_)  =  split

    # 写入文件
    data = {'train':trainPaths,
            'val':valPaths,
            'test':testPaths}
    json_write(data = data,filename = config.DATA_FILE_PATH)
    json_write(data = label_ids,filename = config.LABEL_PATH)

if __name__ == "__main__":
    main()

