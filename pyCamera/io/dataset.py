#encoding:utf-8
import copy
import numpy as np
#import jpeg4py as jpeg
from PIL import Image
from torch.utils.data import Dataset
from ..utils.utils import json_read

class CreateDataset(Dataset):
    '''
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    '''
    def __init__(self,data_path,
                    augmentator,
                    label_path,
                    tta_count = None,
                    expand_dataset = False
                 ):

        self.data_path = data_path
        self._augmentator = augmentator
        self.expand_dataset = expand_dataset
        self.tta_count = tta_count
        self.label_id = json_read(label_path)

        self._restart()

    def _restart(self):
        self._files = []
        self._labels = []
        for record in self.data_path:
            file_path = record
            label = record.split("/")[-2]
            label = self.label_id.get(label,-1)
            self._files.append(file_path)
            self._labels.append(label)
        # 训练数据集不做tta，使用-1表示
        self._aug_labels = [-1] * len(self._files)
        # 一般对val和test数据做TTA，
        # 这时候，需要指定TTA的个数
        if self.expand_dataset:
            files = self._files
            labels = self._labels
            self._files = []
            self._labels = []
            self._aug_labels = []
            for index in range(len(files)):
                self._files.extend([files[index]] * self.tta_count)
                self._labels.extend([labels[index]] * self.tta_count)
                self._aug_labels.extend(list(range(self.tta_count)))

            del files
            del labels

    # 读取数据，数据预处理
    def _preprocess(self,index):
        path = self._files[index]
        image = Image.open(path)
        image = np.array(image)
        # 使用jpeg可以快速加载
        # if ".tif" not in path:
        #     image = jpeg.JPEG(path).decode()
        # else:
        #     image = Image.open(path)
        #     image = np.array(image)
        #　增强
        return self._augmentator(image,self._aug_labels[index])

    def get_label(self):
        return copy.deepcopy(self._labels)

    # 由于我们继承了Dataset类，所以我们需要重写len方法，
    # 该方法主要计算dataset的大小
    # getitem方法主要支持从0到len（self）的索引，即每次如何读取数据
    def __getitem__(self, index):
        image = self._preprocess(index)
        label = self._labels[index]
        return image,label

    def __len__(self):
        return len(self._files)



