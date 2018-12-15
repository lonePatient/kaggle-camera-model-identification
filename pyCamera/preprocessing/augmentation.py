#encoding:utf-8
import cv2
import numpy as np
import random
from PIL import Image
from io import BytesIO
from torchvision import transforms

class Augmentator(object):
    def __init__(self,crop_size = None , is_train_mode = True, proba = 0.5):
        self.mode = is_train_mode
        self.proba = proba
        self.crop_size = crop_size
        self.augs = []
        self._transforms = self._get_transforms()
        self._reset()

    # 总的增强列表
    def _reset(self):
        self.augs.append(lambda image: self._compression(image, qulity_factor=70))
        self.augs.append(lambda image: self._compression(image, qulity_factor=90))

        self.augs.append(lambda image: self._gamma_correction(image,gamma=0.8))
        self.augs.append(lambda image: self._gamma_correction(image, gamma=1.2))

        self.augs.append(lambda image: self._resize(image, scale=0.5))
        self.augs.append(lambda image: self._resize(image, scale=0.8))
        self.augs.append(lambda image: self._resize(image, scale=1.5))
        self.augs.append(lambda image: self._resize(image, scale=2.0))

        self.augs.append(lambda image: self._random_d4(image))

    def _compression(self,image,qulity_factor):
        '''
        压缩图片
        :param qulity_factor:
        :return:
        '''
        buffer = BytesIO()
        image = Image.fromarray(image)
        image.save(buffer,format = "jpeg",quality = qulity_factor)
        buffer.seek(0)
        img = Image.open(buffer)
        np_img = np.array(img)
        del image
        del buffer
        return np_img

    # 图像缩放
    def _resize(self,image,scale,interpolation = cv2.INTER_CUBIC):
        '''
        :param image:
        :param scale: 比例因子
        :param interpolation: 插值方法
        :return:
        '''
        # dsize : 输出图像尺寸
        # 当dsize为（0,0）时，尺寸分别由scale因子算出
        np_img = cv2.resize(image,dsize = (0,0),fx = scale,fy = scale,interpolation = interpolation)
        return np_img

    # gamma校正
    def _gamma_correction(self,image,gamma):
        '''
        这个是针对相机而言，具体的可以搜索关键词查看，
        简单来说，相机内部等进行图像保存时，会隐式地进行图片的Gamma Encode来
        模拟人眼的亮度检测规律。当我们需要对屏幕上对真实的亮度进行显示时，需要进行
        Gamma correction来消除encode带来的影响
        '''
        np_img = np.uint8(cv2.pow(image / 255.,gamma) * 255.)
        return np_img

    # 随机旋转图像
    def _random_rotate(self,img):
        rows, cols, c = img.shape
        angle = np.random.choice([0, 90, 180, 270])
        if angle == 0:
            return img
        # 主要获得图像绕着某一点的旋转矩阵
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        # 实现坐标系仿射变换
        img = cv2.warpAffine(img, M, (cols, rows))
        return img

    # 随机水平翻转，即图像左右对调
    def _random_horizontal_flip(self,img):
        if np.random.random() > 0.5:
            return img
        return cv2.flip(img, 1)

    # 随机的将图像竖直翻转，即图像的上下对调
    def _random_vertical_flip(self,img):
        if np.random.random() > 0.5:
            return img
        return cv2.flip(img, 0)

    # 随机裁剪出crop_size大小
    def  _random_crop(self,img):
        h, w, c = img.shape
        if h == self.crop_size and w == self.crop_size:
            i,j =  0, 0
        else:
            i = 0 if h == self.crop_size else np.random.randint(0, h - self.crop_size)
            j = 0 if w == self.crop_size else np.random.randint(0, w - self.crop_size)
        return img[i: i + self.crop_size, j: j + self.crop_size]

    # 5 crop
    def  _five_crop(self,img):
        h, w, c = img.shape
        assert c == 3, 'Something wrong with channels order'
        if self.crop_size > w or self.crop_size > h:
            raise ValueError(
                "Requested crop size {} is bigger than input size {}".format(self.crop_size, (h, w)))
        tl = img[0: self.crop_size, 0: self.crop_size]
        tr = img[0: self.crop_size, w - self.crop_size: w]
        bl = img[h - self.crop_size: h, 0: self.crop_size]
        br = img[h - self.crop_size: h, w - self.crop_size: w]
        center = self._center_crop(img)
        return tl, tr, bl, br, center

    # 中心裁剪
    def _center_crop(self,img):
        h, w, c = img.shape
        if h == self.crop_size and w == self.crop_size:
            i,j =  0, 0
        else:
            th = tw = self.crop_size
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
        return img[i: i + self.crop_size, j: j + self.crop_size]

    # 转化
    def _get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])
        ])

    #随机d4：Dihedral group
    def _random_d4(self,img):
        img = self._random_horizontal_flip(img)
        img = self._random_vertical_flip(img)
        img = self._random_rotate(img)
        return img

    def __call__(self,image,aug_type):
        '''
        用aug_type区分数据
        '''
        # TTA模式
        if 0 <= aug_type <= 8:
            aug = self.augs[aug_type]
            image = aug(image)
        elif aug_type == 9:
            image = np.rot90(image)
        elif aug_type == 10:
            image = np.rot90(image,k = 2)
        elif aug_type == 11:
            image  = np.rot90(image,k = -1)

        # 训练模式
        if self.mode and  random.random() < self.proba:
            aug = random.choice(self.augs)
            image = aug(image)

        image = self._transforms(image)
        return image
