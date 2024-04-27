import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import SimpleITK as sitk
import glob
import os

def _label_decomp(label_vol, num_cls):
    """
    decompose label for softmax classifier
    original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    numpy version of tf.one_hot
    """
    one_hot = []
    for i in range(num_cls):
        _vol = np.zeros(label_vol.shape)
        _vol[label_vol == i] = 1
        one_hot.append(_vol)
    return np.stack(one_hot, axis=0)

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class Aug_Dataset(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]

        image_itk = sitk.ReadImage(os.path.join(self._base_dir + "/DATASET/" + image_name, 'data.nii.gz'))
        image_aug_itk = sitk.ReadImage(os.path.join(self._base_dir + "/DATASET/" + image_name, 'data_LHY_image.nii.gz')) # data_sobel_image
        label_itk = sitk.ReadImage(os.path.join(self._base_dir + "/DATASET/" + image_name, 'label.nii.gz'))

        image = sitk.GetArrayFromImage(image_itk)
        image_aug = sitk.GetArrayFromImage(image_aug_itk)
        label = sitk.GetArrayFromImage(label_itk)
        # print(len(image_aug))

        if label is not None:
            label[label<0] = 0
        sample = {'image': image, 'image_aug': image_aug,'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class Uni_Dataset(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None, with_idx=False, with_augimg=False):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.with_idx = with_idx
        self.with_augimg = with_augimg

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/DATASET/" + image_name + "/"+ image_name + "_norm.h5", 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        image_itk = sitk.ReadImage(os.path.join(self._base_dir + "/DATASET/" + image_name, 'data.nii.gz'))
        label_itk = sitk.ReadImage(os.path.join(self._base_dir + "/DATASET/" + image_name, 'label.nii.gz'))
        image = sitk.GetArrayFromImage(image_itk)
        label = sitk.GetArrayFromImage(label_itk)
        label[label<0] = 0

        # print(np.unique(image), np.unique(label))

        sample = {'image': image, 'label': label.astype(int)}
        if self.with_augimg:
            image_aug_itk = sitk.ReadImage(
                os.path.join(self._base_dir + "/DATASET/" + image_name, 'data_LHY_image.nii.gz'))
            image_aug = sitk.GetArrayFromImage(image_aug_itk)
            sample['image_aug'] = image_aug
        # print('before randomcrop',image.shape, image_aug.shape, label.shape)
        if self.transform:
            sample = self.transform(sample)
        if self.with_idx:
            sample['idx'] = idx
        return sample

class Uni_Dataset1(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/DATASET/" + image_name + "/"+ image_name + "_norm.h5", 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        image_itk = sitk.ReadImage(os.path.join(self._base_dir + "/DATASET/" + image_name, 'data.nii.gz'))
        label_itk = sitk.ReadImage(os.path.join(self._base_dir + "/DATASET/" + image_name, 'label.nii.gz'))
        image = sitk.GetArrayFromImage(image_itk)
        label = sitk.GetArrayFromImage(label_itk)
        label[label < 0] = 0

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
#######################################################################################################################
class Desco_RandomCrop1(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size,slice_weight):
        self.output_size = output_size
        self.slice_weight=slice_weight

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        weight = np.zeros_like(image)
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        for i in range(0, weight.shape[2]):
            weight[ :, :, i] = self.slice_weight ** abs(i - 43)
        weight[:, weight.shape[1] // 2, :] = 1
        weight = weight[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label, 'weight': weight}

class Desco_RandomCrop2(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size,slice_weight):
        self.output_size = output_size
        self.slice_weight = slice_weight

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        weight = np.zeros_like(image)
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        for i in range(0, weight.shape[1]):
            weight[ :, i, :] = self.slice_weight ** abs(i - weight.shape[1]//2)
        weight[:,:,20]=1
        weight = weight[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label, 'weight': weight}

class Desco_RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        if 'weight' in sample:
            image, label,weight = sample['image'], sample['label'],sample['weight']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            weight=np.rot90(weight, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            weight=np.flip(weight, axis=axis).copy()
            return {'image': image, 'label': label,'weight':weight}
        else:
            image, label = sample['image'], sample['label']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

            return {'image': image, 'label': label}

#######################################################################################################################

class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class Random_Crop_patch(object):
    def __init__(self, output_size):
        self.patch_size = output_size

    def get_bbox(self, seg):
        # seg = seg[0]
        assert len(np.unique(seg)) != 0,'get_bbox: seg无效'
        voxel_dim = np.where(seg != 0)
        minzidx = int(np.min(voxel_dim[0]))
        maxzidx = int(np.max(voxel_dim[0]))
        minxidx = int(np.min(voxel_dim[1]))
        maxxidx = int(np.max(voxel_dim[1]))
        minyidx = int(np.min(voxel_dim[2]))
        maxyidx = int(np.max(voxel_dim[2]))

        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    def random_crop(self, data, start_point, end_point):
        assert len(data.shape) ==3 , 'random_crop:data维度不为3'
        assert len(start_point) == 3, 'random_crop：plan无效'
        image = data[start_point[0]:end_point[0], start_point[1]:end_point[1], start_point[2]:end_point[2]]
        return image

    def pad_patch(self, image, label, pad_size):

        shape = image.shape
        for axis in range(len(self.patch_size)):
            pad_extent = max(0, self.patch_size[axis] - shape[axis])
            # print(pad_extent, patch_size[axis], shape[axis])
            pad_size[axis] = pad_extent
        pad_value = np.min(image)
        for i in range(len(image[0])):
            image[i] = np.pad(image[i], ((0, pad_size[0]), (0, pad_size[1]), (0, pad_size[2])), mode="constant",
                              constant_values=pad_value)
            label[i] = np.pad(label[i], ((0, pad_size[0]), (0, pad_size[1]), (0, pad_size[2])), mode="constant",
                              constant_values=0)
        return image, label

    def __call__(self, sample, all = False, border = True, axis_ = False):
        image, label = sample['image'], sample['label']
        if 'image_aug' in sample:
            image_aug = sample['image_aug']
        shape = image.shape[-3:]
        start_point = [0, 0, 0]
        end_point = [0, 0, 0]
        p = np.random.uniform()

        # 对于全局的裁剪
        if all:
            for axis in range(len(self.patch_size)):
                random_extent = max(0, shape[axis] - self.patch_size[axis])
                start_point[axis] = random.randint(0, random_extent)
                end_point[axis] = start_point[axis] + self.patch_size[axis]
            # 确定patch坐标后的裁剪

            image = self.random_crop(image, start_point, end_point)
            label = self.random_crop(label, start_point, end_point)
            if 'image_aug' in sample:
                image_aug = self.random_crop(image_aug, start_point, end_point)
            # print('全局裁剪：',shape[0],image.shape)
        # 按照最值边界的局部裁剪
        elif border == True:
            seg_bbox = self.get_bbox(label)
            for axis in range(len(self.patch_size)):

                half_patch = self.patch_size[axis] // 2
                suo = half_patch
                # 再进行对中心取值的限制
                if (seg_bbox[axis][1] - suo < seg_bbox[axis][0] + suo):
                    con = random.randint(seg_bbox[axis][0], seg_bbox[axis][1])
                else:
                    con = random.randint(seg_bbox[axis][0] + suo, seg_bbox[axis][1] - suo)

                low = con - half_patch
                if low >= 0:
                    start_point[axis] = low
                else:
                    start_point[axis] = 0
                end_point[axis] = start_point[axis] + self.patch_size[axis]

                if end_point[axis] >= shape[axis]:
                    end_point[axis] = shape[axis]
                    start_point[axis] = end_point[axis] - self.patch_size[axis]
                if start_point[axis] < 0:
                    start_point[axis] = 0

            # 确定patch坐标后的裁剪
            image = self.random_crop(image, start_point, end_point)
            label = self.random_crop(label, start_point, end_point)
            if 'image_aug' in sample:
                image_aug = self.random_crop(image_aug, start_point, end_point)
            # image = self.random_crop(image, start_point, end_point)
            # label = self.random_crop(label, start_point, end_point)
            # print("局部裁剪：",start_point,end_point,shape[0],image.shape)
        # 基于坐标的中心裁剪
        elif axis_ == True:
            label_mask = np.where(label != 0)
            axis_id = np.random.randint(0, len(label_mask[0]))
            axis_point = [label_mask[0][axis_id], label_mask[1][axis_id], label_mask[2][axis_id]]
            for axis in range(len(self.patch_size)):
                half_patch = self.patch_size[axis] // 2
                # 再进行对中心取值的限制
                low = axis_point[axis] - half_patch
                if low >= 0:
                    start_point[axis] = low
                else:
                    start_point[axis] = 0
                end_point[axis] = start_point[axis] + self.patch_size[axis]

                if end_point[axis] >= shape[axis]:
                    end_point[axis] = shape[axis]
                    start_point[axis] = end_point[axis] - self.patch_size[axis]
                if start_point[axis] < 0:
                    start_point[axis] = 0

                if end_point[axis] - start_point[axis] != self.patch_size[axis]:
                    print(start_point[axis], end_point[axis], self.patch_size[axis])
                assert end_point[axis] - start_point[axis] == self.patch_size[
                    axis], 'random_crop_patch：patch长度与原本图像不一样长'

            # 确定patch坐标后的裁剪
            image = self.random_crop(image, start_point, end_point)
            label = self.random_crop(label, start_point, end_point)
            if 'image_aug' in sample:
                image_aug = self.random_crop(image_aug, start_point, end_point)
            # for i in range(len(image[0])):
            #     image[i] = self.random_crop(image[i], start_point, end_point)
            #     label[i] = self.random_crop(label[i], start_point, end_point)
            #     if 'image_aug' in sample:
            #         image_aug[i] = self.random_crop(image_aug[i], start_point, end_point)
            # image = self.random_crop(image, start_point, end_point)
            # label = self.random_crop(label, start_point, end_point)
        if 'image_aug' in sample:
            # print(image.shape, image_aug.shape, label.shape)
            return {'image': image, 'image_aug': image_aug, 'label': label}
        else:
            return {'image': image, 'label': label}
class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}
class RandomCrop_v3(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, image_aug, label = sample['image'], sample['image_aug'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image_aug = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_aug = image_aug[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'image_aug': image_aug, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         image = sample['image']
#         image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
#         if 'onehot_label' in sample:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
#                     'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
#         else:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
# class Desco_ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         image = sample['image']
#         image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
#         if 'onehot_label' in sample:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
#                     'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
#         elif 'weight' in sample:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),'weight': torch.from_numpy(sample['weight'])}
#         else:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        elif 'image_aug' in sample:
            image_aug = sample['image_aug']
            image_aug = image_aug.reshape(1, image_aug.shape[0], image_aug.shape[1], image_aug.shape[2]).astype(
                np.float32)
            return {'image': torch.from_numpy(image), 'image_aug': torch.from_numpy(image_aug),
                    'label': torch.from_numpy(sample['label']).long()}
        elif 'weight' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),'weight': torch.from_numpy(sample['weight'])}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
