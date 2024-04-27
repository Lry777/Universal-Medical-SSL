import cv2
import h5py
import glob
import SimpleITK as sitk
import numpy as np
import os
import shutil
import pywt
import cv2 as cv
from matplotlib import pyplot as plt
import torch
from scipy import ndimage
# from scipy.ndimage import zoom
from skimage import transform as sk_trans

def _compute_stats(voxels):
    if len(voxels) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    median = np.median(voxels)
    mean = np.mean(voxels)
    sd = np.std(voxels)
    mn = np.min(voxels)
    mx = np.max(voxels)
    percentile_99_5 = np.percentile(voxels, 99.5)
    percentile_00_5 = np.percentile(voxels, 00.5)
    return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5


def get_niigz_from_h5(data_path):
    path = glob.glob(os.path.join(data_path, "*.h5"))[0]
    h5f = h5py.File(path, 'r')
    data = h5f['image'][:]
    seg = h5f['label'][:]

    # mask = seg > 0
    # voxel = list(data[mask][::1])
    # median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = _compute_stats(voxel)

    # mean_intensity = mean
    # std_intensity = sd
    # lower_bound = percentile_00_5
    # upper_bound = percentile_99_5
    # data = np.clip(data, lower_bound, upper_bound)
    # data = (data - mean_intensity) / std_intensity

    image = sitk.GetImageFromArray(data)
    label = sitk.GetImageFromArray(seg)

    # B = 1
    # label_cl = np.zeros((B, 2))
    # print(label_cl)
    # for i in range(B):
    #     cl = np.unique(labeled)
    #     a = labeled.flatten()
    #     num = np.bincount(a)
    #     for j in range(len(cl)):
    #         label_cl[i][cl[j]] = num[j]
    #
    #     label_cl[i] = label_cl[i] / label_cl[i].sum()
    # print(label_cl)

    sitk.WriteImage(image, os.path.join(data_path, 'data.nii.gz'))
    sitk.WriteImage(label, os.path.join(data_path, 'label.nii.gz'))


def dwt3d(data):
    # print(data.shape)
    h = data.shape[0]
    w = data.shape[1]
    z = data.shape[2]

    for i in range(z):
        if i == 0:
            LLY, (LHY, HLY, HHY) = pywt.dwt2(data[:, :, i], 'haar')
            LLY = cv2.resize(LLY,(w, h))
            LHY = cv2.resize(LHY,(w, h))
            HLY = cv2.resize(HLY,(w, h))
            HHY = cv2.resize(HHY,(w, h))

            LLY = np.expand_dims(LLY, axis=-1)
            LHY = np.expand_dims(LHY, axis=-1)
            HLY = np.expand_dims(HLY, axis=-1)
            HHY = np.expand_dims(HHY, axis=-1)

        else:
            LLY_i, (LHY_i, HLY_i, HHY_i) = pywt.dwt2(data[:,:,i], 'haar')
            LLY_i = cv2.resize(LLY_i, (w, h))
            LHY_i = cv2.resize(LHY_i, (w, h))
            HLY_i = cv2.resize(HLY_i, (w, h))
            HHY_i = cv2.resize(HHY_i, (w, h))

            LLY = np.concatenate([LLY, np.expand_dims(LLY_i, axis=-1)], axis=-1)
            LHY = np.concatenate([LHY, np.expand_dims(LHY_i, axis=-1)], axis=-1)
            HLY = np.concatenate([HLY, np.expand_dims(HLY_i, axis=-1)], axis=-1)
            HHY = np.concatenate([HHY, np.expand_dims(HHY_i, axis=-1)], axis=-1)

    return LLY, LHY, HLY, HHY

def sobel_edge(data):
    z = data.shape[2]
    for i in range(z):

        if i == 0:
            # 2 计算Sobel卷积结果
            x = cv.Sobel(data[:, :, i], cv.CV_16S, 1, 0, ksize=-1)
            y = cv.Sobel(data[:, :, i], cv.CV_16S, 0, 1, ksize=-1)
            # 3 将数据进行转换
            Scale_absX = cv.convertScaleAbs(x)  # convert 转换  scale 缩放
            Scale_absY = cv.convertScaleAbs(y)
            # 4 结果合成
            result = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
            result =np.expand_dims(result, axis=-1)

        else:
            # 2 计算Sobel卷积结果
            x = cv.Sobel(data[:, :, i], cv.CV_16S, 1, 0, ksize=-1)
            y = cv.Sobel(data[:, :, i], cv.CV_16S, 0, 1, ksize=-1)
            # 3 将数据进行转换
            Scale_absX = cv.convertScaleAbs(x)  # convert 转换  scale 缩放
            Scale_absY = cv.convertScaleAbs(y)
            # 4 结果合成
            result_i = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
            result = np.concatenate([result, np.expand_dims(result_i, axis=-1)], axis=-1)


    return result

def Get_bbox( seg):
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

def Crop(data, start_point, end_point):
        assert len(data.shape) ==3 , 'random_crop:data维度不为3'
        assert len(start_point) == 3, 'random_crop：plan无效'
        image = data[start_point[0]:end_point[0], start_point[1]:end_point[1], start_point[2]:end_point[2]]
        return image
if __name__ == "__main__":

    # base_dir = 'E:\code\MICCAI_CODE\Co-BioNet-main\Co-BioNet-main\data\LA\\2018LA_Seg_Training Set\\'
    base_dir = '/home/lry/Code/NewNet/data/LiTS_mini/DATASET/'

    # data_dir = '/home/lry/DATASET/BTCV/imagesTr'
    # label_dir = '/home/lry/DATASET/BTCV/labelsTr'

    output_path = '/home/lry/Code/NewNet/data/LiTS_mini/DATASET'

    data_path = glob.glob(os.path.join(base_dir, '*'))

####################only make file to Pancreas_h5 ###############################
    # for i, path in enumerate(data_path):
    #     # print(i, path)
    #     data_id = os.path.split(path)[-1][:-8]
    #     print(i, data_id)
    #     os.makedirs(os.path.join(base_dir, data_id))
    #     shutil.move(path, os.path.join(base_dir, data_id))

####################only make file to btvc ###############################
    # lits_data_path = glob.glob(os.path.join(data_dir, '*.nii.gz'))
    # label = dowm(label)
    for i, path in enumerate(data_path):
        # print(i, path)
        data_id = os.path.split(path)[-1]
        print(i, data_id)
        data_path = os.path.join(path, 'data.nii.gz')
        data_aug_path = os.path.join(path, 'data_LHY_image.nii.gz')
        label_path = os.path.join(path, 'label.nii.gz')

        img_itk = sitk.ReadImage(data_path)
        label_itk = sitk.ReadImage(label_path)
        img_aug_itk = sitk.ReadImage(data_aug_path)
        img_data = sitk.GetArrayFromImage(img_itk)
        label_data = sitk.GetArrayFromImage(label_itk)
        img_aug_data = sitk.GetArrayFromImage(img_aug_itk)
        print(img_data.shape,img_aug_data.shape, label_data.shape)

        # label_data[label_data<0] = 0
        # seg_bbox = Get_bbox(label_data)
        # img_data = img_data[seg_bbox[0][0]:seg_bbox[0][1], seg_bbox[1][0]:seg_bbox[1][1], seg_bbox[2][0]:seg_bbox[2][1]]
        # img_aug_data = img_aug_data[seg_bbox[0][0]:seg_bbox[0][1], seg_bbox[1][0]:seg_bbox[1][1], seg_bbox[2][0]:seg_bbox[2][1]]
        # label_data = label_data[seg_bbox[0][0]:seg_bbox[0][1], seg_bbox[1][0]:seg_bbox[1][1], seg_bbox[2][0]:seg_bbox[2][1]]

        # t_img_data = torch.from_numpy(img_data)
        # img_data = dowm(t_img_data.unsqueeze(0).float())
        # t_label_data = torch.from_numpy(label_data)
        # label_data = dowm(t_label_data.unsqueeze(0).float())
        # t_img_aug_data =torch.from_numpy(img_aug_data)
        # img_aug_data = dowm(t_img_aug_data.unsqueeze(0).float())

        # print('out:',img_data.shape, img_aug_data.shape, label_data.shape)
        #
        # os.makedirs(os.path.join(output_path, data_id))

        # print(img_data.shape, label_data.shape)

        # img_data = img_data.transpose((1, 2, 0))
        # label_data = label_data.transpose((1, 2, 0))
        # print(img_data.shape, label_data.shape)

        # img_itk = sitk.GetImageFromArray(img_data)
        # img_aug_itk = sitk.GetImageFromArray(img_aug_data)
        # label_itk = sitk.GetImageFromArray(label_data)
        # sitk.WriteImage(img_itk, os.path.join(output_path, data_id, 'data.nii.gz'))
        # sitk.WriteImage(img_aug_itk, os.path.join(output_path, data_id, 'data_LHY_image.nii.gz'))
        # sitk.WriteImage(label_itk, os.path.join(output_path, data_id, 'label.nii.gz'))



################# turn h5 to nii.gz#############################
    # for i, path in enumerate(data_path):
    #     print(i, path)
    #     get_niigz_from_h5(path)




####################make DWT Edge####################################

    for i, path in enumerate(data_path):
        image = sitk.ReadImage(os.path.join(path, 'data.nii.gz'))
        data = sitk.GetArrayFromImage(image)
        print('[DWT]input data shape:',data.shape)

        LLY, LHY, HLY, HHY = dwt3d(data)
        print('[DWT]output shape:', LLY.shape, LHY.shape, HLY.shape, HHY.shape)
        LLY_image = sitk.GetImageFromArray(LLY)
        LHY_image = sitk.GetImageFromArray(LHY)
        HLY_image = sitk.GetImageFromArray(HLY)
        HHY_image = sitk.GetImageFromArray(HHY)

        # sitk.WriteImage(LLY_image, os.path.join(path, 'data_LLY_image.nii.gz'))
        sitk.WriteImage(LHY_image, os.path.join(path, 'data_LHY_image.nii.gz'))
        # sitk.WriteImage(HHY_image, os.path.join(path, 'data_HHY_image.nii.gz'))

####################make Sobel Edge####################################
    # for i, path in enumerate(data_path):
    #     image = sitk.ReadImage(os.path.join(path, 'data.nii.gz'))
    #     data = sitk.GetArrayFromImage(image)
    #     print('[Sobel]input data shape:', data.shape)
    #     img = sobel_edge(data)
    #     print('[Sobel]output shape:',img.shape)
    #     sobel_image = sitk.GetImageFromArray(img)
    #     sitk.WriteImage(sobel_image, os.path.join(path, 'data_sobel_image.nii.gz'))
