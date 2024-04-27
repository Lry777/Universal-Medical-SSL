import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import os
import SimpleITK as sitk

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA",text = ['Background','Left Atrial'], kp = False, MT = True):
    if dataset_name == "LA":
        with open('../data/LA/test.list', 'r') as f:
            image_list = f.readlines()
        # image_list = ["../data/LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
        image_list = ["../data/LA/DATASET/" + item.replace('\n', '') for item in image_list]
    elif dataset_name == "Pancreas_CT":
        with open('../data/Pancreas/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["../data/Pancreas/DATASET/" + item.replace('\n', '') for item in image_list]
    elif dataset_name == "LiTS":
        with open('../data/LiTS_mini/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["../data/LiTS/DATASET/" + item.replace('\n', '') for item in image_list]
    elif dataset_name == "BTCV":
        with open('../data/BTCV/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["../data/BTCV/DATASET/" + item.replace('\n', '') for item in image_list]

    loader = tqdm(image_list)
    total_dice = [0.0 for _ in range(num_classes)]
    total_dice_s = 0.0
    for image_path in loader:
        image_itk = sitk.ReadImage(
            os.path.join(image_path, 'data.nii.gz'))
        if kp:
            image_aug_itk = sitk.ReadImage(
                os.path.join(image_path, 'data_LHY_image.nii.gz'))
            image_aug = sitk.GetArrayFromImage(image_aug_itk)
        else:
            image_aug = [0]
        label_itk = sitk.ReadImage(
            os.path.join(image_path, 'label.nii.gz'))

        image = sitk.GetArrayFromImage(image_itk)
        label = sitk.GetArrayFromImage(label_itk)

        prediction, score_map = test_single_case_first_output(model, image, image_aug, stride_xy, stride_z, patch_size, num_classes=num_classes, text = text, MT = MT)
        # print(prediction.shape, score_map.shape, np.sum(score_map[0]), np.sum(score_map[1]))

        # print('val',np.unique(prediction), np.sum(prediction))
        if MT:
            prediction = np.argmax(score_map, 0)
            for class_id in range(num_classes):
                seg_pred = np.array(prediction == class_id, dtype=np.int8)
                seg_label = np.array(label == class_id, dtype=np.int8)
                # seg_pred = getLargestCC(seg_pred)

                if np.sum(prediction)==0:
                    dice = 0
                else:
                    dice = metric.binary.dc(seg_pred, seg_label)

                total_dice[class_id] += dice
        else:
            if np.sum(prediction) == 0:
                dice = 0
            else:
                dice = metric.binary.dc(prediction, label)
            total_dice_s += dice
    # avg_dice = total_dice / len(image_list)
    if MT:
        avg_dice = [item/len(image_list) for item in total_dice]
    else:
        avg_dice = total_dice_s / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case(model_name, num_outputs, model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=False, test_save_path=None, preproc_fn=None, metric_detail=1, kp=False, text = ['Background','Left Atrial'], MT = True):

    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = [0.0 for _ in range(num_classes)]
    total_metric_average = 0.0
    for image_path in loader:
        print('TEST SAMPLE:', image_path.split('/')[-1], '#################')
        # h5f = h5py.File(image_path, 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]

        image_itk = sitk.ReadImage(
            os.path.join(image_path, 'data.nii.gz'))
        if kp:
            image_aug_itk = sitk.ReadImage(
                os.path.join(image_path, 'data_LHY_image.nii.gz'))
            image_aug = sitk.GetArrayFromImage(image_aug_itk)
        else:
            image_aug = [0]
        label_itk = sitk.ReadImage(
            os.path.join(image_path, 'label.nii.gz'))

        image = sitk.GetArrayFromImage(image_itk)
        label = sitk.GetArrayFromImage(label_itk)


        if preproc_fn is not None:
            image = preproc_fn(image)
        # print(image.shape, label[:].shape)
        prediction, score_map = test_single_case_first_output(model, image, image_aug, stride_xy, stride_z, patch_size, num_classes=num_classes, text= text, MT = MT)
        # print(prediction.shape, score_map.shape, np.sum(score_map[0]), np.sum(score_map[1]))
        if MT:
            # print('1', score_map.shape)
            prediction = np.argmax(score_map, 0)
            # print('2', prediction.shape)
            # prediction = getLargestCC(prediction)
            # print('3', prediction.shape, np.unique(prediction))
            for class_id in range(num_classes):
                seg_pred = np.array(prediction == class_id, dtype=np.int8)
                seg_label = np.array(label == class_id, dtype=np.int8)
                if len(np.unique(seg_label))==2:
                    # seg_pred = getLargestCC(seg_pred)
                    # prediction = np.argmax(prediction, 1)
                    # print('2', seg_pred.shape, seg_label.shape)
                    # print('2', np.unique(seg_pred), np.unique(seg_label))
                    single_metric = calculate_metric_percase(seg_pred, seg_label)

                    if metric_detail:
                        print('%02d,\t[class id:%01d], %.5f, %.5f, %.5f, %.5f' % (
                        ith, class_id, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
                        with open(test_save_path + '../{}_performance.txt'.format(model_name), 'a') as f:
                            f.writelines('{} sample {} class detail metric is {} \n'.format(image_path.split('/')[-1], class_id, single_metric))
                    # print(single_metric.shape)

                    total_metric[class_id] += np.asarray(single_metric)
                    # total_metric[class_id].append(single_metric)
                else:
                    pass
        else:
            prediction = getLargestCC(prediction)

            if np.sum(prediction) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(prediction, label[:])
            if metric_detail:
                print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
                ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
                with open(test_save_path + '../{}_performance.txt'.format(model_name), 'a') as f:
                    f.writelines('{} sample {} class detail metric is {} \n'.format(image_path.split('/')[-1], class_id, single_metric))

            total_metric += np.asarray(single_metric)
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            # if num_outputs > 1:
            #     nib.save(nib.Nifti1Image(prediction_average.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred_average.nii.gz" % ith)
            #     nib.save(nib.Nifti1Image(score_map_average[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores_average.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_gt.nii.gz" % ith)
        
        ith += 1

    if MT:
        for class_id in range(num_classes):
            # print(len(total_metric), len(total_metric[class_id]))
            # print(total_metric[class_id])
            avg_metric = total_metric[class_id] / len(image_list)
            print('{} class average metric is {}'.format(class_id, avg_metric))

            with open(test_save_path + '../{}_performance.txt'.format(model_name), 'a') as f:
                f.writelines('{} class average metric is {} \n'.format(class_id, avg_metric))
    else:
        avg_metric = total_metric / len(image_list)
        print('average metric is decoder 1 {}'.format(avg_metric))
        with open(test_save_path + '../{}_performance.txt'.format(model_name), 'a') as f:
            f.writelines('{} class average metric is {} \n'.format(1, avg_metric))
    # if num_outputs > 1:
    #     avg_metric_average = total_metric_average / len(image_list)
    #     print('average metric of all decoders is {}'.format(avg_metric_average))
    

        # if num_outputs > 1:
        #     f.writelines('average metric of all decoders is {} \n'.format(avg_metric_average))
    return None


def test_single_case_first_output(model, image, image_aug, stride_xy, stride_z, patch_size, num_classes=1, text = ['Background','Left Atrial'], MT= False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        if len(image_aug) == 1:
            pass
        else:
            image_aug = np.pad(image_aug, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                if len(image_aug) == 1:
                    pass
                else:
                    test_aug_patch = image_aug[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_aug_patch = np.expand_dims(np.expand_dims(test_aug_patch, axis=0), axis=0).astype(np.float32)
                    test_aug_patch = torch.from_numpy(test_aug_patch).cuda()

                with torch.no_grad():
                    # print('xxxxxxxxxxxxxx', image_aug, len(image_aug))
                    if len(image_aug) == 1:
                        if MT:
                            y = model(test_patch)
                            if len(y) > 1: y = y[0]
                            y = F.softmax(y, dim=1)
                        else:
                            y = model(test_patch)
                            y = torch.sigmoid(y)

                    else:
                        y = model(test_patch, test_aug_patch, text = text)
                        if len(y) > 1:
                            y = y[0]
                            if len(y) > 1:
                                y = y[0]
                        y = F.softmax(y, dim=1)
                    # y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                # y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    # label_map = (score_map[0]>0.5).astype(np.int)
    if MT:
        label_map = score_map[0]  # (score_map[0] > 0.5).astype(np.int)
    else:
        label_map = (score_map[0] > 0.5).astype(np.int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map



def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    # print(pred.shape, np.unique(gt))
    if len(np.unique(pred)) == 1:
        hd = 200.0
        asd = 200.0
    else:
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
