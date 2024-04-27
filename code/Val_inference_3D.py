import os
import argparse
import torch

from networks.net_factory import net_factory
# from utils.test_patch import test_all_case
from utils.test_patch import test_all_case

BTCV_text = ['Background','Spleen', 'Right Kidney', 'Left Kidney', 'Gallbladder', 'Esophagus', 'Liver', 'Stomach', 'Aorta', 'Inferior Vena Cava', 'Veins', 'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland']
LiTS_text = ['Background','Liver', 'Liver Tumor']
LA_text = ['Background','Left Atrial']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='../', help='Name of Experiment')
parser.add_argument('--load_name', type=str, default='newNet3d_best_model.pth', help='Name of .pth file')
parser.add_argument('--exp', type=str,  default='newNet3d_LA_textt', help='exp_name')
parser.add_argument('--KP', type=bool,  default=True, help='use Knowledge Prompt')
parser.add_argument('--MT', type=bool,  default=True, help='use Multi Tasks')
parser.add_argument('--num_classes', type=int, default=2, help='num_classes')
parser.add_argument('--model', type=str,  default='newNet3d', help='model_name')
parser.add_argument('--text', type=list,  default=LA_text, help='text')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
test_save_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}_predictions/".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)

if FLAGS.MT:
    num_classes = FLAGS.num_classes
else:
    num_classes = 1
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    FLAGS.root_path = '../data/LA'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/DATASET/" + item.replace('\n', '') for item in image_list]

elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = '../data/Pancreas'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/DATASET/" + item.replace('\n', '')  for item in image_list]
elif FLAGS.dataset_name == "LiTS":
    patch_size = (160, 160, 32)
    FLAGS.root_path = '../data/LiTS'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/DATASET/" + item.replace('\n', '')  for item in image_list]
elif FLAGS.dataset_name == "BTCV":
    patch_size = (128, 128, 64)
    FLAGS.root_path = '../data/BTCV'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/DATASET/" + item.replace('\n', '')  for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

def test_calculate_metric():
    
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test", patch_size=patch_size)
    save_mode_path = os.path.join(snapshot_path, FLAGS.load_name)
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()


    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=patch_size, stride_xy=18, stride_z=4,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, kp=FLAGS.KP, text=FLAGS.text, MT=FLAGS.MT)
    elif FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=patch_size, stride_xy=16, stride_z=16,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, kp=FLAGS.KP, text=FLAGS.text, MT=FLAGS.MT)
    elif FLAGS.dataset_name == "LiTS":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=patch_size, stride_xy=32, stride_z=32,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, kp=FLAGS.KP, text=FLAGS.text, MT=FLAGS.MT)
    elif FLAGS.dataset_name == "BTCV":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=patch_size, stride_xy=32, stride_z=32,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, kp=FLAGS.KP, text=FLAGS.text, MT=FLAGS.MT)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
