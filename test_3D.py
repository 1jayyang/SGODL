import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import shutil
from networks.en_cross_fusion_ssms import SGODL
import torch

from test_3D_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='', help='model_name')


def Inference(FLAGS):
    best_model_path = "".format(FLAGS.exp)
    num_classes = 2
    predict_save_path = "".format(FLAGS.exp)                             # 预测 best_model

    if os.path.exists(predict_save_path):
        shutil.rmtree(predict_save_path)
    os.makedirs(predict_save_path)
    net = SGODL(in_channel=1, out_channel=2).cuda()

    save_mode_path = os.path.join(best_model_path, 'best_model.pth'.format(FLAGS.model))

    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=predict_save_path)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
