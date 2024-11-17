import torch
import torch.nn.functional as F
import sys

sys.path.append("./models")
import numpy as np
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
from models.MSNet import MSNet
from tools.data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=384, help="testing size")
parser.add_argument("--gpu_id", type=str, default="0", help="select gpu id")
parser.add_argument(
    "--test_path",
    type=str,
    default="/home/data1/MSNet/dataset/TestingSet/test/",
    help="test dataset path",
)
opt = parser.parse_args()
dataset_path = opt.test_path

# load the model
model = MSNet()
model.load_state_dict(torch.load("./rem/ckpt8/MSNet_epoch_best.pth"))
model.cuda()
model.eval()
# test
test_datasets = ["'DUT', 'SSD','DES','LFSD','NJU2K','NLPR','SIP','STERE'"]
# test_datasets = ['SSD','SIP','ReDWeb','NJU2K','NLPR','STERE','DES','LFSD','RGBD135','DUT-RGBD']
# test_datasets = ['STEREO']
# test_datasets = ['VT821','VT5000','VT1000']
# test_datasets = ['nju2k','stere','sip']
for dataset in test_datasets:
    save_path = "./res/paper/" + dataset + "/"
    save_path1 = "./res/paper/" + dataset + "/sal1"
    edge_save_path = "./res/paper/edge/" + dataset + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(edge_save_path)
    image_root = dataset_path + dataset + "/RGB/"
    gt_root = dataset_path + dataset + "/GT/"
    depth_root = dataset_path + dataset + "/depth/"
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= gt.max() + 1e-8
        image = image.cuda()
        depth = depth = depth.repeat(1, 3, 1, 1).cuda()
        res, edge, res1 = model(image, depth)  #
        res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
        # res1 = F.upsample(res1, size=gt.shape, mode="bilinear", align_corners=False)
        edge = F.upsample(edge, size=gt.shape, mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # res1 = res1.sigmoid().data.cpu().numpy().squeeze()
        edge = edge.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        print("save img to: ", save_path + name)
        # ndarray to image
        cv2.imwrite(save_path + name, res * 255)
        # cv2.imwrite(save_path1 + name, res1 * 255)
        cv2.imwrite(edge_save_path + name, edge * 255)
    print("Test Done!")
