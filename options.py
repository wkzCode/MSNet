import argparse

# RGBD
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=300, help="epoch number")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--batchsize", type=int, default=8, help="training batch size")
parser.add_argument("--trainsize", type=int, default=384, help="training dataset size")
parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")
parser.add_argument("--decay_rate", type=float, default=0.1, help="decay rate of learning rate")
parser.add_argument("--decay_epoch", type=int, default=100, help="every n epochs decay learning rate")
parser.add_argument("--load", type=str, default="./swin_base_patch4_window12_384_22k.pth",
                    help="train from checkpoints")
parser.add_argument("--load_pre", type=str, default="./rem/ckpt/MSNet_epoch_best.pth", help="train from checkpoints")
parser.add_argument("--gpu_id", type=str, default="1", help="train use gpu")
parser.add_argument("--rgb_root", type=str, default="./dataset/train_dut/train_images/",
                    help="the training rgb images root")
parser.add_argument("--depth_root", type=str, default="./dataset/train_dut/train_depth/",
                    help="the training depth images root")
parser.add_argument("--gt_root", type=str, default="./dataset/train_dut/train_masks/",
                    help="the training gt images root")
parser.add_argument("--edge_root", type=str, default="./dataset/train_dut/edge/", help="the training edge images root")
parser.add_argument("--test_rgb_root", type=str, default="./dataset/TestingSet/test/NLPR/RGB/",
                    help="the test gt images root")
parser.add_argument("--test_depth_root", type=str, default="./dataset/TestingSet/test/NLPR/depth/",
                    help="the test gt images root")
parser.add_argument("--test_gt_root", type=str, default="./dataset/TestingSet/test/NLPR/GT/",
                    help="the test gt images root")
parser.add_argument("--test_edge_root", type=str, default="./dataset/TestingSet/test/NLPR/edge/",
                    help="the test edge images root")
parser.add_argument("--save_path", type=str, default="./rem/ckpt1/", help="the path to save models and logs")
opt = parser.parse_args()
