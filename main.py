import argparse
import test
import train
import os
import ast
import shutil

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=str, default="1dluts_eval")
    parser.add_argument("--use_cuda", type=ast.literal_eval, default=True)
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--testset", type=str, default="data")
    parser.add_argument("--trainset", type=str, default="data")

    parser.add_argument('--ckpt_path', default='checkpoints', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')
    parser.add_argument('--fused_img_path', default='fused_result_1dluts_eval', type=str,
                        metavar='PATH', help='path to save images')
    parser.add_argument('--luts_path', default='luts', type=str,
                        metavar='PATH', help='path to save test luts')
    parser.add_argument('--train_luts_path', default='train_luts', type=str,
                        metavar='PATH', help='path to save train luts')
    parser.add_argument("--low_size", type=int, default=128)
    parser.add_argument("--high_size", type=int, default=512, help='None means random resolution')
    parser.add_argument("--test_high_size", type=int, default=2048)
    parser.add_argument("--n_frames", type=int, default=2)
    parser.add_argument("--offline_test_size", type=int, default=50)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--epochs_per_eval", type=int, default=1)#
    parser.add_argument("--epochs_per_save", type=int, default=10)#
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=1000)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=100)
    return parser.parse_args()

def main(cfg):
    
    print(cfg.status)
    if cfg.status  == "1dluts_eval":
        te = test.Test(cfg)
        print("1d_luts evaluation")
        te.eval_1dluts(0)
    elif cfg.status == "1dluts_train":
        tr = train.Train(cfg)
        print("1d_luts training")
        tr.fit()

if __name__ == "__main__":
    config = parse_config()
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(config.fused_img_path):
        os.makedirs(config.fused_img_path)
    if not os.path.exists(config.train_luts_path):
        os.makedirs(config.train_luts_path)
    main(config)
