import argparse
import test
import os
import ast
import shutil

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=str, default="1dluts_eval")
    parser.add_argument("--use_cuda", type=ast.literal_eval, default=True)
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--testset", type=str, default="data")

    parser.add_argument('--ckpt_path', default='checkpoints', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')
    parser.add_argument('--fused_img_path', default='fused_result_1dluts_eval', type=str,
                        metavar='PATH', help='path to save images')
    parser.add_argument('--luts_path', default='luts', type=str,
                        metavar='PATH', help='path to save luts')
    parser.add_argument("--low_size", type=int, default=128)
    parser.add_argument("--high_size", type=int, default=512, help='None means random resolution')
    parser.add_argument("--test_high_size", type=int, default=2048)
    parser.add_argument("--n_frames", type=int, default=4)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--width", type=int, default=24)
    return parser.parse_args()

def main(cfg):
    t = test.Test(cfg)
    if cfg.status  == "1dluts_eval":
        print("1d_luts evaluation")
        t.eval_1dluts(0)

if __name__ == "__main__":
    config = parse_config()
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(config.fused_img_path):
        os.makedirs(config.fused_img_path)
    main(config)
