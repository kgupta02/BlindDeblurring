# main_deblurring.py

import argparse
import numpy as np
from io_utils import read_image, save_image, save_kernel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/deblurred.png')
    parser.add_argument('--kernel_output', type=str, default='results/kernel.png')
    parser.add_argument('--lambda_val', type=float, default=0.008)
    parser.add_argument('--beta_val', type=float, default=2.0)
    parser.add_argument('--num_levels', type=int, default=5)
    args = parser.parse_args()

    # 1. Read input image
    B = read_image(args.input)