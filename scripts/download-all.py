import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_dataset, KHopTransform
import argparse


parser = argparse.ArgumentParser(description="Download datasets with K-Hop Transformation")
parser.add_argument('--root', type=str, default='/mnt2/luyifeng/ECHO', help='Root directory for dataset storage')
parser.add_argument('--k', type=int, default=None, help='K value for K-Hop Transformation')
parser.add_argument('--task', type=str, default=None, help='Task name (e.g., diam, ecc, sssp, charge, energy)')
args = parser.parse_args()

tasks = ['diam', 'ecc', 'sssp', 'charge', 'energy'] if args.task is None else [args.task]

transform = None if args.k is None else KHopTransform(k=args.k)

for t in tasks:
    print(f"Downloading dataset for task: {t} with K={args.k}")
    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root=args.root,
        task=t, 
        pre_transform=transform,
        k=args.k
    )


