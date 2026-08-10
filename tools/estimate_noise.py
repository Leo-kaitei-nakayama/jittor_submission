"""
Estimate noise level directly from a noisy point cloud (no GT needed), using
local plane-fitting residuals: for each point, fit a plane to its k nearest
neighbors via PCA, measure the point's distance to that plane (along the
normal direction), and take the RMS of these residuals as the noise estimate.

Validates the estimate against the true noise_std recorded by
make_eval_set.py (eval_meta.csv), so we can check whether this cheap,
GT-free signal is good enough to drive a noise-adaptive --niters choice.

Usage:
    python tools/estimate_noise.py \
        --noisy_dir ./eval_noisy --noisy_filename noisy.npy \
        --meta_csv ./tools/results/eval_meta.csv \
        --k 30 --sample_points 5000
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def estimate_noise_level(pc, k=30, sample_points=5000, seed=0):
    """RMS distance of each (sub-sampled) point to the local PCA plane fit
    through its k nearest neighbors. GT-free noise estimate."""
    rng = np.random.RandomState(seed)
    n = pc.shape[0]
    if sample_points is not None and n > sample_points:
        query_idx = rng.choice(n, sample_points, replace=False)
    else:
        query_idx = np.arange(n)

    tree = cKDTree(pc)
    _, nn_idx = tree.query(pc[query_idx], k=k)

    residuals = np.empty(len(query_idx), dtype=np.float64)
    for i, neighbors_idx in enumerate(nn_idx):
        neighbors = pc[neighbors_idx]
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        # smallest-variance direction of the local neighborhood = normal
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        residuals[i] = np.dot(pc[query_idx[i]] - centroid, normal)

    return float(np.sqrt(np.mean(residuals ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_dir', type=str, required=True)
    parser.add_argument('--noisy_filename', type=str, default='noisy.npy')
    parser.add_argument('--meta_csv', type=str, default='',
                         help='eval_meta.csv with true noise_std, for validation (optional)')
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--sample_points', type=int, default=5000,
                         help='subsample this many query points per cloud for speed (0 = use all)')
    parser.add_argument('--out_csv', type=str, default='./tools/results/noise_estimates.csv')
    args = parser.parse_args()

    sample_points = None if args.sample_points == 0 else args.sample_points

    rows = []
    # iterate over the actual files on disk under noisy_dir
    for root, _, files in os.walk(args.noisy_dir):
        if args.noisy_filename in files:
            path = os.path.join(root, args.noisy_filename)
            rel = os.path.relpath(root, args.noisy_dir)
            key = rel.replace(os.sep, '/')
            pc = np.load(path).astype(np.float64)
            est = estimate_noise_level(pc, k=args.k, sample_points=sample_points)
            rows.append((key, est))
            print(f'  {key}  estimated_noise={est:.5f}')

    df = pd.DataFrame(rows, columns=['key', 'estimated_noise'])
    df.to_csv(args.out_csv, index=False)
    print(f'\nSaved estimates to {args.out_csv}')

    if args.meta_csv:
        meta = pd.read_csv(args.meta_csv, dtype={'category': str, 'model_id': str})
        merged = df.merge(meta[['key', 'noise_std']], on='key', how='inner')
        corr = merged['estimated_noise'].corr(merged['noise_std'])
        print()
        print('=' * 65)
        print(f'Validation against true noise_std ({len(merged)} samples)')
        print('=' * 65)
        pd.set_option('display.width', 120)
        print(merged.sort_values('noise_std').to_string(index=False))
        print()
        print(f'corr(estimated_noise, true noise_std) = {corr:+.3f}')


if __name__ == '__main__':
    main()
    