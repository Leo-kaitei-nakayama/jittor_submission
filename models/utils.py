import numpy as np
import jittor as jt


def get_entropy_B(point_clouds):
    """
    Voxel-density entropy per point cloud, mirroring the original torch version
    (dynamic voxel size, unique-voxel counting, 10-bin histogram).

    Computed via numpy internally: no gradients ever flow through this in the
    original either (it is only used to build supervision targets).

    Args:
        point_clouds: (B, N, 3) jt.Var
    Returns:
        (B,) jt.Var of entropies
    """
    entropies = []
    pcs = point_clouds.numpy()

    for pc in pcs:
        vsize = 0.001 * 100000 / pc.shape[0]  # dynamic voxel size (as original)
        min_bound = pc.min(axis=0)
        indices = np.floor((pc - min_bound) / vsize).astype(np.int64)
        _, inverse = np.unique(indices, axis=0, return_inverse=True)
        voxel_densities = np.bincount(inverse).astype(np.float32)

        vmax = voxel_densities.max()
        hist, _ = np.histogram(voxel_densities, bins=10, range=(0.0, float(vmax)))
        hist = hist.astype(np.float32)
        hist[hist == 0] = 1.0
        hist = hist / hist.sum()

        h = hist[hist > 0]
        p = h / h.sum()
        ent = float(-(p * np.log(p)).sum())
        entropies.append(ent)

    return jt.array(np.array(entropies, dtype=np.float32))
