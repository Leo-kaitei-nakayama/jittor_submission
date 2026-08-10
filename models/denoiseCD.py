import numpy as np
import jittor as jt
import jittor.nn as nn

from .feature import FeatureExtraction
from .pointops_jt import knn_points, farthest_point_sampling
from .InfoCD import calc_cd_like_InfoV2


class DenoiseNetCD(nn.Module):
    """
    Jittor port of the PyTorch-Lightning DenoiseNetCD.

    Lightning hooks (configure_optimizers, train/val_dataloader,
    training_step, epoch-end logging) are deliberately NOT in the model —
    plain-Jittor equivalents live in train_ASDN.py. The model itself keeps:
        - get_supervised_loss   (training loss)
        - patch_based_denoise   (inference over a big cloud, patch stitching)
        - denoise_langevin_dynamics (per-patch forward pass)
    """

    def __init__(self, args=None, classify_ckpt=None, classify_frame_knn=32):
        super().__init__()
        self.args = args
        self.feature_nets = FeatureExtraction(
            classify_ckpt=classify_ckpt, classify_frame_knn=classify_frame_knn)

    # ------------------------------------------------------------------
    # Checkpoint loading (from the ORIGINAL torch-lightning .ckpt)
    # ------------------------------------------------------------------
    @classmethod
    def load_from_checkpoint(cls, ckpt_path):
        import torch  # only for deserializing the .ckpt file

        ckpt = torch.load(ckpt_path, map_location='cpu')
        hparams = ckpt.get('hyper_parameters', {}) or {}
        args = hparams.get('args', None)

        model = cls(args)
        state_dict = ckpt['state_dict']
        jt_state = model.state_dict()

        skipped = []
        for name, tensor in state_dict.items():
            if name in jt_state:
                jt_state[name] = jt.array(tensor.detach().cpu().numpy())
            else:
                skipped.append(name)
        model.load_state_dict(jt_state)

        if skipped:
            print(f"[DenoiseNetCD.load_from_checkpoint] warning: {len(skipped)} "
                  f"unmatched tensors skipped (e.g. {skipped[:5]})")
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def curr_iter_add_noise(self, pcl_clean, noise_std):
        new_pcl_clean = pcl_clean + jt.randn_like(pcl_clean) * noise_std.unsqueeze(1).unsqueeze(2)
        return new_pcl_clean.float32()

    def get_supervised_loss(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        """
        Args:
            pcl_noisy:  (B, N, 3)
            pcl_clean:  (B, M, 3), M >= N
            pcl_seeds:  (B, 1, 3)
        """
        B, N_noisy, N_clean = pcl_noisy.shape[0], pcl_noisy.shape[1], pcl_clean.shape[1]

        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)
        pcl_noisy = pcl_noisy - pcl_seeds_1
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2

        pcl_input = pcl_noisy

        feat = pcl_input.reshape(B * N_noisy, -1)[:, 3:]  # (B*N, 0) — empty features
        offset = jt.array(np.array([(i + 1) * N_noisy for i in range(B)]), dtype='int32')
        pred_disp = self.feature_nets(pcl_input, feat, offset)

        pred_pcl = pcl_input + pred_disp

        InfoCD = calc_cd_like_InfoV2(pred_pcl, pcl_clean)
        losses = InfoCD.mean()

        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def patch_based_denoise(self, pcl_noisy, patch_size=1000, seed_k=5,
                             seed_k_alpha=10, num_modules_to_use=None):
        """
        Args:
            pcl_noisy: (N, 3)
        """
        assert pcl_noisy.ndim == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.shape
        pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
        patch_dists, point_idxs_in_main_pcd, patches = knn_points(
            seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        patches = patches[0]  # (num_patches, K, 3)

        # Patch stitching preliminaries
        seed_pnts_1 = seed_pnts.squeeze(0).unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1
        patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]
        patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

        # For each original point, distance-derived weight per covering patch
        all_dists_np = np.full((num_patches, N), np.inf, dtype=np.float32)
        pid_np = point_idxs_in_main_pcd.numpy()
        pdist_np = patch_dists.numpy()
        for pi in range(num_patches):
            all_dists_np[pi, pid_np[pi]] = pdist_np[pi]

        weights = np.exp(-1 * all_dists_np)              # (num_patches, N)
        best_weights_idx = weights.argmax(axis=0)         # (N,)

        patches_denoised = []

        # Denoising
        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            curr_patches = patches[i:i + patch_step]
            patches_denoised_temp = self.denoise_langevin_dynamics(curr_patches)
            patches_denoised.append(patches_denoised_temp)
            i += patch_step

        patches_denoised = jt.concat(patches_denoised, dim=0)
        patches_denoised = patches_denoised + seed_pnts_1

        # Patch stitching: for each original point, take its denoised coordinate
        # from the patch that covers it with the highest weight.
        # (Original used a per-point boolean-mask list comprehension; Jittor
        # can't index with numpy bool masks, so build integer gather indices:
        # pos_map[pi, n] = position j of original point n inside patch pi.)
        pos_map = np.full((num_patches, N), -1, dtype=np.int64)
        col = np.arange(patch_size, dtype=np.int64)
        for pi in range(num_patches):
            pos_map[pi, pid_np[pi]] = col

        point_ids = np.arange(N, dtype=np.int64)
        gather_pos = pos_map[best_weights_idx, point_ids]      # (N,) position in its best patch
        covered = gather_pos >= 0                               # uncovered points -> pad later

        sel_patch = jt.array(best_weights_idx[covered].astype(np.int32))
        sel_pos = jt.array(gather_pos[covered].astype(np.int32))
        pcl_denoised = patches_denoised[sel_patch, sel_pos]     # (N_covered, 3)

        while pcl_denoised.shape[0] != N:
            pcl_denoised = jt.concat(
                (pcl_denoised, pcl_denoised[pcl_denoised.shape[0] - 1].unsqueeze(0)), dim=0)
            print(f'pcl_denoised.shape ===> {pcl_denoised.shape}')

        return pcl_denoised

    def patch_based_denoise_without_stitching(self, pcl_noisy, patch_size=1000,
                                               seed_k=5, seed_k_alpha=10,
                                               num_modules_to_use=None):
        """
        Simpler variant used by test_ASDN.py when --patch_stitching is off:
        denoise patches and concatenate all their points, then FPS back to N.
        """
        assert pcl_noisy.ndim == 2
        N, d = pcl_noisy.shape
        pcl_noisy = pcl_noisy.unsqueeze(0)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
        _, _, patches = knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        patches = patches[0]
        seed_pnts_1 = seed_pnts.squeeze(0).unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1

        patches_denoised = []
        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            curr_patches = patches[i:i + patch_step]
            patches_denoised.append(self.denoise_langevin_dynamics(curr_patches))
            i += patch_step

        patches_denoised = jt.concat(patches_denoised, dim=0) + seed_pnts_1
        all_pts = patches_denoised.reshape(1, -1, 3)
        sampled, _ = farthest_point_sampling(all_pts, N)
        return sampled[0]

    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        pred_disps = []

        with jt.no_grad():
            self.feature_nets.eval()

            feat = pcl_noisy.reshape(B * N, -1)[:, 3:]
            offset = jt.array(np.array([(i + 1) * N for i in range(B)]), dtype='int32')

            pred_points = self.feature_nets(pcl_noisy, feat, offset)
            pred_disps.append(pred_points)

        return pcl_noisy + pred_disps[-1]
