import jittor.nn as nn

from .classifyNet import ScaleNet
from .utils import get_entropy_B


class Classify(nn.Module):
    """
    Jittor port of the PyTorch-Lightning Classify module.

    Lightning hooks (configure_optimizers, dataloaders, epoch-end logging)
    live in train_classifier.py; the model keeps the loss and the
    patch-based validation routine.
    """

    def __init__(self, args=None, frame_knn=32):
        super().__init__()
        self.args = args
        self.frame_knn = getattr(args, 'frame_knn', frame_knn) if args is not None else frame_knn

        self.feature_nets = nn.ModuleList()
        input_dim = 3
        z_dim = 0
        self.feature_nets.append(
            ScaleNet(k=self.frame_knn, input_dim=input_dim, z_dim=z_dim,
                     embedding_dim=256, output_dim=1)
        )

    # ------------------------------------------------------------------
    def get_supervised_loss_nn(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        """
        Args:
            pcl_noisy:  (B, N, 3)
            pcl_clean:  (B, M, 3)
            pcl_seeds:  (B, 1, 3)
        The target is the entropy ratio between noisy and clean patches;
        the network regresses it (MSE, sum reduction as in the original).
        """
        B, N_noisy, N_clean = pcl_noisy.shape[0], pcl_noisy.shape[1], pcl_clean.shape[1]

        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)
        pcl_noisy = pcl_noisy - pcl_seeds_1
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2

        NoiseEntropy = get_entropy_B(pcl_noisy)
        CleanEntropy = get_entropy_B(pcl_clean)

        entropy_ratio = NoiseEntropy / CleanEntropy

        pre, _ = self.feature_nets[0](pcl_noisy, None)
        diff = pre.squeeze(-1) - entropy_ratio
        loss = (diff ** 2).sum()  # MSELoss(reduction='sum')

        return loss
