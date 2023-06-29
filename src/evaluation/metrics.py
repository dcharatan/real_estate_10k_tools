from jaxtyping import Float
from skimage.metrics import peak_signal_noise_ratio
from torch import Tensor


def compute_psnr(
    ground_truth: Float[Tensor, "*batch channel"],
    predicted: Float[Tensor, "*batch channel"],
) -> float:
    return peak_signal_noise_ratio(
        ground_truth.clip(min=0, max=1).detach().cpu().numpy(),
        predicted.clip(min=0, max=1).detach().cpu().numpy(),
        data_range=1.0,
    )
