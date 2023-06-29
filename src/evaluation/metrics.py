from jaxtyping import Float
from skimage.metrics import peak_signal_noise_ratio
from torch import Tensor


def compute_psnr(
    ground_truth: Float[Tensor, "*batch channel"],
    predicted: Float[Tensor, "*batch channel"],
) -> float:
    return peak_signal_noise_ratio(
        ground_truth.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
        data_range=1.0,
    )
