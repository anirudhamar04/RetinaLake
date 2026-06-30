"""
Denoising / smoothing transforms.

Each class stores parameters in ``__init__`` and processes a PIL Image in
``__call__`` by converting to numpy, calling the library function, and
converting back.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image as PILImage

from chaksudb.export.transforms.base import BasePhotometricTransform


def _pil_to_uint8(image: PILImage.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _uint8_to_pil(arr: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class GaussianDenoising(BasePhotometricTransform):
    """Gaussian smoothing (denoising via blur or wavelet)."""

    def __init__(self, sigma: float = 1.0, method: str = "blur"):
        self.sigma = sigma
        self.method = method

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        arr = _pil_to_uint8(image)

        if self.method == "wavelet":
            try:
                from skimage.restoration import denoise_wavelet
            except ImportError as e:
                raise ImportError(
                    "Wavelet denoising requested but scikit-image is not installed; "
                    "install it with: pip install scikit-image  (or choose method='blur')"
                ) from e
            denoised = denoise_wavelet(
                arr, sigma=self.sigma, channel_axis=-1, rescale_sigma=True,
            )
            return _uint8_to_pil((denoised * 255).astype(np.float64))

        ksize = max(3, int(6 * self.sigma + 1)) | 1
        result = cv2.GaussianBlur(arr, (ksize, ksize), self.sigma)
        return _uint8_to_pil(result)

    def __repr__(self) -> str:
        return f"GaussianDenoising(sigma={self.sigma}, method={self.method!r})"


class MedianFiltering(BasePhotometricTransform):
    """Median filter for salt-and-pepper noise."""

    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size | 1

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        arr = _pil_to_uint8(image)
        result = cv2.medianBlur(arr, self.kernel_size)
        return _uint8_to_pil(result)

    def __repr__(self) -> str:
        return f"MedianFiltering(kernel_size={self.kernel_size})"


class BilateralFiltering(BasePhotometricTransform):
    """Edge-preserving bilateral filter."""

    def __init__(self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        arr = _pil_to_uint8(image)
        result = cv2.bilateralFilter(arr, self.d, self.sigma_color, self.sigma_space)
        return _uint8_to_pil(result)

    def __repr__(self) -> str:
        return (
            f"BilateralFiltering(d={self.d}, sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space})"
        )


class Deblurring(BasePhotometricTransform):
    """PSF-based Wiener deconvolution (requires scipy).

    When *psf* is None a simple scalar Wiener filter is used.  When *psf* is
    provided, full FFT-based Wiener deconvolution is applied per channel.
    """

    def __init__(self, psf: np.ndarray | None = None, noise_power: float = 0.01):
        self.psf = psf
        self.noise_power = noise_power

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        arr = _pil_to_uint8(image).astype(np.float64) / 255.0
        result = np.empty_like(arr)

        if self.psf is not None:
            from scipy.fft import fft2, ifft2
            h, w = arr.shape[:2]
            psf_padded = np.zeros((h, w), dtype=np.float64)
            ph, pw = self.psf.shape[:2]
            psf_padded[:ph, :pw] = self.psf / (self.psf.sum() + 1e-12)
            H = fft2(psf_padded)
            H_conj = np.conj(H)
            wiener_filter = H_conj / (np.abs(H) ** 2 + self.noise_power)
            for ch in range(arr.shape[2]):
                G = fft2(arr[:, :, ch])
                result[:, :, ch] = np.real(ifft2(G * wiener_filter))
        else:
            from scipy.signal import wiener
            for ch in range(arr.shape[2]):
                result[:, :, ch] = wiener(arr[:, :, ch], mysize=None, noise=self.noise_power)

        result = np.clip(result * 255, 0, 255)
        return _uint8_to_pil(result)

    def __repr__(self) -> str:
        return f"Deblurring(psf={'set' if self.psf is not None else None}, noise_power={self.noise_power})"


class Deconvolution(BasePhotometricTransform):
    """Richardson-Lucy deconvolution (requires scikit-image)."""

    def __init__(self, psf: np.ndarray | None = None, iterations: int = 30):
        if psf is None:
            size = 5
            psf = np.ones((size, size)) / (size * size)
        self.psf = psf
        self.iterations = iterations

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        from skimage.restoration import richardson_lucy

        arr = _pil_to_uint8(image).astype(np.float64) / 255.0
        result = np.empty_like(arr)
        for ch in range(arr.shape[2]):
            result[:, :, ch] = richardson_lucy(
                arr[:, :, ch], self.psf, num_iter=self.iterations, clip=True,
            )
        result = np.clip(result * 255, 0, 255)
        return _uint8_to_pil(result)

    def __repr__(self) -> str:
        return f"Deconvolution(iterations={self.iterations})"
