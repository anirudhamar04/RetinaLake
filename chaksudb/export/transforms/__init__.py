"""
Spatial and photometric transform library for ChaksuDB export.

Public API: import transform classes directly from this package.
"""

# --- base ---
from chaksudb.export.transforms.base import (
    BaseMorphologicalTransform,
    BasePhotometricTransform,
    BaseSpatialTransform,
    SpatialSample,
)

# --- compose ---
from chaksudb.export.transforms.compose import PhotometricCompose, SpatialCompose

# --- geometric (affine + non-affine + multi-output + annotation-aware) ---
from chaksudb.export.transforms.geometric import (
    BoundingBoxCrop,
    CenterCrop,
    CornerPatchExtraction,
    ElasticTransform,
    FiveCrop,
    Pad,
    PolarTransform,
    RandomAffine,
    RandomCrop,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomRescale,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ROICrop,
    TenCrop,
)

# --- collate ---
from chaksudb.export.transforms.collate import (
    default_collate,
    get_collate_fn,
    packed_collate,
    padded_collate,
)

# --- photometric (re-exports from torchvision) ---
from chaksudb.export.transforms.photometric import (
    ColorJitter,
    GaussianBlur,
    Grayscale,
    Normalize,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomEqualize,
)

# --- retinal-specific ---
from chaksudb.export.transforms.retinal import (
    CLAHE,
    MSRCR,
    BackgroundPolynomialCorrection,
    BlueChannelEmphasis,
    ContrastEnhancement,
    FundusROIMask,
    GammaCorrection,
    GreenChannelExtraction,
    HistogramMatching,
    IlluminationCorrection,
    MultiscaleRetinex,
)

# --- denoising ---
from chaksudb.export.transforms.denoising import (
    BilateralFiltering,
    Deblurring,
    Deconvolution,
    GaussianDenoising,
    MedianFiltering,
)

# --- morphological ---
from chaksudb.export.transforms.morphological import (
    ConnectedComponentFiltering,
    Dilation,
    Erosion,
    MorphologicalClosing,
    Opening,
)

__all__ = [
    # base
    "SpatialSample",
    "BaseSpatialTransform",
    "BasePhotometricTransform",
    "BaseMorphologicalTransform",
    # compose
    "SpatialCompose",
    "PhotometricCompose",
    # geometric
    "Resize",
    "RandomResizedCrop",
    "CenterCrop",
    "RandomCrop",
    "Pad",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "RandomAffine",
    "RandomRescale",
    "RandomPerspective",
    "ElasticTransform",
    "PolarTransform",
    "FiveCrop",
    "TenCrop",
    "CornerPatchExtraction",
    "BoundingBoxCrop",
    "ROICrop",
    # collate
    "default_collate",
    "padded_collate",
    "packed_collate",
    "get_collate_fn",
    # photometric
    "Normalize",
    "ColorJitter",
    "RandomAdjustSharpness",
    "GaussianBlur",
    "RandomAutocontrast",
    "RandomEqualize",
    "Grayscale",
    # retinal
    "CLAHE",
    "HistogramMatching",
    "MultiscaleRetinex",
    "MSRCR",
    "GammaCorrection",
    "ContrastEnhancement",
    "IlluminationCorrection",
    "GreenChannelExtraction",
    "BlueChannelEmphasis",
    "BackgroundPolynomialCorrection",
    "FundusROIMask",
    # denoising
    "GaussianDenoising",
    "MedianFiltering",
    "BilateralFiltering",
    "Deblurring",
    "Deconvolution",
    # morphological
    "Erosion",
    "Dilation",
    "Opening",
    "MorphologicalClosing",
    "ConnectedComponentFiltering",
]
