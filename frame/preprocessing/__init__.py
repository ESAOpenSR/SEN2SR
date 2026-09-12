"""FRAME preprocessing layer.

Converts supported Sentinel-2 input into a validated, model-ready
representation for the upstream, unmodified `sen2sr` SR path, while
preserving provenance/geospatial metadata and a separate validity mask.

See docs/FRAME_TECHNICAL_SPEC.md Section 7 for the design this implements,
and README.md in this package for the concrete contract and usage.

Public API:
    preprocess_rgbn, PreprocessedInput, RGBN_BANDS, RGBN_RESOLUTION_M
    RasterMetadata
    ValidityMask, DEFAULT_SCL_INVALID_CLASSES
    to_reflectance
    Exceptions: PreprocessingError and its subclasses (see .errors)
"""

from frame.preprocessing.errors import (
    InvalidInputScaleError,
    InvalidShapeError,
    MissingMetadataError,
    PreprocessingError,
    UnsupportedBandsError,
    UnsupportedResolutionError,
)
from frame.preprocessing.masks import DEFAULT_SCL_INVALID_CLASSES, ValidityMask
from frame.preprocessing.metadata import RasterMetadata, validate_metadata_matches_array
from frame.preprocessing.pipeline import (
    PROVEN_PATCH_SIZE,
    RGBN_BANDS,
    RGBN_RESOLUTION_M,
    RGBN_SR_VARIANT,
    PreprocessedInput,
    preprocess_rgbn,
)
from frame.preprocessing.reflectance import RAW_DIGITAL_NUMBER, REFLECTANCE, to_reflectance
from frame.preprocessing.validation import (
    reorder_bands,
    validate_bands,
    validate_resolution,
    validate_shape,
)

__all__ = [
    "preprocess_rgbn",
    "PreprocessedInput",
    "RGBN_BANDS",
    "RGBN_RESOLUTION_M",
    "RGBN_SR_VARIANT",
    "PROVEN_PATCH_SIZE",
    "RasterMetadata",
    "validate_metadata_matches_array",
    "ValidityMask",
    "DEFAULT_SCL_INVALID_CLASSES",
    "to_reflectance",
    "RAW_DIGITAL_NUMBER",
    "REFLECTANCE",
    "validate_shape",
    "validate_bands",
    "reorder_bands",
    "validate_resolution",
    "PreprocessingError",
    "InvalidShapeError",
    "UnsupportedBandsError",
    "UnsupportedResolutionError",
    "InvalidInputScaleError",
    "MissingMetadataError",
]
