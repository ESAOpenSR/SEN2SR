# Changelog

## [Unreleased]

## 0.8.6

Changes below are relative to the published 0.8.5 package.

### Fixed

- Correct tiled prediction for rectangular images, single tiles, single-tile axes,
  and images smaller than 128 pixels. Small tiles use edge padding and are cropped
  back to the input bounds. Output dimensions, dtype, and CPU placement are preserved.
- Validate tile overlap, input dimensions, and model output shapes to report invalid
  inputs clearly instead of producing incomplete images or assignment errors.
- Pass tensors to the Gaussian and sigmoid filters' exponentials so these filters
  run without scalar input errors.
- Repair local attribution maps' NumPy/PyTorch conversions, preserve dimensions
  for singleton inputs, and return finite maps for zero gradients. Use NumPy's
  supported trapezoidal integration API.
- Support CNN convolution fusion when `bias=False`.
- Import Swin helpers from `timm.layers` instead of its deprecated import path.
- Restore the `timm` runtime dependency declared in the published 0.8.5 wheel
  and require a modern `torchvision` for compatible dependency resolution.

### Added

- CPU regression tests covering tiling, SR wrappers, spectral constraints,
  attribution, CNN/Swin predictions, gradients, and checkpoint round trips.
- Python 3.10 and 3.12 CI, contribution guidance, pull request and issue templates.
- Publish to PyPI automatically when a stable GitHub release is published, after
  building and testing the installed wheel on Python 3.10 and 3.12. Requires the
  project owner to configure PyPI Trusted Publishing once.
- Machine-readable paper citation in `CITATION.cff`.

### Changed

- Document the intentional CNN block flow used by published SEN2SRLite checkpoints;
  predictions from those checkpoints retain the existing architecture.
- Replace the placeholder changelog with release notes for SEN2SR. Historical
  release details have not been reconstructed.
- Align the README license badge and package metadata with the existing CC0-1.0
  `LICENSE`, and use absolute README image URLs for PyPI rendering.

### Contributors

Thanks to [Akash-design-prog](https://github.com/Akash-design-prog) for the tiled
prediction fixes in [#6](https://github.com/ESAOpenSR/SEN2SR/pull/6) and
[#7](https://github.com/ESAOpenSR/SEN2SR/pull/7). The original contributor commits
are retained in the Git history.

[Unreleased]: https://github.com/ESAOpenSR/SEN2SR/compare/main...dev
