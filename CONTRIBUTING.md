# Contributing

Contributions to [SEN2SR](https://github.com/ESAOpenSR/SEN2SR) are welcome.
Fork the repository, create a branch for your change, and submit a pull request
against `dev`. Include regression tests for bug fixes and document changes to
public behavior.

`dev` is the integration branch for testing changes. Merge contributor pull
requests into `dev` using a merge commit to preserve their original commits and
authorship. Keep follow-up fixes in separate commits. Once the combined changes
are tested, open a `dev` to `main` pull request and merge it with a merge commit
to retain that history.

## Running the tests

The tests run on the CPU with synthetic tensors and small PyTorch models.
They do not need a GPU, pretrained weights, downloaded imagery, or network access
after installation. CI runs them on Python 3.10 and 3.12.

From the repository root, create an environment and install the test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tests/requirements.txt
```

`timm` and its matching `torchvision` dependency are required by the package's
import checks, even though these tests do not load a pretrained model. Install
PyTorch first using the CPU index above to avoid downloading CUDA dependencies.

Run the suite and the formatting checks for the tiling code:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 python -m pytest -q
ruff check --no-fix sen2sr/utils.py tests
ruff format --check sen2sr/utils.py tests
```

Test files follow the package modules:

- `test_predict_large.py`: tiling, overlap cropping, small images, scaling, and invalid inputs.
- `test_utils.py`: tile iteration and boundary alignment.
- `test_nonreference.py`: non-reference SR wrapper and hard-constraint preparation.
- `test_referencex2.py`: Sentinel-2 band resampling, reconstruction, and the 2x wrapper.
- `test_referencex4.py`: the 4x pipeline, band order, and spectral constraint wiring.
- `test_tricks.py`: frequency filters and hard constraints.
- `test_xai.py`: attribution objectives, gradients, Gini coefficients, and KDE.
- `test_cnn.py`: CNN predictions, gradients, convolution fusion, attention maps, and checkpoint round trips.
- `test_swin.py`: Swin attention windows, rectangular inputs, gradients, and checkpoint round trips.

These tests verify the processing pipeline rather than pretrained model quality.

## Reporting issues and opening pull requests

Use the bug report form to provide a minimal reproduction, package versions,
and relevant model/tensor details. Use the feature request form to describe the
problem, proposed behavior, and how the improvement could be verified.

The pull request template asks for the change, related issue, and validation.
Target `dev` for development changes. GitHub uses these templates from the
default branch, so they become available after this work reaches `main`.

To measure package coverage, preload PyTorch and timm before enabling tracing
(their native imports failed under coverage tracing in the tested environment):

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 python - <<'PY'
import torch
import timm
import pytest

raise SystemExit(pytest.main(["-q", "--cov=sen2sr", "--cov-report=term-missing"]))
PY
```

The cases from [PR #7](https://github.com/ESAOpenSR/SEN2SR/pull/7) are covered by
ordinary pytest regression tests in `test_predict_large.py`, along with the
smaller images that required follow-up fixes. Run just these with:

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_predict_large.py -q
```
