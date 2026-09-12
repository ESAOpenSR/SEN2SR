# SEN2SR
### *A radiometrically and spatially consistent super-resolution framework for Sentinel-2*
<p align="center">
  <img src="https://github.com/user-attachments/assets/9d6345e3-0c5a-4b53-b8ff-e0aa28f99c96" width="100%">
</p>




<p align="center">
<a href='https://pypi.python.org/pypi/sen2sr'>
    <img src='https://img.shields.io/pypi/v/sen2sr.svg' alt='PyPI' />
</a>
<a href="https://creativecommons.org/publicdomain/zero/1.0/" target="_blank">
    <img src="https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg" alt="License">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://pycqa.github.io/isort/" target="_blank">
    <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
</a>
<a href="https://huggingface.co/tacofoundation/SEN2SR" target="_blank">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg" alt="isort">
</a>
</p>




---

**GitHub**: [https://github.com/ESAOpenSR/sen2sr](https://github.com/ESAOpenSR/sen2sr) 🌐

**PyPI**: [https://pypi.org/project/sen2sr/](https://pypi.org/project/sen2sr/) 🛠️

**Preprint**: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5247739](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5247739) 📄

---

## **Table of Contents**

- [**Overview**](#overview-)
- [**Installation**](#installation-)
- [**From 10m and 20m Sentinel-2 bands to 2.5m**](#from-10m-and-20m-sentinel-2-bands-to-25m)
- [**From 10m Sentinel-2 bands to 2.5m**](#from-10m-sentinel-2-bands-to-25m)
- [**From 20m Sentinel-2 bands to 10m**](#from-20m-sentinel-2-bands-to-10m)
- [**Predict on large images**](#predict-on-large-images)
- [**Estimate the Local Attention Map**](#estimate-the-local-attention-map)

## **Overview**

**sen2sr** is a Python package designed to enhance the spatial resolution of Sentinel-2 satellite images to 2.5 meters using a set of neural network models. 

| Model | Description | Run Link |
|--------|-------------|---------|
| **Run SENSRLite** | A lightweight SR model optimized for running fast! | <a target="_blank" href="https://colab.research.google.com/drive/1x65GoI5hOfgX61LhtbATSBm7HySUHSw9?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **Run SENSR** | Our lightweight SR model! | <a target="_blank" href="https://colab.research.google.com/drive/1MdhdsPwJyV3f0jUgW_WaVeO3-aqA80OG?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **Run LDSR-S2 & SEN2SR** | ESAOpenSR's flagship  latent diffusion model | <a target="_blank" href="https://colab.research.google.com/drive/1NJuyswsquOLMFc_AP93P_5QcZnbNhGuB?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## **Installation**

Install the **SEN2SRLite** version using pip:

```bash
pip install sen2sr mlstac git+https://github.com/ESDS-Leipzig/cubo.git
```

For using the full version of **SEN2SR**, which employs the Mamba architecture, install as follows:

> ⚠️ **Warning**  
> `mamba-ssm` **requires a GPU runtime** and **CUDA version > 12** to function properly.  
> Installation may take a significant amount of time. **Please avoid interrupting the process.**


1. Create a fresh Conda environment:

```bash
conda create -n test_env python=3.11
conda activate test_env
```

2. Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. Attempt to install mamba-ssm using --no-build-isolation:

```bash
pip install mamba-ssm --no-build-isolation
```

4. Install the remaining dependencies:
```bash
pip install sen2sr mlstac git+https://github.com/ESDS-Leipzig/cubo.git
```

Adapted from this [state-spaces/mamba issue](https://github.com/state-spaces/mamba/issues/662).


## From 10m and 20m Sentinel-2 bands to 2.5m


This example demonstrates the use of the `SEN2SRLite` model to enhance the spatial resolution of Sentinel-2 imagery. A 
Sentinel-2 L2A data cube is created over a specified region and time range using the cubo library, including both 10 m 
and 20 m bands. The pretrained model, downloaded via mlstac, takes a single normalized sample as input and predicts a 
HR output. The visualization compares the original RGB composite to the super-resolved result.  
LDSR-S2 is also available.


```python
import mlstac
import torch
import cubo

# Download the model
mlstac.download(
  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
  output_dir="model/SEN2SRLite",
)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlstac.load("model/SEN2SRLite").compiled_model(device=device)
model = model.to(device)

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=128,
    resolution=10
)

# Prepare the data to be used in the model, select just one sample 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)

# Apply model
superX = model(X[None]).squeeze(0)
```

<p align="center">
  <img src="assets/srimg01.png" width="100%">
</p>


## From 10m Sentinel-2 bands to 2.5m


This example demonstrates the use of the `SEN2SRLite NonReference_RGBN_x4` model variant to enhance the spatial resolution 
of only the 10 m Sentinel-2 bands: red (B04), green (B03), blue (B02), and near-infrared (B08). A Sentinel-2 L2A data cube is created using the cubo library for a specific location and date range. The input is normalized and passed to a pretrained non-reference model optimized for RGB+NIR inputs. 


```python
import mlstac
import torch
import cubo

# Download the model
mlstac.download(
  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
  output_dir="model/SEN2SRLite_RGBN",
)

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B04", "B03", "B02", "B08"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=128,
    resolution=10
)


# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)
X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Load the model
model = mlstac.load("model/SEN2SRLite_RGBN").compiled_model(device=device)

# Apply model
superX = model(X[None]).squeeze(0)
```

<p align="center">
  <img src="assets/srimg02.png" width="100%">
</p>


## From 20m Sentinel-2 bands to 10m

This example demonstrates the use of the `SEN2SRLite Reference_RSWIR_x2` model variant to enhance the spatial resolution of the 20 m Sentinel-2 bands: red-edge (B05, B06, B07), shortwave infrared (B11, B12), and near-infrared (B8A) to 10 m. 


```python
import mlstac
import torch
import cubo

# Download the model
mlstac.download(
  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/Reference_RSWIR_x2/mlm.json",
  output_dir="model/SEN2SRLite_Reference_RSWIR_x2",
)

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=128,
    resolution=10
)

# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)

# Load the model
model = mlstac.load("model/SEN2SRLite_Reference_RSWIR_x2").compiled_model(device=device)
model = model.to(device)

# Apply model
superX = model(X[None]).squeeze(0)
```

<p align="center">
  <img src="assets/srimg03.png" width="100%">
</p>


## **Predict on large images**

This example demonstrates the use of `SEN2SRLite NonReference_RGBN_x4` for super-resolving large Sentinel-2 RGB+NIR images by chunking the 
input into smaller overlapping tiles. Although the model is trained to operate on fixed-size 128×128 patches, the `sen2sr.predict_large` utility automatically segments larger inputs into these tiles, applies the model to each tile independently, and then reconstructs the full image. An overlap margin (e.g., 32 pixels) is introduced between tiles to minimize edge artifacts and ensure continuity across tile boundaries.

```python
import mlstac
import sen2sr
import torch
import cubo

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=1024,
    resolution=10
)

# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)
X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Load the model
model = mlstac.load("model/SEN2SRLite").compiled_model(device=device)


# Apply model
superX = sen2sr.predict_large(
    model=model,
    X=X, # The input tensor
    overlap=32, # The overlap between the patches
)
```

<p align="center">
  <img src="assets/srimg05.png" width="95%">
</p>


### Estimate the Local Attention Map


This example computes the Local Attention Map (LAM) to analyze the model's spatial sensitivity 
and robustness. The input image is scanned with a sliding window, and the model's attention is 
estimated across multiple upscaling factors. The resulting KDE map highlights regions where 
the model focuses more strongly, while the robustness vector quantifies the model's stability 
to spatial perturbations.


```python
import mlstac
import sen2sr
import torch
import cubo

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B04", "B03", "B02", "B08"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=128,
    resolution=10
)


# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)
X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Load the model
#mlstac.download(
#  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
#  output_dir="model/SEN2SRLite_RGBN",
#)
model = mlstac.load("model/SEN2SRLite_RGBN").compiled_model(device=device)

# Apply model
kde_map, complexity_metric, robustness_metric, robustness_vector = sen2sr.lam(
    X=X, # The input tensor
    model=model, # The SR model
    h=240, # The height of the window
    w=240, # The width of the window
    window=32, # The window size
    scales = ["2x", "3x", "4x", "5x", "6x"]
)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(kde_map)
ax[0].set_title("Kernel Density Estimation")
ax[1].plot(robustness_vector)
ax[1].set_title("Robustness Vector")
plt.show()
```

<p align="center">
  <img src="assets/srimg04.png" width="95%">
</p>

