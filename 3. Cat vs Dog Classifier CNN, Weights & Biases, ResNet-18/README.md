# Cat vs Dog Classifier 🐱🐕
A CNN built from scratch in PyTorch, tracked with Weights & Biases(link to the statistics below), tuned with a Bayesian hyperparameter sweep, and benchmarked against a pretrained ResNet-18.


## Overview

Three phases building on each other:

1. **Scratch CNN** — a 3-block convolutional network written from the ground up in PyTorch (~9M parameters)
2. **Experiment tracking** — every run logged to W&B, followed by a 20-trial Bayesian hyperparameter sweep
3. **Transfer learning** — a frozen pretrained ResNet-18 with a replaced classification head compared head-to-head against the scratch CNN

## Results

| Model | Trainable params | Val accuracy | Test accuracy |
|---|---|---|---|
| Scratch CNN | ~9,537,346 | 91.37% | 91.41% |
| ResNet-18 (frozen) | ~1,026 | 97.35% | 97.69% |

I have also uploaded the trained parameters of the two models for the particular architectures

## Dataset

[`microsoft/cats_vs_dogs`](https://huggingface.co/datasets/microsoft/cats_vs_dogs) from HuggingFace — 23,410 images, downloaded automatically at runtime. No manual download needed.

Split 70 / 10 / 20 into train / validation / test.



## Tech stack

- **PyTorch / torchvision** — model, training loop, transforms, pretrained ResNet
- **HuggingFace `datasets`** — data loading
- **Weights & Biases** — experiment tracking, hyperparameter sweeps, model artifacts
- **matplotlib** — visualisation


## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Weights & Biases account

You need a free W&B account — the notebook logs every run, sweep, parameters, and model artifact there.

1. Sign up at [wandb.ai](https://wandb.ai)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. `wandb.login()` will prompt you for it on first run

### 3. Hardware

Training on CPU is possible but slow (~90 min for one 15-epoch run; the 20-trial sweep would take many hours). A GPU is strongly recommended.

**Free GPU via Google Colab:**
- Open the notebook in Colab → `Runtime` → `Change runtime type` → `T4 GPU`
- Or use the official Colab VS Code extension to connect a local notebook to a Colab runtime

The `DEVICE` variable auto-detects CUDA, so the same code runs unchanged on CPU or GPU.


## Important Note on Checkpoints

The sweep runs `train()` 20 times, and every trial saves to the same `CHECKPOINT` path — so the file left on disk after a sweep is whatever the **last** trial produced, not necessarily the best one.

To get the best model:

1. Open your W&B project → **Sweeps** tab → sort runs by `val/acc`
2. Click into the best run → **Artifacts** tab
3. Download that specific `.pth` file
4. Point `CHECKPOINT` at it

If running on Colab, the checkpoint lives in temporary storage and is wiped when the session ends. Download it (or pull it from the W&B artifact) before closing.


## Predicting on your own images

`predict_your_image(model, image_path)` runs inference on any image you upload and displays it with the predicted label and confidence.
On Colab, mount your Drive, upload the image, and add the correct file path to the function

If running locally, just use a normal file path.


## Concepts covered

- Custom PyTorch `Dataset` (`__init__` / `__len__` / `__getitem__`) and `DataLoader`
- Data augmentation — random crop, horizontal flip, colour jitter
- CNN internals — Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout
- The PyTorch training loop: `zero_grad` → forward → `loss.backward()` → `optimizer.step()`
- `model.train()` vs `model.eval()` and why BatchNorm/Dropout behave differently
- `torch.no_grad()` during evaluation
- Adam optimiser, weight decay, learning rate scheduling (`StepLR`)
- W&B — `init`, `log`, `watch`, sweeps, artifacts
- Bayesian hyperparameter search
- Transfer learning — freezing a backbone, replacing the classification head
- Why pretrained models need ImageNet normalisation values
- Test set discipline — evaluate once, never tune against it
