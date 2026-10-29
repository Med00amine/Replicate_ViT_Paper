# Replicating the Vision Transformer (ViT) Paper in PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Med00amine/Replicate_ViT_Paper/blob/main/VIT_Paper.ipynb)

This notebook builds the Vision Transformer from **[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)** (Dosovitskiy et al., 2020) from scratch in PyTorch. It then compares the from-scratch model with a pretrained `torchvision` ViT-B/16 used for transfer learning on a small 3-class food dataset (pizza, steak, sushi).

## Key Takeaway

| Model | Params | Size | Test loss | Test accuracy (epoch 10) |
|---|---:|---:|---:|---:|
| ViT-Base, trained from scratch | ~85.8M | ~327 MB | 1.03 | 52.4% |
| Pretrained ViT-B/16 (frozen backbone, new head) | ~85.8M (only ~2.3K trainable) | ~327 MB | 0.21 | **91.8%** |

The from-scratch model stays near **chance level (~33%)** for most of training. Its 52% at the final epoch is noisy, not a sign that it learned well. This matches a main point of the paper: transformers lack the built-in assumptions about images that CNNs have, so they need **large-scale pretraining** (ImageNet-21k / JFT-300M) to work well. With a few hundred images, transfer learning wins clearly.

## What's Implemented

Each part maps to an equation or table in the paper:

| Component | Paper reference | Class |
|---|---|---|
| Patch + position embeddings, class token | Eq. 1 | `PatchEmbedding` |
| Multi-head self-attention with pre-LayerNorm | Eq. 2 | `MultiheadSelfAttentionBlock` |
| MLP block (GELU, dropout) | Eq. 3 | `MLPBlock` |
| Transformer encoder block with residuals | Eq. 2–3 | `TransformerEncoderBlock` |
| Full model + classification head | Eq. 4, Table 1 | `VITEncoder` |

Hyperparameters follow **ViT-Base** (Table 1): 16×16 patches, 768-dim embeddings, 12 layers, 12 heads, 3072-dim MLP, dropout 0.1.

## Notebook Walkthrough

1. Environment setup (torch ≥ 1.12, torchvision ≥ 0.13)
2. Download the pizza, steak, sushi dataset
3. Transforms (resize to 224×224) and DataLoaders (batch size 32)
4. Build the ViT components step by step
5. Train the from-scratch ViT (Adam, lr 1e-3, 10 epochs)
6. Load pretrained `vit_b_16`, freeze the backbone, and replace the head
7. Train the classifier head (10 epochs)
8. Predict on a custom image

## How to Run

**Easiest:** click the *Open in Colab* badge above and use a GPU runtime.

**Locally:**

```bash
git clone https://github.com/Med00amine/Replicate_ViT_Paper.git
cd Replicate_ViT_Paper
pip install -r requirements.txt
jupyter notebook VIT_Paper.ipynb
```

The notebook downloads the dataset and helper modules (`going_modular`, `helper_functions.py`) from [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning) automatically. A CUDA GPU is recommended; CPU works but is slow.

## What I'd Try Next

- Data augmentation (RandAugment, Mixup) and a learning-rate warmup + cosine schedule, as used in the paper
- A smaller ViT (e.g. ViT-Tiny) to see whether from-scratch training improves on small data
- Fine-tuning the whole pretrained model instead of only the head
- Visualizing attention maps

## References

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Daniel Bourke, [Zero to Mastery: PyTorch for Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning), Section 08 (Paper Replicating), which this project follows
