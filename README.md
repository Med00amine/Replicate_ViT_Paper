# Replicate ViT Paper

This project is a PyTorch notebook that reproduces the main ideas behind the Vision Transformer (ViT) paper, then compares a custom ViT implementation with a pretrained `torchvision` ViT model on a small image classification dataset.

The notebook trains on the `pizza_steak_sushi` dataset from Daniel Bourke's `pytorch-deep-learning` repository and classifies images into three classes:

- pizza
- steak
- sushi

## Project Files

```text
.
|-- VIT_Paper.ipynb
|-- README.md
`-- .gitattributes
```

## What the Notebook Does

`VIT_Paper.ipynb` walks through:

1. Installing/checking PyTorch and TorchVision versions.
2. Downloading the pizza, steak, sushi dataset.
3. Creating image transforms, datasets, and dataloaders.
4. Building ViT components from scratch:
   - patch embedding
   - multi-head self-attention block
   - MLP block
   - transformer encoder block
   - full ViT encoder/classifier
5. Training the custom ViT model.
6. Loading a pretrained `torchvision.models.vit_b_16` model.
7. Freezing pretrained weights and replacing the classifier head.
8. Training the pretrained ViT classifier head.
9. Running prediction on a custom pizza image.

## Requirements

The notebook expects Python 3 and the following main packages:

```bash
pip install torch torchvision torchaudio torchinfo matplotlib requests
```

The notebook also downloads helper code from:

```text
https://github.com/mrdbourke/pytorch-deep-learning
```

Specifically, it uses:

- `going_modular.going_modular.data_setup`
- `going_modular.going_modular.engine`
- `going_modular.going_modular.predictions`
- `helper_functions.py`

## Recommended Environment

- Python 3.9+
- PyTorch 1.12+ or PyTorch 2.x
- TorchVision 0.13+
- Jupyter Notebook, JupyterLab, or Google Colab
- CUDA-compatible GPU recommended, but CPU can run the notebook more slowly

## How to Run

1. Clone the repository:

```bash
git clone <your-repo-url>
cd Replicate_ViT_Paper
```

2. Install dependencies:

```bash
pip install torch torchvision torchaudio torchinfo matplotlib requests
```

3. Start Jupyter:

```bash
jupyter notebook
```

4. Open and run:

```text
VIT_Paper.ipynb
```

The notebook will download the dataset and helper scripts automatically if they are not already available.

## Model Results

The notebook reports approximately:

| Model | Model size | Test loss | Test accuracy |
| --- | ---: | ---: | ---: |
| Custom replicated ViT | 327 MB | ~1.0334 | ~52.37% |
| Pretrained ViT-B/16 | 327 MB | ~0.2060 | ~91.76% |

The pretrained ViT performs much better because it was pretrained on a much larger dataset, while the custom ViT is trained from scratch on a small dataset of roughly a few hundred images.

## Notes

- The custom model uses ViT-Base-style hyperparameters such as 16x16 patches, 768-dimensional embeddings, 12 transformer layers, and 12 attention heads.
- The dataset is small, so the from-scratch model is mainly useful for learning the architecture rather than achieving state-of-the-art accuracy.
- The pretrained model is better suited for practical transfer learning.

## Reference

This project is inspired by:

- Vision Transformer paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Daniel Bourke's PyTorch deep learning materials: `mrdbourke/pytorch-deep-learning`
