# SA-CMF: Secondary Attention-Based Cross-Modal Fusion Framework

## 1. Introduction

This repository implements a **Secondary Attention–Based Cross-Modal Fusion (SA-CMF) framework** for deep-space target recognition. The proposed method integrates **2D spatial image features** and **1D temporal sequence features** through transformer-based encoding, cross-attention alignment, and **mutual information–driven adaptive fusion**.

Unlike conventional cross-modal fusion methods that rely solely on attention weights, SA-CMF explicitly models **inter-modal statistical dependency** using a **Copula-based mutual information (MI) formulation**, enabling dynamic importance reweighting, redundancy suppression, and robustness under heterogeneous and low-SNR conditions.

---

## 2. Framework Overview

### 2.1 Overall Pipeline

The SA-CMF framework consists of the following stages:

1. **Modality-specific feature extraction**  
   - 2D spatial image features are extracted using an InceptionV3 backbone with enhancement preprocessing.
   - 1D temporal sequence features are constructed from radiation intensity time-series via temporal segmentation and embedding.

2. **Transformer encoding**  
   Independent Transformer Encoders are applied to each modality to preserve intra-modal structural characteristics.

3. **Cross-attention alignment**  
   Bidirectional cross-attention explicitly models inter-modal dependencies between spatial and temporal representations.

4. **Mutual information–driven adaptive fusion**  
   - Feature distributions are normalized via ECDF mapping.
   - A Gumbel Copula is employed to estimate cross-modal mutual information.
   - Estimated MI is used to adaptively update modality importance weights, suppressing redundant information.

5. **Prediction and inference**  
   The fused representation supports multi-object detection, classification, and visualization with consistent training–inference behavior.

---

## 3. Environment & Dependencies

- Python >= 3.8
- PyTorch >= 1.10
- torchvision
- numpy
- scipy
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Dataset Preparation

Due to data confidentiality, the dataset itself is not publicly released. However, the expected directory structure and data format are as follows:

```text
data/
├── train/
│   ├── Image-A/
│   ├── Image-B/
│   ├── Seq-A/
│   └── Seq-B/
├── val/
└── test/
```

- **Image-A / Image-B**: Paired 2D images from different views or sensors.
- **Seq-A / Seq-B**: Corresponding 1D temporal sequences (e.g., `.npy` or `.txt`).
- Files are aligned by identical filenames or indices across modalities.

The data loading logic is implemented in `train.py` and `test.py`.

---

## 5. Training

### 5.1 Training Objective

The training loss consists of:

- **Classification loss (Cls)**
- **Cross-modal similarity loss (Sim)**
- **Mutual information regularization (MI)**

Total loss:

```
L = L_cls + L_sim + λ * L_mi
```

MI is **only used during training** to update modality importance weights and does not introduce additional statistical estimation during inference.

### 5.2 Training Command

```bash
python train.py \
  --data_root ./data/train \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-4 \
  --lambda_mi 0.1
```

Model checkpoints and evaluation metrics are saved automatically.

---

## 6. Testing and Inference

The inference pipeline supports **dual-folder input**, **dynamic target detection**, and **multi-object inference**.

### 6.1 Inference Command

```bash
python test.py \
  --image_dir ./data/test/Image-A \
  --seq_dir ./data/test/Seq-A \
  --ckpt ./checkpoints/best.pth \
  --vis
```

### 6.2 Outputs

```text
outputs/
├── detections/
├── heatmaps/
└── fusion_weights/
```

Visualization results include detection overlays and modality importance maps.

---

## 7. Project Structure

```text
project/
│
├── train.py        # training loop and optimization
├── test.py         # inference and visualization
├── model.py        # Transformer and cross-attention modules
├── fusion.py       # Copula-based MI estimation and importance update
├── image_feature.py# image preprocessing and feature extraction
├── text_feature.py # temporal sequence feature construction
└── utils/
    ├── metrics
    ├── similarity
    └── evaluation tools
```

---

## 8. Notes

- Mutual information is used for **adaptive fusion control**, not as a direct decision variable.
- The framework ensures **training–inference consistency** by avoiding test-time distribution estimation.
- The design supports extension to additional modalities with minimal modification.

