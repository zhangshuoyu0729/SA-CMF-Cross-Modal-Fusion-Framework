# SA-CMF: Secondary Attention-Based Cross-Modal Fusion Framework

## 1. Introduction

This repository implements a **Secondary Attention-Based Cross-Modal Fusion (SA-CMF)** framework for deep-space target recognition. The proposed method integrates **2D spatial features** and **1D sequence features** through transformer-based encoding, cross-attention alignment, and **mutual-information-driven adaptive fusion**.

Unlike conventional cross-modal fusion methods that rely only on attention weights, SA-CMF explicitly models **inter-modal statistical dependency** using a **Copula-based mutual information (MI)** formulation. This design supports dynamic importance reweighting, redundancy suppression, and robust recognition under heterogeneous and low-SNR sensing conditions.

---

## 2. Framework Overview

### 2.1 Overall Pipeline

The SA-CMF framework consists of the following stages:

1. **Modality-specific feature extraction**

   - 2D spatial features are extracted using an InceptionV3 backbone with enhancement preprocessing.
   - 1D sequential features are extracted using a Transformer-based encoder with sequence-aware preprocessing.

2. **Transformer encoding**

   Independent Transformer encoders are applied to each modality to preserve intra-modal structural characteristics.

3. **Cross-attention alignment**

   Cross-attention explicitly models inter-modal dependencies between spatial and sequence representations.

4. **Mutual-information-driven adaptive fusion**

   - Feature distributions are normalized through ECDF mapping.
   - A Gumbel Copula is used to estimate cross-modal mutual information.
   - Estimated MI adaptively updates modality importance weights and suppresses redundant information.

5. **Prediction and inference**

   The fused representation supports multiclass target recognition with consistent training-inference behavior.

---

## 3. Environment and Dependencies

- Python >= 3.8
- TensorFlow
- OpenCV
- NumPy
- Pillow
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Dataset Preparation

The expected directory structure and data format are as follows:

```text
data/
|-- train/
|   |-- 2D/
|   `-- 1D/
|-- val/
`-- test/
```

The data loading logic is implemented in `train.py` and `test.py`.

---

## 5. Training

### 5.1 Training Objective

The training loss consists of:

- **Classification loss (Cls)**
- **Cross-modal similarity loss (Sim)**
- **Mutual information regularization (MI)**

Total loss:

```text
L = L_cls + lambda_sim * L_sim + lambda_mi * L_mi
```

MI is used during training to update modality importance weights and guide redundancy-aware fusion. The final prediction is still produced from the fused representation.

### 5.2 Training Command

```bash
python train.py
```

Model checkpoints and evaluation metrics are saved automatically according to the paths configured in `train.py`.

---

## 6. Testing and Inference

The inference pipeline supports **dual-folder input**, **dynamic target detection**, and **multi-object inference**.

### 6.1 Inference Command

```bash
python test.py
```

### 6.2 Outputs

```text
outputs/
|-- detections/
|-- heatmaps/
`-- fusion_weights/
```

Visualization results include detection overlays and modality importance maps.

---

## 7. Evaluation Metrics

SA-CMF targets practical deep-space multiclass recognition. The evaluation protocol combines metrics for recognition accuracy, fusion quality, missing-modality robustness, and deployment efficiency.

The metric definitions are collected in `metrics.py` for reusable research-code evaluation. The repository provides the framework structure and metric implementations; dataset files, trained weights, and deployment-specific reproduction scripts are maintained separately.

### 7.1 Multiclass Precision, Recall, and F1

Precision, Recall, and F1 are computed from multiclass prediction statistics. The main reported F1 score uses a sample-distribution-aware weighted multiclass setting for imbalanced target categories and engineering deployment scenarios.

The evaluation code reports weighted, micro, macro, and per-class statistics with explicit names. Weighted F1 is computed directly from multiclass predictions and labels, while macro scores provide supplementary class-balanced references.

### 7.2 Modality Complementarity Gain (MCG)

MCG measures the effective complementary improvement provided by the fused representation. In the implementation, MCG is evaluated after modality alignment, common-scale normalization, and redundancy-aware regulation.

The code represents this chain as MI-guided redundancy suppression -> fused representation -> complementary gain. Cross-attention and MI-guided importance updating produce modality-aware representations. mcg_components_from_fusion_chain() estimates redundant overlap and reports the residual complementary gain of the fused representation over the single-modality/redundancy baseline.

### 7.3 Robustness@Missing-Modality (R@MM)

R@MM evaluates normalized robustness retention when one sensing modality is unavailable. The implementation follows a missing-modality evaluation path rather than only reporting the final classification score.

During evaluation, one modality can be replaced by a missing input, and missing_modality_evaluation() reports accuracy retention, confidence retention, feature-response stability, and the final R@MM score. The final value is computed as the normalized retention of the missing-modality setting relative to the full-modality setting.

### 7.4 Inference Time and Samples/s

Inference time and samples/s describe different deployment properties. Inference time reports latency-oriented measurements, while samples/s reports throughput-oriented measurements.

The code separates the two benchmark views. summarize_latency_benchmark() reports single-sample and batch-forward latency, while summarize_throughput_benchmark() reports batched processing rate in samples/s. These values are reported as separate deployment measurements because batch size, hardware scheduling, preprocessing, and parallel execution affect them differently.

---

## 8. Project Structure

```text
project/
|-- train.py        # Training loop and evaluation metrics
|-- test.py         # Inference, visualization, and timing statistics
|-- metrics.py      # Multiclass, fusion, robustness, and timing metrics
|-- model.py        # Transformer and cross-attention modules
|-- fusion.py       # Copula-based MI estimation and importance update
|-- TWOD_feature.py # 2D spatial feature preprocessing and extraction
|-- ONED_feature.py # 1D sequence feature construction
|-- attention.py    # Attention utilities
`-- transfor.py     # Feature transformation
```
