# SA-CMF-Cross-Modal-Fusion-Framework
We propose a Secondary Attention-Based Cross-Modal Fusion (SA-CMF) framework, which exploits modality-specific attention to preserve unique feature characteristics, employs cross-attention to model inter-modal dependencies, and incorporates mutual information–based adjustment to suppress redundancy and mitigate distribution discrepancies.

This project implements a Mutual Information–Driven Cross-Modal Fusion Framework for deep-space target recognition, integrating 2D spatial image features and 1D sequence features via cross-attention, transformer encoding, and Copula-based mutual information modeling.

The project implements a complete cross-modal learning pipeline integrating spatial image features and temporal sequence features via transformer-based encoding, cross-attention alignment, and mutual information–driven adaptive fusion. The framework explicitly models cross-modal dependency using a Copula-based mutual information formulation, enabling dynamic importance reweighting and effective redundancy suppression during fusion. The design ensures strong consistency between training and inference while supporting multi-object detection and visualization in complex scenes.

project/
│
├── train.py
│   ├── dataset loading & batching
│   ├── training loop
│   ├── loss computation (Cls + Sim + MI)
│   └── metric evaluation & checkpointing
│
├── test.py
│   ├── dual-folder input (Image-A/B, Text-A/B)
│   ├── dynamic target detection
│   ├── multi-object inference
│   └── visualization & post-processing
│
├── model.py
│   ├── TransformerEncoder
│   ├── CrossAttention
│   └── CrossAttentionModel (core network)
│
├── fusion.py
│   ├── ECDF mapping
│   ├── Gumbel Copula
│   ├── Mutual Information estimation
│   └── Importance update (Im_img', Im_onedim')
│
├── image_feature.py
│   ├── image enhancement
│   ├── InceptionV3 feature extraction
│   └── dual 2D feature generation
│
├── text_feature.py
│   ├── sequence loading
│   ├── temporal split
│   └── 1D feature tensor construction
│
└── utils/
    ├── metrics
    ├── similarity
    └── evaluation tools
