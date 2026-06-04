# Approximating Video Semantic Space: A Linear-Time Hybrid Mamba-ResNet Architecture

**Multimodal Research**  
**Author:** Krishna Anchal

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Mamba%20%7C%20ResNet--18-success.svg)]()

##  Project Overview
The computational complexity of traditional Vision Transformers creates severe bottlenecks in multimodal audio-visual retrieval. This repository contains the code, evaluation suites, and training logs for a highly efficient, **linear-time $O(N)$ architecture**. 

The primary objective is to determine if a complete video sequence can be accurately approximated using only a **single static visual frame** and its **corresponding audio track**, entirely bypassing the massive computational overhead of dense temporal video processing.

##  Main Structural Contributions
To achieve this highly efficient approximation, this architecture moves beyond standard MLP concatenation baselines by introducing:
1. **Adaptive Gating Mechanism:** A dynamic fusion layer inspired by CMRF-Net that contextually weights visual anchors and continuous audio waveforms to suppress background noise and prevent single-modality dominance.
2. **$O(N)$ Mamba Fusion Core:** The complete replacement of quadratic cross-attention with a hardware-aware parallel sequence engine (State Space Model).
3. **Dual-Objective Distillation Protocol:** A custom joint objective function combining MSE and Cosine similarities. This aligns the temporal trajectory while maximizing global angular proximity, preventing the dimensional collapse chronically observed in pure latent space regression models.
4. **Stable Latent Anchoring:** The strategic deployment of a frozen ResNet-18 visual teacher to provide a dense, highly localized 512-D continuous latent space anchor.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/3ed756f2-a0ba-4c5a-ae07-cd87c72e602f" />


##  Repository Structure
The codebase is modularized for scalable engineering and rapid ablation testing.

**Main Training Notebooks** are **multi-modal-resnet-5k.ipynb** & **multi-modal-resnet-10k.ipynb**

```text
AIMS Multi-modal Research Intern/
│
├── data/                           # Dataset loaders and augmentations (VATEX, UCF101)
├── modules/                        # Core Architectural Components
│   ├── evaluation.py               # Downstream retrieval task logic
│   ├── gating.py                   # Adaptive Softmax Gating mechanism
│   ├── loss.py                     # Dual-Objective Distillation protocol
│   ├── mamba_engine.py             # O(N) State Space Model (SSM) blocks
│   └── pipeline.py                 # Master Hybrid Mamba-ResNet integration
│
├── tests/                          # Unit Testing & Benchmarking
│   ├── benchmark_suite.py          # Master evaluation and terminal reporting
│   ├── test_dataset.py             
│   ├── test_eval.py                
│   ├── test_gating.py              
│   ├── test_loss.py                
│   ├── test_mamba.py               
│   └── test_pipeline.py            
│
├── train.py                        # Execution loop for local training
├── train.py
├── train.py
├── multi-modal-resnet-5k.ipynb     # Cloud sprint log (5,000 video pairs)
├── multi-modal-resnet-10k.ipynb    # Cloud sprint log (10,000 video pairs)
└── README.md
