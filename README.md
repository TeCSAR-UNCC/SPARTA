# SPARTA

## Overview

SPARTA is a novel approach for **human-centric video anomaly detection (VAD)** utilizing **Spatio-Temporal Pose and Relative Pose (ST-PRP) tokenization** combined with a **transformer-based architecture, SPARTA**. This repository provides the implementation of the **SPARTA model**, which introduces an innovative **Unified Encoder Twin Decoders (UETD) Transformer** for enhanced anomaly detection in human activity videos.

This method outperforms existing pose-based anomaly detection techniques and competes with pixel-based approaches while ensuring better privacy and lower bias.

## Features
- **Spatio-Temporal Pose and Relative Pose (ST-PRP) Tokenization**
- **Unified Encoder Twin Decoders (UETD) Transformer Architecture**
- **State-of-the-Art Performance on Multiple Benchmark Datasets**
- **Self-Supervised Learning for Robust Generalization**

## Installation
To set up the environment and dependencies:
```bash
pip install -r requirements.txt
```
Ensure you have a compatible GPU for training and inference.

## Datasets
The following benchmark datasets were used for evaluation:
- **ShanghaiTech (SHT)**
- **HR-ShanghaiTech (HR-SHT)**
- **Charlotte Anomaly Dataset (CHAD)**
- **Northwestern Polytechnical University Campus (NWPUC)**

## Model Architecture
SPARTA consists of:
1. **ST-PRP Tokenization** - A novel method capturing human motion.
2. **UETD Transformer Core** - A unified encoder with twin decoders:
   - **Current Target Decoder (CTD)**
   - **Future Target Decoder (FTD)**
3. **Hybrid Scoring (HS)** - Combination of CTD and FTD for anomaly detection.

### Key Figures
#### Figure 1: SPARTA Architecture
![SPARTA Architecture](figs/SPARTA.png)

#### Figure 2: ST-PRP Tokenization Schema
![ST-PRP Tokenization](figs/tokenization.png)

### Key Tables
#### Table 1: Performance Comparison with Pose-Based Methods
| Method | SHT AUC (%) | HR-SHT AUC (%) | CHAD AUC (%) | NWPUC AUC (%) | Avg. AUC (%) |
|--------|------------|---------------|-------------|-------------|-------------|
| STG-NF | 85.90 | 87.40 | 60.60 | 62.56 | 74.11 |
| SPARTA-C | 85.10 | 86.70 | 66.12 | 62.69 | 75.15 |
| SPARTA-F | 83.19 | 83.70 | 66.61 | 62.29 | 73.94 |
| **SPARTA-H** | **85.75** | **87.23** | **67.04** | **63.48** | **75.87** |

#### Table 2: Performance Comparison with Pixel-Based Methods
| Method | SHT AUC (%) | NWPUC AUC (%) | Avg. AUC (%) |
|--------|------------|-------------|-------------|
| MNAD | 70.50 | 62.50 | 66.50 |
| OG-Net | - | 62.50 | - |
| MemAE | 71.20 | 61.90 | 66.55 |
| **SPARTA-H** | **85.75** | **63.48** | **74.62** |

## Usage
### Training
```bash
python train.py --dataset SHT --epochs 30 --batch_size 256
```
### Evaluation
```bash
python evaluate.py --dataset SHT --model_path checkpoints/sparta_h.pth
```
### Inference
```bash
python infer.py --video_path data/test_video.mp4 --model_path checkpoints/sparta_h.pth
```

## Citation
If you find this work useful, please cite:
```
@article{posewatch2024,
  title={Human-Centric Video Anomaly Detection Through Spatio-Temporal Pose Tokenization and Transformer},
  author={Ghazal Alinezhad Noghre, Armin Danesh Pazho, Hamed Tabkhi},
  journal={IEEE},
  year={2024}
}
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any inquiries or collaborations, contact **[Author Name]** at **[email@example.com]** or open an issue on GitHub.

---
This repository provides a strong foundation for **privacy-aware, pose-based anomaly detection** in videos. Contributions and pull requests are welcome!

