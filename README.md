# "Why the face?": Exploring Robot Error Detection Using Instrumented Bystander Reactions

<div align="center">

**[Maria Teresa Parreira](https://www.mariateresaparreira.com/)<sup>1,2</sup>, [Ruidong Zhang](https://www.ruidongzhang.com/)<sup>1</sup>, [Sukruth Gowdru Lingaraju](https://www.sukruthgl.com/)<sup>2</sup>, [Alexandra Bremers](https://alexbremers.com/)<sup>2</sup>, Xuanyu Fang<sup>2</sup>, [Adolfo Ramirez-Aristizabal](https://www.ramirez-aristizabal.com/)<sup>3</sup>, [Manaswi Saha](https://homes.cs.washington.edu/~manaswi/)<sup>3</sup>, [Michael Kuniavsky](https://www.kuniavsky.com/)<sup>3</sup>, [Cheng Zhang](https://www.chengzhang.org/)<sup>1</sup>, [Wendy Ju](https://wendyju.com/)<sup>1,2</sup>**

<sup>1</sup>Cornell University | <sup>2</sup>Cornell Tech | <sup>3</sup>Accenture Labs

[Paper](submission/main.pdf) | [Supplementary Material](submission/supp_material.pdf)

</div>

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/a898c13e-4acc-45e9-a4c3-a1361067b23d" alt="Study Overview" width="100%">
</p>

## Abstract

How do humans recognize and rectify social missteps? We achieve social competence by looking around at our peers, decoding subtle cues from bystanders — a raised eyebrow, a laugh — to evaluate the environment and our actions. Robots, however, struggle to perceive and make use of these nuanced reactions.

By employing a novel **neck-mounted device** that records facial expressions from the chin region, we explore the potential of previously untapped data to capture and interpret human responses to robot error. First, we develop **NeckNet-18**, a 3D facial reconstruction model to map the reactions captured through the chin camera onto facial points and head motion. We then use these facial responses to develop a robot error detection model which **outperforms standard methodologies** such as using OpenFace or video data, generalizing well both for within- and across-participant data.

Through this work, we argue for expanding human-in-the-loop robot sensing, fostering more seamless integration of robots into diverse human environments, pushing the boundaries of social cue detection and opening new avenues for adaptable and sustainable robotics.

## Key Contributions

1. **NeckNet-18**: A lightweight 3D facial reconstruction model (ResNet-18 based) that converts IR camera data from a neck-mounted device into 52 facial Blendshapes and 3 head rotation parameters.

2. **Error Detection Models**: Machine learning models that detect robot errors from human facial reactions, achieving:
   - **84.7% accuracy** for single-participant models with only 5% training data
   - **5% better accuracy** than OpenFace-based methods for cross-participant generalization
   - **Superior performance** compared to RGB camera approaches

3. **Comprehensive Benchmark**: First study to systematically compare neck-mounted device data against conventional methods (OpenFace, RGB cameras) for error detection in HRI.

## System Overview

<p align="center">
  <img src="submission/FIGURE_setup.png" alt="Study Setup" width="70%">
</p>

Our system consists of two main components:

### 1. NeckNet-18: 3D Facial Reconstruction
- Converts IR camera images from NeckFace device into 3D facial expressions
- Outputs 52 Blendshape parameters + 3 head rotation angles
- Requires a short calibration round (~5 minutes) per participant

### 2. Error Detection Model
- Trained on reconstructed facial reactions to detect errors
- Supports both cross-participant and single-participant generalization
- Multiple architectures tested (GRU, LSTM, Transformer, MiniRocket, etc.)

## Repository Structure

```
.
├── submission/                           # Paper and supplementary materials
│   ├── main.pdf                         # Main paper
│   ├── supp_material.pdf                # Supplementary materials
│   └── *.png                            # Figures used in paper
│
├── code_neckface_training/              # NeckNet-18 training code
│   ├── train.py                         # Training script
│   ├── eval.py                          # Inference script
│   ├── data_preparation.py              # Data preprocessing
│   └── README.md                        # Detailed instructions
│
├── code_preprocessing/                   # Dataset creation pipelines
│   └── create_datasets/
│       ├── neckface_rgb/                # NeckFace IR & RGB frame extraction
│       └── neckface_openface_feature_extraction/  # OpenFace processing
│
├── code_timeseries/                     # Error detection models (time-series)
│   └── # Models: GRU, LSTM, BiLSTM, Transformer, MiniRocket, etc.
│
├── code_rgb/                            # Error detection models (frame-based)
│   └── # Models: ResNet34-based CNNs
│
└── stimulus_dataset_information.xlsx    # Stimulus video dataset metadata
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)
- NeckFace device (for data collection)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/badrobots-feat-neckface.git
cd badrobots-feat-neckface

# Install dependencies for NeckNet-18 training
cd code_neckface_training
pip install -r requirements.txt

# Install dependencies for error detection models
cd ../code_timeseries
pip install tsai torch torchvision
```

## Quick Start

### 1. NeckNet-18: Training 3D Facial Reconstruction

```bash
cd code_neckface_training

# Prepare your data (see code_neckface_training/README.md for details)
python data_preparation.py -p /path/to/participant/data

# Train NeckNet-18
python train.py -p /path/to/participant/data -ts 01 -g 0 -t 1 -o experiment_name

# Generate predictions
python eval.py --resume /path/to/checkpoint.tar -v /path/to/video.avi
```

**Pre-trained Model**: Download the NeckFace meta model trained on the original user study: [neckface_us_full_model.tar](https://cornell.box.com/s/cuqqeuz9eqyauwbkb1rkakqi6vul21op)

### 2. Error Detection: Training Classification Models

```bash
cd code_timeseries

# Train error detection model on NeckData
python train_model.py --dataset neckdata --model gru_fcn --folds 5

# Evaluate on test set
python evaluate_model.py --checkpoint best_model.pth --dataset neckdata
```

## Datasets

Our work uses four datasets for benchmarking:

1. **NeckData**: 55 features (52 Blendshapes + 3 head rotations) reconstructed from NeckFace IR cameras using NeckNet-18
2. **OpenData**: 49 features (Action Units, gaze, pose) extracted using OpenFace from RGB videos
3. **NeckFaceIR**: Raw IR video frames from NeckFace cameras (224×224, 12 fps)
4. **RGBData**: RGB video frames from participant-facing camera (224×224, 12 fps)

**Stimulus Videos**: 30 videos (10 human errors, 10 robot errors, 10 control) - see `stimulus_dataset_information.xlsx`

## Results

### NeckNet-18 Performance
- **MAE (Facial Motion)**: 34.0 ± 6.9
- **MAE (Head Rotation)**: 7.0 ± 1.3

### Error Detection Performance

| Model Type | Dataset | Accuracy | F1-Score |
|------------|---------|----------|----------|
| GRU_FCN | NeckData | **65.8%** | **63.7%** |
| gMLP | OpenData | 60.6% | 53.5% |
| GRU_FCN (Single-Participant, 5% training) | NeckData | **84.7%** | **84.2%** |
| InceptionTime (Single-Participant, 5% training) | OpenData | 78.8% | 78.2% |

**Key Finding**: NeckData models outperform OpenData models, especially with limited training data, demonstrating the value of 3D facial reconstruction from neck-mounted sensors.

## Hardware Requirements

### NeckFace Device
- **Cameras**: 2× IR cameras (Arducam) mounted on neck band
- **Lighting**: 2× 850 nm IR LEDs per camera
- **Controller**: Raspberry Pi 4B with Arducam multi-camera adapter
- **Calibration**: iPhone 11 Pro with TrueDepth camera (for ground truth)

For device fabrication details, see [NeckFace: Continuously Tracking Full Facial Expressions on Neck-mounted Wearables](https://doi.org/10.1145/3463511)

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{parreira2025whyface,
  title={"Why the face?": Exploring Robot Error Detection Using Instrumented Bystander Reactions},
  author={Parreira, Maria Teresa and Zhang, Ruidong and Lingaraju, Sukruth Gowdru and Bremers, Alexandra and Fang, Xuanyu and Ramirez-Aristizabal, Adolfo and Saha, Manaswi and Kuniavsky, Michael and Zhang, Cheng and Ju, Wendy},
  booktitle={TBD},
  year={2025},
  organization={TBD}
}
```

## Related Work

- **NeckFace**: [Chen et al., 2021](https://doi.org/10.1145/3463511) - Original neck-mounted facial expression tracking system
- **BAD Dataset**: [Bremers et al., 2023](https://arxiv.org/abs/2301.11972) - Bystander Affect Detection dataset for HRI failure detection
- **Err@HRI Challenge**: [Spitale et al., 2024](https://arxiv.org/abs/2407.06094) - Multimodal error detection challenge

## Acknowledgments

This work was supported by Cornell University and Accenture Labs. We thank all participants in our user study. Special thanks to the original NeckFace team for providing the foundational device and pre-trained models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Maria Teresa Parreira: mb2554@cornell.edu

---

<div align="center">
  <sub>Built with ❤️ by the IRL-CT and Claude Code</sub>
</div>
