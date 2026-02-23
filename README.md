# 💓 Cardiac Sensing FM: A Multi-Modal Foundation Model for Cardiac Sensing Biosignals

**CSFM** is a large-scale, multi-modal foundation model developed for decoding cardiac sensing biosignals such as ECG and PPG. Trained with self-supervised learning on data from **1.7 million individuals**, it captures rich physiological patterns and can be applied to a wide range of downstream healthcare tasks.

### 🔍 Key Features

- 🧠 **Multi-modal input support** — Integrates ECG, PPG, and textual reports
- 🔬 **Self-supervised pretraining** — Scales across diverse, heterogeneous cardiac datasets
- 📊 **Massive training corpus** — Pretrained on biosignals from 1.7 million individuals
- 🏥 **Broad applications** — Enables:
  - Cardiovascular disease (CVD) classification  
  - Demographic recognition 
  - Vital sign measurement
  - ECG-based question answering
  - Clinical outcome prediction
  - Cross-modality generation (e.g., wearable → 12-lead ECG, PPG → ECG)
  - ...

---

## ⚙️ Installation

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.11
- Other dependencies (see [`requirements.txt`](./requirements.txt))

Normally takes shorter than 10 minutes to install

### Set up environment with Conda

```bash
conda create -n CSFM python=3.11
conda activate CSFM
```

### Install required packages
```bash
git clone https://github.com/guxiao0822/Cardiac-Sensing-FM.git
cd Cardiac-Sensing-FM
pip install -r requirements.txt
```

---

## 📈 Quickstart: Using Cardiac-Sensing-FM as a Feature Extractor

Use `CSFM` directly as a feature extractor to generate meaningful representations from raw biosignals.

### 🔹 Download the Pretrained Model Weights
Download the pretrained weights from the provided shared link (currently only available to a limited group, please contact xiao.gu@eng.ox.ac.uk for access). 

### 🔹 Load the Pretrained Model

```python
from network.model import CSFM_model
import torch 

# initialize the model with optional size: 'Tiny', 'Base', 'Large' 
model = CSFM_model('Tiny') 

model = model.cuda()
model.eval()

# Load pretrained weights if available
checkpoint_path = 'pretrained/<checkpoint_file>.pth'
print('load from ', checkpoint_path)
checkpoint = torch.load(checkpoint_path)
encoder_state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.') and 'mlp_head' not in k}

model.load_state_dict(encoder_state_dict, strict=False)

```

### 🔹 Extract Features from a Biosignal

To extract features, you need to provide:
1. A **preprocessed biosignal** (e.g., Any-channel ECG and/or PPG), formatted as a NumPy array, shape : `[channels, time]`  
2. The **input channels**, which specify the type of signal:
   - `0–11`: Standard 12-lead ECG
   - `12`: PPG
   - Wearable ECG treated as lead II, i.e., use channel = `1`

### examples of preprocessing a Lead II ECG + PPG
```python
import torch
import torch.nn as nn
from utils import preprocess_ecg, preprocess_ppg

# Load and preprocess
ecg = ...  # shape: (1, time)
ppg = ...  # shape: (1, time)
ecg_fs, ppg_fs = 100, 100

# Clean. Normalize and Standard
ecg = preprocess_ecg(ecg, ecg_fs) 
ppg = preprocess_ppg(ppg, ppg_fs)

# Concatenate signals (channels x time)
signal = np.concatenate((ecg, ppg), axis=0) 

# Channel Settings
channels = np.asarray([1, 12])  # ECG lead II + PPG

# Disable classification head
model.mlp_head = nn.Identity()

# Extract features
features = model(torch.tensor(signal).unsqueeze(0), channels)  # shape: (1, hidden_dim)
```

You can then use `features` for tasks such as clustering, retrieval, or downstream model training.

---

## 🩺 Fine-tuning CSFM on Downstream Tasks

For full instructions (~0.5hr) on how to fine-tune CSFM on your own dataset:

📖 See [`TUTORIAL`](./tutorial/finetune_tutorial.ipynb)

The tutorial covers:
- Data preprocessing scripts
- Finetuning configurations
- Training and evaluation examples


---

## 📚 Citation

If you use **Cardiac Sensing FM (CSFM)** in your research or applications, please cite:

```bibtex
@article{gu2026cardiac,
  title={Cardiac health assessment across scenarios and devices using a multi-modal foundation model pretrained on data from 1.7 million individuals},
  author={Gu, Xiao and Tang, Wei and Han, Jinpei and Sangha, Veer and Liu, Fenglin and Gowda, Shreyank N and Ribeiro, Antonio H and Schwab, Patrick and Branson, Kim and Clifton, Lei and others},  journal={Nature Machine Intelligence},
  year={2026},
  publisher={Springer Nature}
}
```