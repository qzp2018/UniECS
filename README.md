# UniECS

This repo contains the codebase for the paper "[UniECS: Unified Multimodal E-Commerce Search Framework
with Gated Cross-modal Fusion](https://arxiv.org/abs/2508.13843)"

## 📢 Updates

- 🔥[2024-08-20]: Initial release of the codebase and paper
- 🔥[Coming Soon]: Release of pre-trained models and datasets

## 📖 Overview

 We introduce **UniECS**, a unified multimodal e-commerce search framework that handles all retrieval scenarios across image, text, and their combinations.
Our main contributions include:
- **Unified Architecture**: A flexible framework is developed that processes diverse visual and textual inputs via a single pipeline, supporting arbitrary modality combinations for both queries and candidates in e-commerce search.
- **Technical Advances**: A gated multimodal encoder with adaptive fusion capabilities is introduced, which effectively handles missing modalities. It is paired with a training strategy that enhances cross-modal alignment and representation quality through specialized loss functions. 
- **Benchmark**: M-BEER, a comprehensive benchmark, is created, containing 50K product pairs for evaluating e-commerce search. Each sample includes trigger text, trigger image, recall text, and recall image, enabling standardized evaluation of nine distinct retrieval scenarios.

<img width="3830" height="838" alt="image" src="https://github.com/user-attachments/assets/d28e8870-bb5f-4268-a10e-1b2344e4b714" />


## 🗂️ Dataset
 [🤗 Dataset](YOUR_HUGGINGFACE_DATASET_LINK) coming soon!.


## 🛠️ Installation

Prepare the codebase and Conda environment just following Chinese-clip:

```bash
git clone https://github.com/qzp2018/UniECS.git
cd UniECS
conda env create -n uniecs python=3.8
conda activate uniecs
pip install -r requirements.txt
pip install cn_clip
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0 (for GPU training)

## 🚀 Quick Start

### Training

To train the model from scratch:

```bash
cd scripts/
bash train.sh
```

Before running, please modify the data root in `train.sh` to point to your dataset and checkpoint directory.

**Noting!!:** For training efficiency, the validation results shown during training are evaluated within a batch, so **they are not the true evaluation results!!**

### Evaluation

To evaluate a trained model:

```bash
cd scripts/
bash eval.sh
```

Before running, please:
- Modify the `MODEL_PATH` in `eval.sh` to point to your trained model
- Modify the `DATA_DIR` in `eval.sh` to point to your test data
- Set evaluation configurations in `configs/eval_config.yaml`

### Inference

For quick inference on new data:

```python
from src.model import YourModel
from src.utils import load_config

# Load pre-trained model
config = load_config('configs/model_config.yaml')
model = YourModel.load_pretrained('path/to/checkpoint')

# Run inference
result = model.predict(input_data)
```

## 📁 Project Structure

```
├── src/                     # Source code
│   ├── model/              # Model implementations
│   ├── data/               # Data loading and processing
│   ├── training/           # Training scripts
│   └── utils/              # Utility functions
├── scripts/                # Training and evaluation scripts
│   ├── train.sh
│   └── eval.sh
├── notebooks/              # Jupyter notebooks for analysis
└── tests/                  # Unit tests
```


## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{yourname2024title,
  title={Your Paper Title},
  author={Your Name and Co-author Name},
  journal={arXiv preprint arXiv:2508.13843},
  year={2024}
}
```
