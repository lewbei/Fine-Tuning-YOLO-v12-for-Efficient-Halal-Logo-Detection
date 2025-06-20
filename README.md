# Fine-Tuning YOLOv12 for Efficient Halal Logo Detection

## Abstract

Counterfeit or missing Halal certification marks threaten consumer trust and impede regulatory oversight, yet manual logo inspection remains slow, error-prone, and unsuited to high-volume workflows. We present a lightweight, edge-deployable Halal-logo detection system that fine-tunes two recent YOLO-v12 backbones, nano and small on a 50 class public dataset which has 1,292 images with 640 × 640 resolution. Two training strategies are applied to the models, train from scratch and transfer learning. The transfer learning is fine tune the model with COCO-pre-trained weights while freezing the first three layers. Fine-tuned YOLO-v12 small lifts accuracy from 0.919 to 0.945 mAP@0.5:0.95 and halves GPU training time, whereas YOLO-v12 nano attains 0.947 mAP with only 2.6 M parameters, 4.7 ms single-image latency, and 509 MB peak VRAM. Transfer learning trims training energy from 0.14 kWh to 0.04 kWh and reduces estimated CO₂ emissions to 0.02 kg, underscoring its sustainability advantage. The study provides the first systematic evaluation of YOLO-v12 nano versus small for Halal-logo detection, evidence that lightweight backbones can match larger models when fine-tuned, and an open-source, reproducible training and inference pipeline. These results demonstrate that modern nano detectors deliver high accuracy with one-third of the energy footprint, paving the way for scalable, real-time Halal-logo verification in production lines, retail checkpoints, and mobile auditing applications.

## Overview

This project provides a set of scripts to train, evaluate, and analyze YOLOv12 models for Halal logo detection using both training from scratch and transfer learning approaches.

This project utilizes code from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed under AGPL-3.0.

## Key Features

- **Lightweight Models**: YOLO-v12 nano (2.6M parameters) and small variants
- **High Accuracy**: YOLO-v12 nano achieves 0.947 mAP@0.5:0.95
- **Fast Inference**: 4.7 ms single-image latency with nano model
- **Energy Efficient**: Transfer learning reduces training energy by 71% (0.14 kWh → 0.04 kWh)
- **Low Carbon Footprint**: CO₂ emissions reduced to 0.02 kg with transfer learning
- **Edge Deployment Ready**: 509 MB peak VRAM requirement
- **Two Training Strategies**: Train from scratch and transfer learning approaches

## Dataset Information

- **Dataset Size**: 1,292 images
- **Image Resolution**: 640 × 640 pixels
- **Number of Classes**: 50 Halal logo classes
- **Format**: YOLO format annotations
- **Source**: Public Halal logo dataset (available on Roboflow)
- **Data Split**: Train (72%) / Validation (26%) / Test (2%)

## Model Performance

### Accuracy Results

| Model | Training Strategy | mAP@0.5:0.95 | AP@0.5 | Precision | Recall |
|-------|------------------|--------------|--------|-----------|--------|
| YOLO-v12 Small | From Scratch | 0.919 | 0.995 | 0.917 | 0.984 |
| YOLO-v12 Small | Transfer Learning | 0.945 | 0.995 | 0.902 | 1.000 |
| YOLO-v12 Nano | Transfer Learning | 0.947 | 0.995 | 1.000 | 0.981 |

### Inference Performance

| Model | Training Strategy | Parameters | Latency (ms) | Peak VRAM (MB) |
|-------|------------------|------------|--------------|----------------|
| YOLO-v12 Small | From Scratch | 9,250,230 | 15.8 | 975 |
| YOLO-v12 Small | Transfer Learning | 9,250,230 | 12.3 | 904 |
| YOLO-v12 Nano | Transfer Learning | 2,566,478 | 4.7 | 509 |

### Energy Consumption and Carbon Footprint

| Model | Training Strategy | GPU Hours | Energy (kWh) | CO₂ Emissions (kg) |
|-------|------------------|-----------|--------------|-------------------|
| YOLO-v12 Small | From Scratch | 0.56 | 0.14 | 0.06 |
| YOLO-v12 Small | Transfer Learning | 0.26 | 0.06 | 0.02 |
| YOLO-v12 Nano | Transfer Learning | 0.18 | 0.04 | 0.02 |

## Project Structure

- `train.py`: Train YOLOv12 models from scratch or fine-tune pre-trained models.
- `evaluate_test.py`: Evaluate trained models on a test set and measure performance metrics.
- `convert_yolo_to_cocojson.py`: Convert YOLO format labels to COCO JSON format.
- `estimate_energy.py`: Estimate the energy consumption and CO2 emissions of the training process.
- `requirements.txt`: A list of required Python packages for this project.

## Methodology

### Training Strategies

1. **Train from Scratch**: Models are trained from randomly initialized weights
2. **Transfer Learning**: Models are fine-tuned from COCO pre-trained weights with the first three layers frozen

### Model Variants

- **YOLO-v12 Nano**: Lightweight model optimized for edge deployment (2.6M parameters)
- **YOLO-v12 Small**: Balanced model for accuracy and efficiency (9.3M parameters)

### Training Configuration

| Hyperparameter | Training from Scratch | Transfer Learning |
|----------------|----------------------|-------------------|
| **Epochs** | 100 | 50 |
| **Initial Learning Rate** | 0.01 (default) | 0.001 (default) |
| **Optimizer** | SGD with Momentum | SGD with Momentum |
| **Momentum** | 0.937 | 0.937 |
| **Weight Decay** | 0.0005 | 0.0005 |
| **Batch Size** | 16 | 16 |
| **Input Image Size** | 640 × 640 | 640 × 640 |
| **Backbone Layers Frozen** | None | First three layers |
| **Data Augmentation** | Mosaic, HSV, Scaling, Translation | Mosaic, HSV, Scaling, Translation |
| **Early Stopping (Patience)** | 15 epochs | 15 epochs |
| **GPU Hardware** | NVIDIA RTX 2080 Ti | NVIDIA RTX 2080 Ti |

### Evaluation Metrics

- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 in 0.05 steps
- **AP@0.5**: Average Precision at IoU threshold of 0.50
- **Precision**: Ratio of true positive detections to total positive detections
- **Recall**: Ratio of true positive detections to total ground truth objects
- **Latency**: End-to-end inference time for a single 640×640 image on GPU
- **Energy Consumption**: Electrical energy consumed during training (kWh)
- **Carbon Emissions**: Estimated CO₂ emissions associated with energy consumption

### Energy and Carbon Footprint Calculation

**Energy Consumption:**
```
E(kWh) = (H × P) / 1000
```
Where H = GPU training time (hours), P = average GPU power draw (250W for RTX 2080 Ti)

**Carbon Emissions:**
```
CO₂(kg) = E(kWh) × EF
```
Where EF = emission factor (0.4 kg CO₂ per kWh in Malaysia)

## Requirements

### Software Requirements
- Python 3.8+
- PyTorch 1.8+
- Ultralytics YOLO package
- CUDA-capable GPU drivers

### Hardware Requirements

#### For Training
- **GPU**: NVIDIA RTX 2080 Ti or equivalent (11GB VRAM recommended)
- **CPU**: Multi-core processor (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB+ system memory
- **Storage**: 2GB+ free space for datasets and model weights

#### For Inference
- **Minimum**: 4GB GPU VRAM (for YOLO-v12 Nano)
- **Recommended**: 8GB+ GPU VRAM (for optimal performance)
- **Edge Deployment**: 509MB VRAM minimum (YOLO-v12 Nano)

#### Power Consumption
- **Training**: ~250W GPU power draw (RTX 2080 Ti)
- **Inference**: Varies by model size and batch size

### Performance Benchmarks
- **Training Time**: 0.18-0.56 hours (depending on model and strategy)
- **Inference Speed**: 4.7-15.8ms per image (640×640 resolution)
- **Memory Usage**: 509-975MB VRAM during inference

## Setup

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your dataset:**
    -   Organize your dataset in the YOLO format.
    -   Create a `data.yaml` file that points to your training and validation data. See the Ultralytics documentation for more details.

## Usage

### Training

The `train.py` script is used to train or fine-tune YOLOv12 models.

-   **Modify `train.py`:**
    -   Set the `data_path` variable to the path of your `data.yaml` file.
-   **Run the script:**
    ```bash
    python train.py
    ```

### Evaluation

The `evaluate_test.py` script is used to evaluate the performance of your trained models.

-   **Modify `evaluate_test.py`:**
    -   Update `DATA_YAML_PATH` to your `data.yaml` file.
    -   Update `model_weights_paths` with the paths to your trained model weights (`.pt` files).
-   **Run the script:**
    ```bash
    python evaluate_test.py
    ```

### Label Conversion

The `convert_yolo_to_cocojson.py` script converts YOLO format labels to COCO JSON format.

-   **Modify `convert_yolo_to_cocojson.py`:**
    -   Set `DATA_YAML` to the path of your `data.yaml` file.
    -   Set `OUT_JSON` to the desired output file path.
-   **Run the script:**
    ```bash
    python convert_yolo_to_cocojson.py
    ```

### Energy Estimation

The `estimate_energy.py` script estimates the energy consumption and CO2 emissions of the training process.

-   **Modify `estimate_energy.py`:**
    -   Update the `results_paths` dictionary with the paths to the `results.csv` files generated during training.
-   **Run the script:**
    ```bash    python estimate_energy.py
    ```

## Applications

This Halal logo detection system is designed for various real-world applications:

- **Production Lines**: Real-time quality control and verification
- **Retail Checkpoints**: Automated scanning at point-of-sale systems
- **Mobile Auditing**: Portable verification tools for field inspections
- **Regulatory Oversight**: Automated compliance monitoring systems
- **Supply Chain Management**: Verification of Halal certification throughout the supply chain

## Results Summary

### Key Findings

- **Best Performing Model**: YOLO-v12 Nano with 0.947 mAP@0.5:0.95, surpassing the larger Small model
- **Accuracy Improvement**: Transfer learning boosts YOLO-v12 Small from 0.919 to 0.945 mAP@0.5:0.95
- **Perfect Precision**: YOLO-v12 Nano achieves 1.000 precision with transfer learning
- **Speed Advantage**: YOLO-v12 Nano runs 3× faster than scratch-trained Small model (4.7ms vs 15.8ms)
- **Memory Efficiency**: YOLO-v12 Nano uses 50% less VRAM (509MB vs 975MB)
- **Energy Savings**: Transfer learning reduces training energy by 71% (0.14 kWh → 0.04 kWh for Nano)
- **Carbon Footprint**: CO₂ emissions reduced by 67% with transfer learning (0.06 kg → 0.02 kg)
- **Training Time**: Transfer learning cuts training time by more than half

### Performance Highlights

- **Real-time Capability**: All models achieve real-time inference (<16ms)
- **Edge Deployment**: YOLO-v12 Nano is optimized for resource-constrained environments
- **Sustainability**: Transfer learning demonstrates significant environmental benefits
- **Scalability**: Lightweight models enable deployment across various platforms

### Practical Implications

- **Production Ready**: Models suitable for industrial deployment
- **Cost Effective**: Reduced computational requirements lower operational costs
- **Environmental Impact**: Sustainable AI approach with reduced carbon footprint
- **Versatile Application**: Multiple deployment scenarios from edge to cloud

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

### Third-Party Components

This project utilizes code from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. The use of Ultralytics YOLO code requires that this entire project be licensed under AGPL-3.0 to ensure compliance.

### Attribution

- **Ultralytics YOLO**: This project uses the YOLOv12 implementation from Ultralytics. Original repository: https://github.com/ultralytics/ultralytics
- **License**: GNU Affero General Public License v3.0
- **Website**: https://www.ultralytics.com

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{your_paper_2025,
  title={Fine-Tuning YOLO-v12 for Efficient Halal Logo Detection},
  author={[Your Name(s)]},
  booktitle={AIP Conference Proceedings},
  year={2025},
  publisher={AIP Publishing},
  url={https://github.com/lewbei/Fine-Tuning-YOLO-v12-for-Efficient-Halal-Logo-Detection}
}
```

Please also cite the original Ultralytics YOLO:

```bibtex
@software{ultralytics_yolo,
  title={Ultralytics YOLO},
  author={Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year={2023},
  url={https://github.com/ultralytics/ultralytics},
  license={AGPL-3.0}
}
```