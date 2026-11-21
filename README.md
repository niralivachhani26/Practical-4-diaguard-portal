# DiaGuard: Explainable Multi-Modal Deep Learning Framework for Joint Diabetic Retinopathy and Foot Ulcer Risk Screening

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Zenodo Preprint](https://img.shields.io/badge/Zenodo-Preprint%20DOI-blue)](https://zenodo.org/)

DiaGuard is an end-to-end, explainable multi-modal screening framework and clinical decision-support portal. It unifies five-stage **Diabetic Retinopathy (DR)** fundus grading and binary **Diabetic Foot Ulcer (DFU)** dermal detection within a single interface, featuring real-time **Gradient-weighted Class Activation Mapping (Grad-CAM)** spatial attention heatmaps and fused microvascular risk scoring.

---

## 📌 Executive Summary & Clinical Context

Diabetes mellitus induces systemic microvascular damage affecting both ocular capillaries and peripheral extremities. Traditional computer-aided diagnosis (CAD) pipelines treat ophthalmology and podiatry screening as isolated silos and operate as opaque black boxes. 

**DiaGuard** bridges this gap by offering:
1. **Dual-Organ Diagnostic Fusion:** Integrated assessment of retinal fundus scans and plantar dermal images.
2. **Photometric-Safe Augmentation:** Spatial transformations bounded by strict hue tolerances ($\Delta H \le 0.03$) to preserve diagnostic inflammatory erythema and ischemic chromaticity.
3. **Hard-Margin Optimization:** Focal Loss ($\gamma = 2.0$) optimization to prioritize gradient updates on ambiguous, early-stage lesion boundaries.
4. **Real-Time Visual Verification:** Live Grad-CAM overlay generation integrated directly into a web dashboard for point-of-care clinical triage.

---

## 🛠 System Architecture

```
                          +-----------------------------------+
                          |     Patient Input Scans           |
                          | (Retinal Fundus + Plantar Dermal) |
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------+-----------------+
                          | Photometric Preprocessing Engine  |
                          | (224x224 RGB, ImageNet Norm)     |
                          +--------+-----------------+--------+
                                   |                 |
                +------------------+                 +------------------+
                |                                                       |
                v                                                       v
+---------------+---------------+                       +---------------+---------------+
|    Ocular Branch: ResNet-50   |                       | Plantar Branch: EfficientNet  |
|  (5-Stage DR Severity Model)  |                       |   (Binary DFU Focal Loss)     |
+---------------+---------------+                       +---------------+---------------+
                |                                                       |
                v                                                       v
+---------------+---------------+                       +---------------+---------------+
|  Grad-CAM Hook (layer4[-1])   |                       |  Grad-CAM Hook (features[-1]) |
+---------------+---------------+                       +---------------+---------------+
                |                                                       |
                +------------------+                 +------------------+
                                   |                 |
                                   v                 v
                          +--------+-----------------+--------+
                          |   Max-Weighted Risk Fusion Engine  |
                          |      & FastAPI Inference API      |
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------+-----------------+
                          |   Interactive Web Portal UI       |
                          | (Dual Heatmaps & Clinical Alerts) |
                          +-----------------------------------+
```

---

## 📊 Benchmark Quantitative Results

Evaluated on a held-out stratified validation partition ($n = 211$ scans from $N = 1,055$ clinical skin images):

| Metric | Validation Score | Clinical Significance |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.9985** | Near-optimal class separability across decision thresholds |
| **Macro F1-Score** | **0.9763** | Balanced cross-fold harmonic classification performance |
| **Sensitivity (Recall)** | **98.06%** | High sensitivity minimizing missed early-stage foot ulcers ($FN = 2$) |
| **Specificity** | **99.07%** | Reduces unnecessary specialist referrals on intact skin ($FP = 1$) |
| **Precision (PPV)** | **99.02%** | High diagnostic reliability for positive ulcer flags |
| **Overall Accuracy** | **98.58%** | 208 out of 211 validation samples correctly classified |

### Confusion Matrix ($N = 211$)
- **True Positives (TP):** 101
- **True Negatives (TN):** 107
- **False Positives (FP):** 1
- **False Negatives (FN):** 2

---

## ⚙️ Mathematical Formulations

### 1. Class-Weighted Focal Loss
To prevent easily classified healthy skin samples from dominating training gradients, the DFU branch uses class-weighted Focal Loss:
$$\mathcal{L}_{focal}(p_{t}) = -\alpha_{t} (1 - p_{t})^{\gamma} \log(p_{t})$$
where $\gamma = 2.0$ acts as the focusing parameter to concentrate learning on ambiguous lesion boundaries.

### 2. Grad-CAM Spatial Heatmap Localization
Neuron importance weights $\alpha_{k}^{c}$ for target logit $y^{c}$ at the final convolutional layer are computed via global average pooling of backpropagated gradients over spatial dimensions $(i, j)$:
$$\alpha_{k}^{c} = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^{c}}{\partial A_{ij}^{k}}$$
The spatial localization heatmap $L_{\text{Grad-CAM}}^{c}$ is generated by passing the weighted combination of forward feature maps $A^{k}$ through a ReLU non-linearity:
$$L_{\text{Grad-CAM}}^{c} = \text{ReLU}\left(\sum_{k} \alpha_{k}^{c} A^{k}\right)$$

### 3. Max-Weighted Microvascular Risk Index
Individual ocular risk score $R_{DR} \in [0, 100]$ and plantar ulcer risk score $R_{DFU} \in [0, 100]$ are fused into a patient-level Microvascular Risk Index $R_{total}$:
$$R_{total} = 0.7 \cdot \max(R_{DR}, R_{DFU}) + 0.3 \cdot \frac{R_{DR} + R_{DFU}}{2}$$
This formulation guarantees that severe pathology in either individual modality triggers an immediate **High-Risk Triage Alert** ($R_{total} \ge 65$).

---

## 📁 Repository Structure

```text
Practical-4-diaguard-portal/
├── backend/
│   ├── api.py                         # FastAPI application & REST endpoint routes
│   └── inference_engine.py            # Dual-model loading, Grad-CAM hooks & inference engine
├── static/
│   └── index.html                     # Responsive clinical portal dashboard UI
├── weights/
│   ├── best_dfu_model.pth             # EfficientNet-V2-S DFU model checkpoint (~79.1 MB)
│   └── resnet50_retinopathy_final.pth # ResNet-50 DR model checkpoint (~92.0 MB)
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA enabled)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/niralivachhani26/Practical-4-diaguard-portal.git
cd Practical-4-diaguard-portal
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Activate environment (Linux/macOS)
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the DiaGuard Application
Start the FastAPI server:
```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```
or run directly via:
```bash
python main.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 📡 API Endpoints Summary

### `GET /health`
Returns the status of the inference engine, device allocation (CPU/CUDA), and loaded checkpoints.

### `POST /api/predict`
Accepts multi-part form data containing dual clinical scans.

**Request Payload:**
- `patient_id` (string): Unique or anonymized patient identifier.
- `ocular_file` (file): Retinal fundus image (`.png`, `.jpg`, `.jpeg`).
- `plantar_file` (file): Plantar dermal image (`.png`, `.jpg`, `.jpeg`).

**Response Schema (JSON):**
```json
{
  "status": "success",
  "patient_id": "PT-2026-0841",
  "ocular_result": {
    "predicted_grade": 3,
    "grade_label": "Severe NPDR",
    "confidence": 0.884,
    "risk_score": 54.8,
    "gradcam_base64": "data:image/png;base64,..."
  },
  "plantar_result": {
    "prediction": "Abnormal (Diabetic Foot Ulcer)",
    "confidence": 0.982,
    "risk_score": 82.5,
    "gradcam_base64": "data:image/png;base64,..."
  },
  "fused_assessment": {
    "microvascular_risk_index": 78.3,
    "triage_category": "HIGH RISK",
    "recommendation": "Urgent referral to Ophthalmology / Diabetic Foot Clinic within 48 hours."
  },
  "telemetry": {
    "inference_time_ms": 2420
  }
}
```

---

## 📄 Citation & Attribution

If you use this project, code, or architecture in your research, please cite the associated preprint:

```bibtex
@article{vachhani2026diaguard,
  title     = {DiaGuard: An Explainable Multi-Modal Deep Learning Framework for Joint Diabetic Retinopathy and Foot Ulcer Risk Screening},
  author    = {Vachhani, Nirali},
  journal   = {Zenodo Preprint},
  year      = {2026},
  doi       = {10.5281/zenodo.xxxxxxx},
  url       = {https://doi.org/10.5281/zenodo.xxxxxxx}
}
```

---

## 👤 Author & Contact

**Nirali Vachhani**  
*Independent Medical AI Researcher*  
Surat, Gujarat, India  
📧 Email: [vnirali261997@gmail.com](mailto:vnirali261997@gmail.com)  
octokit GitHub: [@niralivachhani26](https://github.com/niralivachhani26)
