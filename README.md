# Mitigating Bias in Facial Recognition Systems: DB-VAE Replication

A corrected replication of the Debiasing Variational Autoencoder (DB-VAE) algorithm on UTKFace and Tiny ImageNet datasets.

**Paper:** Amini et al. (2019) - [Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure](http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf)

**Key Contribution:** Identifies and corrects a loss-function bug in the MIT 6.S191 Lab 2 template that affects the classification gradient on non-face samples.

---

## 📋 Project Overview

This research examines how well the DB-VAE algorithm generalizes across dataset compositions by replicating the method on:
- **Positive class:** UTKFace (facial images with race/gender labels)
- **Negative class:** Tiny ImageNet (non-face images)
- **Evaluation:** 10-group (5 races × 2 genders) held-out split

The project conducts a comprehensive hyperparameter sweep over α ∈ {0.001, 0.01, 0.05, 0.1, ∞} across three random seeds.

**Finding:** On this well-balanced dataset with cleanly-cropped faces, baseline performance is near-ceiling (99.82% accuracy across all groups), leaving minimal opportunity for debiasing interventions to improve performance.

---

## 📁 Project Structure

```
├── Project_PT_Part2_Debiasing_Kaggle.ipynb    # Main experimental notebook
├── report/
│   ├── main.tex                                # LaTeX source (2-column format)
│   ├── main.pdf                                # Final PDF report
│   ├── main.docx                               # Word format report
│   ├── references.bib                          # Bibliography
│   └── figures/
│       ├── per_group_accuracy.png              # Accuracy by demographic group
│       └── aggregate_metrics.png               # Aggregate performance metrics
├── src/
│   ├── __init__.py
│   ├── data_download.py                        # Dataset download utilities
│   ├── dataset.py                              # Data loading and preprocessing
│   ├── artifacts.py                            # Model checkpoint management
│   ├── evaluate.py                             # Evaluation metrics
│   └── util.py                                 # Utility functions
├── scripts/
│   └── make_paper_figures.py                   # Figure generation script
└── data/                                       # Data directory (local only, not in repo)
    ├── utkface/
    ├── tiny-imagenet-200/
    └── cache/
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for first-time users)

1. Click here to open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kwadwot/capstone-project/blob/main/Project_PT_Part2_Debiasing_Kaggle.ipynb)

2. The notebook will handle environment setup automatically:
   - Install dependencies
   - Download datasets
   - Run training and evaluation

3. Outputs will be saved to your Colab instance and can be downloaded

### Option 2: Local Setup

#### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training, optional but recommended)
- ~50 GB disk space for datasets

#### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd capstone_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Download Datasets

The notebook will automatically download datasets on first run, or use the utility:

```python
from src.data_download import download_utkface, download_tiny_imagenet

download_utkface()
download_tiny_imagenet()
```

#### Run the Experiment

```bash
# Option 1: Run the Jupyter notebook
jupyter notebook Project_PT_Part2_Debiasing_Kaggle.ipynb

# Option 2: Run programmatically (after setup in notebook)
python -c "
import torch
# Run training/evaluation pipeline
"
```

---

## 📊 Experiment Details

### Algorithm: DB-VAE (Debiasing Variational Autoencoder)

**Main idea:** Adaptively re-weight training samples based on rarity in learned latent space to mitigate bias.

**Loss function:**
- VAE loss: Reconstruction + KL divergence
- Classification loss: Binary cross-entropy (face/non-face)
- Combined with α-weighted sampling for debiasing

**Key parameter:** α
- α → 0: Standard training (no debiasing)
- α ∈ (0, 1): Debiasing active, controlled by parameter
- α → ∞: Extreme uniform resampling

### Hyperparameter Sweep

```
α values: {0.001, 0.01, 0.05, 0.1, ∞}
Random seeds: {0, 1, 2}
Evaluation: Per-group accuracy across 10 demographic groups
```

### Datasets

| Dataset | Role | Size | Labels |
|---------|------|------|--------|
| **UTKFace** | Positive (faces) | ~23,000 images | Age, gender, race |
| **Tiny ImageNet** | Negative (non-faces) | ~100,000 images | Object classes |

**Demographic groups:** 5 races × 2 genders = 10 groups
- Races: White, Black, Asian, Indian, Other
- Genders: Male, Female

---

## 📈 Results

### Baseline Performance
- **Mean accuracy across all groups:** 99.82%
- **Min group accuracy:** 99.60%
- **Max group accuracy:** 100%
- **Cross-group variance:** < 0.4%

### Key Finding
The α-sweep produces **no measurable improvement** over the baseline because:
1. Clean, well-aligned face crops reduce feature variance
2. Class-balanced dataset leaves minimal underrepresentation
3. Near-ceiling performance limits headroom for intervention

This characterizes the **regime where density-based resampling has limited effect**.

---

## 🔧 Reproducing Results

### Full Pipeline
1. Notebook automatically downloads and prepares datasets
2. Trains DB-VAE models with different α values
3. Evaluates per-group accuracy
4. Generates performance plots

### Expected Runtime
- **Colab GPU (T4):** ~30-45 minutes per configuration
- **Local GPU (RTX 3080):** ~15-20 minutes per configuration
- **CPU only:** Not recommended (4+ hours per configuration)

### Output Files
- `results/alpha_*/` — Model checkpoints and metrics per configuration
- `report/figures/` — Performance visualizations
- `sweep_summary.json` — Aggregated results across all configurations

---

## 📝 Report

The full research report is available in multiple formats:

- **PDF:** `report/main.pdf` (recommended for printing/formal reading)
- **DOCX:** `report/main.docx` (editable format)
- **Source:** `report/main.tex` (LaTeX source code)

**Report structure:**
1. Introduction: Motivation and related work
2. Methodology: DB-VAE algorithm and experimental design
3. Experiments: Hyperparameter sweep results
4. Discussion: Interpretation and implications
5. Conclusion: Future directions

---

## 🔍 Bug Fix

This project includes a **correction to the MIT 6.S191 Lab 2 template**:

**Original bug:** The loss function incorrectly swaps the face-indicator factor between classification and VAE terms, zeroing out the classification gradient on non-face samples.

**Impact:** This causes the classifier to not properly learn the binary classification task.

**Fix:** Correct assignment of loss weights ensures proper gradient flow to both tasks.

See `report/main.pdf` (Methodology section) for technical details.

---

## 📚 Dependencies

Core packages:
- `torch` ≥ 1.9.0
- `torchvision` ≥ 0.10.0
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`

Install all with:
```bash
pip install -r requirements.txt
```

---

## 💡 Key Takeaways

1. **Algorithmic debiasing is data-regime dependent** — effectiveness depends on underrepresentation severity
2. **Dataset quality matters** — clean, well-aligned data reduces the need for debiasing interventions
3. **Bug detection is crucial** — small implementation errors can invalidate research conclusions
4. **Generalization testing is essential** — findings from one dataset/setup may not transfer

---

## 📖 Citation

If you use this code or findings, please cite:

```bibtex
@article{twumasi2024debiasing,
  title={Mitigating Bias in Facial Recognition Systems: A Corrected Replication of DB-VAE on UTKFace and Tiny ImageNet},
  author={Twumasi, Kwadwo},
  school={CUNY Lehman College},
  year={2024}
}

@article{amini2019uncovering,
  title={Uncovering and mitigating algorithmic bias through learned latent structure},
  author={Amini, Alexander and Schaefer, Ava and Sap, Maarten and LeCun, Yann and Darrell, Trevor},
  journal={arXiv preprint arXiv:1811.10598},
  year={2019}
}
```

---

## 🙋 Support & Questions

**Issues or questions?**
1. Check the report (`report/main.pdf`) for methodology details
2. Review the notebook comments for implementation details
3. Open an issue on GitHub

**For Colab setup help:**
- The notebook includes detailed Colab-specific instructions
- Kaggle API setup will be guided in the notebook if needed

---

## 📄 License

This project replicates research from the MIT Deep Learning course. Please respect the original authors' copyright and the terms of use from the MIT Deep Learning course materials.

---

## 🙏 Acknowledgments

- **Amini et al. (2019)** — Original DB-VAE paper and MIT Deep Learning Lab template
- **Professor Liang Zhao** — Mentor and advisor
- **AI Tools** — GitHub Copilot CLI for writing assistance and initial LaTeX generation

---

## 📞 Contact

**Author:** Kwadwo Twumasi  
**Institution:** CUNY Lehman College  
**Advisor:** Professor Liang Zhao
