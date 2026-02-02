# Calibrated Geo-Social Link Prediction for Household–School Connectivity

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-green.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Paper Reference

This repository provides the **official implementation** for:

> **"Calibrated geo-social link prediction for household–school connectivity in community resilience"**  
> Himadri Sen Gupta, Saptadeep Biswas, Charles D. Nicholson  
> *International Journal of Disaster Risk Reduction*, Volume 131, December 2025, 105872  
> DOI: [https://doi.org/10.1016/j.ijdrr.2025.105872](https://doi.org/10.1016/j.ijdrr.2025.105872)

### Citation

If you use this code in your research, please cite:

```bibtex
@article{gupta2025calibrated,
  title={Calibrated geo-social link prediction for household–school connectivity in community resilience},
  author={Gupta, Himadri Sen and Biswas, Saptadeep and Nicholson, Charles D.},
  journal={International Journal of Disaster Risk Reduction},
  volume={131},
  pages={105872},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.ijdrr.2025.105872}
}
```

---

## Abstract

Understanding community resilience requires models that reason over heterogeneous, geospatially structured relations between households and schools. We engineer a calibrated geo–social link prediction pipeline that fuses:

- **Heterogeneous Graph Transformer (HGT)** — relation-aware message passing
- **LightGCN branch** — collaborative propagation on the household–school bipartite graph
- **Multi-scale geospatial Fourier features** — via a learnable gate

The model is pretrained with denoising reconstruction and an InfoNCE contrastive objective, then fine-tuned with hard negatives; temperature scaling turns scores into well-calibrated probabilities.

---

## Model Architecture

| Component | Description |
|-----------|-------------|
| **Feature Encoder** | MLP with multi-scale Fourier positional encoding for geographic coordinates and categorical embeddings for demographics |
| **HGT Backbone** | Heterogeneous Graph Transformer with relation-aware message passing across node and edge types |
| **LightGCN Branch** | Collaborative propagation on the bipartite household-school attendance graph |
| **Fusion Gate** | Learnable gating mechanism to adaptively combine HGT and LightGCN representations |
| **Calibrated Decoder** | MLP scoring function with temperature scaling for probability calibration |

---

## Results Summary

### Standard Test Set (A)
| Metric | Value |
|--------|-------|
| AUC | 0.997 |
| AP | 0.996 |
| Brier | 0.029 |
| ECE | 0.057 |

### Cold-Start Households (B)
| Metric | Value |
|--------|-------|
| AUC | 0.864 |
| AP | 0.752 |

### Cold-Start Schools (C, UNSEEN-only pool)
| Metric | Value |
|--------|-------|
| AUC | ≈1.0 |
| AP | ≈1.0 |
| Hit@k | 1.0 (k≥3) |
| NDCG@k | 1.0 (k≥3) |

---

## Requirements

### Software Dependencies

```bash
# Core dependencies
python>=3.8
torch>=2.0.0
torch-geometric>=2.3.0
torch-cluster>=1.6.0

# Data processing
pandas>=1.5.0
numpy>=1.21.0

# Evaluation
scikit-learn>=1.0.0

# Optional (recommended)
psutil>=5.9.0
matplotlib>=3.5.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/himadri-gupta/geo-social-school-gnn.git
cd geo-social-school-gnn

# Create conda environment (recommended)
conda create -n geosocial python=3.10
conda activate geosocial

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install remaining dependencies
pip install pandas numpy scikit-learn psutil matplotlib
```

---

## Data Requirements

The pipeline expects three CSV files from the Housing Unit Inventory (HUI) and Person-Record (PREC) synthetic population datasets:

| File | Description | Required Columns |
|------|-------------|------------------|
| `hui_*.csv` | Household Unit Information | `huid`, `ownershp`, `race`, `hispan`, `randincome`, `numprec` |
| `prec_*_students.csv` | Student enrollment records | `huid`, `NCESSCH`, `SCHNAM09`, `hcb_lat`, `hcb_lon`, `ncs_lat`, `ncs_lon` |
| `prec_*_schoolstaff.csv` | Staff employment records | `huid`, `SIName`, `hcb_lat`, `hcb_lon` |

**Data Source**: Lumberton, NC 2010 synthetic population from [DesignSafe-CI Project PRJ-2961](https://www.designsafe-ci.org/).

---

## Usage

### Quick Start (Jupyter Notebook)

Open `geo_social_school_AI.ipynb` and run the cells sequentially:

1. **Cell 2**: Main GNN training pipeline
2. **Cell 3**: Generate LaTeX tables
3. **Cell 4**: Generate publication figures

```python
# Set file paths
paths = {
    'households': 'hui_v0-1-0_Lumberton_NC_2010_rs9876.csv',
    'students': 'prec_v0-2-0_Lumberton_NC_2010_rs9876_students.csv',
    'staff': 'prec_v0-2-0_Lumberton_NC_2010_rs9876_schoolstaff.csv'
}

# Run full robustness suite
run_robustness_suite(paths, device='cuda', seed=42)
```

### Command Line Interface

```bash
python geo_social_school_AI.py \
    --mode robust \
    --households hui_v0-1-0_Lumberton_NC_2010_rs9876.csv \
    --students prec_v0-2-0_Lumberton_NC_2010_rs9876_students.csv \
    --staff prec_v0-2-0_Lumberton_NC_2010_rs9876_schoolstaff.csv \
    --seed 42 \
    --device cuda
```

---

## Evaluation Protocol

| Experiment | Description |
|------------|-------------|
| **A. Standard Split** | Random 90/10 train/test split |
| **A2. Proximity-Controlled** | Split that prevents "near" leakage |
| **B. Cold-Start Households** | 20% of households held out entirely (inductive) |
| **C. Cold-Start Schools** | 20% of schools held out entirely (inductive) |

Candidate pools: **ALL-candidate** vs. **UNSEEN-only** for fair comparison.

---

## Output Structure

```
outputs/
├── FULL_YYYYMMDD_HHMMSS/
│   ├── seed_42/
│   │   ├── test_edge_scores.csv
│   │   └── model_best.pt
│   ├── roc_by_seed.pdf
│   ├── pr_by_seed.pdf
│   ├── reliability_by_seed.pdf
│   └── fairness_by_group.tex
├── ROBUST_YYYYMMDD_HHMMSS/
│   ├── A_standard/
│   ├── B_coldstart_households/
│   ├── C_coldstart_schools/
│   └── robustness_table_min.tex
└── paper_tables.tex
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch_cluster` | `pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html` |
| `FileNotFoundError: CSV files` | Update paths dictionary to point to your data location |
| `CUDA out of memory` | Reduce `FT_BATCH` in `DEFAULT_CFG` or use `device='cpu'` |
| `No FULL_* directory found` | Run the training pipeline first |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This research was partially supported by:

- **National Institute of Standards and Technology (NIST) Center of Excellence for Risk-Based Community Resilience Planning** through a cooperative agreement with Colorado State University (Grant Numbers: 70NANB20H008 and 70NANB15H044)
- **CSU Pueblo Foundation**
- **School of Engineering at Colorado State University Pueblo**

We thank:

- **Dr. Nathanael Rosenheim** for curating and sharing the Housing Unit Inventory (HUI) dataset and replication code on DesignSafe-CI (Project PRJ-2961)
- **DesignSafe-CI and IN-CORE teams** for data hosting, curation, and research infrastructure support
- **HUA and PREC workflow maintainers** (Dr. N. Rosenheim, M. Safayet, Dr. A. Beck) for open-sourcing their tools
- **Dr. John van de Lindt** and collaborators for the IN-CORE platform

---

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
