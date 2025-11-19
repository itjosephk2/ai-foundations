
# AI Foundations

Foundations-first study repo for maths, Python, and core machine learning ideas.

The goal of this project is to build a **clear, reproducible trail** of notebooks and small code modules that move from algebra and vectors → linear algebra → probability & statistics → neural networks and ML models.

---

## Project Structure

```text
AI-FOUNDATIONS/
├─ .venv/                     # Local virtual environment (not tracked)
├─ notebooks/
│  ├─ 00_setup_check.ipynb    # Environment sanity check
│  ├─ algebra_basics/         # Algebra refresh & core skills
│  ├─ linear_algebra/         # Vectors, matrices, transformations, etc.
│  ├─ ml_models/              # Classical ML & model experiments
│  ├─ nnfs/                   # "Neural Networks from Scratch" companion notebooks
│  ├─ numpy_course/           # Numpy practice & utilities
│  ├─ prob_&_stats/           # Probability & statistics notes / exercises
│  └─ python_basics/          # Python fundamentals & patterns
├─ src/
│  ├─ __init__.py
│  ├─ tensors_arrays_vectors.py  # Reusable tensor/vector helpers
│  ├─ linear_algebra/            # LA-focused library code (WIP)
│  └─ nnfs/                       # NNFS-style layers, activations, etc. (WIP)
├─ tests/                      # Unit tests for src/ (to grow over time)
├─ requirements.txt            # Python dependencies for the project
└─ README.md                   # You are here
```

---

## Notebooks Overview

All learning happens first in notebooks, then any reusable logic gets pulled into `src/`.

### `notebooks/algebra_basics/`
- Refresher on manipulation of equations, functions, inequalities, etc.
- Focus: confidence with algebra used later in calculus, linear algebra and ML.

### `notebooks/linear_algebra/`
- Vectors, vector operations, norms, dot products
- Matrices, matrix operations, and matrix–vector products
- Geometric intuition (transformations, projections, etc.)

### `notebooks/python_basics/`
- Core Python syntax, control flow, functions, OOP patterns
- Aimed at fluency for writing clean numerical / ML code

### `notebooks/numpy_course/`
- Numpy arrays, indexing, broadcasting
- Vectorisation and performance-focused patterns

### `notebooks/nnfs/`
- Companion notebooks for **Neural Networks from Scratch**
- Step-by-step implementation of layers, activations, loss functions, and training loops

### `notebooks/prob_&_stats/`
- Probability distributions, expectations, variance
- Inference topics (confidence intervals, hypothesis testing etc.)
- Mirrors college module topics where useful

### `notebooks/ml_models/`
- Classical ML models (e.g. regression, classification, clustering)
- Small experiments using scikit-learn and custom implementations

---

## `src/` – Reusable Library Code

Everything in `src/` is meant to be **importable**, independent of the notebooks.

- `tensors_arrays_vectors.py`  
  Utility functions for working with vectors/arrays/tensors that appear across notebooks.

- `src/linear_algebra/`  
  Room for small modules like:
  - `vector_ops.py`
  - `matrix_ops.py`
  - `decompositions.py` (e.g. eigenvalues, SVD, etc.)

- `src/nnfs/`  
  Implementations inspired by **NNFS**:
  - Dense layers, activation functions
  - Loss functions and accuracy metrics
  - Optimisers and simple training loops

The idea is: notebooks prototype things, then stable pieces graduate into `src/`.

---

## `tests/` – Tests & Regression Checks

This folder will hold tests for `src/`:

- Unit tests for vector/matrix helpers
- Tests for NN layers, activations, and loss functions
- Simple “numerical check” style tests (e.g. comparing analytic vs numerical gradients)

Example future layout:

```text
tests/
├─ test_tensors_arrays_vectors.py
├─ test_linear_algebra_ops.py
└─ test_nnfs_layers.py
```

---

## Getting Started

1. **Clone the repo**

```bash
git clone <your-repo-url> ai-foundations
cd ai-foundations
```

2. **Create & activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch JupyterLab**

```bash
jupyter lab
```

Open any notebook under `notebooks/` and start exploring.

---

## Roadmap / Progress

High-level learning roadmap, roughly in this order:

- [x] Python basics & patterns
- [x] Algebra refresh
- [x] Vectors & basic linear algebra
- [ ] Deeper linear algebra (eigenvalues, eigenvectors, decompositions)
- [ ] Probability & statistical inference
- [ ] Classical ML models
- [ ] Neural networks from scratch (full training loop)
- [ ] Optimisation & regularisation
- [ ] Larger projects / case studies

This list is intentionally flexible and will evolve as the project grows.

---

## Notes

- Notebooks are kept clean in git using `nbstripout` (no large outputs committed).
- The repository is designed to be self-contained and reproducible for future revisits.
- Anything in `src/` should be importable without needing Jupyter.
