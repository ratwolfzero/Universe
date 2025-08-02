# Golomb Universe Simulation

This repository contains a Python implementation of the simulation code for the **Golomb Universe: A Combinatorial Axiomatization of Physical Reality** — a theoretical framework that models the emergence of spacetime and physical quantities from a minimal set of logical distinctions forming a greedy Golomb ruler.

The simulation generates Golomb rulers, computes informational metrics, checks for dimensional transitions, and visualizes the emergent spacetime structure as described in the theoretical framework.

---

## 📘 Overview

The simulation models the universe as a sequence of **irreducible distinctions**, represented as a **greedy Golomb ruler** — a set of integers where all pairwise differences are unique.

It computes mutual information, informational curvature, and spectral embeddings to simulate the emergence of **1D (temporal), 2D, 3D, and 4D** spacetime structures. The code validates key axioms:

* **Axiom II (Irreducibility)**
* **Axiom VII (Informational Structure)**

### Features

* **Golomb Ruler Generation**
* **Informational Metrics**: Computes:

  * Mutual information (\$I\_n\$)
  * Informational scale (\$\ell\_{\text{info}}\$)
  * Curvature (\$R\_n\$)
* **Dimensional Transitions**
* **Spectral Embeddings**
* **Validation**: Axiom compliance
* **Visualizations**: Embeddings, metrics, growth
* **Summary Outputs**

---

## 📦 Prerequisites

Install required Python packages:

```bash
pip install numpy matplotlib scipy numba tqdm
```

---

## 📊 Key Outputs

### Plots

* `golomb_ruler_growth.png`
* `mutual_information_matrix.svg`
* `eigenvalue_ratios.png`
* `informational_curvature.png`
* `embedding_2D.png`
* `embedding_3D.png`
* `embedding_spacetime_4D.png`
* `temporal_spatial_projection_4D.png`
* `embedding_comparison.png`

### Console Output

* Progress updates
* Summary table of metrics
* Stability comparisons
* Validation parameters

---

## 🧪 Example Output

```bash
Starting spacetime simulation with intrinsic transition conditions...
Generating Golomb ruler...
Running simulation calculations...
Simulation: 100%|██████████████████████████████████████| 1000/1000 [00:20<00:00]


Summary of Essential Calculated Values

| n    | d\_min | l\_info | R\_n  | λ₂/λ₁ | λ₃/λ₂ | λ₄/λ₃ | Note                       |
| ---- | ------ | ------- | ----- | ----- | ----- | ----- | -------------------------- |
| 19   | 0.168  | 0.254   | 1.333 | 1.320 | 1.089 | 1.169 | 2D Transition              |
| 76   | 0.106  | 0.188   | 2.312 | 1.182 | 1.083 | 1.056 | 3D Transition              |
| 308  | 0.077  | 0.149   | 3.261 | 1.141 | 1.038 | 1.049 | 4D Spacetime Stabilization |
| 1000 | 0.061  | 0.126   | 4.066 | 1.182 | 1.019 | 1.004 | Final Step                 |

Final Results

* **1D → 2D transition** at `n = 19`
* **2D → 3D transition** at `n = 76`
* **4D Spacetime Stabilization** at `n = 308`
```

---

## 🧩 Code Structure

### Main Functions

* `generate_golomb(n)`
* `compute_metrics(G)`
* `compute_embedding(G, dim)`
* `compute_embedding_spacetime(G, spatial_dim)`
* `check_transitions(G, d_min, l_info, R_n)`
* `validate_golomb(G)`
* `plot_results(...)`
* `print_summary(...)`
* `print_validation(...)`
* `simulate(n_max)`

---

## 📐 Theoretical Alignment

| Concept / Axiom | Implementation                                      |
| --------------- | --------------------------------------------------- |
| Axiom II        | Unique pairwise differences (Golomb ruler)          |
| Axiom III       | Monotonic temporal ordering (np.diff > 0)           |
| Axiom V         | Energy functional (\$E\_n\$) via validation         |
| Axiom VII       | \$I\_n\$, \$\ell\_{\text{info}}\$, \$R\_n\$ metrics |
| Appendix D      | 4D embedding via spectral + time structure          |
| Appendix E      | Informational scale implementation                  |
| Appendix H      | Spectral gaps / curvature for validation            |

---

## ⚠️ Limitations

* **High computational cost** for `n_max > 1500`
