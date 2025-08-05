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

Growth of Golomb Ruler (Temporal Dimension)

Display: Shows $t_i$ to 2M at 500, per generate_golomb.
Relation: Basis for Plots 7, 8, 9’s temporal dominance.

Mutual Information Matrix at n=500

Display: Correctly depicts 500x500 $W$ heatmap, with title and axes matching the full ruler.
Relation: Drives Laplacian for Plots 3, 5-9’s embeddings.

Eigenvalue Ratios for Dimensional Transitions

Display: Expected to show blue ($\lambda_2/\lambda_1$) at 19, orange ($\lambda_3/\lambda_2$) at 76, green ($\lambda_4/\lambda_3$) at 308, with overlap confirming 4D.
Relation: Guides Plots 5-9’s dimensional structures.

Informational Curvature Evolution of $R_n$

Display: Shows $R_n$ from 1.333 to 3.595, with transitions at 19, 76, 308.
Relation: Supports Plot 3’s logic, influencing Plots 5-9.

2D Spatial Embedding at n=76 (1D→2D)

Display: Displays linear 2D scatter, matching $\lambda_2/\lambda_1 = 1.182$.
Relation: Evolves into Plot 6, influenced by Plots 2 and 3.

3D Spatial Embedding at n=76 (2D→3D)

Display: Shows scattered 3D points, with $\lambda_3/\lambda_2 = 1.083$.
Relation: Transitions to Plots 7 and 9, driven by Plots 2 and 3.

3D Spatial Embedding at n=308 (4D Spacetime Stabilization)

Display: Displays 3D curve, with $\lambda_4/\lambda_3 = 1.049$.
Relation: Links to Plot 8’s correlation and Plot 9’s collapse.

Temporal-Spatial Projection at n=308

Display: Shows smooth $t_i$ vs. $s_{i,1}$ curve, due to $\lambda_1$ alignment.
Relation: Complements Plot 7, influencing Plot 9.

Comparison of First 76 Distinctions

Display: Shows blue scatter vs. red dot, with collapse due to 4D dominance.
Relation: Builds on Plots 6 and 7, reflecting Plot 8.

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
