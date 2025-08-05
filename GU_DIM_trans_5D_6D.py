import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from numba import njit
from tqdm import tqdm


@njit
def generate_golomb(n: int) -> np.ndarray:
    """
    Generates the first n Golomb rulers using an optimized growing algorithm.
    """
    G = np.zeros(n, dtype=np.int64)
    D_size = 1024
    D = np.zeros(D_size, dtype=np.bool_)
    temp_size = 1024
    temp = np.zeros(temp_size, dtype=np.bool_)
    G[0] = 0
    current_length = 1

    while current_length < n:
        m = G[current_length - 1] + 1
        while True:
            valid = True
            max_diff = 0
            for i in range(current_length):
                diff = m - G[i]
                if diff >= D_size:
                    new_size = max(D_size * 2, diff + 1)
                    new_D = np.zeros(new_size, dtype=np.bool_)
                    new_D[:D_size] = D
                    D = new_D
                    D_size = new_size
                if D[diff]:
                    valid = False
                    break
                if diff > max_diff:
                    max_diff = diff
            if valid:
                if max_diff >= temp_size:
                    new_temp_size = max(temp_size * 2, max_diff + 1)
                    new_temp = np.zeros(new_temp_size, dtype=np.bool_)
                    new_temp[:temp_size] = temp
                    temp = new_temp
                    temp_size = new_temp_size
                temp[:max_diff + 1] = False
                for i in range(current_length):
                    diff = m - G[i]
                    if temp[diff]:
                        valid = False
                        break
                    temp[diff] = True
            if valid:
                for i in range(current_length):
                    diff = m - G[i]
                    D[diff] = True
                G[current_length] = m
                current_length += 1
                break
            else:
                m += 1
    return G.astype(np.float64)


def compute_metrics(G):
    """Numerically stable metric calculation for mutual information and curvature."""
    n = len(G)
    if n < 2:
        return 1.0, 1.0, 0.0, np.zeros((n, n))
    diffs = np.abs(np.subtract.outer(G, G))
    np.fill_diagonal(diffs, np.inf)
    mean_diff = np.mean(diffs[diffs < np.inf])
    norm_diffs = diffs / (mean_diff + 1e-16)
    W = np.log(1 + 1/norm_diffs)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    I_max = np.max(W)
    d_min = 1/(1 + I_max)
    l_info = 1/(1 + np.log(n))
    R_n = max(0, (1/l_info) * (1 - d_min/l_info))
    return d_min, l_info, R_n, W


def compute_embedding(G, dim):
    """Compute spectral embedding in 2D or 3D using Laplacian eigenvectors."""
    n = len(G)
    if n < 2:
        return np.zeros((n, dim))
    d_min, l_info, R_n, W = compute_metrics(G)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    try:
        _, eigenvectors = eigh(L, eigvals_only=False,
                               subset_by_index=[1, min(dim, n-1)])
        return eigenvectors[:, :dim]
    except Exception as e:
        print(f"Error in compute_embedding for dim={dim}: {e}")
        return np.zeros((n, dim))


def compute_embedding_spacetime(G, spatial_dim=3):
    """
    Compute spectral embedding with explicit temporal coordinate for 4D spacetime.
    Returns (n, 4) array: [t_i, s_{i,1}, s_{i,2}, s_{i,3}].
    """
    n = len(G)
    if n < 2:
        return np.zeros((n, spatial_dim + 1))
    d_min, l_info, R_n, W = compute_metrics(G)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    try:
        _, eigenvectors = eigh(L, eigvals_only=False, subset_by_index=[
                               1, min(spatial_dim, n-1)])
        embedding = np.zeros((n, spatial_dim + 1))
        embedding[:, 0] = G  # Temporal coordinate
        embedding[:, 1:spatial_dim + 1] = eigenvectors[:,
                                                       :spatial_dim]  # Spatial coordinates
        return embedding
    except Exception as e:
        print(
            f"Error in compute_embedding_spacetime for spatial_dim={spatial_dim}: {e}")
        return np.zeros((n, spatial_dim + 1))


def check_transitions(G, d_min, l_info, R_n):
    """
    Check dimensional transitions based on eigenvalue ratios and curvature.
    Extended for D5 and D6 transitions with intrinsic conditions.
    """
    n = len(G)
    if n < 10:
        return False, False, False, False, False, 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        _, _, _, W = compute_metrics(G)
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, 6])
        λ0, λ1, λ2, λ3, λ4, λ5, λ6 = eigenvalues
        λ1 = max(λ1, 1e-8)
        λ2 = max(λ2, 1e-8)
        λ3 = max(λ3, 1e-8)
        λ4 = max(λ4, 1e-8)
        λ5 = max(λ5, 1e-8)
        λ6 = max(λ6, 1e-8)
        r1 = λ2 / λ1  # 1D→2D
        r2 = λ3 / λ2  # 2D→3D
        r3 = λ4 / λ3  # 4D Spacetime Stabilization
        r4 = λ5 / λ4  # D5 Transition
        r5 = λ6 / λ5  # D6 Transition
        transition_2D = (r1 > r2 and R_n > r1)
        transition_3D = (r2 > r3 and R_n > r1+r2)
        transition_4D = (r3 > r2 and R_n > r1+r2+r3)
        transition_D5 = (r4 > r3 and R_n > r1+r2+r3+r4)
        transition_D6 = (r5 > r4 and R_n > r1+r2+r3+r4+r5)
        return transition_2D, transition_3D, transition_4D, transition_D5, transition_D6, r1, r2, r3, r4, r5
    except Exception as e:
        print("Error in check_transitions:", e)
        return False, False, False, False, False, 0.0, 0.0, 0.0, 0.0, 0.0


def validate_golomb(G):
    """Validate that G is a Golomb ruler."""
    n = len(G)
    diffs = np.abs(np.subtract.outer(G, G))
    np.fill_diagonal(diffs, np.inf)
    unique_diffs = np.unique(diffs[diffs < np.inf])
    expected_diffs = n * (n - 1) // 2
    is_valid = len(unique_diffs) == expected_diffs
    entropy = expected_diffs
    return is_valid, entropy


def plot_results(G_full, results, metrics_history):
    """
    Generate plots for Golomb ruler, mutual information, eigenvalues, curvature,
    embeddings, temporal-spatial projection, and subset comparison.
    """
    ns, d_mins, l_infos, R_ns, r1s, r2s, r3s, r4s, r5s = metrics_history

    # Plot 1: Golomb Ruler Growth (Temporal Coordinate)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(G_full) + 1), G_full,
             'o-', label='Golomb Ruler Marks (t_i)')
    plt.xlabel('Index (n)')
    plt.ylabel('Temporal Coordinate (t_i)')
    plt.title('Growth of Golomb Ruler (Temporal Dimension)')
    plt.grid(True)
    plt.legend()
    plt.savefig('golomb_ruler_growth.png')
    plt.close()

    # Plot 2: Mutual Information Matrix at n_max
    _, _, _, W = compute_metrics(G_full)
    plt.figure(figsize=(8, 6))
    plt.imshow(W, cmap='magma', interpolation='bilinear')
    plt.colorbar(label='Mutual Information I(X_i; X_j)')
    plt.title(f'Mutual Information Matrix at n={len(G_full)}')
    plt.xlabel('Distinction i')
    plt.ylabel('Distinction j')
    plt.savefig('mutual_information_matrix.svg')
    plt.close()

    # Plot 3: Eigenvalue Ratios
    plt.figure(figsize=(10, 6))
    plt.plot(ns, r1s, label='λ₂/λ₁ (1D→2D)')
    plt.plot(ns, r2s, label='λ₃/λ₂ (2D→3D)')
    plt.plot(ns, r3s, label='λ₄/λ₃ (4D Spacetime Stabilization)')
    plt.plot(ns, r4s, label='λ₅/λ₄ (D5 Transition)')
    plt.plot(ns, r5s, label='λ₆/λ₅ (D6 Transition)')
    if results.get('2D') is not None:
        plt.axvline(x=results.get('2D'), color='r', linestyle=':',
                    label=f'1D→2D at n={results.get("2D")}')
    if results.get('3D') is not None:
        plt.axvline(x=results.get('3D'), color='g', linestyle=':',
                    label=f'2D→3D at n={results.get("3D")}')
    if results.get('4D') is not None:
        plt.axvline(x=results.get('4D'), color='b', linestyle=':',
                    label=f'4D Stabilization at n={results.get("4D")}')
    if results.get('D5') is not None:
        plt.axvline(x=results.get('D5'), color='m', linestyle=':',
                    label=f'D5 Transition at n={results.get("D5")}')
    if results.get('D6') is not None:
        plt.axvline(x=results.get('D6'), color='c', linestyle=':',
                    label=f'D6 Transition at n={results.get("D6")}')
    plt.xlabel('Number of Distinctions (n)')
    plt.ylabel('Eigenvalue Ratios')
    plt.title('Eigenvalue Ratios for Dimensional Transitions')
    plt.grid(True)
    plt.legend()
    plt.savefig('eigenvalue_ratios.png')
    plt.close()

    # Plot 4: Informational Curvature
    plt.figure(figsize=(10, 6))
    plt.plot(ns, R_ns, label='R_n')
    if results.get('2D') is not None:
        plt.axvline(x=results.get('2D'), color='r', linestyle=':',
                    label=f'1D→2D at n={results.get("2D")}')
    if results.get('3D') is not None:
        plt.axvline(x=results.get('3D'), color='g', linestyle=':',
                    label=f'2D→3D at n={results.get("3D")}')
    if results.get('4D') is not None:
        plt.axvline(x=results.get('4D'), color='b', linestyle=':',
                    label=f'4D Stabilization at n={results.get("4D")}')
    if results.get('D5') is not None:
        plt.axvline(x=results.get('D5'), color='m', linestyle=':',
                    label=f'D5 Transition at n={results.get("D5")}')
    if results.get('D6') is not None:
        plt.axvline(x=results.get('D6'), color='c', linestyle=':',
                    label=f'D6 Transition at n={results.get("D6")}')
    plt.xlabel('Number of Distinctions (n)')
    plt.ylabel('Informational Curvature (R_n)')
    plt.title('Informational Curvature Evolution')
    plt.grid(True)
    plt.legend()
    plt.savefig('informational_curvature.png')
    plt.close()

    # Plot 5: 2D Embedding at 1D→2D Transition
    if results.get('2D') is not None:
        G_2D = G_full[:results['2D']]
        embedding_2D = compute_embedding(G_2D, 2)
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_2D[:, 0], embedding_2D[:,
                    1], c='black', label='Distinctions')
        plt.xlabel('X (Eigenvector 1)')
        plt.ylabel('Y (Eigenvector 2)')
        plt.title(f'2D Spatial Embedding at n={results["2D"]} (1D→2D)')
        plt.grid(True)
        plt.legend()
        plt.savefig('embedding_2D.png')
        plt.close()

    # Plot 6: 3D Embedding at 2D→3D Transition
    if results.get('3D') is not None:
        G_3D = G_full[:results['3D']]
        embedding_3D = compute_embedding(G_3D, 3)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding_3D[:, 0], embedding_3D[:, 1],
                   embedding_3D[:, 2], c='blue', label='Distinctions')
        ax.set_xlabel('X (Eigenvector 1)')
        ax.set_ylabel('Y (Eigenvector 2)')
        ax.set_zlabel('Z (Eigenvector 3)')
        ax.set_title(f'3D Spatial Embedding at n={results["3D"]} (2D→3D)')
        plt.legend()
        plt.savefig('embedding_3D.png')
        plt.close()

    # Plot 7: 3D Spatial Embedding at 4D Spacetime Stabilization
    if results.get('4D') is not None:
        G_4D = G_full[:results['4D']]
        embedding_4D = compute_embedding_spacetime(G_4D, spatial_dim=3)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding_4D[:, 1], embedding_4D[:, 2],
                   embedding_4D[:, 3], c='red', label='Distinctions')
        ax.set_xlabel('X (Eigenvector 1)')
        ax.set_ylabel('Y (Eigenvector 2)')
        ax.set_zlabel('Z (Eigenvector 3)')
        ax.set_title(
            f'3D Spatial Embedding at n={results["4D"]} (4D Spacetime Stabilization)')
        plt.legend()
        plt.savefig('embedding_spacetime_4D.png')
        plt.close()

    # Plot 8: Temporal-Spatial Projection at 4D Spacetime Stabilization
    if results.get('4D') is not None:
        G_4D = G_full[:results['4D']]
        embedding_4D = compute_embedding_spacetime(G_4D, spatial_dim=3)
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_4D[:, 0], embedding_4D[:,
                    1], c='purple', label='Distinctions')
        plt.xlabel('Temporal Coordinate (t_i)')
        plt.ylabel('Spatial Coordinate (s_{i,1})')
        plt.title(
            f'Temporal-Spatial Projection at n={results["4D"]} (4D Spacetime Stabilization)')
        plt.grid(True)
        plt.legend()
        plt.savefig('temporal_spatial_projection_4D.png')
        plt.close()

    # Plot 9: Comparison of First 76 Distinctions at n=76 vs. n=308
    if results.get('3D') is not None and results.get('4D') is not None:
        G_3D = G_full[:results['3D']]
        G_4D = G_full[:results['4D']]
        embedding_3D = compute_embedding(G_3D, 3)
        embedding_4D_subset = compute_embedding_spacetime(G_4D, spatial_dim=3)[
            :results['3D'], 1:4]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding_3D[:, 0], embedding_3D[:, 1],
                   embedding_3D[:, 2], c='blue', label=f'n={results["3D"]}')
        ax.scatter(embedding_4D_subset[:, 0], embedding_4D_subset[:, 1], embedding_4D_subset[:,
                   2], c='red', label=f'n={results["4D"]} (first {results["3D"]})')
        ax.set_xlabel('X (Eigenvector 1)')
        ax.set_ylabel('Y (Eigenvector 2)')
        ax.set_zlabel('Z (Eigenvector 3)')
        ax.set_title(f'Comparison of First {results["3D"]} Distinctions')
        plt.legend()
        plt.savefig('embedding_comparison.png')
        plt.close()


def print_summary(G_full, results, metrics_history):
    """Print a summary table of essential calculated values and stability metrics."""
    ns, d_mins, l_infos, R_ns, r1s, r2s, r3s, r4s, r5s = metrics_history
    print("\nSummary of Essential Calculated Values:")
    print("-" * 110)
    print(f"{'n':>5} | {'d_min':>8} | {'l_info':>8} | {'R_n':>8} | "
          f"{'λ₂/λ₁':>8} | {'λ₃/λ₂':>8} | {'λ₄/λ₃':>8} | {'λ₅/λ₄':>8} | {'λ₆/λ₅':>8} | {'Note'}")
    print("-" * 110)

    def print_row(idx, note):
        print(f"{ns[idx]:>5} | {d_mins[idx]:>8.3f} | {l_infos[idx]:>8.3f} | {R_ns[idx]:>8.3f} | "
              f"{r1s[idx]:>8.3f} | {r2s[idx]:>8.3f} | {r3s[idx]:>8.3f} | {r4s[idx]:>8.3f} | {r5s[idx]:>8.3f} | {note}")

    if results["2D"] is not None:
        print_row(results["2D"] - 1, "2D Transition")
    if results["3D"] is not None:
        print_row(results["3D"] - 1, "3D Transition")
    if results["4D"] is not None:
        print_row(results["4D"] - 1, "4D Spacetime Stabilization")
    if results["D5"] is not None:
        print_row(results["D5"] - 1, "D5 Transition")
    if results["D6"] is not None:
        print_row(results["D6"] - 1, "D6 Transition")
    print_row(len(ns) - 1, "Final Step")
    print("-" * 110)

    # Stability Metrics Comparison
    print("\nStability Metrics Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} | {'n=76':>10} | {'n=308':>10}")
    print("-" * 50)
    if results["3D"] is not None and results["4D"] is not None:
        G_3D = G_full[:results["3D"]]
        G_4D = G_full[:results["4D"]]
        d_min_3D, l_info_3D, R_n_3D, W_3D = compute_metrics(G_3D)
        d_min_4D, l_info_4D, R_n_4D, W_4D = compute_metrics(G_4D)
        d_ij_3D = 1 / (1 + W_3D)
        np.fill_diagonal(d_ij_3D, np.inf)
        d_ij_4D = 1 / (1 + W_4D)
        np.fill_diagonal(d_ij_4D, np.inf)
        l_eff_3D = np.min(d_ij_3D)
        l_eff_4D = np.min(d_ij_4D)
        E_n_3D = np.sum(1 / l_eff_3D**2 - 1 / d_ij_3D[d_ij_3D < np.inf]**2) / 2
        E_n_4D = np.sum(1 / l_eff_4D**2 - 1 / d_ij_4D[d_ij_4D < np.inf]**2) / 2
        embedding_3D = compute_embedding(G_3D, 3)
        embedding_4D = compute_embedding_spacetime(G_4D, spatial_dim=3)[:, 1:4]
        euclidean_3D = np.sqrt(
            np.sum((embedding_3D[:, None] - embedding_3D)**2, axis=2))
        euclidean_4D = np.sqrt(
            np.sum((embedding_4D[:, None] - embedding_4D)**2, axis=2))
        np.fill_diagonal(d_ij_3D, 0)
        np.fill_diagonal(d_ij_4D, 0)
        distortion_3D = np.mean((euclidean_3D - d_ij_3D)
                                ** 2 / (d_ij_3D**2 + 1e-16))
        distortion_4D = np.mean((euclidean_4D - d_ij_4D)
                                ** 2 / (d_ij_4D**2 + 1e-16))
        alt_distortion_3D = np.mean(np.abs(euclidean_3D - d_ij_3D))
        alt_distortion_4D = np.mean(np.abs(euclidean_4D - d_ij_4D))
        norm_distortion_3D = distortion_3D / \
            (results["3D"] * (results["3D"] - 1) / 2)
        norm_distortion_4D = distortion_4D / \
            (results["4D"] * (results["4D"] - 1) / 2)
        print(f"{'Distortion':<15} | {distortion_3D:>10.3f} | {distortion_4D:>10.3f}")
        print(
            f"{'Abs Distortion':<15} | {alt_distortion_3D:>10.3f} | {alt_distortion_4D:>10.3f}")
        print(
            f"{'Norm Distortion':<15} | {norm_distortion_3D:>10.6f} | {norm_distortion_4D:>10.6f}")
        print(f"{'E_n':<15} | {E_n_3D:>10.3f} | {E_n_4D:>10.3f}")
        print(f"{'R_n':<15} | {R_n_3D:>10.3f} | {R_n_4D:>10.3f}")
    print("-" * 50)


def print_validation(G, results):
    """
    Print validation parameters to confirm compliance with the framework.
    Updated to include D5 and D6 transitions.
    """
    is_valid, entropy = validate_golomb(G)
    print("\nValidation Parameters:")
    print("-" * 80)
    print(
        f"Golomb Ruler Validity (Axiom II): {'Valid' if is_valid else 'Invalid'}")
    print(f"Entropy (S_n = n(n-1)/2, Appendix C): {entropy}")
    temporal_valid = np.all(np.diff(G) > 0)
    print(
        f"Temporal Order (Axiom III, G[i] < G[i+1]): {'Valid' if temporal_valid else 'Invalid'}")

    def validate_transition(n_transition, transition_name, condition, r_values, prev_r_sum):
        G_t = G[:n_transition]
        d_min, l_info, R_n, W = compute_metrics(G_t)
        _, _, _, _, _, r1, r2, r3, r4, r5 = check_transitions(
            G_t, d_min, l_info, R_n)
        n = len(G_t)
        d_ij = 1 / (1 + W)
        np.fill_diagonal(d_ij, np.inf)
        l_eff = np.min(d_ij[d_ij < np.inf])
        E_n = np.sum(1 / l_eff**2 - 1 / d_ij[d_ij < np.inf]**2) / 2
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, 6])
        λ0, λ1, λ2, λ3, λ4, λ5, λ6 = eigenvalues
        gap1 = λ1
        gap2 = λ2 - λ1
        gap3 = λ3 - λ2
        gap4 = λ4 - λ3
        gap5 = λ5 - λ4
        gap6 = λ6 - λ5
        embedding = compute_embedding(G_t, 3) if transition_name in [
            "2D", "3D"] else compute_embedding_spacetime(G_t, spatial_dim=3)[:, 1:4]
        euclidean_dists = np.sqrt(
            np.sum((embedding[:, None] - embedding)**2, axis=2))
        np.fill_diagonal(d_ij, 0)
        distortion = np.mean((euclidean_dists - d_ij)**2 / (d_ij**2 + 1e-16))
        alt_distortion = np.mean(np.abs(euclidean_dists - d_ij))
        norm_distortion = distortion / (n * (n - 1) / 2)
        print(f"\n{transition_name} Transition at n={n_transition}:")
        print(
            f" Intrinsic condition: {condition}: {'Valid' if eval(condition) else 'Invalid'}")
        print(f" {', '.join(f'λ{i+2}/λ{i+1} = {r:.3f}' for i, r in enumerate(r_values))
                  }, R_n = {R_n:.3f}, prev_r_sum = {prev_r_sum:.3f}")
        print(
            f" Energy Functional (Axiom V): E_n = {E_n:.3f} (>= 0: {'Valid' if E_n >= 0 else 'Invalid'})")
        print(
            f" Spectral Gaps (Annex H): λ₁ = {gap1:.3f}, λ₂-λ₁ = {gap2:.3f}, λ₃-λ₂ = {gap3:.3f}, λ₄-λ₃ = {gap4:.3f}, λ₅-λ₄ = {gap5:.3f}, λ₆-λ₅ = {gap6:.3f}")
        print(f" Embedding Distortion (Annex D): {distortion:.3f}")
        print(
            f" Alternative Distortion (Absolute Error): {alt_distortion:.3f}")
        print(f" Normalized Distortion (per pair): {norm_distortion:.6f}")
        print(
            f" W (I_n) Summary: Max = {np.max(W):.3f}, Mean = {np.mean(W[W > 0]):.3f}")

    # Validate 1D→2D transition
    if results["2D"] is not None:
        G_t = G[:results["2D"]]
        _, _, _, _, _, r1, r2, r3, r4, r5 = check_transitions(
            G_t, *compute_metrics(G_t)[:3])
        validate_transition(results["2D"], "1D→2D", "r1 > r2 and R_n > r1", [
                            r1, r2, r3, r4, r5], r1)

    # Validate 2D→3D transition
    if results["3D"] is not None:
        G_t = G[:results["3D"]]
        _, _, _, _, _, r1, r2, r3, r4, r5 = check_transitions(
            G_t, *compute_metrics(G_t)[:3])
        validate_transition(
            results["3D"], "2D→3D", "r2 > r3 and R_n > r1+r2", [r1, r2, r3, r4, r5], r1 + r2)

    # Validate 4D Spacetime Stabilization
    if results["4D"] is not None:
        G_t = G[:results["4D"]]
        _, _, _, _, _, r1, r2, r3, r4, r5 = check_transitions(
            G_t, *compute_metrics(G_t)[:3])
        validate_transition(results["4D"], "4D Spacetime Stabilization",
                            "r3 > r2 and R_n > r1+r2+r3", [r1, r2, r3, r4, r5], r1 + r2 + r3)

    # Validate D5 Transition
    if results["D5"] is not None:
        G_t = G[:results["D5"]]
        _, _, _, _, _, r1, r2, r3, r4, r5 = check_transitions(
            G_t, *compute_metrics(G_t)[:3])
        validate_transition(results["D5"], "D5", "r4 > r3 and R_n > r1+r2+r3+r4", [
                            r1, r2, r3, r4, r5], r1 + r2 + r3 + r4)

    # Validate D6 Transition
    if results["D6"] is not None:
        G_t = G[:results["D6"]]
        _, _, _, _, _, r1, r2, r3, r4, r5 = check_transitions(
            G_t, *compute_metrics(G_t)[:3])
        validate_transition(results["D6"], "D6", "r5 > r4 and R_n > r1+r2+r3+r4+r5", [
                            r1, r2, r3, r4, r5], r1 + r2 + r3 + r4 + r5)

    print("NOTE: All transition conditions are intrinsic with zero free parameters.")


def simulate(n_max):
    """
    Simulation with 3D spatial embedding and 4D spacetime embedding.
    Extended for D5 and D6 transitions with intrinsic conditions.
    """
    results = {"2D": None, "3D": None, "4D": None, "D5": None, "D6": None}
    print("Generating Golomb ruler...")
    G_full = generate_golomb(n_max)
    ns, d_mins, l_infos, R_ns = [], [], [], []
    r1s, r2s, r3s, r4s, r5s = [], [], [], [], []

    print("Running simulation calculations...")
    with tqdm(range(1, n_max + 1), desc="Simulation",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              unit=" steps") as pbar:
        for n in pbar:
            G = G_full[:n]
            d_min, l_info, R_n, _ = compute_metrics(G)
            t2d, t3d, t4d, tD5, tD6, r1, r2, r3, r4, r5 = check_transitions(
                G, d_min, l_info, R_n)
            ns.append(n)
            d_mins.append(d_min)
            l_infos.append(l_info)
            R_ns.append(R_n)
            r1s.append(r1)
            r2s.append(r2)
            r3s.append(r3)
            r4s.append(r4)
            r5s.append(r5)
            if t2d and results["2D"] is None:
                results["2D"] = n
            if t3d and results["2D"] is not None and results["3D"] is None:
                results["3D"] = n
            if t4d and results["3D"] is not None and results["4D"] is None:
                results["4D"] = n
            if tD5 and results["4D"] is not None and results["D5"] is None:
                results["D5"] = n
            if tD6 and results["D5"] is not None and results["D6"] is None:
                results["D6"] = n
            if n % 100 == 0 or n == n_max:
                tqdm.write(
                    f"Progress: n={n}, d_min={d_min:.3f}, l_info={l_info:.3f}, R_n={R_n:.3f}")

    plot_results(G_full, results, (ns, d_mins, l_infos,
                 R_ns, r1s, r2s, r3s, r4s, r5s))
    print_summary(G_full, results, (ns, d_mins, l_infos,
                  R_ns, r1s, r2s, r3s, r4s, r5s))
    print_validation(G_full, results)
    return results, G_full


# Run simulation
print("Starting spacetime simulation with intrinsic transition conditions...")
results, G_full = simulate(1300)
print("\nFinal Results:")
print(f"1D→2D transition at n={results['2D']}")
print(f"2D→3D transition at n={results['3D']}")
print(f"4D Spacetime Stabilization at n={results['4D']}")
print(f"D5 Transition at n={results['D5']}")
print(f"D6 Transition at n={results['D6']}")
