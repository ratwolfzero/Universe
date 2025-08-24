import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import argparse
import os             


@njit
def generate_golomb(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)  # Match return type to float64
    
    # Pre-allocate D with estimated max size (tuned for n=5100; ~3e9)
    estimated_max = 3000000000
    D = np.zeros(estimated_max, dtype=np.bool_)
    G = np.zeros(n, dtype=np.int64)  # Keep int64 internally for precision
    G[0] = 0
    current_length = 1

    while current_length < n:
        m = G[current_length - 1] + 1
        while True:
            valid = True                                            
            for i in range(current_length):
                diff = m - G[i]
                if diff >= len(D):
                    raise ValueError("Exceeded pre-allocated size; increase estimated_max")
                if D[diff]:
                    valid = False
                    break
            if valid:
                for i in range(current_length):
                    diff = m - G[i]
                    D[diff] = True
                G[current_length] = m                  
                current_length += 1
                break
            m += 1
    return G.astype(np.float64)  # Consistent float64 return
    

																																			 
def compute_metrics(G):
    n = len(G)
    if n < 2:
        return 1.0, 1.0, 0.0, np.zeros((n, n))
    diffs = np.abs(np.subtract.outer(G, G))
    np.fill_diagonal(diffs, np.inf)
    mean_diff = np.mean(diffs[diffs < np.inf])
    norm_diffs = diffs / (mean_diff + 1e-16)
    W = np.log(1 + 1/norm_diffs)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    
    return W

def plot_mutual_information_matrix(W, output_file=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(W, cmap='magma', interpolation='bilinear')
    plt.colorbar(label='Mutual Information I(X_i; X_j)')
    plt.title(f'Mutual Information Matrix (n={W.shape[0]})')
    plt.xlabel('Distinction i')
    plt.ylabel('Distinction j')
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()                     


def main(n_max: int, output_dir: str):                      
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating Golomb ruler of length n={n_max}...")
    G = generate_golomb(n_max)

    print("Computing mutual information matrix...")
    
    W = compute_metrics(G)

    print("Plotting mutual information matrix...")
    output_path = os.path.join(output_dir, f"mutual_information_matrix_n{n_max}.png")
    plot_mutual_information_matrix(W, output_path)
    print(f"Saved mutual information heatmap to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golomb Ruler Mutual Information Matrix Simulation")
    parser.add_argument('--n_max', type=int, default=1000, help='Maximum number of distinctions (default: 100)')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save plots (default: ./results)')
    args = parser.parse_args()

    main(args.n_max, args.output_dir)

