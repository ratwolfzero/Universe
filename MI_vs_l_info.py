import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import math

# --- Core Functions ---
@njit
def generate_golomb(n: int) -> np.ndarray:
    """Optimized Golomb ruler generator"""
    G = np.zeros(n, dtype=np.int64)
    D = set()
    G[0] = 0
    current_length = 1
    
    while current_length < n:
        m = G[current_length - 1] + 1
        while True:
            valid = True
            new_diffs = set()
            for i in range(current_length):
                diff = m - G[i]
                if diff in D or diff in new_diffs:
                    valid = False
                    break
                new_diffs.add(diff)
            if valid:
                G[current_length] = m
                D.update(new_diffs)
                current_length += 1
                break
            m += 1
    return G.astype(np.float64)

def compute_mi_matrix(G):
    """Numerically stable MI matrix calculation"""
    n = len(G)
    diffs = np.abs(np.subtract.outer(G, G))
    np.fill_diagonal(diffs, np.inf)
    mean_diff = np.mean(diffs[diffs < np.inf])
    norm_diffs = diffs / (mean_diff + 1e-16)
    W = np.log(1 + 1/norm_diffs)
    return np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

# --- Visualization ---
def create_final_visualization(n=200):
    """Produces publication-ready MI matrix plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm'
    })
    
    G = generate_golomb(n)
    W = compute_mi_matrix(G)
    l_info = 1/(1 + math.log(n))
    threshold = np.percentile(W[W > 0], 90)  # 90th percentile
    
    fig, ax = plt.subplots(figsize=(10, 8.5))
    
    # Main heatmap
    im = ax.imshow(W, cmap='inferno', 
                  norm=plt.Normalize(0, np.max(W)*1.05),
                  interpolation='nearest')
    
    # Colorbar with consistent formatting
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mutual Information $I_n(i,j)$', 
                  rotation=270, 
                  labelpad=20,
                  fontsize=12)
    
    # Precise halo boundary
    halo_mask = (W >= threshold*0.98) & (W <= threshold*1.02)
    halo_i, halo_j = np.where(halo_mask)
    ax.scatter(halo_j, halo_i, color='lime', s=12, marker='o', 
              edgecolor='black', linewidth=0.5,
              label=fr'Halo boundary ($I_n$={threshold:.2f})')
    
    # Theoretical scale markers
    l_info_pos = int(n * l_info)
    ax.axvline(l_info_pos, color='dodgerblue', linestyle=':', linewidth=2.5,
              label=fr'Informational scale ($n \cdot \ell$={l_info_pos})')
    ax.axhline(l_info_pos, color='dodgerblue', linestyle=':', linewidth=2.5)
    
    # Diagonal reference
    ax.plot([0, n-1], [0, n-1], 'w-', alpha=0.7, linewidth=1.8, 
           label='Diagonal ($i=j$)')
    
    # Annotation and formatting
    ax.set_title(f'Mutual Information Matrix ($n$={n})', 
                fontsize=14, pad=15)
    ax.set_xlabel('Distinction Index $j$', fontsize=12, labelpad=10)
    ax.set_ylabel('Distinction Index $i$', fontsize=12, labelpad=10)
    
    # Improved legend
    legend = ax.legend(loc='upper right', framealpha=1, fontsize=10)
    legend.get_frame().set_edgecolor('0.2')
    legend.get_frame().set_boxstyle('Round, pad=0.2')
    
    # Custom tick marks
    tick_step = max(1, n//8)
    ticks = np.arange(0, n, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig('Figure_x_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Verification output
    print("\n" + "="*60)
    print(f"Final Analysis (n = {n})".center(60))
    print("="*60)
    print(f"{'Parameter':<25} {'Value':>15} {'Units':>10}")
    print("-"*60)
    print(f"{'Informational scale (ℓ)':<25} {l_info:>15.4f} {'-':>10}")
    print(f"{'Theoretical position':<25} {l_info_pos:>15d} {'-':>10}")
    print(f"{'Halo threshold':<25} {threshold:>15.4f} {'bits':>10}")
    print(f"{'Maximum mutual info':<25} {np.max(W):>15.4f} {'bits':>10}")
    print(f"{'Minimum mutual info':<25} {np.min(W[W>0]):>15.4f} {'bits':>10}")
    print("="*60)

# --- Execution ---
if __name__ == "__main__":
    create_final_visualization(n=1000)
