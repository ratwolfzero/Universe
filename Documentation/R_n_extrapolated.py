import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from numba import njit
from tqdm import tqdm
from scipy.optimize import curve_fit
import pickle

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
    """Numerically stable metric calculation."""
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

def logarithmic_fit(x, a, b, c):
    """Logarithmic function for curve fitting: a * ln(x) + b + c/x."""
    return a * np.log(x) + b + c / x

def extrapolate_R_n(ns, R_ns, n_max=10000):
    """Extrapolate R_n using a logarithmic fit."""
    # Fit the data to a logarithmic model
    popt, _ = curve_fit(logarithmic_fit, ns, R_ns, p0=[1.0, 0.0, 0.0], maxfev=10000)
    a, b, c = popt
    print(f"Fit parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}")
    
    # Generate extrapolation points
    n_extra = np.arange(1001, n_max + 1)
    R_n_extra = logarithmic_fit(n_extra, a, b, c)
    
    return n_extra, R_n_extra

def further_extrapolate_R_n(ns, R_ns, current_max_n, new_max_n):
    """Further extrapolate R_n to a higher n value using the existing fit."""
    # Use the same logarithmic fit parameters from the initial extrapolation
    popt, _ = curve_fit(logarithmic_fit, ns, R_ns, p0=[1.0, 0.0, 0.0], maxfev=10000)
    a, b, c = popt
    print(f"Further extrapolation fit parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}")
    
    # Generate new extrapolation points from current_max_n + 1 to new_max_n
    n_extra_further = np.arange(current_max_n + 1, new_max_n + 1)
    R_n_extra_further = logarithmic_fit(n_extra_further, a, b, c)
    
    return n_extra_further, R_n_extra_further

def display_further_extrapolation(ns, R_ns, n_extra, R_n_extra, n_extra_further, R_n_extra_further):
    """Display the computed and further extrapolated R_n values with plots and summary."""
    # Combine all data
    ns_all = np.concatenate([ns, n_extra, n_extra_further])
    R_ns_all = np.concatenate([R_ns, R_n_extra, R_n_extra_further])
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(ns, R_ns, 'b-', label='Computed R_n (n ≤ 1000)')
    plt.plot(n_extra, R_n_extra, 'r--', label=f'Extrapolated R_n (n ≤ 10000)')
    plt.plot(n_extra_further, R_n_extra_further, 'g-.', label=f'Further Extrapolated R_n (n ≤ {n_extra_further[-1]})')
    plt.xlabel('Number of Distinctions (n)')
    plt.ylabel('Informational Curvature (R_n)')
    plt.title('Informational Curvature: Computed and Extrapolated')
    plt.grid(True)
    plt.legend()
    plt.savefig('R_n_further_extrapolation.png')
    plt.show()
    plt.close()
    
    # Save results
    results = {
        'ns': ns_all,
        'R_ns': R_ns_all,
        'computed_ns': ns,
        'computed_R_ns': R_ns,
        'extrapolated_ns_10000': n_extra,
        'extrapolated_R_ns_10000': R_n_extra,
        'further_extrapolated_ns': n_extra_further,
        'further_extrapolated_R_ns': R_n_extra_further
    }
    with open('R_n_results_extended.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Extended results saved to 'R_n_results_extended.pkl'")
    
    # Print summary
    print("\nSummary of Key Values:")
    print("-" * 60)
    print(f"{'n':>10} | {'R_n':>10}")
    print("-" * 60)
    for n in [100, 1000, 5000, 10000, n_extra_further[-1]]:
        idx = np.where(ns_all == n)[0]
        if len(idx) > 0:
            print(f"{n:>10} | {R_ns_all[idx[0]]:>10.3f}")
    print("-" * 60)

def main(n_max=1000, extrapolate_to=10000, further_extrapolate_to=50000):
    """Main function to compute and extrapolate informational curvature."""
    print("Generating Golomb ruler...")
    G_full = generate_golomb(n_max)
    
    ns = []
    R_ns = []
    
    print("Computing informational curvature...")
    with tqdm(range(1, n_max + 1), desc="Computing R_n", unit=" steps") as pbar:
        for n in pbar:
            G = G_full[:n]
            _, _, R_n, _ = compute_metrics(G)
            ns.append(n)
            R_ns.append(R_n)
            if n % 100 == 0 or n == n_max:
                tqdm.write(f"Progress: n={n}, R_n={R_n:.3f}")
    
    ns = np.array(ns)
    R_ns = np.array(R_ns)
    
    print("Extrapolating R_n...")
    n_extra, R_n_extra = extrapolate_R_n(ns, R_ns, extrapolate_to)
    
    print("Further extrapolating R_n...")
    n_extra_further, R_n_extra_further = further_extrapolate_R_n(ns, R_ns, extrapolate_to, further_extrapolate_to)
    
    # Display results
    display_further_extrapolation(ns, R_ns, n_extra, R_n_extra, n_extra_further, R_n_extra_further)
    
if __name__ == "__main__":
    main(n_max=1000, extrapolate_to=10000, further_extrapolate_to=50000)
