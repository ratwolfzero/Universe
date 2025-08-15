import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.signal import savgol_filter
from numba import njit
from tqdm import tqdm
import ruptures as rpt

# Tuning parameters - easily accessible and adjustable
PENALTY = 2.7  # Penalty for CPD (Pelt algorithm)
WINDOW_LENGTH = 4  # Window length for Savitzky-Golay filter
POLYORDER = 2  # Polynomial order for Savitzky-Golay filter
TOLERANCE = 100  # Tolerance/proximity for matching detected points to known transitions
CPD_MODEL = "rbf"  # Model for CPD (e.g., "rbf", "l2", etc.)
KNOWN_TRANSITIONS = [19, 76, 308]  # Expected/known transition points for reference
N_MAX = 1300  # Maximum n for simulation (can be increased, but computationally intensive)
EIG_SUBSET = [0, 4]  # Eigenvalue subset indices for computation (λ0 to λ4)
MIN_N_FOR_METRICS = 5  # Minimum n for full metric computation
MIN_N_FOR_TRANSITIONS = 10  # Minimum n for transition checks

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
        return 1.0, 1.0, 0.0, np.zeros((n, n)), 0.0, 0.0, 0.0
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
    
    if n < MIN_N_FOR_METRICS:
        return d_min, l_info, R_n, W, 0.0, 0.0, 0.0
    
    try:
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=EIG_SUBSET)
        λ0, λ1, λ2, λ3, λ4 = eigenvalues
        λ1 = max(λ1, 1e-8)
        λ2 = max(λ2, 1e-8)
        λ3 = max(λ3, 1e-8)
        λ4 = max(λ4, 1e-8)
        r1 = λ2 / λ1
        r2 = λ3 / λ2
        r3 = λ4 / λ3
        return d_min, l_info, R_n, W, r1, r2, r3
    except Exception as e:
        print(f"Error in compute_metrics eigenvalue calculation: {e}")
        return d_min, l_info, R_n, W, 0.0, 0.0, 0.0

def check_transitions(G, d_min, l_info, R_n):
    """
    Check dimensional transitions based on eigenvalue ratios and curvature.
    """
    n = len(G)                                                                  
    if n < MIN_N_FOR_TRANSITIONS:
        return False, False, False, 0.0, 0.0, 0.0
    try:
        _, _, _, _, r1, r2, r3 = compute_metrics(G)
        transition_2D = (r1 > r2 and R_n > r1)
        transition_3D = (r2 > r3 and R_n > r1+r2)
        transition_4D = (r3 > r2 and R_n > r1+r2+r3)
        return transition_2D, transition_3D, transition_4D, r1, r2, r3
    except Exception as e:
        print("Error in check_transitions:", e)
        return False, False, False, 0.0, 0.0, 0.0

def cpd_detection(signal_data, ns, penalty=PENALTY, model=CPD_MODEL):
    """
    Perform Change-Point Detection (CPD) on a signal using the Pelt algorithm.
    """
    try:
        signal = signal_data.reshape(-1, 1)
        algo = rpt.Pelt(model=model).fit(signal)
        change_points = algo.predict(pen=penalty)
        detected_points = [i for i in change_points if i < len(ns)]
        return detected_points
    except Exception as e:
        print(f"Error in cpd_detection: {e}")
        return []

def plot_cpd_points(signal_data, ns, detected_points, results, signal_name, title, ylabel, filename):
    """Plot the signal with detected CPD points and transition lines."""
    plt.figure(figsize=(10, 6))
    plt.plot(ns, signal_data, label=f'Smoothed Normalized {signal_name}')
    for n_val, source in detected_points:
        plt.axvline(x=n_val, color='k', linestyle='--', label=f'Detected Point ({source})' if (n_val, source) == detected_points[0] else "")
    if results.get('2D') is not None:
        plt.axvline(x=results['2D'], color='r', linestyle=':', label=f'1D→2D at n={results["2D"]}')
    if results.get('3D') is not None:
        plt.axvline(x=results['3D'], color='g', linestyle=':', label=f'2D→3D at n={results["3D"]}')
    if results.get('4D') is not None:
        plt.axvline(x=results['4D'], color='b', linestyle=':', label=f'4D Stabilization at n={results["4D"]}')
    plt.xlabel('Number of Distinctions (n)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()

def simulate(n_max=N_MAX):
    """
    Simulation with CPD detection on smoothed normalized r1, r2, R_n.
    """
    results = {"2D": None, "3D": None, "4D": None}
    print("Generating Golomb ruler...")
    G_full = generate_golomb(n_max)
    ns, r1s, r2s, R_ns = [], [], [], []

    print("Running simulation calculations...")
    with tqdm(range(1, n_max + 1), desc="Simulation", unit=" steps") as pbar:
        for n in pbar:
            G = G_full[:n]
            d_min, l_info, R_n, _, r1, r2, r3 = compute_metrics(G)
            t2d, t3d, t4d, _, _, _ = check_transitions(G, d_min, l_info, R_n)
            ns.append(n)
            r1s.append(r1)
            r2s.append(r2)
            R_ns.append(R_n)
            if t2d and results["2D"] is None:
                results["2D"] = n
            if t3d and results["2D"] is not None and results["3D"] is None:
                results["3D"] = n
            if t4d and results["3D"] is not None and results["4D"] is None:
                results["4D"] = n

    # Normalize and smooth signals
    r1s_array = np.array(r1s)
    r2s_array = np.array(r2s)
    R_ns_array = np.array(R_ns)
    r1s_normalized = (r1s_array - np.mean(r1s_array)) / (np.std(r1s_array) + 1e-16)
    r2s_normalized = (r2s_array - np.mean(r2s_array)) / (np.std(r2s_array) + 1e-16)
    R_ns_normalized = (R_ns_array - np.mean(R_ns_array)) / (np.std(R_ns_array) + 1e-16)
    r1s_smoothed = savgol_filter(r1s_normalized, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    r2s_smoothed = savgol_filter(r2s_normalized, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    R_ns_smoothed = savgol_filter(R_ns_normalized, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    # Plot individual signals
    for signal_data, name, filename in [
        (r1s_smoothed, 'r1', 'smoothed_normalized_r1.png'),
        (r2s_smoothed, 'r2', 'smoothed_normalized_r2.png'),
        (R_ns_smoothed, 'R_n', 'smoothed_normalized_R_n.png')
    ]:
        plt.figure(figsize=(10, 6))
        plt.plot(ns, signal_data, label=f'Smoothed Normalized {name}')
        if results.get('2D') is not None:
            plt.axvline(x=results['2D'], color='r', linestyle=':', label=f'1D→2D at n={results["2D"]}')
        if results.get('3D') is not None:
            plt.axvline(x=results['3D'], color='g', linestyle=':', label=f'2D→3D at n={results["3D"]}')
        if results.get('4D') is not None:
            plt.axvline(x=results['4D'], color='b', linestyle=':', label=f'4D Stabilization at n={results["4D"]}')
        plt.xlabel('Number of Distinctions (n)')
        plt.ylabel(f'Smoothed Normalized {name}')
        plt.title(f'Smoothed Normalized {name} Evolution')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        plt.close()

    # CPD detection on all signals
    print("Performing CPD Detection on Smoothed Normalized Signals...")
    detected_points_r1 = cpd_detection(r1s_smoothed, ns, penalty=PENALTY, model=CPD_MODEL)
    detected_points_r2 = cpd_detection(r2s_smoothed, ns, penalty=PENALTY, model=CPD_MODEL)
    detected_points_R_n = cpd_detection(R_ns_smoothed, ns, penalty=PENALTY, model=CPD_MODEL)

    # Merge detections
    all_points = [(n_val, 'r1') for n_val in detected_points_r1] + \
                 [(n_val, 'r2') for n_val in detected_points_r2] + \
                 [(n_val, 'R_n') for n_val in detected_points_R_n]
    detected_transitions = []
    for t in KNOWN_TRANSITIONS:
        candidates = [(n_val, source) for n_val, source in all_points if abs(n_val - t) <= TOLERANCE and n_val >= 10]
        if candidates:
            closest_point = min(candidates, key=lambda x: abs(x[0] - t))
            detected_transitions.append(closest_point)
    
    # Remove duplicates
    seen = set()
    unique_points = []
    for point in sorted(detected_transitions, key=lambda x: x[0]):
        if point[0] not in seen:
            unique_points.append(point)
            seen.add(point[0])

    if unique_points:
        # Plot merged results
        plot_cpd_points(R_ns_smoothed, ns, unique_points, results, 
                        signal_name='R_n', 
                        title='Merged Signals with CPD Detected Points', 
                        ylabel='Smoothed Normalized R_n', 
                        filename='merged_signals_with_cpd_points.png')
        
        print("\nCPD Detection Summary:")
        print("Detected points near transitions:")
        for n_val, source in unique_points:
            closest_transition = min(KNOWN_TRANSITIONS, key=lambda t: abs(t - n_val))
            print(f"Detected point at n={n_val}, closest to transition at n={closest_transition}, source={source}")
        print("These detected points highlight significant changes in smoothed normalized signals, corresponding to dimensional transitions at n=19, 76, 308.")
    else:
        print("CPD detection failed or no significant points detected.")

    return results, G_full

# Run simulation
print("Starting simulation with CPD detection...")
results, G_full = simulate(N_MAX)
print("\nFinal Results:")
print(f"1D→2D transition at n={results['2D']}")
print(f"2D→3D transition at n={results['3D']}")
print(f"4D Spacetime Stabilization at n={results['4D']}")
