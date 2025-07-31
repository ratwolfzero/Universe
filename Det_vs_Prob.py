import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- 1. Generate Greedy Golomb Ruler ---


def generate_greedy_golomb_ruler(n_elements):
    if n_elements <= 0:
        return []
    ruler = [0]
    all_differences = set()
    for _ in range(n_elements - 1):
        m = max(ruler) + 1
        while True:
            new_differences = set()
            is_valid = True
            for g in ruler:
                diff = abs(m - g)
                if diff in all_differences or diff in new_differences:
                    is_valid = False
                    break
                new_differences.add(diff)
            if is_valid:
                ruler.append(m)
                all_differences.update(new_differences)
                break
            m += 1
    return ruler

# --- 2. Compute Normalized Distinction Distances and Axiomatic MI ---


def calculate_d_ij(golomb_ruler):
    all_raw_diffs = [abs(golomb_ruler[i] - golomb_ruler[j])
                     for i in range(len(golomb_ruler))
                     for j in range(i + 1, len(golomb_ruler))]

    raw_diffs_array = np.array(all_raw_diffs)
    avg_raw_diff = np.mean(raw_diffs_array)

    d_ij_values = []
    your_i_n_values = []

    for i in range(len(golomb_ruler)):
        for j in range(i + 1, len(golomb_ruler)):
            raw_diff = abs(golomb_ruler[i] - golomb_ruler[j])
            d_val = raw_diff / \
                avg_raw_diff if avg_raw_diff != 0 else float('inf')
            d_ij_values.append(d_val)
            your_i_n_values.append(
                np.log(1 + 1/d_val) if d_val > 0 else np.inf)

    return np.array(d_ij_values), np.array(your_i_n_values)

# --- 3. Generic Decay Models ---


def inverse_power_law(d, A, C):
    return A / (d**C)


def exponential_decay(d, A, B):
    return A * np.exp(-B * d)

# --- 4. Plotting Routine ---


def plot_mi_vs_decay(n_elements=20, normalize=False, save_fig=True, fig_name="Figure_J1.pdf"):
    # Generate ruler and MI values
    golomb_ruler = generate_greedy_golomb_ruler(n_elements)
    print(f"Generated Golomb Ruler ({n_elements} elements): {golomb_ruler}")
    d_ij_values, your_i_n_values = calculate_d_ij(golomb_ruler)

    # Filter out infinite values
    finite_idx = np.isfinite(your_i_n_values)
    d_ij = d_ij_values[finite_idx]
    mi_vals = your_i_n_values[finite_idx]

    # Sort for smooth plotting
    sort_idx = np.argsort(d_ij)
    d_ij = d_ij[sort_idx]
    mi_vals = mi_vals[sort_idx]

    if normalize:
        mi_vals /= np.max(mi_vals)

    # Fit inverse power law
    try:
        popt_power, _ = curve_fit(inverse_power_law, d_ij, mi_vals, p0=[
                                  1.0, 1.0], bounds=([0, 0], [np.inf, 10.0]))
        mi_fit_power = inverse_power_law(d_ij, *popt_power)
        r2_power = r2_score(mi_vals, mi_fit_power)
    except RuntimeError:
        popt_power = [np.nan, np.nan]
        r2_power = np.nan

    # Fit exponential
    try:
        popt_exp, _ = curve_fit(exponential_decay, d_ij, mi_vals, p0=[
                                1.0, 0.1], bounds=([0, 0], [np.inf, np.inf]))
        mi_fit_exp = exponential_decay(d_ij, *popt_exp)
        r2_exp = r2_score(mi_vals, mi_fit_exp)
    except RuntimeError:
        popt_exp = [np.nan, np.nan]
        r2_exp = np.nan

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.scatter(d_ij, mi_vals, color='blue', s=50, alpha=0.7,
                label='Axiomatic MI: $I_n(i,j) = \\log(1 + 1/d_{ij})$')

    if not np.isnan(r2_power):
        plt.plot(d_ij, mi_fit_power, 'r--',
                 label=f'Best Fit $A/d^C$: A={popt_power[0]:.2f}, C={popt_power[1]:.2f}, $R^2$={r2_power:.3f}')
    if not np.isnan(r2_exp):
        plt.plot(d_ij, mi_fit_exp, 'g:',
                 label=f'Best Fit $Ae^{{-Bd}}$: A={popt_exp[0]:.2f}, B={popt_exp[1]:.2f}, $R^2$={r2_exp:.3f}')

    plt.title('Functional Comparison: Axiomatic MI vs. Probabilistic Decay Models')
    plt.xlabel(
        r'Normalized Distinction Distance $d_{ij} = |x_i - x_j| / \langle |x_k - x_l| \rangle$')
    plt.ylabel('Mutual Information $I(i,j)$' +
               (' (Normalized)' if normalize else ''))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)

    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
        print(f"Plot saved as: {fig_name}")

    plt.show()

# --- Execute ---


if __name__ == "__main__":
    plot_mi_vs_decay(n_elements=20, normalize=False,
                     save_fig=True, fig_name="Figure_J.1.png")
