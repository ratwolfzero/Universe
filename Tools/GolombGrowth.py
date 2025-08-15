import matplotlib.pyplot as plt
import networkx as nx
import random
from itertools import combinations
from math import log


def golomb_grow(n: int) -> list[int]:
    """
    Generates a Golomb ruler of 'n' marks using an iterative growth approach,
    as derived from the axiomatic framework.

    The algorithm iteratively adds marks to the ruler. For each new mark,
    it starts by testing the smallest possible integer greater than the
    last added mark (`G[-1] + 1`). It then **increments this candidate mark (`m`)**
    by one until it finds a value that generates only **unique distinctions**.
    "Unique distinctions" means that the absolute differences between the
    candidate mark and all existing marks in the ruler (`G`) must not
    already exist in the set of previously found distinctions (`D`) and
    must not duplicate among themselves. This iterative search for the
    smallest valid `m` ensures the construction of a minimal Golomb ruler.

    Args:
        n (int): The desired number of marks for the Golomb ruler.

    Returns:
        list[int]: A list representing the Golomb ruler sequence.
    """
    G = [0]  # The Golomb ruler sequence, initialized with 0 (Axiom 0)
    D = set()  # Set to store all unique distinctions found so far (Axiom II)

    while len(G) < n:
        m = G[-1] + 1  # Start searching for the next mark from the previous mark + 1
        new_diffs = [abs(m - x) for x in G]
        is_unique = True
        temp_new_diffs_set = set()
        for diff in new_diffs:
            if diff in D or diff in temp_new_diffs_set:  # Embodies Axiom II: Irreducible Uniqueness
                is_unique = False
                break
            temp_new_diffs_set.add(diff)

        while not is_unique:  # Iterate 'm' until a unique candidate is found
            m += 1
            new_diffs = [abs(m - x) for x in G]
            is_unique = True
            temp_new_diffs_set = set()
            for diff in new_diffs:
                if diff in D or diff in temp_new_diffs_set:
                    is_unique = False
                    break
                temp_new_diffs_set.add(diff)

        G.append(m)  # Morphism f: G_k -> G_{k+1} (Axiom III)
        D.update(temp_new_diffs_set)  # Update set of unique differences

    return G


def entropy_fn(n: int) -> int:
    """
    Calculates the maximum possible number of unique distinctions for 'n' marks,
    representing combinatorial entropy for the system.
    """
    return n * (n - 1) // 2


def spatial_energy(G: list[int]) -> float:
    """
    Calculates 'spatial energy': a figurative metric based on
    the sum of inverse distances between all pairs of marks.

    Higher spatial energy implies more compact and constrained configurations.
    """
    if not G or len(G) < 2:
        return float('inf')  # Or 0.0 depending on interpretation
    return sum(1.0 / abs(a - b) for a, b in combinations(G, 2))


def estimated_max_energy(n: int) -> float:
    """
    Estimate the maximum theoretical spatial energy for n marks,
    assuming distances are the first n(n-1)/2 integers.
    Uses harmonic number approximation: H_m ≈ ln(m) + γ,
    where m = number of unique pairs = n(n-1)/2.
    """
    gamma = 0.5772156649  # Euler–Mascheroni constant
    m = n * (n - 1) // 2  # number of unique pairs
    return log(m) + gamma


def plot_golomb_graph(G: list[int]):
    """
    Visualize the Golomb ruler as a graph with nodes as marks and edges as distances.
    Displays actual energy and estimated maximum energy in the title.
    """
    g = nx.Graph()
    g.add_nodes_from(G)
    edges = [(a, b, {'weight': abs(a-b)}) for a, b in combinations(G, 2)]
    g.add_edges_from(edges)

    pos = nx.circular_layout(g)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(g, pos, node_size=700,
                           node_color='lightblue', alpha=0.9, linewidths=1.0, edgecolors='gray')
    nx.draw_networkx_edges(g, pos, width=1.5, alpha=0.7, edge_color='darkgray')
    nx.draw_networkx_labels(g, pos, font_size=14,
                            font_weight='bold', font_color='black')

    edge_labels = {(u, v): d['weight'] for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                 font_color='darkgreen', font_size=10,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3'))

    actual_energy = spatial_energy(G)
    max_energy = estimated_max_energy(len(G))
    plt.title(
        f"\nGolomb Graph for Sequence {G}\n"
        f"Combinatorial Entropy (max distinctions): {entropy_val}\n"
        f"Spatial Energy: {actual_energy:.4f}, Theoretical Maximum: {max_energy:.4f}",
        fontsize=12
    )

    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()


def uniqueness_ratio(G: list[int]) -> float:
    n = len(G)
    max_diffs = n * (n - 1) // 2
    diffs = set(abs(a - b) for a, b in combinations(G, 2))
    return len(diffs) / max_diffs


def compute_stats_for_n(n: int):
    lattice = list(range(n))
    random_seq = sorted(random.sample(range(10 * n), n))
    golomb_seq = golomb_grow(n)

    data = []
    for label, seq in [('Lattice', lattice), ('Random', random_seq), ('Golomb', golomb_seq)]:
        uniq = uniqueness_ratio(seq)
        energy = spatial_energy(seq)
        data.append((label, n, uniq, energy))
    return data


def plot_uniqueness_vs_energy(ns):
    all_data = []
    for n in ns:
        all_data.extend(compute_stats_for_n(n))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {'Lattice': 'o', 'Random': 's', 'Golomb': '^'}
    colors = {'Lattice': 'red', 'Random': 'orange', 'Golomb': 'green'}

    for label in markers.keys():
        xs = [d[2] for d in all_data if d[0] == label]
        ys = [d[3] for d in all_data if d[0] == label]
        ns_plot = [d[1] for d in all_data if d[0] == label]
        sc = ax.scatter(
            xs, ys, label=label, marker=markers[label], color=colors[label], s=30, alpha=0.7)
        for x, y, n in zip(xs, ys, ns_plot):
            if n in [100]:  # annotate key points only for clarity
                ax.annotate(f'n={n}', (x, y), textcoords="offset points", xytext=(
                    5, 5), ha='left', fontsize=9)

    ax.set_xlabel(
        'Uniqueness Ratio (Unique Distances / Max Distances)', fontsize=12)
    ax.set_ylabel('Spatial Energy (Sum of Inverse Distances)', fontsize=12)
    ax.set_title(
        'Uniqueness vs Spatial Energy for Different Sequence Types', fontsize=14)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# --- Example Usage (Main Execution Block) ---
if __name__ == "__main__":
    n_marks = 17  # Example for an n-mark Golomb ruler
    golomb_sequence = golomb_grow(n_marks)
    print(f"Golomb sequence for {n_marks} marks: {golomb_sequence}")

    entropy_val = entropy_fn(len(golomb_sequence))
    print(f"Combinatorial Entropy (max distinctions): {entropy_val}")

    energy_val = spatial_energy(golomb_sequence)
    print(f"Spatial Energy (sum of inverse distances): {energy_val:.4f}")

    max_energy_val = estimated_max_energy(len(golomb_sequence))
    print(
        f"Estimated max energy for {len(golomb_sequence)} marks: {max_energy_val:.4f}")

    plot_golomb_graph(golomb_sequence)

    ns = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    plot_uniqueness_vs_energy(ns)
