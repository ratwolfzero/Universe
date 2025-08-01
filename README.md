# The Golomb Universe: A Combinatorial Axiomatization of Physical Reality

A formally grounded model of emergent spacetime and matter, where distinctions form the basis of physical structure. We present a minimal axiomatic synthesis combining Spencer-Brown’s logic of form, modal logic, and category theory, each supplying a distinct foundational function—form, possibility, and compositionality, respectively. While none alone suffices to account for physical emergence from first principles, their integration produces a logically generative, combinatorially rich framework. From a purely logical act—distinction-making—emerge time, space, energy, and matter as structured outcomes of an irreversible, informational process.
Author: Ralf Becker
Affiliation: Nuremberg, Germany
Contact: <ratwolf@duck.com>
Version: July 2025
License: Creative Commons Attribution 4.0 International

1. Abstract
We propose a formally minimal, combinatorial model of quantum gravitational structure in which space, time, energy, and matter arise from a single, irreducible principle: the generation of distinctions without repetition. The model operationalizes this principle via a growth rule derived from a foundational axiomatic system. This process produces sequences analogous to greedy Golomb rulers—integer sets with all pairwise differences unique—imposing strict structural novelty at each generative step. Standard physical features such as causality, entropy, and mass-energy distribution emerge as invariants or constraints within this distinction-generating process. The resulting framework unifies logical form with geometric evolution, without assuming prior spacetime structure.

2. Axiomatic Foundations
We characterize the universe not as an ontological substance but as a formal consequence of logical operations—specifically, the iterative act of distinction. Let $\mathcal{C}$ be a category whose objects represent distinct informational states, and morphisms represent irreducible transitions (distinctions) between them.
2.1 Axiom Interpretation
Axiom 0: The Void
The undifferentiated origin. The initial object $0$ in $\mathcal{C}$. Contains no distinctions. Logically unstable:
$$\diamondsuit D$$
Axiom I: First Distinction
Symmetry breaking initiates existence via a morphism:
$$f : 0 \rightarrow A$$
This denotes the primal distinction, represented by the operator $\Box$.
Axiom II: Irreducibility
Growth proceeds only via irreducibly novel distinctions. Nested distinctions are idempotent:
$$\Box(\Box(X)) = \Box(X)$$
Axiom III: Temporal Order
Morphisms induce an irreversible partial order:
$$f: A \rightarrow B \Rightarrow A \prec_t B$$
Axiom IV: Spatial Structure
Spatiality emerges from relative independence among distinctions. Greater informational difference implies greater spatial separation.
Axiom V: Energetic Cost
Distinctions require energy, defined as:
$$E: \text{Obj}(\mathcal{C}) \rightarrow \mathbb{R}_{\geq 0}, \quad E(0) = 0, \quad E(X) \leq E(\Box(X))$$
Axiom VI: Causal Closure
Reality is self-contained, with all morphisms arising from modal necessity:
$$\diamondsuit (\exists f: 0 \rightarrow A)$$

🔧 Axiom VII: Informational Structure
Mutual information between distinctions induces an emergent geometry. Let the system of distinctions be a finite Golomb ruler $G_n \subset \mathbb{N}$, where each element corresponds to an irreducible distinction per Axiom II.
For each distinction $x_i \in G_n$, define a random variable $X_i$. The mutual information matrix $I_n$ is given by:
$$I_n(i,j) = \log \left(1 + \frac{1}{d_{ij}}\right)$$
where the normalized distinction distance is:
$$d_{ij} = \frac{|x_i - x_j|}{\langle |x_k - x_l| \rangle}$$
and the average pairwise difference is:
$$\langle |x_k - x_l| \rangle = \frac{2}{n(n - 1)} \sum_{k < l} |x_k - x_l|$$
This normalization ensures scale-invariant mutual information. Larger $d_{ij}$ yields lower $I_n(i,j)$, approximating a decaying interaction strength.

2.2 Notational Conventions

Symbol
Definition

$0, \emptyset$
The initial (null) object in category $\mathcal{C}$

$\Box$
Spencer-Brown mark: denotes a logical distinction

$E(X)$
Energy associated with object $X$

$\prec_t$
Temporal or causal precedence

$\diamondsuit$
Modal possibility operator

$f: A \to B$
Morphism denoting a distinction from $A$ to $B$

$D(G_n)$
Set of all pairwise differences in configuration $G_n$

3. Distinction-Driven Growth Rule
Growth Theorem
Let $G_0 = {0}$. Given Axioms I–III and the irreducibility constraint from Axiom II, define an inductive sequence ${G_n}$ such that:
$$G_{n+1} = G_n \cup {m}$$
where:
$$m = \min {k > \max(G_n) \mid \forall g \in G_n, |k - g| \notin D(G_n)}$$
Each step adds a minimal integer $m$ ensuring all new pairwise differences are distinct from existing ones, constructing a greedy Golomb ruler.
3.1 Properties

Irreversibility: Deletion removes irreducible differences (violates Axiom II).
Determinism: The choice of $m$ is unique and ordered.
Openness: The growth process is infinite in potential extent.

3.2 Example Construction: $G_0$ to $G_3$

$G_0 = {0} \Rightarrow D = \emptyset$
$G_1 = {0, 1} \Rightarrow D = {1}$
Attempt $2$: $2 - 1 = 1 \in D$ → invalid. Try $3$: $3 - 1 = 2$, $3 - 0 = 3$ → all new → accept. $G_2 = {0, 1, 3}, D = {1, 2, 3}$
Try $4, 5, 6$: all yield repeats. $7$: $7 - 3 = 4$, $7 - 1 = 6$, $7 - 0 = 7$ → all new. $G_3 = {0, 1, 3, 7}$

4. Emergent Physical Quantities
4.1 Quantities

Quantity
Interpretation

Time
Defined by the ordering of morphisms: $A \prec_t B LillingtonB$

Space
Emergent from relative distinctions; modeled as relational independence

Entropy
Number of distinctions: $S_n = \binom{n}{2}$

Energy
Cost of uniqueness: $E(X) \propto \text{number of new differences in } X$

Matter
Stable substructures within $\mathcal{C}$ that preserve distinction locally

Causality
Irreversible morphism chains without cycles

4.2 Matter Subclasses

Component
Interpretation

Dark Matter
Stable but causally or observationally isolated substructures

Observable Matter
Substructures with sufficient internal symmetry to interact or couple

Dark Energy
Residual capacity for distinction—unused potential, not a substance per se

5. Energetic Optimality of Golomb Growth
Let $S = {x_0, x_1, \ldots, x_n}$, with $x_i \prec x_j$ for $i < j$. Define the pairwise differences:
$$D_{ij} = |x_j - x_i|$$
The energy is proportional to distinctiveness:
$$E(x_i) \propto \text{distinctiveness}(x_i)$$
The total energy is:
$$\forall i < j, \forall k < l, \quad D_{ij} \neq D_{kl} \Rightarrow E(S) = \sum_{i<j} D_{ij}$$
where $f$ is strictly decreasing. Golomb rulers minimize total energy under maximal pairwise distinctiveness.

6. Falsifiability and Predictive Content
6.1 Testable Predictions

No duplicate pairwise differences should exist at any scale.
Growth is deterministic and irreversible.
Removing any distinction violates Axiom II (structural inconsistency).

6.2 Falsification Criterion

Observation of repeated relational distances within fundamental physical structure would falsify the model.

7. Comparative Summary

Framework
Free Parameters
Background Structure
Growth Mechanism

String Theory
$>10^{500}$
10D manifold
Perturbative (string loops)

Loop Quantum Gravity
Few
Spin network topology
Topological transition

Golomb Universe
0
None
Irreducible distinctions

7.1 Analogy

String Theory → Complete spacecraft, lacks launch path
LQG → High-performance engine, lacks chassis
Golomb Universe → New propulsion principle under theoretical test

8. Conclusion
This work introduces a novel axiomatic approach to quantum gravity and fundamental physics, wherein physical reality emerges as a combinatorial unfolding of distinctions. The growth rule derived from logical axioms produces structures akin to greedy Golomb rulers, encoding time, space, and energy as emergent properties. The model is falsifiable, structurally minimal, and unifies logic with physical interpretation in a background-free, deterministic framework.
The following appendices provide supporting context and detailed derivations. Appendices A-C formalize the growth rule and its properties, while Appendix D onwards further develops the framework, detailing the emergence of 4D spacetime and quantum structure, which is expanded upon in Part II.

Appendices
Appendix A: Formal Proof of the Growth Rule
Let $G_n = {g_0, g_1, \ldots, g_{n-1}}$ be a set of $n$ distinct natural numbers, ordered such that:
$$g_0 < g_1 < \cdots < g_{n-1}$$
Define the set of pairwise differences:
$$D(G_n) = {|g_i - g_j| \mid g_i, g_j \in G_n, i \neq j}$$
Assume $D(G_n)$ contains unique elements, making $G_n$ a Golomb ruler of size $n$.
Growth Rule
The next set is:
$$G_{n+1} = G_n \cup {m}$$
where $m$ is the smallest natural number satisfying:

$m > \max(G_n)$

For all $g \in G_n$:
$$|m - g| \notin D(G_n)$$

We prove that such an $m$ exists, is unique, and that $G_{n+1}$ remains a Golomb ruler.
A.1 Lemma — Existence of $m$
Statement: There exists a natural number $m > \max(G_n)$ such that:
$$|m - g| \notin D(G_n) \quad \text{for all } g \in G_n$$
Proof: Let $M = \max(G_n)$. Since $D(G_n)$ is finite and the set of candidates:
$${k \in \mathbb{N} \mid k > M}$$
is infinite, there are infinitely many $k$ such that the differences:
$${|k - g| \mid g \in G_n}$$
do not overlap with $D(G_n)$. Thus, the set:
$$S = {k \in \mathbb{N} \mid k > M \text{ and } \forall g \in G_n, (k - g) \notin D(G_n)}$$
is non-empty.
A.2 Lemma — Uniqueness of $m$
Statement: The value $m = \min(S)$ is unique.
Proof: Since $S$ is a non-empty subset of $\mathbb{N}$, the well-ordering principle ensures that $m = \min(S)$ exists and is unique.
A.3 Lemma — Golomb Property of $G_{n+1}$
Statement: The set $G_{n+1} = G_n \cup {m}$ is a Golomb ruler.
Proof: We must show all pairwise differences in $G_{n+1}$ are distinct.

New vs. Old Differences: By construction:
$$|m - g| \notin D(G_n) \quad \text{for all } g \in G_n$$
Thus, new differences do not duplicate existing ones.

New Differences Are Unique: Suppose:
$$|m - g_i| = |m - g_j| \quad \text{for } g_i, g_j \in G_n, g_i \neq g_j$$
Since $m > \max(G_n) \geq g_i, g_j$:
$$m - g_i = m - g_j \Rightarrow g_i = g_j$$
a contradiction. Thus, new differences are distinct.

Therefore, $G_{n+1}$ is a Golomb ruler.
A.4 Conclusion
The inductive growth rule is valid, as $m$ exists (A.1), is unique (A.2), and $G_{n+1}$ is a Golomb ruler (A.3).

Appendix B: Proof of Irreversibility
Statement: Let $G_n$ be a Golomb ruler of size $n$, and:
$$G_{n+1} = G_n \cup {m}$$
Removing $m$ from $G_{n+1}$ destroys information, making the process irreversible.
Proof: Adding $m$ introduces:
$$n = |G_n|$$
new distinct differences:
$${|m - g| \mid g \in G_n}$$
These are not in $D(G_n)$, are mutually distinct, and necessary for $G_{n+1}$’s Golomb property.
Connection to Axiom II: Axiom II implies idempotency:
$$\Box(\Box(X)) = \Box(X)$$
Removing $m$ destroys $m$, its $n$ unique differences, and the Golomb property, violating Axiom II’s irreducibility principle.
Conclusion: The growth process is strictly additive, history-dependent, and irreversible.

Appendix C: Entropy and Structural Complexity
The entropy of $G_n$ with $n$ distinctions is the number of unique pairwise differences:
$$S_n = \binom{n}{2} = \frac{n(n-1)}{2}$$
This grows monotonically with $n$, reflecting increasing structural complexity.

Appendix D: Mathematically Rigorous Concept for 4D Spacetime Emergence
The transition from a 1D combinatorial sequence to 4D spacetime emerges from informational complexity, as per Axiom VII, through structural bifurcations driven by the Graph Laplacian’s eigenvalue spectrum and informational curvature.
D.1 Stage 1: The 1D Temporal Continuum
The universe’s state at step $n$ is a discrete sequence of distinctions:
$$G_n = {t_0, t_1, \ldots, t_n}$$
Here, $t_i$ represents the temporal coordinate, per Axiom III, forming a 1D structure.
D.2 Stage 2: The Emergence of 2D Spatiality
The first spatial dimension emerges at a critical informational complexity threshold.

Informational Graph: Construct $G_{\text{info}}$ with vertices $V = {X_0, X_1, \ldots, X_n}$ and edge weights:
$$w_{ij} = I(X_i; X_j) = \log \left(1 + \frac{1}{d_{ij}}\right)$$
where:
$$d_{ij} = \frac{|x_i - x_j|}{\langle |x_k - x_l| \rangle}$$

Graph Laplacian: The Laplacian matrix is:
$$L = D - I_n$$
where $D$ is the degree matrix and $I_n$ is the mutual information matrix. Its eigenvalues are:
$$0 = \lambda_0 \leq \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$$

Bifurcation to 2D: The transition occurs when:
$$\frac{\lambda_2}{\lambda_1} > 1.3$$
and the informational curvature:
$$R_n = \frac{1}{\ell_{\text{info}}^2} \left(1 - \frac{d_{\min}^{(n)}}{\ell_{\text{info}}}\right) > 1.3$$
where:
$$\ell_{\text{info}} = \frac{1}{1 + \log n}, \quad d_{\min}^{(n)} = \min_{i \neq j} \frac{1}{1 + I_n(i,j)}$$
This indicates the graph cannot be embedded in 1D without distortion.

2D Embedding: Each distinction $X_i$ maps to coordinates $(s_{i,1}, s_{i,2})$ using eigenvectors of $\lambda_1$ and $\lambda_2$.

D.3 Stage 3: The Bifurcation to 3D Spatiality
The third spatial dimension emerges when:
$$\frac{\lambda_3}{\lambda_2} > 1.15 \quad \text{and} \quad R_n > 2.2$$
indicating the graph requires a 3D embedding. Each $X_i$ maps to $(s_{i,1}, s_{i,2}, s_{i,3})$ using eigenvectors of $\lambda_1, \lambda_2, \lambda_3$.
D.4 Stage 4: The 4D Spacetime Manifold
The 3D spatial manifold combines with the 1D temporal continuum, mapping each distinction $X_i$ to:
$$P_i = (t_i, s_{i,1}, s_{i,2}, s_{i,3})$$
where $t_i$ is the discrete Golomb ruler integer and $(s_{i,1}, s_{i,2}, s_{i,3})$ are continuous spectral coordinates.

Appendix E: Emergent Informational Scale and Dimensional Dynamics
This appendix extends Appendix D, deriving spacetime dynamics and quantum structure from Axiom VII without invoking the Planck length.
E.1 Defining the Informational Scale $\ell_{\text{info}}$
Axiom VII defines the pseudometric:
$$d(i,j) = \frac{1}{1 + I(X_i; X_j)}$$
where:
$$I(X_i; X_j) = \log \left(1 + \frac{1}{d_{ij}}\right), \quad d_{ij} = \frac{|x_i - x_j|}{\langle |x_k - x_l| \rangle}$$
The minimal distance is:
$$d_{\min}^{(n)} = \min_{i \neq j} d(i,j)$$
The informational scale is:
$$\ell_{\text{info}} = \frac{1}{1 + \log n}$$
Derivation:
The entropy is:
$$S_n = \binom{n}{2}$$
Mutual information is bounded:
$$I(X_i; X_j) \leq \log \binom{n}{2} \approx 2 \log n - \log 2$$
Thus:
$$\ell_{\text{info}} \approx \frac{1}{1 + 2 \log n}$$
Physical Interpretation: $\ell_{\text{info}}$ is the minimal resolvable separation, analogous to a fundamental length scale.
E.2 Informational Curvature and Dimensional Bifurcation
The informational scalar curvature is:
$$R_n = \frac{1}{\ell_{\text{info}}^2} \left(1 - \frac{d_{\min}^{(n)}}{\ell_{\text{info}}}\right)$$
Theorem E.1: When $R_n \geq \ell_{\text{info}}^{-2}$, the distinction graph requires a higher-dimensional embedding.
Proof:

1D Limitation: In 1D, $\delta_{\min}^{(n)} \sim \frac{1}{n}$, so:
$$d_{\min}^{(n)} \sim \frac{1}{1 + \log n} \to \ell_{\text{info}}, \quad R_n \to \ell_{\text{info}}^{-2}$$

2D Impossibility: Per the Erdős–Anning theorem, infinite unique distances are impossible in $\mathbb{R}^2$.

3D Sufficiency: The Johnson–Lindenstrauss lemma ensures $n$ points can be embedded in $\mathbb{R}^3$ with minimal distortion.

Minimality: Higher dimensions offer no energetic advantage.

Physical Interpretation: $R_n$ drives the transition to 3D due to informational crowding.
E.3 Informational Energy Functional
The energy functional is:
$$E_n = \sum_{i < j} \left( \frac{1}{d(i,j)^2} - \frac{1}{\ell_{\text{info}}^2} \right)$$
When $d(i,j) \to \ell_{\text{info}}$, $E_n$ diverges, signaling a higher-dimensional embedding. In 3D, $d(i,j) \sim n^{-1/3}$, stabilizing $E_n$.

Appendix F: Spacetime Dynamics via Informational Action
F.1 Discrete Action Functional
The action is:
$$S_G = \sum_{\langle i,j \rangle} I(X_i; X_j) \cdot d(i,j)^2 + \sum_i E(X_i)$$
where:
$$d(i,j) = \frac{1}{1 + I(X_i; X_j)}$$
$$E(X_i) = \sum_{j \in N_i} \log \left(1 + \frac{|t_i - t_j|}{d(i,j)}\right)$$
The sum is over edges in $G_{\text{info}}$ with $d(i,j) \leq r$, where $r \propto \ell_{\text{info}}$.
F.2 Equations of Motion
Varying $S_G$ with respect to $P_i = (t_i, s_{i,1}, s_{i,2}, s_{i,3})$:
$$\sum_{j \in N_i} I(X_i; X_j) (P_i - P_j) = -\nabla_{P_i} E(X_i)$$
F.3 Continuum Limit
As $n \to \infty$, the graph approximates a 4D manifold with metric:
$$g_{\mu\nu}(x) = \lim_{R \to \ell_{\text{info}}} \frac{1}{N_R} \sum_{\langle i,j \rangle \in B_R(x)} \Delta P_i^\mu \Delta P_j^\nu I(X_i; X_j)$$
The action converges to:
$$S_G \to \int \sqrt{-g} (R + 2\Lambda_{\text{info}}) , d^4x$$
where:
$$\Lambda_{\text{info}} \propto \ell_{\text{info}}^{-2}$$
Physical Interpretation: The dynamics resemble geodesic motion in general relativity.

Appendix G: Quantum Structure from Informational Contingency
G.1 Probabilistic Amplitudes
The state space is:
$$\Psi(X_i) = \mathbb{C}^{|N_i|}$$
The transition amplitude is:
$$\langle X_j | f | X_i \rangle = \exp\left(-i \sum_{k \in N_i \cap N_j} I(X_k; X_i, X_j)\right)$$
Physical Interpretation: The phase encodes informational context, mimicking quantum amplitudes.
G.2 Emergent Uncertainty Principle
Positional uncertainty:
$$\Delta x_i \sim d_{\min}^{(n)} = \min_{j \in N_i} \frac{1}{1 + I(X_i; X_j)}$$
Momentum-like uncertainty:
$$\Delta p_i \sim \frac{\partial E_n}{\partial d_{\min}^{(n)}} \approx \frac{2}{(d_{\min}^{(n)})^3}$$
Uncertainty product:
$$\Delta x_i \Delta p_i \sim \frac{2}{(d_{\min}^{(n)})^2}$$
When $d_{\min}^{(n)} \to \ell_{\text{info}}$:
$$\Delta x_i \Delta p_i \sim \ell_{\text{info}}^{-2}$$
Physical Interpretation: This resembles the Heisenberg uncertainty principle.

Appendix H: Experimental Signatures
H.1 Informational Spectral Gaps
Energy gaps are given by:
$$\Delta E \sim \frac{1}{\ell_{\text{info}}} \cdot \left| \sum_{j \in N_i} I(X_i; X_j) - \sum_{j \in N_k} I(X_k; X_j) \right|$$
Test: Measure spectral gaps in quantum simulators.
H.2 CMB Fluctuations
Curvature fluctuations:
$$\lambda_k \sim \frac{1}{\ell_{\text{info}}^2} \cdot \left(1 - \frac{\langle I(X_i; X_j) \rangle}{\log \binom{n}{2}}\right)$$
Test: Search for discrete “informational modes” in CMB B-mode polarization.

Appendix I: Open Research Topics

Topic
Priority
Description
Axiom Link
Goal

Gauge Fields
High
Derive gauge symmetries from morphism structures in $\mathcal{C}$
I, IV
Model fundamental interactions (e.g., electromagnetism)

Quantum Measurement
High
Formalize observer-dependent collapse in the distinction network
I, VI
Develop a contextual measurement theory

Physical Constants
Medium
Connect $\ell_{\text{info}}$ to physical constants and derive $\hbar$, $G$, $c$ from combinatorial constraints
VII
Anchor $\ell_{\text{info}}$ to observable physics (e.g., Planck length)

Thermodynamics
Medium
Relate $S_n$ to statistical entropy and the second law, including deriving $I_n$ from Shannon’s definition:  $I(X;Y) = H(X) + H(Y) - H(X,Y)$
III
Establish thermodynamic consistency and refine informational structure

Simulations
Low
Simulate $G_{\text{info}}$ growth and validate transitions, noting heuristic gaps (e.g., $I_n, \ell_{\text{info}}$)
VII
Validate predictions computationally and enhance transparency

Appendix J: Conclusion (Part II, Appendix E -I)
This axiomatic framework derives spacetime, curvature, and quantum structure from informational principles, with $\ell_{\text{info}}$ as the fundamental scale. The approach maintains the Golomb Universe’s minimalism while offering testable predictions, paving the way for a unified theory of quantum gravity. From distinctions alone, the universe weaves its geometry and dynamics—one irreducible difference at a time.
The close functional alignment between the axiomatic mutual information:
$$I_n(i,j) = \log\left(1 + \frac{1}{d_{ij}}\right)$$
and standard probabilistic decay models (Figure J.1) reinforces a central insight of this framework: deterministic informational structure can naturally reproduce behaviors typically attributed to stochastic systems. Specifically, mutual information—often treated as inherently probabilistic—emerges here as a purely structural function of distinction distance in a minimal, rule-based universe. This suggests that laws grounded in probability, such as those found in quantum theory or statistical mechanics, may be emergent signatures of deeper combinatorial architectures. For instance, the deterministic chaos of the Hopalong attractor, a system defined by iterative non-linear equations (e.g., Martin, 1986), generates fractal patterns resembling stochastic noise, paralleling how the Golomb Universe’s mutual information matrix produces halo structures and logarithmic decay. While formal Shannon entropy requires probabilistic state assumptions incompatible with the Golomb Universe’s foundational minimalism, the observed decay in $I_n$ over normalized distinction distance mirrors the expected decay of mutual information in statistical ensembles. This alignment supports the view that probability is not a primitive feature of reality, but a macroscopic manifestation of informational inaccessibility or coarse-graining within an underlying deterministic substrate built entirely from irreducible distinctions.
*Figure J.1 — Functional comparison between the axiomatic mutual information:
$$I_n(i,j) = \log\left(1 + \frac{1}{d_{ij}}\right)$$
derived from deterministic distinction structure, and best-fit inverse power law and exponential decay models typical of probabilistic systems. Despite involving no probabilistic assumptions, the axiomatic curve closely tracks both fits across all ranges of normalized distinction distances, suggesting that familiar information-theoretic behaviors can emerge from fundamentally non-statistical foundations.*

References
Awodey, S. (2010). Category Theory. Oxford University Press.
Barwise, J., Seligman, J. (1997). Information Flow. Cambridge University Press.
Bloom, G.S., Golomb, S.W. (1977). Applications of Numbered Undirected Graphs. IEEE Proceedings, 65(4), 562–570.
