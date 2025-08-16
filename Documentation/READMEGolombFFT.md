# 📐 Golomb Signal Spectral Analysis

This Python program generates **Golomb ruler signals**, performs **Fourier analysis**, and visualizes:

* **Time-domain representation**
* **FFT magnitude spectrum**
* **Power spectrum**

It explores the effects of **DC removal** and **windowing** on spectral behavior.

---

## 📌 What Is a Golomb Ruler?

A **Golomb ruler** is a set of integers such that all pairwise differences are unique. These rulers are used in:

* Radar & sonar
* Sensor array design
* Coding theory

The program constructs a Golomb ruler using a **greedy growing algorithm**.

---

## 🔧 What the Program Does

1. **Generate a Golomb Ruler**
   → Using a fast greedy algorithm to generate `n` unique-mark positions.

2. **Construct a Binary Signal**
   → Signal length = `G[-1] + 1`.
   → Set 1s at Golomb ruler positions, 0s elsewhere.

3. **Remove DC Component (Optional)**
   → Subtracts signal mean to center the signal around 0.

4. **Normalize Signal (Always Applied)**
   → Scales to ensure maximum absolute amplitude is 1.

5. **Apply Spectral Window (Optional)**
   → Supports `hanning`, `hamming`, `blackman`, or `rectangular`.

6. **Compute FFT & Power Spectrum**
   → Uses NumPy’s FFT to compute spectral content.

7. **Plot Results**
   → Shows time-domain, FFT magnitude, and log power spectrum.

---

## 📈 Output Interpretation

### 1. 🟦 Time-Domain Golomb Signal

A stem plot showing signal values after optional DC removal and normalization.

#### 🔵 Blue Stems

* Correspond to Golomb ruler positions.
* Always normalized to **+1.0**.

#### 🔴 Red Stems (When DC Removed)

* Represent values at **non-Golomb positions** after DC removal.
* Typically **negative**, shown in red.

#### ⚠️ Why Red Stem Amplitudes Are < 1 (No Windowing Case)

Let:

* `n` = number of Golomb marks
* `L` = signal length = `G[-1] + 1`
* `μ = n / L` = mean before DC removal

Then after DC removal:

* Golomb positions = `1 - μ`
* Non-Golomb positions = `-μ`

And after normalization:

| Type             | Value   | Final Amplitude |
| ---------------- | ------- | --------------- |
| Golomb Positions | `1 - μ` | **+1.0**        |
| Non-Golomb Pos.  | `-μ`    | `-μ / (1 - μ)`  |

Example:
If `μ = 0.38`, then red stems ≈ `-0.38 / 0.62 ≈ -0.61`.

🛑 **Important:** This applies **only without windowing** (rectangular window).
Applying spectral windows modifies amplitudes and breaks this clean ratio.

---

### 2. 🔵 FFT Magnitude Spectrum

* Displays energy across normalized frequencies \[0, 0.5].
* **DC component (0 Hz)** corresponds to average signal value.
* Sparse frequency content leads to scattered spectral peaks.

---

### 3. 🔷 Power Spectrum (Log Scale)

* Power = square of FFT magnitude.
* Plotted on a **logarithmic scale** for visibility.
* Highlights dominant spectral bands.

---

## 🧠 Key Observations

### ✅ DC Component Reflects Total Mark Count (Rectangular Window Only)

When DC is **not removed**, the first FFT bin (index 0) contains the sum of the signal:

$$
\text{DC Magnitude} = n \quad (\text{number of marks})
$$

$$
\text{DC Power} = n^2
$$

#### Example:

```bash
Magnitude of DC component (mag[0]): 10.0
Power of DC component (power[0]): 100.0
```

🛑 This relationship holds **only when no spectral window** is applied.

---

### 🔴 Red Stems Appear When DC Is Removed

When subtracting the mean:

* Golomb entries drop from 1 to `1 - μ`
* All 0 entries become `-μ`
* After normalization, the signal ranges from `-μ / (1 - μ)` to `+1.0`
* Red stems (negative) are plotted for visual emphasis

---

## 🪟 What Is Windowing?

Spectral windows reduce **leakage** in FFTs by tapering the signal:

| Window      | Description                   |
| ----------- | ----------------------------- |
| Hanning     | Smooth taper to zero          |
| Hamming     | Less aggressive than Hanning  |
| Blackman    | Best roll-off, most smoothing |
| Rectangular | No tapering (raw signal)      |

Applying windows helps reveal **true frequency components**.

---

## 🧪 Example Usage

Run from terminal:

```bash
python golomb_fft.py
```

It runs two configurations:

1. **DC Removed**, with **Hanning Window**
2. **DC Kept**, with **Hanning Window**

Each shows:

* Time-domain signal (with stem colors)
* FFT magnitude spectrum
* Power spectrum (log scale)

---

## ⚙️ Function Overview

### `golomb_grow(n)`

Returns the first `n` Golomb marks using greedy difference-checking.

### `create_signal_from_golomb(G, remove_dc=True)`

Creates a binary signal from `G`, subtracts mean if requested, and normalizes.

### `compute_spectrum(signal, apply_window, window_type)`

Applies optional window, computes FFT, and returns:

* Frequencies
* Magnitude
* Power

### `plot_results(...)`

Generates three aligned plots:

* Time-domain signal
* FFT magnitude
* Log power spectrum

### `main(n, DC=False, windowing=False, window_type='hanning')`

Executes the full analysis:

* `n`: number of Golomb marks
* `DC`: keep DC or remove
* `windowing`: apply spectral window
* `window_type`: which window to use

---

## 📚 Plot Layout

```
+--------------------------+
|     Golomb Signal        |
+-------------+------------+
|  FFT Mag    |   Power    |
|  Spectrum   |  Spectrum  |
+-------------+------------+
```

---

## 📤 Sample Console Output

```bash
--- Example 3: DC Removed, Hanning Window Applied ---
Value of signal mean BEFORE FFT: 0.0
--- Applied hanning window ---
Magnitude of DC component (mag[0]): 2.17e-16
Power of DC component (power[0]): 4.72e-32

--- Example 4: DC Kept, Hanning Window Applied ---
Value of signal mean BEFORE FFT: 0.175
--- Applied hanning window ---
Magnitude of DC component (mag[0]): 10.0
Power of DC component (power[0]): 100.0
```

---

## 📎 Dependencies

Install via pip:

```bash
pip install numpy matplotlib numba
```

---

## 📌 Conclusion

This tool demonstrates:

* Efficient Golomb ruler construction
* Sparse signal visualization
* Spectrum analysis with and without DC component
* Effect of windowing on frequency content

Ideal for DSP education, sparse sampling studies, or Golomb ruler research.
