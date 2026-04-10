# Reinterpreting ABC-DRF: Joint Posterior Estimation via Distributional Random Forests

This repository contains the code for **our thesis project**. We implemented and reinterpreted the **ABC-DRF** framework (based on the work by Raynal et al.) to estimate the **full joint posterior distribution** of parameters, rather than just computing isolated marginals.

## Our Approach

Traditional Approximate Bayesian Computation (ABC) struggles with the selection of summary statistics. To test the robustness of **Distributional Random Forests (DRF)**, we designed a specific stress test:

* **Noise Injection:** We flooded our reference table with **50 completely random noise variables**. The goal is to prove that DRF performs **automatic feature selection**, ignoring irrelevant data and isolating the true signal.
* **Joint vs. Marginal:** Standard ABC-RF builds separate forests for each parameter. We train a **single forest** to extract a universal set of weights, allowing us to reconstruct a mathematically coherent **2D joint posterior distribution**.

## The Toy Model

To verify that the machine learning algorithm actually learns the truth, we used a **Gaussian - Inverse Gamma** model. Because this model has an **exact analytical posterior**, we can overlay the theoretical contours directly on top of our DRF predictions as an absolute ground-truth check.

* **Likelihood:** `Y ~ N(θ1, θ2)`
* **Variance Prior:** `θ2 ~ Inverse-Gamma(a0, b0)`
* **Mean Prior:** `θ1 | θ2 ~ N(0, θ2)`

## Key Code Features

* **Parallel Computing:** We use `mclapply` (via the `parallel` package) to generate the **10,000-simulation** reference table efficiently across multiple CPU cores.
* **Sheather-Jones Bandwidth:** For marginal density estimation, we strictly use `bw="SJ"` to prevent oversmoothing, which is critical for highly skewed distributions like the Inverse-Gamma.
* **Log-Scale Computations:** To avoid numeric underflow when generating the exact 2D contours, all joint probability math is handled in **log-scale**.

## How to Run

1. Make sure you have R installed and install the required packages:
   ```R
   install.packages(c("abcrf", "MASS", "invgamma", "drf"))
2. Run the main script.

The code will generate the simulated reference table, train a DRF with 5,000 trees, predict the posterior weights based on the observed data, and automatically plot the results.

## Output

The script outputs a figure with three panels:

* Joint Posterior **Heatmap**

* Marginal Densities of θ1 and θ2


You will see the DRF predictions (blue lines / heatmap) plotted directly against the exact mathematical formulas (red lines / contours) to demonstrate the algorithm's accuracy despite the injected noise.
