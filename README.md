# Portfolio Diversification Analysis (Evans & Archer Replication)

This Python script replicates the core empirical analysis presented in the paper:

**Evans, John L., and Stephen H. Archer. "Diversification and the Reduction of Dispersion: An Empirical Analysis." *The Journal of Finance*, vol. 23, no. 5, 1968, pp. 761–67.**

The script demonstrates how portfolio risk (measured by the standard deviation of returns) decreases as the number of randomly selected securities in the portfolio increases. It shows that this risk reduction follows a predictable pattern, diminishing rapidly at first and then leveling off, approaching the level of non-diversifiable systematic risk.

## Overview

The script performs the following steps:

1.  **Loads historical stock return data** from a CSV file.
2.  **Preprocesses the data:** Filters by date range, selects common stocks (optional), handles missing values, and pivots the data for time-series analysis.
3.  **Simulates portfolio creation:** For portfolio sizes ranging from 1 to a specified maximum (e.g., 40):
    *   Randomly selects securities without replacement.
    *   Calculates the equally-weighted portfolio return for each time period.
    *   Calculates the standard deviation of the *logarithms of the portfolio value relatives* (`log(1 + Return)`) over the analysis period. This matches the methodology of the Evans & Archer paper.
    *   Repeats this simulation multiple times for statistical robustness.
4.  **Aggregates results:** Calculates the average standard deviation for each portfolio size across all simulations.
5.  **Fits a hyperbolic curve:** Fits the function `Y = A + B/X` to the results, where:
    *   `Y` is the average portfolio standard deviation.
    *   `X` is the number of securities in the portfolio.
    *   `A` represents the asymptote, an estimate of the non-diversifiable systematic risk.
    *   `B` is a coefficient related to the diversifiable (unsystematic) risk.
6.  **Generates a plot:** Visualizes the relationship between portfolio size and average standard deviation, including the fitted curve and the systematic risk asymptote, similar to Figure 1 in the original paper.

## Requirements

*   Python 3.x
*   Libraries:
    *   pandas
    *   numpy
    *   matplotlib
    *   scipy
    *   tqdm (optional, for progress bar)

You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib scipy tqdm