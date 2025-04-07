import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm # for progress bar

# --- Configuration ---
DATA_FILE = 'Markets_Pod/Stocks_Data/data.csv' # <---  YOUR CSV FILENAME
OUTPUT_PLOT_FILE = 'diversification_curve.png'
START_DATE = '2000-01-01' # <--- Adjust analysis period start
END_DATE = '2024-12-31'   # <--- Adjust analysis period end
DATE_COLUMN = 'date'
PERMNO_COLUMN = 'PERMNO' # Unique Security ID
RETURN_COLUMN = 'RET'    # Return column
SHRCD_COLUMN = 'SHRCD'   # Share Code column (for filtering common stock)
COMMON_STOCK_CODES = [10, 11] # Typical CRSP codes for US common stock

MAX_PORTFOLIO_SIZE = 40   # Max number of stocks in a portfolio (like the paper)
NUM_SIMULATIONS = 200     # Number of random portfolios per size (more = smoother curve)
# Increase NUM_SIMULATIONS for better results (e.g., 500, 1000), paper used 60.

# --- Data Loading and Preparation ---

print(f"Loading data from {DATA_FILE}...")
try:
    # Try reading with standard encoding first
    df = pd.read_csv(DATA_FILE, low_memory=False)
except UnicodeDecodeError:
    print("UTF-8 decoding failed, trying latin1...")
    try:
        df = pd.read_csv(DATA_FILE, low_memory=False, encoding='latin1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found.")
    print("Please make sure the file is in the same directory or provide the full path.")
    exit()

print("Preprocessing data...")

# Convert date column to datetime
try:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
except KeyError:
    print(f"Error: Date column '{DATE_COLUMN}' not found in the CSV.")
    exit()
df = df.dropna(subset=[DATE_COLUMN]) # Drop rows where date conversion failed

# Convert return column to numeric, coerce errors to NaN
try:
    df[RETURN_COLUMN] = pd.to_numeric(df[RETURN_COLUMN], errors='coerce')
except KeyError:
    print(f"Error: Return column '{RETURN_COLUMN}' not found.")
    exit()

# Filter for relevant date range
df = df[(df[DATE_COLUMN] >= START_DATE) & (df[DATE_COLUMN] <= END_DATE)]

# Filter for common stocks
try:
    df = df[df[SHRCD_COLUMN].isin(COMMON_STOCK_CODES)]
except KeyError:
    print(f"Warning: Share code column '{SHRCD_COLUMN}' not found. Cannot filter for common stocks.")
    # Decide whether to proceed without filtering or stop
    # proceed = input("Proceed without stock type filtering? (yes/no): ")
    # if proceed.lower() != 'yes':
    #     exit()

# Check if PERMNO column exists
if PERMNO_COLUMN not in df.columns:
     print(f"Error: Security ID column '{PERMNO_COLUMN}' not found.")
     exit()

# Pivot to get returns time series: columns=PERMNO, index=date, values=RET
print("Pivoting data...")
try:
    returns_pivot = df.pivot_table(index=DATE_COLUMN, columns=PERMNO_COLUMN, values=RETURN_COLUMN)
except Exception as e:
    print(f"Error during pivoting: {e}")
    print("Check for duplicate PERMNO entries for the same date.")
    # Example check: duplicates = df[df.duplicated(subset=[DATE_COLUMN, PERMNO_COLUMN], keep=False)]
    # print(duplicates)
    exit()


# Data Cleaning after Pivot:
# 1. Drop columns (securities) with any missing values within the period
returns_pivot = returns_pivot.dropna(axis=1, how='any')

# 2. Ensure we have enough data
num_dates = len(returns_pivot)
available_securities = returns_pivot.columns.tolist()
num_available_securities = len(available_securities)

print(f"Analysis period: {returns_pivot.index.min().date()} to {returns_pivot.index.max().date()}")
print(f"Number of time periods (e.g., months): {num_dates}")
print(f"Number of securities with complete data: {num_available_securities}")

if num_available_securities < MAX_PORTFOLIO_SIZE:
    print(f"Warning: Only {num_available_securities} securities available, less than MAX_PORTFOLIO_SIZE ({MAX_PORTFOLIO_SIZE}).")
    MAX_PORTFOLIO_SIZE = num_available_securities
    if MAX_PORTFOLIO_SIZE == 0:
        print("Error: No securities available after filtering. Check data and date range.")
        exit()

if num_dates < 2:
    print("Error: Need at least 2 time periods to calculate standard deviation.")
    exit()

# --- Simulation ---

portfolio_sizes = range(1, MAX_PORTFOLIO_SIZE + 1)
average_std_devs = []
all_simulation_results = {size: [] for size in portfolio_sizes} # Store individual sim results

print(f"\nRunning simulations for portfolio sizes 1 to {MAX_PORTFOLIO_SIZE} ({NUM_SIMULATIONS} runs each)...")

# Use tqdm for progress bar if installed
for m in tqdm(portfolio_sizes, desc="Portfolio Size"):
    portfolio_std_devs_for_size_m = []
    for _ in range(NUM_SIMULATIONS):
        # 1. Randomly select m securities without replacement
        selected_permnos = random.sample(available_securities, m)
        portfolio_component_returns = returns_pivot[selected_permnos]

        # 2. Calculate equally weighted portfolio returns for each period
        # mean(axis=1) calculates the average return across the selected stocks for each date
        portfolio_period_returns = portfolio_component_returns.mean(axis=1)

        # 3. Calculate log of value relatives (log(1 + R))
        # Add small epsilon to avoid log(0) if a return is exactly -1
        log_value_relatives = np.log(1 + portfolio_period_returns + 1e-10)

        # 4. Calculate the standard deviation of the log value relatives over time
        # ddof=1 for sample standard deviation
        std_dev = log_value_relatives.std(ddof=1)

        portfolio_std_devs_for_size_m.append(std_dev)
        all_simulation_results[m].append(std_dev)


    # 5. Average the standard deviations for this portfolio size
    average_std_dev = np.mean(portfolio_std_devs_for_size_m)
    average_std_devs.append(average_std_dev)

# --- Curve Fitting ---
print("\nFitting hyperbolic curve Y = B*(1/X) + A...")

# Define the function for curve fitting (Y = A + B/X)
def hyperbolic_func(x, a, b):
    return a + b / x

x_data = np.array(portfolio_sizes)
y_data = np.array(average_std_devs)

try:
    # Provide initial guesses if needed, e.g., a ~ last std dev, b ~ initial drop
    initial_guess = [y_data[-1], (y_data[0] - y_data[-1])]
    params, covariance = curve_fit(hyperbolic_func, x_data, y_data, p0=initial_guess)
    a_fit, b_fit = params
    print(f"Fit complete.")
    print(f"  Asymptote (A - Systematic Risk estimate): {a_fit:.6f}")
    print(f"  Coefficient (B): {b_fit:.6f}")

    # Calculate R-squared for the fit
    residuals = y_data - hyperbolic_func(x_data, a_fit, b_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  R-squared of fit: {r_squared:.4f}")

    fit_successful = True
except Exception as e:
    print(f"Error during curve fitting: {e}")
    print("Curve fitting failed. Plot will show only simulation results.")
    fit_successful = False
    a_fit, b_fit, r_squared = None, None, None


# --- Plotting ---
print(f"Generating plot '{OUTPUT_PLOT_FILE}'...")

plt.figure(figsize=(12, 7))

# Plot individual simulation results (optional, can be noisy)
# for size in portfolio_sizes:
#    plt.plot([size]*len(all_simulation_results[size]), all_simulation_results[size], '.', color='lightgray', alpha=0.1)

# Plot average standard deviation per portfolio size
plt.plot(x_data, y_data, 'bo-', label='Average Portfolio Std Dev (Simulated)', markersize=5)

# Plot the fitted curve and asymptote if successful
if fit_successful:
    plt.plot(x_data, hyperbolic_func(x_data, a_fit, b_fit), 'r--', label=f'Fitted Curve: Y = {a_fit:.4f} + {b_fit:.4f}/X\n$R^2 = {r_squared:.4f}$')
    plt.axhline(y=a_fit, color='g', linestyle=':', label=f'Asymptote (Systematic Risk) = {a_fit:.4f}')

plt.xlabel("Number of Securities in Portfolio (X)")
plt.ylabel("Average Standard Deviation of Log Value Relatives (Y)")
plt.title(f"Portfolio Diversification Effect ({START_DATE} to {END_DATE})\nReplicating Evans & Archer (1968)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(0, MAX_PORTFOLIO_SIZE + 1)
# Adjust ylim if necessary, e.g., start slightly below asymptote
min_y = min(y_data) * 0.95 if not fit_successful else min(min(y_data), a_fit) * 0.95
max_y = max(y_data) * 1.05
plt.ylim(min_y, max_y)


plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE)
print(f"Plot saved to {OUTPUT_PLOT_FILE}")
# plt.show() # Uncomment to display the plot directly

print("\nAnalysis complete.")