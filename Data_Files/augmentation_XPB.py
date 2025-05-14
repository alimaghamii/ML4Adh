import pandas as pd
import numpy as np
import mpmath as mp
import math

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def _gamma_product(b_list):
    prod = 1
    for b in b_list:
        prod *= mp.gamma(b)
    return prod

def hyper_reg(a_list, b_list, z):
    a_f = [float(a) for a in a_list]
    b_f = [float(b) for b in b_list]
    return mp.hyper(a_f, b_f, float(z), force_series=True) / _gamma_product(b_list)

def I_value(n, nu, Gamma):
    pre   = (2 ** (-3 - 2 * n)) * (math.pi ** (-1.5 - n)) / ((n - 1) * nu)
    x_sq  = (Gamma / (4 * math.pi * nu)) ** 2
    z     = float(-x_sq)

    hyper1 = mp.hyper(
        [-0.5],
        [float(0.5 - n / 2), float(1 - n / 2)],
        z,
        force_series=True
    )

    term1 = -(4 ** (1 + n)) * (math.pi ** (0.5 + n)) * (
        Gamma - 2 * (n - 1) * math.pi * nu * hyper1
    )

    hyper2 = hyper_reg([float((n - 1) / 2)], [0.5, float((2 + n) / 2)], z)
    hyper3 = hyper_reg([float(n / 2)],       [1.5, float((3 + n) / 2)], z)

    term2 = (
        2
        * math.pi
        * (Gamma / nu) ** n
        * (
            4 * math.pi * nu * mp.gamma(1 - n / 2) * hyper2
            + Gamma * mp.gamma(1.5 - n / 2) * hyper3
        )
    )

    return pre * (term1 + term2)

def gamma_eff_PB(n, k, nu, tol=1e-10, max_iter=200):
    Gamma = 1.0
    for _ in range(max_iter):
        I = I_value(n, nu, Gamma)
        Gamma_next = 1.0 / (1.0 - (1.0 - k) * I)
        if abs(Gamma_next - Gamma) < tol:
            return float(Gamma_next)
        Gamma = Gamma_next
    raise RuntimeError(f"Convergence failed at ν̂ = {nu:g}")

# --------------------------------------------------------------------------
# Constants for r_u → ν̂ conversion
# --------------------------------------------------------------------------
C1 = 2.887
C2 = 3.24 * np.pi ** (2 / 3)

# --------------------------------------------------------------------------
# Load and clean the data
# --------------------------------------------------------------------------
df = pd.read_csv("Data_Files/comb_1to9.csv")

# Ensure runload, n, k are numeric and drop obviously bad rows
for col in ["runload", "n", "k"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(subset=["runload", "n", "k"], inplace=True)

# --------------------------------------------------------------------------
# Compute ν̂ from runload
# --------------------------------------------------------------------------
df["nu_hat"] = C1 * (abs(df["runload"] / C2) ** 1.171)

# --------------------------------------------------------------------------
# Safely compute PB and build a new list of valid rows
# --------------------------------------------------------------------------
valid_rows = []
for i, row in df.iterrows():
    try:
        pb = gamma_eff_PB(row["n"], row["k"], row["nu_hat"])
        row_with_pb = row.copy()
        row_with_pb["PB"] = pb
        valid_rows.append(row_with_pb)
    except Exception as e:
        print(f"⚠️ Skipping row {i} due to error: {e}")

# --------------------------------------------------------------------------
# Build clean DataFrame from successful rows
# --------------------------------------------------------------------------
df_clean = pd.DataFrame(valid_rows)
df_clean.drop(columns=["nu_hat"], inplace=True)
df_clean.to_csv("Data_Files/aug_comb_1to9.csv", index=False)

print(f"✅ Done — Saved {len(df_clean)} valid rows to 'aug_comb_1to9.csv'.")
