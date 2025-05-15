import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpmath as mp
import os
from joblib import load

from utilities import file_PB, save_dir_PARPML_6inputs

# --------------------------------------------------------------------------
# XPB model functions
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
    pre = (2 ** (-3 - 2 * n)) * (np.pi ** (-1.5 - n)) / ((n - 1) * nu)
    x_sq = (Gamma / (4 * np.pi * nu)) ** 2
    z = float(-x_sq)

    hyper1 = mp.hyper([-0.5], [float(0.5 - n / 2), float(1 - n / 2)], z, force_series=True)
    term1 = -(4 ** (1 + n)) * (np.pi ** (0.5 + n)) * (Gamma - 2 * (n - 1) * np.pi * nu * hyper1)

    hyper2 = hyper_reg([float((n - 1) / 2)], [0.5, float((2 + n) / 2)], z)
    hyper3 = hyper_reg([float(n / 2)], [1.5, float((3 + n) / 2)], z)

    term2 = (
        2 * np.pi * (Gamma / nu) ** n *
        (4 * np.pi * nu * mp.gamma(1 - n / 2) * hyper2 + Gamma * mp.gamma(1.5 - n / 2) * hyper3)
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
# Settings and parameters
# --------------------------------------------------------------------------
alphabet = 'b'

if alphabet == 'a':
    n_fixed = 0.6
elif alphabet == 'b':
    n_fixed = 1.6

mu_fixed = 3.24
delta_l = np.log10(73)
alphaload_fixed = delta_l
k_values = [0.5,1/5, 0.1, 1/20, 1/50]

runload_values = np.linspace(-2, 9, 100)
r_u_values = 10 ** runload_values

C1 = 2.887
C2 = 3.24 * np.pi ** (2 / 3)

save_dir = save_dir_PARPML_6inputs
os.makedirs(save_dir, exist_ok=True)

# --------------------------------------------------------------------------
# Load trained model
# --------------------------------------------------------------------------
model_path = os.path.join(save_dir, 'XGBoost_model.pkl')
trained_model = load(model_path)

# --------------------------------------------------------------------------
# Plot setup
# --------------------------------------------------------------------------
font_size = 20
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'axes.linewidth': 1,
    'figure.figsize': (7, 4)
})

colors = ['#D10070', '#007FFF', '#009970', '#FF8C00', '#8B00FF']
colors = ['#8B0000', '#00008B', '#006400', '#F0E442', '#0072B2']  # Colorblind-friendly options
colors = ['purple', 'blue', 'red', 'green', 'darkorange']  # Updated colors with red replaced by purple

fig = plt.figure()

# --------------------------------------------------------------------------
# Loop through k values
# --------------------------------------------------------------------------
for i, k in enumerate(k_values):
    gamma_xpb = []
    ml_input_rows = []

    for r in r_u_values:
        nu_hat = C1 * ((r / C2) ** 1.171)
        try:
            gamma_val = gamma_eff_PB(n_fixed, k, nu_hat)
        except Exception:
            gamma_val = np.nan
        gamma_xpb.append(gamma_val)

        if not np.isnan(gamma_val):
            ml_input_rows.append([
                np.log10(r),       # runload
                alphaload_fixed,   # alphaload
                mu_fixed,          # mu
                np.log10(k),       # log10(k)
                n_fixed,           # n
                gamma_val          # XPB output (raw)
            ])
        else:
            ml_input_rows.append([np.nan] * 6)

    # Plot XPB curve
    plt.plot(r_u_values, gamma_xpb, linestyle='--', color=colors[i], label=f'$k={k:.3f}$ (XPB)', linewidth=2)

    # ML prediction using XPB output as input feature
    ml_input_df = pd.DataFrame(ml_input_rows, columns=["runload", "alphaload", "mu", "k", "n", "PB"])
    valid_ml_df = ml_input_df.dropna()

    if not valid_ml_df.empty:
        # Sample 20 evenly spaced points from valid input
        sample_indices = np.linspace(0, len(valid_ml_df) - 1, 21, dtype=int)
        ml_sampled_df = valid_ml_df.iloc[sample_indices]

        gamma_ml_sampled = trained_model.predict(ml_sampled_df)
        r_u_sampled = 10 ** ml_sampled_df["runload"]

        # Scatter ML predictions
        plt.scatter(r_u_sampled, 10 ** gamma_ml_sampled, label=f'$k={k:.3f}$ (ML)', color=colors[i]
                    , lw=5, alpha=0.4)

# plt.scatter(r_u, w_po_predicted, label=r'ML Prediction', color='blue', alpha=0.4, lw=2, s=90)

# --------------------------------------------------------------------------
# Finalize and save plot
# --------------------------------------------------------------------------
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\widehat{r}_u$')
plt.ylabel(r'$\widehat{\Gamma}_{eff}$')
# plt.legend(fontsize=12, loc='lower right')
plt.xlim(10**-2, 10**9)
font_size = 18
# Add a text arrow
plt.annotate(f'$k= {k_values}$', xytext=(10 ** -1.95, 10 ** 1.59), xy=(10 ** 3, 10 ** 0.25),
             arrowprops=dict(facecolor='black', arrowstyle='<-'),
             fontsize=font_size)

label = f'$\\widehat{{\\delta}}_l = {10**delta_l:.1f}$'
plt.text(10**-1.95, 10**1.2, label, fontsize=font_size)
plt.text(10**-1.95, 10**1.0, f'$n={n_fixed}$', fontsize=font_size, color='black')
plt.text(10**-1.95, 10**0.8, f'$\\mu={mu_fixed}$', fontsize=font_size, color='black')

plt.text(10**8.05, 10**0.001, f'({alphabet})', fontsize=font_size)


plt.tight_layout()

filename = f"XPB_vs_ML_GammaEff_n_{n_fixed}_mu_{mu_fixed}_sampled_ML.png"
plt.savefig(os.path.join(save_dir, filename))
plt.show()

print(f"✅ Figure saved to: {os.path.join(save_dir, filename)}")
