# Gaussian Mixture Model (GMM) selection with AIC/BIC and visualizations (1D)
# - This notebook-style code demonstrates how to find the best number of Gaussian components
#   using AIC and BIC, then visualizes the fitted mixture and responsibilities.
# - To apply to your own data, replace the "data" array with your 1D array (e.g. np.array(my_values)).
# - Rules followed: matplotlib used for plotting, each chart is created separately, no explicit colors set.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Example data (replace this with your own 1D data)
np.random.seed(0)
data = np.concatenate([
    np.random.normal(-1.0, 0.4, 200),
    np.random.normal(2.5, 0.3, 150),
    np.random.normal(6.0, 0.6, 120)
])
X = data.reshape(-1, 1)

# -----------------------------
# Function: fit GMMs across a range of k and compute AIC/BIC
def fit_gmms_and_scores(X, k_min=1, k_max=8):
    ks = list(range(k_min, k_max + 1))
    models = {}
    aics = []
    bics = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        gmm.fit(X)
        models[k] = gmm
        aics.append(gmm.aic(X))
        bics.append(gmm.bic(X))
    return ks, models, np.array(aics), np.array(bics)

ks, models, aics, bics = fit_gmms_and_scores(X, k_min=1, k_max=8)

# -----------------------------
# Plot 1: AIC and BIC vs number of components
plt.figure(figsize=(8, 4))
plt.plot(ks, aics, marker='o', label='AIC')
plt.plot(ks, bics, marker='o', label='BIC')
plt.xlabel('Number of components (k)')
plt.ylabel('Score')
plt.title('AIC and BIC for different number of GMM components')
plt.legend()
plt.grid(axis='y', linestyle=':', linewidth=0.5)
plt.show()

# -----------------------------
# Choose best k (minimum BIC and AIC)
best_k_bic = ks[np.argmin(bics)]
best_k_aic = ks[np.argmin(aics)]
print(f"Best k by BIC: {best_k_bic}, Best k by AIC: {best_k_aic}")

# Refit the GMM chosen by BIC (commonly preferred for model selection)
gmm = models[best_k_bic]
labels = gmm.predict(X)                     # hard labels (argmax of responsibilities)
probs = gmm.predict_proba(X)                # responsibilities (N x K)
weights = gmm.weights_                      # mixture weights
means = gmm.means_.flatten()                # component means
covs = gmm.covariances_.reshape(best_k_bic, 1, 1)  # cov matrices (k, 1, 1)
stds = np.sqrt(covs).flatten()              # component stds

# -----------------------------
# Plot 2: Histogram with GMM component PDFs overlaid
x_grid = np.linspace(np.min(X) - 1, np.max(X) + 1, 1000)
total_pdf = np.zeros_like(x_grid)
component_pdfs = []
for k in range(best_k_bic):
    comp_pdf = weights[k] * norm.pdf(x_grid, loc=means[k], scale=stds[k])
    component_pdfs.append(comp_pdf)
    total_pdf += comp_pdf

plt.figure(figsize=(8, 4))
plt.hist(X.flatten(), bins=40, density=True, alpha=0.6)
plt.plot(x_grid, total_pdf, linewidth=2, label='Mixture PDF')
# plot each component's PDF
for k, comp_pdf in enumerate(component_pdfs):
    plt.plot(x_grid, comp_pdf, linestyle='--', linewidth=1, label=f'Comp {k} PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title(f'Histogram + GMM (k={best_k_bic}) PDFs')
plt.legend()
plt.show()

# -----------------------------
# Plot 3: 1D scatter with assigned labels and component means
plt.figure(figsize=(10, 1.6))
plt.scatter(X.flatten(), np.zeros_like(X.flatten()), c=labels, s=30, alpha=0.7)
for m in means:
    plt.axvline(m, linestyle='-', linewidth=2)
plt.yticks([])
plt.xlabel('Value')
plt.title('1D Scatter of samples colored by assigned GMM component (vertical lines = means)')
plt.show()

# -----------------------------
# Plot 4: Responsibilities (stacked area) across the value axis
# We'll compute responsibilities across the x_grid to show where each component dominates.
probs_grid = np.vstack([weights[k] * norm.pdf(x_grid, loc=means[k], scale=stds[k]) for k in range(best_k_bic)]).T
# normalize to get responsibilities across grid
probs_grid_norm = probs_grid / probs_grid.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 4))
plt.stackplot(x_grid, probs_grid_norm.T)
plt.xlabel('Value')
plt.ylabel('Responsibility (component probability)')
plt.title('GMM responsibilities across value axis (stacked area)')
plt.show()

# -----------------------------
# Optional: Show a small table-like summary for components
import pandas as pd
summary = pd.DataFrame({
    'component': np.arange(best_k_bic),
    'weight': weights,
    'mean': means,
    'std': stds
})
# Use display for the user; python_user_visible will render the DataFrame.
summary

# -----------------------------
# Quick note printed for user instructions
print("\nHow to use with your own data:")
print(" - Replace the `data` array above with your 1D numpy array (e.g. data = np.array(my_values)).")
print(" - Adjust the k range in fit_gmms_and_scores if you expect more/fewer components.")
print(" - Use BIC (more conservative) or AIC (less conservative) as the selection criterion.")

