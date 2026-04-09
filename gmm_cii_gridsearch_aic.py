# This code is meant to apply scikit learn GMM on CII lines, 
# hopefully to have a glance on the distributions of the spectra. 
# Optionally, the map can be masked by the moment0 map later on.
# reference: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py

from astropy.io import fits
from spectral_cube import SpectralCube
from astropy import units as u
u.add_enabled_units(u.def_unit(['K (Tmb)'], represents=u.K))
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap
from codes.cube_rms_estimate import rms_negative
from astropy.visualization.wcsaxes import add_scalebar
from sklearn.model_selection import GridSearchCV
plt.style.use('./codes/astro.mplstyle')

# user defined variable
cube_path = 'carta/SOFIA/new_pca/W40_CII_PCA_20_8_0p3_clean.fits'
vmin, vmax = -12, 20
mask_threshold = 10
check_expansion = False

cube = SpectralCube.read(cube_path).with_spectral_unit(u.km/u.s)
slab = cube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)
spectra = slab.unmasked_data[:].value

# we mask based on maximum value > N sigma let say
spectra_max = np.max(spectra, axis=0)
spectra_rms,_ = rms_negative(slab)
spectra_snr = spectra_max/spectra_rms
mask = spectra_snr>mask_threshold

expanded_mask = mask[np.newaxis, :, :]
masked_spectra = np.where(expanded_mask, spectra, np.nan)

v, y, x = masked_spectra.shape
spectra_2d = masked_spectra.reshape(v, y*x).T
data_max = np.max(spectra_2d, axis=1, keepdims=True)
data_sum = np.sum(spectra_2d, axis=1, keepdims=True)

# Normalize each spectrum
normalized_data = spectra_2d/data_sum
normalized_data = np.nan_to_num(normalized_data)

X = normalized_data

# Perform GMM clustering with n clusters
print('running gmm grid search, this might takes a while...')

def gmm_aic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the AIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.aic(X)

param_grid = {
    "n_components": range(1, 30),
}

grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_aic_score, verbose=5
)
grid_search.fit(X)

import pandas as pd

df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "mean_test_score": "AIC score",
    }
)
print(df.sort_values(by="AIC score"))