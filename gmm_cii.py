# This code is meant to apply scikit learn GMM on CII lines, 
# hopefully to have a glance on the distributions of the spectra. 
# Optionally, the map can be masked by the moment0 map later on.

from astropy.io import fits
from spectral_cube import SpectralCube
from astropy import units as u
u.add_enabled_units(u.def_unit(['K (Tmb)'], represents=u.K))
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap
from codes.cube_rms_estimate import CubeNoise
from astropy.visualization.wcsaxes import add_scalebar

# matplotlib configuration
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'

# user defined variable
cube_path = 'carta/SOFIA/new_pca/W40_CII_PCA_20_8_0p3_clean.fits'
vmin, vmax = -20, 30
mask_threshold = 10
gmm_ncomp = 10
use_saved_model = True
check_expansion = False

cube = SpectralCube.read(cube_path).with_spectral_unit(u.km/u.s)
slab = cube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)
spectra = slab.unmasked_data[:].value

# or we can mask based on maximum value > N sigma let say
spectra_max = np.max(spectra, axis=0)
cube_noise = CubeNoise(slab)
spectra_rms,_ = cube_noise.rms_negative()
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
n_comp = gmm_ncomp

if use_saved_model:
    print("using previously run gmm model...")
    gmm = joblib.load(f"gmm_model_ncomp{gmm_ncomp}.pkl")

else:
    # Perform GMM clustering with n clusters
    print('running gmm, this might takes a while...')
    gmm = GaussianMixture(n_components=n_comp)
    gmm.fit(normalized_data)

    # Save model
    joblib.dump(gmm, f"gmm_model_ncomp{gmm_ncomp}.pkl")


# Get the predicted cluster labels for each data point
labels = gmm.predict(normalized_data)

labels_2d = labels.reshape(y,x)

# Combine the labels with the normalized data
labeled_data = np.hstack((normalized_data, labels[:, np.newaxis]))
labeled_data_unnormalized = np.hstack((spectra_2d, labels[:, np.newaxis]))

# Compute the average spectra (and their normalized ones) for each label
average_spectra = np.zeros((n_comp, normalized_data.shape[1]))
average_spectra_std = np.zeros((n_comp, normalized_data.shape[1]))

average_normalized_spectra = np.zeros((n_comp, normalized_data.shape[1]))
average_normalized_spectra_std = np.zeros((n_comp, normalized_data.shape[1]))

for i in range(n_comp):
    # the below takes the mean of the normalized spectra.
    average_normalized_spectra[i] = np.mean(labeled_data[labeled_data[:, -1] == i, :-1], axis=0)
    average_normalized_spectra_std[i] = np.std(labeled_data[labeled_data[:, -1] == i, :-1], axis=0)
    # alternatively, we can take the mean of the original spectra
    average_spectra[i] = np.mean(labeled_data_unnormalized[labeled_data_unnormalized[:, -1] == i, :-1], axis=0)
    average_spectra_std[i]  = np.std(labeled_data_unnormalized[labeled_data_unnormalized[:, -1] == i, :-1], axis=0)
# (optional) normalize the average spectra
# average_spectra_normalized = average_spectra / np.max(average_spectra, axis=1, keepdims=True)

# Plot the cluster spatial distribution, and their average spectra
# add wcs ticks and labels, scalebar
cluster = np.unique(labels)
fig = plt.figure(figsize=(1820/162,1000/162), dpi=162) #figsize
gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])
ax_map = fig.add_subplot(gs[0, 0], projection=cube.wcs.celestial)
cmap = plt.get_cmap('Paired')
cluster_colors = [cmap(i) for i in cluster]
listed_cmap = ListedColormap(cluster_colors)
img = ax_map.imshow(labels_2d, origin='lower', cmap=listed_cmap)
cbar = fig.colorbar(img, ax=ax_map)
selected_clusters = cluster
add_scalebar(ax_map, length=411*u.arcsec, label='1 pc', 
             corner='bottom right',color='black')
ax_map.set_ylabel('DEC')
ax_map.set_xlabel('RA')

# plot all averaged spectra in 1 plot
# ax_spec = fig.add_subplot(gs[0, 1])
# for idx, cluster_idx in enumerate(selected_clusters):
    # ax_spec.step(
        # slab.spectral_axis,
        # average_spectra_normalized[cluster_idx],
        # average_spectra[cluster_idx],
        # label=f'Cluster {cluster_idx}',
        # color=cluster_colors[cluster_idx],
        # where='mid' 
    # )
    
# plot averaged spectra separatedly in a column
gs_spec = gs[0, 1].subgridspec(len(cluster), 1,
                               hspace=0, wspace=0)

for idx in range(len(cluster)):
    ax = fig.add_subplot(gs_spec[idx,0])
    x = slab.spectral_axis.value
    y = average_spectra[cluster[idx]]
    y_up = y+average_spectra_std[cluster[idx]]
    y_down = y-average_spectra_std[cluster[idx]]
    ax.step(x, y, color=cluster_colors[cluster[idx]])
    
    ax.text(0.95, 0.95, f'Cluster {cluster[idx]}', ha='right', va='top', 
            transform=ax.transAxes)
    # also plot their std as uncertainties
    ax.fill_between(x, y_up, y_down, alpha=0.5, color=cluster_colors[cluster[idx]])
    # ax.set_xticks(np.arange(-10, 22, 5))
    ax.minorticks_on()
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='minor', size=5)
    ax.tick_params(axis='both', which='major', size=8)
    
    if idx == 0:
        ax.set_title('Average Spectra')
    if idx == len(cluster)//2:
        ax.set_ylabel('Intensity (K)')
    if idx != len(cluster)-1:
        ax.set_xticklabels([])
    
    if check_expansion:
        ax.axhline(0, linewidth=0.3, color='black')
        ax.set_ylim(-1.0, 2.0)
        
ax.set_xlabel('Velocity (km/s)')


# also plot the normalized version
gs_spec_norm = gs[0, 2].subgridspec(len(cluster), 1,
                               hspace=0, wspace=0)

for idx in range(len(cluster)):
    ax = fig.add_subplot(gs_spec_norm[idx,0])
    x = slab.spectral_axis.value
    y = average_normalized_spectra[cluster[idx]]
    y_up = y+average_normalized_spectra_std[cluster[idx]]
    y_down = y-average_normalized_spectra_std[cluster[idx]]
    ax.step(x, y, color=cluster_colors[cluster[idx]])
    
    ax.text(0.95, 0.95, f'Cluster {cluster[idx]}', ha='right', va='top', 
            transform=ax.transAxes)
    # also plot their std as uncertainties
    ax.fill_between(x, y_up, y_down, alpha=0.5, color=cluster_colors[cluster[idx]])
    # ax.set_xticks(np.arange(-10, 22, 5))
    ax.minorticks_on()
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='minor', size=5)
    ax.tick_params(axis='both', which='major', size=8)
    if idx == 0:
        ax.set_title('Average Normalized Spectra')
    if idx == len(cluster)//2:
        ax.set_ylabel('Normalized Intensity (K)')        
    if idx != len(cluster)-1:
        ax.set_xticklabels([])
        
ax.set_xlabel('Velocity (km/s)')

# check if the maximum variation of normalized spectra of 
# each cluster is beyond the mean rms noise
# print('maximum 

plt.legend()
plt.show()