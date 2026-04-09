# This code is meant to apply scikit learn GMM on CII lines, 

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
plt.style.use('./codes/astro.mplstyle')

# user defined variable
cube_path = 'carta/SOFIA/new_pca/W40_CII_PCA_20_8_0p3_clean.fits'
# vmin, vmax = -20, 30 # previous range
vmin, vmax = -12, 20 # new range determined from dynamic singal windowing 
mask_threshold = 10
gmm_ncomp = 3
use_saved_model = False
check_expansion = False
random_state = 42 # for reproducible results

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

# minmax normalize each spectrum
normalized_data = spectra_2d/data_sum
normalized_data = np.nan_to_num(normalized_data)

gmm_ncomp = gmm_ncomp

if use_saved_model:
    print("using previously run gmm model...")
    gmm = joblib.load(f"gmm_model_ncomp{gmm_ncomp}.pkl")

else:
    # Perform GMM clustering with n clusters
    print('running gmm, this might takes a while...')
    gmm = GaussianMixture(n_components=gmm_ncomp, random_state=random_state)
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
average_spectra = np.zeros((gmm_ncomp, normalized_data.shape[1]))
average_spectra_std = np.zeros((gmm_ncomp, normalized_data.shape[1]))

average_normalized_spectra = np.zeros((gmm_ncomp, normalized_data.shape[1]))
average_normalized_spectra_std = np.zeros((gmm_ncomp, normalized_data.shape[1]))

for i in range(gmm_ncomp):  
    # Select only the spectra for this cluster (dropping the label column)
    cluster_mask = (labeled_data[:, -1] == i)
    cluster_samples = labeled_data[cluster_mask, :-1]
    
    # the below takes the mean and standard deviation of the normalized spectra of each cluster.
    average_normalized_spectra[i] = np.mean(cluster_samples, axis=0)
    average_normalized_spectra_std[i] = np.std(cluster_samples, axis=0)

    # we also take the mean and standard deviation of the original unnormalized spectra
    average_spectra[i] = np.mean(labeled_data_unnormalized[labeled_data_unnormalized[:, -1] == i, :-1], axis=0)
    average_spectra_std[i]  = np.std(labeled_data_unnormalized[labeled_data_unnormalized[:, -1] == i, :-1], axis=0)

# Plot the cluster spatial distribution, and their average spectra
# add wcs ticks and labels, scalebar
cluster = np.unique(labels)

# identify the cluster number of empty spectrum and remove it from the list
empty_cluster_idx = np.where((average_normalized_spectra == 0).all(axis=1))[0]
empty_cluster = cluster[empty_cluster_idx]

cluster_clean = np.delete(cluster,empty_cluster_idx)

labels_2d = np.where(labels_2d==empty_cluster, np.nan, labels_2d)

# breakpoint()    
fig = plt.figure(figsize=(1820/162,1000/162), dpi=162) #figsize
gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.25)
ax_map = fig.add_subplot(gs[0, 0], projection=cube.wcs.celestial)
cmap = plt.get_cmap('Paired', gmm_ncomp)
cluster_colors = [cmap(i) for i in cluster]
img = ax_map.imshow(labels_2d, origin='lower', cmap=cmap)
selected_clusters = cluster
add_scalebar(ax_map, length=411*u.arcsec, label='1 pc', 
             corner='bottom right',color='black')
ax_map.set_ylabel('DEC')
ax_map.set_xlabel('RA')
    
# plot averaged spectra separatedly in a column
gs_spec = gs[0, 1].subgridspec(gmm_ncomp-1, 1,
                               hspace=0, wspace=0)

for idx in range(gmm_ncomp-1):
    ax = fig.add_subplot(gs_spec[idx,0])
    x = slab.spectral_axis.value
    y = average_spectra[cluster_clean[idx]]
    y_up = y+average_spectra_std[cluster_clean[idx]]
    y_down = y-average_spectra_std[cluster_clean[idx]]
    ax.step(x, y, color=cluster_colors[cluster_clean[idx]])
    
    ax.text(0.95, 0.95, f'Cluster {cluster_clean[idx]}', ha='right', va='top', 
            transform=ax.transAxes)
    # also plot their std as uncertainties
    ax.fill_between(x, y_up, y_down, alpha=0.5, color=cluster_colors[cluster_clean[idx]])
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='minor', size=5)
    ax.tick_params(axis='both', which='major', size=8)
    
    if idx == 0:
        ax.set_title('Average Spectra')
    if idx == (gmm_ncomp-1)//2:
        ax.set_ylabel('Intensity (K)')
    if idx != (gmm_ncomp-1)-1:
        ax.set_xticklabels([])
    
    if check_expansion:
        ax.axhline(0, linewidth=0.3, color='black')
        ax.set_ylim(-1.0, 2.0)
        
ax.set_xlabel('Velocity (km/s)')


# also plot the normalized version
gs_spec_norm = gs[0, 2].subgridspec(gmm_ncomp-1, 1,
                               hspace=0, wspace=0)

for idx in range(gmm_ncomp-1):
    ax = fig.add_subplot(gs_spec_norm[idx,0])
    x = slab.spectral_axis.value
    y = average_normalized_spectra[cluster_clean[idx]]
    std = average_normalized_spectra_std[cluster_clean[idx]]
    y_up = y+std
    y_down = y-std
    ax.step(x, y, color=cluster_colors[cluster_clean[idx]])
    
    ax.text(0.95, 0.95, f'Cluster {cluster_clean[idx]}', ha='right', va='top', 
            transform=ax.transAxes)

    # also plot their std as uncertainties
    ax.fill_between(x, y_up, y_down, alpha=0.5, color=cluster_colors[cluster_clean[idx]])
    
    # plot the variation per channel in the same plot
    ax.step(x, std, color='red', linestyle='dashed', linewidth=1)
        
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='minor', size=5)
    ax.tick_params(axis='both', which='major', size=8)
    if idx == 0:
        ax.set_title('Average Normalized Spectra')
    if idx == (gmm_ncomp-1)//2:
        ax.set_ylabel('Normalized Intensity (K)')        
    if idx != (gmm_ncomp-1)-1:
        ax.set_xticklabels([])
        
ax.set_xlabel('Velocity (km/s)')

# plot all averaged spectra in 1 plot and display in a new figure
fig2 = plt.figure(figsize=(1100/162, 750/162))
ax_all = fig2.add_subplot(111)
x = slab.spectral_axis.value
for idx in range(gmm_ncomp-1):
    y = average_spectra[cluster_clean[idx]]
    ax_all.step(x, y, color=cluster_colors[cluster_clean[idx]])

ax_all.set_ylabel('Intensity (K)')
ax_all.set_xlabel('Velocity (km/s)')



plt.legend()
plt.show()