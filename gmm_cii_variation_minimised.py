# This code is meant to apply scikit learn GMM on CII lines, iterating with different number of components (ncomp)
# the results will then be used in another code of next part to determine the best ncomp based on total w

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
max_ncomp = 30 # what is the maximum number of ncomp to iterate?
run_per_ncomp = 10 # how many times to run the model per ncomp?
use_saved_model = True
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

# minmax normalize each spectrum
normalized_data = spectra_2d/data_sum
normalized_data = np.nan_to_num(normalized_data)

# Perform GMM clustering with n clusters

array_total_wcss = []

for ncomp in range(1, max_ncomp+1):
    print(f'running gmm with ncomp of {ncomp}, this might takes a while...')
    
    array_ncomp_variances = []
    
    # repeat for run_per_ncomp number of times
    for k in range(run_per_ncomp):
        gmm = GaussianMixture(n_components=ncomp)
        gmm.fit(normalized_data)

        # Get the predicted cluster labels for each data point
        labels = gmm.predict(normalized_data)

        labels_2d = labels.reshape(y,x)

        # Combine the labels with the normalized data
        labeled_data = np.hstack((normalized_data, labels[:, np.newaxis]))
        labeled_data_unnormalized = np.hstack((spectra_2d, labels[:, np.newaxis]))

        # Compute the average spectra (and their normalized ones) for each label
        average_spectra = np.zeros((ncomp, normalized_data.shape[1]))
        average_spectra_std = np.zeros((ncomp, normalized_data.shape[1]))

        average_normalized_spectra = np.zeros((ncomp, normalized_data.shape[1]))
        average_normalized_spectra_std = np.zeros((ncomp, normalized_data.shape[1]))

        total_wcss = 0

        for i in range(ncomp):  
            # Select only the spectra for this cluster (dropping the label column)
            cluster_mask = (labeled_data[:, -1] == i)
            cluster_samples = labeled_data[cluster_mask, :-1]
            
            # the below takes the mean and standard deviation of the normalized spectra of each cluster.
            average_normalized_spectra[i] = np.mean(cluster_samples, axis=0)
            average_normalized_spectra_std[i] = np.std(cluster_samples, axis=0)

            # we also take the mean and standard deviation of the original unnormalized spectra
            average_spectra[i] = np.mean(labeled_data_unnormalized[labeled_data_unnormalized[:, -1] == i, :-1], axis=0)
            average_spectra_std[i]  = np.std(labeled_data_unnormalized[labeled_data_unnormalized[:, -1] == i, :-1], axis=0)
            
            # calculate the Within-Cluster Sum of Squares (wcss) of each cluster
            squared_diffs = (cluster_samples - average_normalized_spectra[i])**2
            # breakpoint()
            cluster_ss = np.sum(squared_diffs)
            total_wcss += cluster_ss

        print(f"total wcss for ncomp of {ncomp} for run {k+1} is {total_wcss}")
        
        array_ncomp_variances.append(total_wcss)
        
    array_total_wcss.append(array_ncomp_variances)
    
# save the total_wcss array into a file
array_total_wcss = np.array(array_total_wcss)
np.save('codes/gaussian_mixture_model/total_wcss.npy',array_total_wcss)
