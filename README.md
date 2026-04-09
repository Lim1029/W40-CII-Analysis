This project applies Gaussian Mixture Model (GMM) to W40 CII spectralcube to cluster spectra of similar shape together.

The steps are as follow:

1) Determining the optimal number of cluster (ncomp), using several methods:  
1.1) Using the built-in GridSearchCV method in Scikit-learn to iterate ncomp from 1 to 30. The optimal ncomp is chosen as lowest:  
1.1.1) Akaike information criterion (AIC) score: [`gmm_cii_gridsearch_aic.py`](./gmm_cii_gridsearch_aic.py)  
1.1.2) Bayesian information criterion (BIC) score: [`gmm_cii_gridsearch_bic.py`](./gmm_cii_gridsearch_bic.py)  

1.2) Finding the "knee" of Total within cluster sum of square (wcss) against ncomp plot: [`gmm_cii_variation_minimised.py`](./gmm_cii_variation_minimised.py) and [`gmm_cii_variation_minimised_cont.py`](./gmm_cii_variation_minimised_cont.py) 

2) performs GMM with the chosen ncomp: [`gmm_cii.py`](./gmm_cii.py)