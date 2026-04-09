# This code continues the analysis by importing the total_wcss.npy and select the best ncomp based on some metric

import numpy as np 
import matplotlib.pyplot as plt

# plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 162

array_total_wcss = np.load('codes/gaussian_mixture_model/total_wcss.npy')

# calculate mean and standard deviation
mean_wcss = np.mean(array_total_wcss, axis=1)
std_wcss = np.std(array_total_wcss, axis=1)

# plot the mean_wcss with error bar std_wcss array              
x = np.arange(1,len(array_total_wcss)+1,1)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.errorbar(x=x, y=mean_wcss, yerr=std_wcss,
            linestyle='-', marker='*',
            linewidth=1, capsize=5, elinewidth=1, ms=5
            )

# calculate the derivative
dx = x[2:] - x[:-2]
dy = mean_wcss[2:] - mean_wcss[:-2]
dxdy = dy/dx
dydx_err = np.sqrt(std_wcss[2:]**2 + std_wcss[:-2]**2) / dx

ax2 = ax.twinx()

# ax2.plot(x[1:-1], dxdy, marker='x', color='red')
ax2.errorbar(x=x[1:-1], y=dxdy, yerr=dydx_err,
            linestyle='-', marker='*',
            linewidth=1, capsize=5, elinewidth=1, ms=5, color='red'
            )

ax.set_xlabel('ncomp')
ax.set_ylabel('Total WCSS', color='blue')
ax.tick_params(axis='y', which='both', colors='blue', labelcolor='blue')
ax2.set_ylabel('derivative of total wcss', color='red')
ax2.tick_params(axis='y', which='both', colors='red', labelcolor='red')

# detecting the elbow and kneed
from kneed import KneeLocator

kl = KneeLocator(x, mean_wcss, curve="convex", 
                 direction="decreasing")

print(kl.knee)    
print(kl.knee_y) 

# plot the kneed
ax.axvline(kl.knee, linestyle='dashed', color='black')
# ax.text(x=kl.knee, y=, s=kl.knee, 

plt.show()
