import numpy as np

array_path = input('input path to the .npy file: ')

array = np.load(array_path)
mean_array = np.mean(array, axis=1)

std_array = np.std(array, axis=1)

import matplotlib.pyplot as plt
plt.style.use('./codes/astro.mplstyle')

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ninput = np.arange(1,20.1,2)

dnoutput = mean_array[2:] - mean_array[:-2]

dninput = ninput[2:] - ninput[:-2]

derivative = dnoutput/dninput

ax.errorbar(ninput, mean_array, yerr=std_array, marker='.', color='blue')
ax2.plot(ninput[1:-1],derivative, marker='x', color='red')
ax.set_ylabel('noutput', color='blue')
ax.set_xlabel('ninput')
ax2.set_ylabel('derivative', color='red')

ax.tick_params(axis='y', which='both', colors='blue', labelcolor='blue')
ax2.tick_params(axis='y', which='both', colors='red', labelcolor='red')
plt.show()