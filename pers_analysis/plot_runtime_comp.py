import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# smoothing
array = [pd.read_csv(str(i) + '.csv', sep=' ').cost.values for i in range(1, 6)]
max_value = max([len(array[i]) for i in range(len(array))])
for i in range(len(array)):
    array[i] = list(array[i]) + [array[i][-1]] * (max_value - len(array[i]))

array = [savgol_filter(array[i], 51, 3) for i in range(len(array))]
array_means = np.mean(array, axis=0)
array_means = np.array([array_means[i] - 900 for i in range(len(array_means))])
array_stds = np.std(array, axis=0)

## egna

array_egna = [pd.read_csv('egna' + str(i) + '.csv', sep=' ').cost.values for i in range(1, 6)]
max_value = max([len(array_egna[i]) for i in range(len(array_egna))])
for i in range(len(array_egna)):
    array_egna[i] = list(array_egna[i]) + [array_egna[i][-1]] * (max_value - len(array_egna[i]))

array_egna = [savgol_filter(array_egna[i], 51, 3) for i in range(len(array_egna))]
array_means_egna = np.mean(array_egna, axis=0)
array_means_egna = np.array([array_means_egna[i] - 900 for i in range(len(array_means_egna))])
array_stds_egna = np.std(array_egna, axis=0)

plt.figure(figsize=(15, 5))
plt.errorbar(range(len(array_means)), array_means, yerr=array_stds, color='blue', label='SPEDA')
plt.errorbar(range(len(array_means_egna)), array_means_egna, yerr=array_stds_egna, color='green', label='EGNA')

plt.xlabel('iteration', fontsize=15)
plt.ylabel('cost', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Best cost for each generation during runtime for cec8', fontsize=15)
plt.tight_layout()
plt.legend(fontsize=15)
plt.show()
plt.savefig('comp_runtime_egna_speda.png')
