import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("dark", 5)


def calculate_mean_std(results):
    means = []
    stds = []
    percentages = []

    for size in sorted(results.keys()):
        values = results[size]
        if len(values) > 0:
            percentages.append(int(size * 100))
            means.append(np.mean(values))
            stds.append(np.std(values))

    return percentages, means, stds
            

# === AccuracyByDataSize ===
sizes = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
means = [np.float64(0.2198830395936966), np.float64(0.7625730991363525), np.float64(0.8407017469406128), np.float64(0.9069005966186523), np.float64(0.9296686172485351), np.float64(0.9546331524848938)]
stds = [np.float64(0.039593708018591446), np.float64(0.051461999118331006), np.float64(0.014578173082278009), np.float64(0.00503603423785939), np.float64(0.009993282786952117), np.float64(0.000651014537135622)]

results_random = {0.75: [0.9477, 0.9395, 0.9512, 0.9304, 0.9500],
                  0.50: [0.9380, 0.9322, 0.9380, 0.9260, 0.9328],
                  0.25: [0.9044, 0.9041, 0.9068, 0.8989, 0.9073],
                  0.10: [0.8585, 0.8626, 0.8667, 0.8559, 0.8614],
                  0.05: [0.8389, 0.8290, 0.8281, 0.8243, 0.8319]
                  }

results_balanced_initial_split = {
                                   0.50: [0.9386, 0.9410, 0.9377, 0.9389, 0.9351],
                                   0.25: [0.9000, 0.9044, 0.9027, 0.9068, 0.9030],
                                   0.10: [0.8612, 0.8629, 0.8711, 0.8670, 0.8603],
                                   0.05: [0.8366, 0.8264, 0.8316, 0.8270, 0.8460]
                                  }

percentages_random, means_random, stds_random = calculate_mean_std(results_random)

percentages_balanced_initial_split, means_balanced_initial_split, stds_balanced_initial_split = calculate_mean_std(results_balanced_initial_split)



percentages = [int(size * 100) for size in sizes]
plt.figure(figsize=(6, 4))
fontsize = 16
plt.errorbar(percentages, means, yerr=stds, fmt='-o', capsize=5, elinewidth=2, label = 'Accuracy by training data size')
for x, y in zip(percentages, means):
    plt.text(x + 5, y - 0.05, f"{y:.3f}", ha='center', fontsize=10, color='black')

plt.errorbar(percentages_random, means_random, yerr=stds_random, fmt='-o', capsize=5, elinewidth=2, color='red', label='SSL (randomly initialized)') # Semi-supervised (class-balanced init)
for x, y in zip(percentages_random, means_random):
    plt.text(x - 3, y + 0.03, f"{y:.3f}", ha='center', fontsize=10, color='red')

plt.errorbar(percentages_balanced_initial_split, means_balanced_initial_split, yerr=stds_balanced_initial_split, fmt='-o', capsize=5, elinewidth=2, color='green', label='SSL (balanced initial split)') 
for x, y in zip(percentages_balanced_initial_split, means_balanced_initial_split):
    plt.text(x + 0, y - 0.05, f"{y:.3f}", ha='center', fontsize=10, color='green')

plt.xlabel("Percentage of data used [%]", fontsize=fontsize)
plt.ylabel("Accuracy - test", fontsize=fontsize)
plt.ylim((0, 1.0))
plt.xlim((0, 105))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('images/SSL_both.png')
print('Plot has been saved as SSL_random.png')
plt.show()
