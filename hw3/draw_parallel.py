import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "nBodies": [1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 2048, 2048, 4096, 4096, 4096, 4096, 4096, 4096,
                8192, 8192, 8192, 8192, 8192, 8192, 16384, 16384, 16384, 16384, 16384, 16384],
    "BLOCK_SIZE": [8, 16, 32, 64, 128, 256, 8, 16, 32, 64, 128, 256, 8, 16, 32, 64, 128, 256,
                   8, 16, 32, 64, 128, 256, 8, 16, 32, 64, 128, 256],
    "Interactions_Per_Second_Billion": [0.381, 0.397, 0.386, 0.385, 0.395, 0.399, 13.294, 1.506, 1.491, 1.513, 1.551,
                                        1.465, 5.464, 5.464, 5.386, 5.441, 5.556, 5.521, 18.385, 19.728, 19.168, 19.608,
                                        19.046, 18.488, 35.603, 57.591, 57.820, 57.216, 56.752, 57.780],
}

df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
for block_size in df["BLOCK_SIZE"].unique():
    subset = df[df["BLOCK_SIZE"] == block_size]
    plt.plot(subset["nBodies"], subset["Interactions_Per_Second_Billion"], label=f'BLOCK_SIZE={block_size}', marker='o')

# Add log scale
# plt.xscale('log')  # Log scale for x-axis
plt.yscale('log')  # Log scale for y-axis

plt.xlabel('nBodies (log scale)')
plt.ylabel('Interactions Per Second (Billion, log scale)')
plt.title('Interactions Per Second vs nBodies (Log Scale)')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)  # Gridlines for log scale
plt.show()
plt.savefig('outputs/parallel_analysis.png')
plt.savefig('outputs/parallel_analysis.pdf')
