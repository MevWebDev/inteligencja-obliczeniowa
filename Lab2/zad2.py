import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Load the Iris dataset
iris_data = pd.read_csv("./Lab2/iris.csv")

# Extract numeric features and target variable
numeric_features = iris_data.select_dtypes(include=["number"])
species = iris_data['variety']

# Standardize the numeric features
standardizer = StandardScaler()
scaled_features = standardizer.fit_transform(numeric_features)

# Apply PCA to reduce dimensions to 2 components
pca_transformer = PCA(n_components=2)
reduced_features = pca_transformer.fit_transform(scaled_features)

# Calculate explained variance for each principal component
variance_explained = np.round(pca_transformer.explained_variance_ratio_ * 100, 2)
print("Explained variance per component:", variance_explained)

# Plot the PCA results
plt.figure(figsize=(10, 6))
unique_species = species.unique()
color_palette = ['red', 'green', 'blue']

for idx, sp in enumerate(unique_species):
    species_mask = species == sp
    plt.scatter(reduced_features[species_mask, 0], 
                reduced_features[species_mask, 1], 
                c=color_palette[idx], 
                label=sp, 
                alpha=0.7)

plt.xlabel(f"Principal Component 1 ({variance_explained[0]}%)")
plt.ylabel(f"Principal Component 2 ({variance_explained[1]}%)")
plt.title("2D PCA Visualization of Iris Dataset")
plt.legend()
plt.grid(True)
plt.savefig("pca.png")
plt.close()

# Display PCA loadings
pca_loadings = pd.DataFrame(
    pca_transformer.components_, 
    columns=numeric_features.columns, 
    index=[f'PC{i+1} ({variance_explained[i]}%)' for i in range(pca_transformer.n_components_)]
)
print("\nPCA Loadings:")
print(pca_loadings)