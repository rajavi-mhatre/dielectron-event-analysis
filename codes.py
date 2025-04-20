import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('dielectron.csv')
data.columns = data.columns.str.strip()

# Print the first few rows to check the data
print(data.head())

# Check for missing data
print(data.isnull().sum())

# Invariant mass calculation
def calculate_invariant_mass(E1, E2, px1, px2, py1, py2, pz1, pz2):
    """Calculate the invariant mass using the energy and momentum components."""
    try:
        mass_squared = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
        if mass_squared < 0:
            raise ValueError("Negative mass squared detected, invalid calculation.")
        return np.sqrt(mass_squared)
    except ValueError as e:
        print(f"Error in invariant mass calculation: {e}")
        return np.nan

# Apply the invariant mass calculation to the dataset
data['M_New'] = np.vectorize(calculate_invariant_mass)(
    data['E1'], data['E2'], data['px1'], data['px2'], data['py1'], data['py2'], data['pz1'], data['pz2']
)

# Visualize the results

# Scatter plot of M_New vs M (Original mass)
plt.scatter(x=data['M_New'], y=data['M'])
plt.xlabel('Calculated Invariant Mass (M_New)')
plt.ylabel('Original Invariant Mass (M)')
plt.title('M_New vs M')
plt.show()

# Histogram of original mass (M)
plt.hist(data['M'], bins=500, color='mediumorchid', edgecolor='mediumorchid')
plt.xlabel("Invariant Mass (M)")
plt.ylabel("Frequency")
plt.title("Distribution of Original Invariant Mass")
plt.grid(True)
plt.show()

# Histogram of calculated invariant mass (M_New)
plt.hist(data['M_New'], bins=500, color='blue', edgecolor='blue')
plt.xlabel("Calculated Invariant Mass (M_New)")
plt.ylabel("Frequency")
plt.title("Distribution of Calculated Invariant Mass")
plt.grid(True)
plt.show()

# Remove rows with missing values in the calculated invariant mass
clean_data = data.dropna(subset=['M_New']) #imp

# Histogram for cleaned data
plt.hist(clean_data['M_New'], bins=500, color='dodgerblue', edgecolor='dodgerblue')
plt.xlabel("M_New (Cleaned Data)")
plt.ylabel("Frequency")
plt.title("Histogram of Cleaned M_New Values")
plt.grid(True)
plt.show()

# Compare the original and calculated invariant mass distributions

plt.hist(data['M'].dropna(), bins=100, alpha=0.5, label='Original M', color='orchid', edgecolor='black')
plt.hist(clean_data['M_New'], bins=100, alpha=0.5, label='Calculated M_New', color='dodgerblue', edgecolor='black')
plt.xlabel("Invariant Mass")
plt.ylabel("Frequency")
plt.title("Comparison of M vs M_New")
plt.legend()
plt.grid(True)
plt.show()

# Calculate the difference between the original and calculated invariant mass
data['delta_M'] = data['M'] - data['M_New']

# Histogram of differences (delta_M)
plt.hist(data['delta_M'].dropna(), bins=500, color='crimson', edgecolor='crimson')
plt.xlabel("Difference Between M and M_New (delta_M)")
plt.ylabel("Frequency")
plt.title("Distribution of Differences Between M and M_New")
plt.grid(True)
plt.show()

# Identify and print outliers keeping threshold = 1
outliers = data[np.abs(data['delta_M']) > 1]
print(f"{len(outliers)} significant outliers found.")
print(outliers[['M', 'M_New', 'delta_M']].head())

# Scatter plot of Energy (E1) vs. Difference in Invariant Mass (delta_M)
plt.scatter(data['E1'], data['delta_M'], alpha=0.5)
plt.xlabel("Energy (E1)")
plt.ylabel("Difference in Invariant Mass (delta_M)")
plt.title("Energy vs. Difference in Invariant Mass")
plt.grid(True)
plt.show()