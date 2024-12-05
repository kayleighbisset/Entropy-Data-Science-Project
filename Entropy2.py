import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def entropy(p):
    # Calculates the entropy H(X) = -sum(P(x) * log2(P(x)))
    p = p[p > 0]  # Ignore 0 probabilities
    return -np.sum(p * np.log2(p))

def joint_entropy(df, col1, col2):
    # Calculates the joint entropy of 2 variables H(X,Y) = -sum(P(x,y) * log2(P(x,y)))
    joint_prob = df.groupby([col1, col2]).size().div(len(df))
    return entropy(joint_prob.values)

def mutual_information(df, col1, col2):
    # Calculates the mutual information between two variables
    H_X = entropy(df[col1].value_counts(normalize=True))
    H_Y = entropy(df[col2].value_counts(normalize=True))
    H_XY = joint_entropy(df, col1, col2)
    return H_X + H_Y - H_XY

# Load data
file_path = "iris_dataset_new.xlsx"
data = pd.read_excel(file_path)
df = pd.DataFrame(data)

# Bin continuous variables into bins & adjust number of bins based on how many we want
num_bins = 10
df['SepalLengthCm_binned'] = pd.cut(df['SepalLengthCm'], bins=num_bins, labels=False)
df['SepalWidthCm_binned'] = pd.cut(df['SepalWidthCm'], bins=num_bins, labels=False)

# Display the number of bins for each feature
sepal_length_bins = df['SepalLengthCm_binned'].nunique()
sepal_width_bins = df['SepalWidthCm_binned'].nunique()

print(f"Number of bins for SepalLengthCm: {sepal_length_bins}")
print(f"Number of bins for SepalWidthCm: {sepal_width_bins}")

# Calculate entropy and mutual information
H_SepalLengthCm = entropy(df['SepalLengthCm_binned'].value_counts(normalize=True))
H_SepalWidthCm = entropy(df['SepalWidthCm_binned'].value_counts(normalize=True))
H_XY = joint_entropy(df, 'SepalLengthCm_binned', 'SepalWidthCm_binned')
MI_XY = mutual_information(df, 'SepalLengthCm_binned', 'SepalWidthCm_binned')

# Print results
print(f"Entropy of SepalLengthCm (binned): {H_SepalLengthCm}")
print(f"Entropy of SepalWidthCm (binned): {H_SepalWidthCm}")
print(f"Joint Entropy of SepalLengthCm and SepalWidthCm (binned): {H_XY}")
print(f"Mutual Information between SepalLengthCm and SepalWidthCm (binned): {MI_XY}")

# Create results array to hold data
results = [
    ('SepalLengthCm', H_SepalLengthCm, MI_XY),
    ('SepalWidthCm', H_SepalWidthCm, MI_XY)
]

# Use Pandas to display it
results_df = pd.DataFrame(
    results, columns=["Feature", "Entropy", "Mutual Information"]
)

# Use matplotlib to display chart
plt.figure(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.35

plt.bar(x - width / 2, results_df["Entropy"], width, label="Entropy")
plt.bar(x + width / 2, results_df["Mutual Information"], width, label="Mutual Information")

plt.xticks(x, results_df["Feature"], rotation=45)
plt.ylabel("Value")
plt.title("Entropy and Mutual Information with Binning")
plt.legend()
plt.tight_layout()
plt.show()
