import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def entropy(p):
    # Calculates the entropy H(X) = -sum( P(x) * log2( P(x) ) )
    p = p[p > 0]  # Ignore 0 probabilities
    return -np.sum(p * np.log2(p))

def joint_entropy(df, col1, col2):
    # Calculates the joint entropy of 2 variables H(X,Y) = -sum( P(x,y) * log2( P(x,y) ) )
    joint_prob = df.groupby([col1, col2]).size().div(len(df))
    return entropy(joint_prob.values)

def mutual_information(df, col1, col2):
    # Calculates the mutual information between two variables I(X;Y) = H(X) + H(Y) - H(X,Y)
    # uses the joint_entropy function above
    return entropy(df[col1].value_counts(normalize=True)) + entropy(df[col2].value_counts(normalize=True)) - joint_entropy(df, col1, col2)

# read it from a spreadsheet
file_path = "iris_dataset_new.xlsx"
excel_file = pd.ExcelFile('iris_dataset_new.xlsx')
print(excel_file.sheet_names) 
data = pd.read_excel(file_path)

# use pandas to create Data Frame
df = pd.DataFrame(data)

H_SepalLengthCm = entropy(df['SepalLengthCm'].value_counts(normalize=True))
H_SepalWidthCm = entropy(df['SepalWidthCm'].value_counts(normalize=True))
H_XY = joint_entropy(df, 'SepalLengthCm', 'SepalWidthCm')
MI_XY = mutual_information(df, 'SepalLengthCm', 'SepalWidthCm')
# Print results
print(f"Entropy of SepalLengthCm: {H_SepalLengthCm}")
print(f"Entropy of SepalWidthCm: {H_SepalWidthCm}")
print(f"Joint Entropy of SepalLengthCm and SepalWidthCm: {H_XY}")
print(f"Mutual Information between SepalLengthCm and SepalWidthCm: {MI_XY}")
#Create results array to hold data
results = []
results.append(('SepalLengthCm', H_XY, MI_XY))
results.append(('SepalWidthCm', H_XY, MI_XY))
# Use Pandas to display it nicely
results_df = pd.DataFrame(
    results, columns=["Feature", "Joint Entropy", "Mutual Information"]
)

# Use matplotlib to display it
plt.figure(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.35

plt.bar(x - width / 2, results_df["Joint Entropy"], width, label="Joint Entropy")
plt.bar(x + width / 2, results_df["Mutual Information"], width, label="Mutual Information")

plt.xticks(x, results_df["Feature"], rotation=45)
plt.ylabel("Value")
plt.title("Entropy and Mutual Information")
plt.legend()
plt.tight_layout()
plt.show()