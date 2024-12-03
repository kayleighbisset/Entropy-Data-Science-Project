import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import math


def calculate_mutual_information(vector1, vector2):
    # Calculate entropy for vector1
    entropy1 = 0
    total_elements1 = len(vector1)
    min_value1 = min(vector1)
    max_value1 = max(vector1)

    for x in range(min_value1, max_value1 + 1):
        num_occurrences = 0
        for value in vector1:
            if x == value:
                num_occurrences += 1

        if num_occurrences > 0:
            probability = num_occurrences / total_elements1
            entropy1 += -probability * math.log(probability, 2)

    # Calculate entropy for vector2
    entropy2 = 0
    total_elements2 = len(vector2)
    min_value2 = min(vector2)
    max_value2 = max(vector2)

    for x in range(min_value2, max_value2 + 1):
        num_occurrences = 0
        for value in vector2:
            if x == value:
                num_occurrences += 1

        if num_occurrences > 0:
            probability = num_occurrences / total_elements2
            entropy2 += -probability * math.log(probability, 2)

    # Calculate joint entropy
    total_pairs = []
    for x in range(len(vector1)):
        pair = f"{vector1[x]},{vector2[x]}"
        total_pairs.append(pair)

    unique_pairs = set(total_pairs)
    joint_entropy = 0
    total_pair_count = len(total_pairs)

    for pair in unique_pairs:
        num_occurrences = total_pairs.count(pair)
        probability_pair = num_occurrences / total_pair_count
        joint_entropy += -probability_pair * math.log(probability_pair, 2)

    # Calculate mutual information
    mutual_information = entropy1 + entropy2 - joint_entropy
    return mutual_information


# Load the Iris dataset
file_path = '/mnt/data/Iris_Dataset.csv'
iris_data = pd.read_csv(file_path)

# Convert the selected columns to discrete integer values (bins for mutual information calculation)
iris_data['SepalLengthCm'] = pd.cut(iris_data['SepalLengthCm'], bins=10, labels=range(10)).astype(int)
iris_data['SepalWidthCm'] = pd.cut(iris_data['SepalWidthCm'], bins=10, labels=range(10)).astype(int)

# Select two columns for mutual information calculation
vector1 = iris_data['SepalLengthCm'].tolist()
vector2 = iris_data['SepalWidthCm'].tolist()

# Calculate and print mutual information
mi = calculate_mutual_information(vector1, vector2)
print("Mutual Information:",mi)

# Plot the selected columns
plt.scatter(iris_data['SepalLengthCm'], iris_data['SepalWidthCm'], alpha=0.7)
plt.title("Scatter Plot of SepalLengthCm vs SepalWidthCm")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()