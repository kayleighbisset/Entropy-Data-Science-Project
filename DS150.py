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

    unique_pairs = list(set(total_pairs))
    joint_entropy = 0
    total_pair_count = len(total_pairs)

    for pair in unique_pairs:
        num_occurrences = total_pairs.count(pair)
        probability_pair = num_occurrences / total_pair_count
        joint_entropy += -probability_pair * math.log(probability_pair, 2)

    # Calculate mutual information
    mutual_information = entropy1 + entropy2 - joint_entropy
    return mutual_information


# Input vectors
vector1 = [4, 3, 2, 1, 3]
vector2 = [4, 3, 2, 1, 3]

# Calculate and print mutual information
i = calculate_mutual_information(vector1, vector2)
print(i)
