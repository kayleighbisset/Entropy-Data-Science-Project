# Entropy-Data-Science-Project

def calculate_mutual_information(vector1, vector2):
    
    entropy1 = 0
    total_elements1 = len(vector1)
    min_value1 = min(vector1)
    max_value1 = max(vector1)
    
    for x in range(min_value1, max_value1 + 1):
        num_occurences = 0
        for value in vector1:
            if x == value:
                num_occurences +=1
                
        probability = (num_occurences / total_elements1)
        entropy1 += -(probability) * (math.log(probability, 2))

    entropy2 = 0
    total_elements2 = len(vector2)
    min_value2 = min(vector2)
    max_value2 = max(vector2)
    
    for x in range(min_value2, max_value2 + 1):
        num_occurences = 0
        for value in vector2:
            if x == value:
                num_occurences +=1
                
        probability = (num_occurences / total_elements2)
        entropy2 += -(probability) * (math.log(probability, 2))

    vectors = [vector1, vector2]
    uniquepairs = []
    totalpairs = []
    joint_entropy = 0
    
    for x in range(len(vector1)):
        pair = f'{vector1[x]}, {vector2[x]}'
        totalpairs.append(pair)
        
    for x in totalpairs:
        if x not in uniquepairs:
            uniquepairs.append(x)
            
    amountpairs = len(totalpairs)
    
    for i in uniquepairs:
        num_occurences = 0
        for pair in totalpairs:
            if i == pair:
                num_occurences +=1
                
        probability_pair = (num_occurences / amountpairs)
        joint_entropy += -(probability_pair) * (math.log(probability_pair, 2))
        
    mutual_information = (entropy1 + entropy2 - joint_entropy)
    return mutual_information
vector1 = [4,3,2,1,3]
vector2 = [4,3,2,1,3]

i = calculate_mutual_information(vector1, vector2)
print(i)
