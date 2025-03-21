import numpy as np
from sklearn.svm import SVC

#Here, I'm going to mock an example of input with 4 features and output with 10 labels. Alternatively, you can provide your own real, preprocessed, and ready dataset. However, for this demonstration, I will mimic the dataset of technical debt as explained in the README.
# Number of samples, input features, and output labels
n_samples = 1000  # Dataset size
n_features = 4    # Input features
n_labels = 10     # Output labels

# Generate random input data (features)
X = np.random.rand(n_samples, n_features)

# Generate random output labels (binary multi-label)
Y = np.random.randint(2, size=(n_samples, n_labels))


def calculate_margin(X, Y, label_order):
    """
    Calculate the margin for a given label order.
    :param X: Input features (n_samples x n_features)
    :param Y: Output labels (n_samples x n_labels)
    :param label_order: List of labels in the current order
    :return: List of margins for each label in the order
    """
    margins = []
    for i, label in enumerate(label_order):
        # Train an SVM for the current label
        svm = SVC(kernel='linear')
        svm.fit(X, Y[:, label])
        
        # Compute the margin (1 / ||w||^2)
        w = svm.coef_[0]
        margin = 1 / np.linalg.norm(w) ** 2
        margins.append(margin)
        
        # Update X with the current label as an additional feature
        X = np.hstack((X, Y[:, label].reshape(-1, 1)))
    
    return margins



def gmlo_dp(X, Y, n_labels):
    """
    Find the optimal order of labels using Dynamic Programming.
    :param X: Input features (n_samples x n_features)
    :param Y: Output labels (n_samples x n_labels)
    :param n_labels: Number of labels
    :return: Optimal order of labels and the corresponding cost
    """
    # Initialize the DP table
    dp = {}  # Key: (last_label, subset_size), Value: (cost, subset)
    
    # Step 1: Initialize for subsets of size 1
    for i in range(n_labels):
        margins = calculate_margin(X, Y, [i])
        cost = 1 / margins[0] ** 2
        dp[(i, 1)] = (cost, [i])
    
    # Step 2: Iterate over subset sizes from 2 to n_labels
    for k in range(2, n_labels + 1):
        new_dp = {}
        for (last_label, subset_size), (cost, subset) in dp.items():
            for i in range(n_labels):
                if i not in subset:
                    # Compute the new cost
                    new_subset = subset + [i]
                    margins = calculate_margin(X, Y, new_subset)
                    new_cost = cost + (1 / margins[-1] ** 2)
                    
                    # Update the DP table
                    if (i, k) not in new_dp or new_cost < new_dp[(i, k)][0]:
                        new_dp[(i, k)] = (new_cost, new_subset)
        dp = new_dp
    
    # Step 3: Find the optimal order
    optimal_cost, optimal_order = min(dp.values(), key=lambda x: x[0])
    return optimal_order, optimal_cost



# Find the optimal order of labels
optimal_order, optimal_cost = gmlo_dp(X, Y, n_labels)

# Print the results
print("Optimal Order of Labels:", optimal_order)
print("Optimal Cost:", optimal_cost)



# an example of output might look like this 
#Optimal Order of Labels: [3, 7, 1, 5, 9, 2, 8, 4, 6, 0]
#Optimal Cost: 0.123456789