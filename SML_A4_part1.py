#!/usr/bin/env python
# coding: utf-8

# In[22]:


from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import random  # Add import for random module
random.seed(10)

class Node:
    def __init__(self, data_indices, depth):
        self.data_indices = data_indices
        self.depth = depth
        self.left = None
        self.right = None
        self.split_dim = None
        self.split_value = None
        self.label = None

def create_data_matrix( X_train ):
    X = X_train.reshape(-1, 784)
    X = X.T
    return X


def center_data(X, mean=None):
    if mean is None:
        mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean
    return X_centered, mean

def apply_pca(X_centered, p):
    covariance_matrix = np.matmul(X_centered, X_centered.T) / (X_centered.shape[1] - 1)
    V, U = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(V)[::-1]
    U_sorted = U[:, sorted_indices][:, :p]

    Y = U_sorted.T @ X_centered
    return U_sorted, Y

def reconstruct_data(U_sorted, Y):
    X_recon = U_sorted @ Y
    return X_recon

def calculate_mse(X_centered, X_recon):
    mse = np.sum((X_centered - X_recon) ** 2) / X_centered.size
    return mse

def plot_reconstructed_images(X_recon_p, p):
    fig, axes = plt.subplots(10, 5, figsize=(10, 10))
    for i in range(10):
        for j in range(5):
            axes[i, j].imshow(X_recon_p[:, i * 100 + j].reshape(28, 28), cmap='cubehelix_r')
            axes[i, j].axis('off')

    plt.suptitle(f"Reconstructed Images with p={p}")
    plt.show()

def calculate_class_accuracy(y_true, y_pred):

    num_classes = 3
    accuracy_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for true_label, pred_label in zip(y_true, y_pred):
        total_per_class[true_label] += 1
        if true_label == pred_label:
            accuracy_per_class[true_label] += 1

    accuracy_per_class = accuracy_per_class / total_per_class
    return accuracy_per_class



def print_tree(root, level=0, prefix="Root:"):
    if root is not None:
        print(" " * (level * 4) + prefix, root.label)
        if root.left is not None or root.right is not None:
            if root.left is not None:
                print_tree(root.left, level + 1, prefix="L--")
            else:
                print(" " * ((level + 1) * 4) + "L--None")
            if root.right is not None:
                print_tree(root.right, level + 1, prefix="R--")
            else:
                print(" " * ((level + 1) * 4) + "R--None")



def weighted_misclassification_loss(y, weight_matrix):
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    most_frequent_label = np.argmax(counts)
    incorrect_samples = np.sum(weight_matrix[y != most_frequent_label])
    return incorrect_samples,np.sum([y != most_frequent_label])

def find_best_split(X, y, data_indices, weight_matrix):
    n_samples, n_features = X[data_indices].shape
    
    best_loss = float('inf')
    best_split_dim = None
    best_split_value = None
    
    for dim in range(n_features):
        unique_values = np.unique(X[data_indices, dim])
        
        for i in range(len(unique_values) - 1):
            value = (unique_values[i] + unique_values[i + 1]) / 2  # Midpoint split
            left_indices = data_indices[X[data_indices, dim] <= value]
            right_indices = data_indices[X[data_indices, dim] > value]

            loss_left ,l = weighted_misclassification_loss(y[left_indices], weight_matrix[left_indices])
            loss_right ,r= weighted_misclassification_loss(y[right_indices], weight_matrix[right_indices])
            
            # Calculate weighted misclassification loss
            
            loss = ( loss_left + loss_right) / np.sum(weight_matrix)
            
            # print(loss)
            if loss < best_loss:
                # print("Loss ",len(X)-l-r)    
                best_loss = loss
                best_split_dim = dim
                best_split_value = value
                
    return best_split_dim, best_split_value,best_loss
    



def assign_label_for_node(node, y):
    unique_labels, label_counts = np.unique(y[node.data_indices], return_counts=True)
    node.label = unique_labels[np.argmax(label_counts)]
    return node

def check_stopping_criteria(y, data_indices, total_leaf_nodes, max_leaf_nodes):
    if len(np.unique(y[data_indices])) == 1 or (max_leaf_nodes is not None and total_leaf_nodes >= max_leaf_nodes):
        return True
    return False



def grow_tree(X, y, data_indices=None, depth=0, max_depth=1,weight_matrix=None):
    global total_leaf_nodes
    if weight_matrix is None:
        weight_matrix = np.array([1.0/len(X)]*len(X))
    if data_indices is None:
        data_indices = np.arange(X.shape[0])

    n_samples, n_features = X[data_indices].shape

    node= Node(data_indices=data_indices, depth=depth)

    if depth == max_depth or check_stopping_criteria(y, data_indices, total_leaf_nodes, 2):
        assign_label_for_node(node, y)
        total_leaf_nodes += 1
        return node

    best_split_dim, best_split_value,best_loss = find_best_split(X, y, data_indices,weight_matrix)
    
    left_indices = data_indices[X[data_indices, best_split_dim] <= best_split_value]
    right_indices = data_indices[X[data_indices, best_split_dim] > best_split_value]

    node.split_dim = best_split_dim
    node.split_value = best_split_value

    node.left = grow_tree(X, y, left_indices, depth=depth + 1, max_depth=max_depth)
    node.right = grow_tree(X, y, right_indices, depth=depth + 1, max_depth=max_depth)

    return node,best_loss

def predict(x, node):
    if node.label is not None:
        return node.label
    if x[node.split_dim] <= node.split_value:
        return predict(x, node.left)
    else:
        return predict(x, node.right)


import numpy as np

# Load MNIST dataset
mnist_data = np.load(r"D:\Downloads\mnist.npz")
x_train_all, y_train_all = mnist_data['x_train'], mnist_data['y_train']
x_test_all, y_test_all = mnist_data['x_test'], mnist_data['y_test']

# Select classes 0 and 1
selected_train_indices = np.where((y_train_all == 0) | (y_train_all == 1))[0]
selected_test_indices = np.where((y_test_all == 0) | (y_test_all == 1))[0]

x_selected_train = x_train_all[selected_train_indices]
y_selected_train = y_train_all[selected_train_indices]
x_selected_test = x_test_all[selected_test_indices]
y_selected_test = y_test_all[selected_test_indices]

# Sample 1000 samples randomly from each class for validation
num_val_samples_per_class = 1000

# Initialize empty lists to store indices
val_indices = []

# For each class (0 and 1)
for class_label in [0, 1]:
    # Find indices of samples belonging to the current class in the training set
    class_indices_train = np.where(y_selected_train == class_label)[0]

    # Randomly select validation samples
    val_indices.extend(np.random.choice(class_indices_train, size=num_val_samples_per_class, replace=False))

# Convert list to numpy array
val_indices = np.array(val_indices)

# Remove validation samples from the training set
x_train = np.delete(x_selected_train, val_indices, axis=0)
y_train = np.delete(y_selected_train, val_indices)

# Separate validation set
x_val = x_selected_train[val_indices]
y_val = y_selected_train[val_indices]

# Verify shapes
print("Shapes:")
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_val:", x_val.shape)
print("y_val:", y_val.shape)

# x_test should contain all samples from the original x_test_all
x_test = x_selected_test
y_test = y_selected_test

# Verify shapes
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

X = create_data_matrix(x_train)
X_centered,X_train_mean = center_data(X)
p = 5
U_sorted, x_reduced = apply_pca(X_centered, p)
x_reduced = x_reduced.T
print(x_reduced.shape)


X = create_data_matrix(x_val)
X_centered_val,X_val_mean = center_data(X,X_train_mean)
p = 5
x_reduced_val = U_sorted.T @ X_centered_val
x_reduced_val = x_reduced_val.T
print(x_reduced_val.shape)



X = create_data_matrix(x_test)
X_centered_test,X_test_mean = center_data(X,X_train_mean)
p = 5
x_reduced_test = U_sorted.T @ X_centered_test
x_reduced_test = x_reduced_test.T
print(x_reduced_test.shape)





# def ada_boost(X, y, num_iterations):
#     weights = np.ones(len(X),dtype=np.float64) / len(X)  # Initialize weights uniformly
#     classifiers = []  # Store the decision trees
#     alphas = []  # Store the alpha values
#     global total_leaf_nodes
#     for _ in tqdm(range(num_iterations)):
#         # Create a decision tree using the weighted dataset
#         total_leaf_nodes = 0
        
        
#         tree,best_loss = grow_tree(X, y, weight_matrix=weights)
#         print(" unique weights " , np.unique(weights))
#         predictions = np.array([predict(x, tree) for x in X])
        
#         accuracy = np.mean(predictions == y)
#         print(f"Iteration {_+1}: Accuracy = {accuracy}")
        
#         print(np.bincount(predictions))
#         misclassified = np.where(predictions != y)[0]
#         weighted_error = np.sum(weights[misclassified]) / np.sum(weights)
#         print(" weighted_error " ,weighted_error )
        
#         # Calculate alpha
#         alpha = np.log((1 - weighted_error) / weighted_error)
#         print("alpha ",alpha)
#         alphas.append(alpha)
#         classifiers.append((tree,alpha))
        

#         predictions = ada_boost_predict(classifiers, X,y)
#         accuracy = np.mean(predictions == y)
#         print(f"Iteration {_+1}: Accuracy = {accuracy}")

#         # Update weights
#         weights[misclassified] *= np.exp(alpha)

       
#     return classifiers
def ada_boost(X, y, X_val, y_val, num_iterations):
    
    weights = np.ones(len(X), dtype=np.float64) / len(X)  # Initialize weights uniformly
    classifiers = []  # Store the decision trees
    alphas = []  # Store the alpha values
    val_accuracies = []  # Store validation accuracies
    
    global total_leaf_nodes
    for i in tqdm(range(num_iterations)):
        # Create a decision tree using the weighted dataset
        total_leaf_nodes = 0
        tree, best_loss = grow_tree(X, y, weight_matrix=weights)
        print_tree(tree)
        predictions = np.array([predict(x, tree) for x in X])
        accuracy = np.mean(predictions == y)
        print(f"Iteration {i+1}: Training Accuracy  of Tree= {accuracy}")

        misclassified = np.where(predictions != y)[0]
        weighted_error = np.sum(weights[misclassified]) / np.sum(weights)

        # Calculate alpha
        alpha = np.log((1 - weighted_error) / weighted_error)
        alphas.append(alpha)
        classifiers.append((tree, alpha))
        print('alpha ', alpha)
        print('weighted_error ',weighted_error)
        # Update weights
        weights[misclassified] *= np.exp(alpha)

        # Calculate validation accuracy
        val_predictions = ada_boost_predict(classifiers, X_val, y_val)
        val_accuracy = np.mean(val_predictions == y_val)
        val_accuracies.append(val_accuracy)
        print(f"Iteration {i+1}: Validation Accuracy ADA Boost= {val_accuracy}")

    return classifiers, val_accuracies

def ada_boost_predict(classifiers, X,y):
    predictions = np.zeros(len(X))
    
    for tree, alpha in classifiers:
        prediction = np.array([predict(x, tree) for x in X],dtype = np.int64)
        # prediction[prediction == 0] = -1  # Change 0 to -1
        
        
        predictions += (alpha * ((prediction * 2 )- 1))
    
    print(((prediction * 2 )- 1)[980:1020])
    
    # # Final prediction based on the sign
    final_prediction = np.sign(predictions)
    final_prediction = (final_prediction+1)/2
   # print(final_prediction[980:1020])
    
    return final_prediction


# Example usage:
num_iterations = 300
classifiers,val_accuracies = ada_boost(x_reduced, y_train,x_reduced_val,y_val, num_iterations)#

# Example usage:
# predictions = ada_boost_predict(classifiers, x_val)


# In[28]:


import matplotlib.pyplot as plt

def plot_accuracies( val_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Number of Tree')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracies against Number of Tree')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_adaboost(classifiers, X_test, y_test):
    test_predictions = ada_boost_predict(classifiers, X_test, y_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Test Accuracy ADA Boost: {test_accuracy}")

# Example usage
plot_accuracies( val_accuracies)
test_adaboost(classifiers, x_reduced_test, y_test)


# In[24]:


tree, _ = classifiers[np.argmax(val_accuracies)]


# In[30]:


prediction = np.array([predict(x, tree) for x in x_reduced_test],dtype = np.int64)
print("Test Accuracy on Best Tree ",np.mean(prediction == y_test))


# In[27]:


test_predictions = ada_boost_predict(classifiers, x_reduced_test, y_test)
np.mean(test_predictions == y_test)


# In[ ]:




