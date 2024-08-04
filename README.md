# 🌟 MNIST Dataset Analysis: Implementing AdaBoost.M1 for Digit Classification

Welcome to the MNIST Dataset project focusing on digits 0 and 1. This repository contains code and documentation to perform a series of tasks on the MNIST dataset, specifically implementing the AdaBoost.M1 algorithm with decision stumps as weak classifiers. Let's get started! 🚀

## 📁 Dataset

Use the [MNIST dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) for the following tasks, focusing on digits 0 and 1, labeled as -1 and 1.

## 🧩 Task 1: Data Preparation

1. **Divide the Dataset**:
   - Split the train set into train and validation sets.
   - Keep **1000 samples** from each class for validation.
   - Validation set should be used to evaluate the performance of the classifier and must not be used in obtaining the PCA matrix.

## 📊 Task 2: Principal Component Analysis (PCA)

1. **Apply PCA** and reduce the dimension to **p = 5**.
2. Use the training set of the two classes to obtain the PCA matrix.
3. For the remaining tasks, use the reduced dimension dataset.

## 🌲 Task 3: Decision Stump Learning

1. **Learn a Decision Stump** using the training set:
   - For each dimension, find the unique values and sort them in ascending order.
   - Evaluate splits at the midpoint of two consecutive unique values.
   - Find the best split by minimizing weighted misclassification error.
   - Denote this as **h1(x)**.

## 🔄 Task 4: AdaBoost.M1 Implementation

1. **Compute α1** and update weights.
2. Build another tree **h2(x)** using the training set with updated weights.
3. Compute **α2** and update weights.
4. Repeat to grow **300 such stumps**.

## 📈 Task 5: Evaluation

1. After every iteration, find the **accuracy on validation set** and report.
2. Show a **plot of accuracy on validation set vs. number of trees**.
3. Use the tree that gives the highest accuracy to evaluate on the test set.
4. **Report test accuracy**.

## 🤝 Contributing
Feel free to open issues or submit pull requests if you have any suggestions or improvements.

## 📄 License
This project is licensed under the MIT License.
