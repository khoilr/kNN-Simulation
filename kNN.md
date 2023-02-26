# K-Nearest Neighbor

## Introduction

K-nearest neighbor (kNN) is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). kNN has been used in statistical estimation and pattern recognition already in the beginning of 1970's as a non-parametric technique. However, it was not until the machine learning boom in the late 1990's that kNN became a popular algorithm for machine learning.

## Why do we need a kNN algorithm?

kNN is a non-parametric method used for classification and regression. Non-parametric means that it does not make any assumptions on the underlying data. kNN is widely used in industry because it is simple to implement, versatile, and effective. The algorithm is considered to be a lazy learner since it does not learn a discriminative function from the training data but memorizes the training dataset instead. However, the algorithm should not be confused with instance-based learning or case-based reasoning (CBR) since kNN does not use any generalization, reasoning, or inference mechanism.

## How does the kNN algorithm work?

The kNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

The kNN algorithm uses 'feature similarity' to predict the values of new data points. This means that the new data points are classified by a majority vote of its neighbors, with the data points being assigned to the class most common among its k nearest neighbors measured by a distance function (e.g., Euclidean distance).

The 'k' in kNN is a parameter that refers to the number of nearest neighbors to include in the majority vote. In other words, it is the number of nearest neighbors that is used for classification.

The following figure shows the classification of a new data point based on the k nearest neighbors. The new data point is assigned to the class of the majority of its k nearest neighbors.

![kNN](https://miro.medium.com/max/1400/0*34SajbTO2C5Lvigs.png)

### Step by step explanation

1. Load the data
2. Initialize k to your chosen number of neighbors
3. For each example in the data
    1. Calculate the distance between the query example and the current example from the data.
    2. Add the distance and the index of the example to an ordered collection
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
5. Pick the first k entries from the sorted collection
6. Get the labels of the selected k entries
7. If regression, return the mean of the k labels
8. If classification, return the mode of the k labels

## How to choose the value of k?

The value of k is usually odd. In case of tie, the algorithm chooses the smaller class. Choosing a small k means that noise will have a higher influence on the result and choosing a large k means that the algorithm is less sensitive to noise. A good way to choose k is to try different values and see which one gives the best results.

## Calculating the distance

The distance can be calculated using different distance metrics, such as Euclidean distance, Manhattan distance, Minkowski distance, Hamming distance, and Chebyshev distance. The most commonly used distance metric is Euclidean distance.

### Euclidean distance

The Euclidean distance between two points p and q is the length of the line segment connecting them. The Euclidean distance is given by the following equation:

$$ d(p,q) = \sqrt{(q_1-p_1)^2 + (q_2-p_2)^2 + \cdots + (q_n-p_n)^2} $$

Illustration: ![Euclidean distance](https://miro.medium.com/max/562/1*hUrrJWqXysnlF4zMbNrKVg.png)

### Manhattan distance

The Manhattan distance between two points p and q is the sum of the absolute differences of their Cartesian coordinates. The Manhattan distance is given by the following equation:

$$ d(p,q) = |q_1-p_1| + |q_2-p_2| + \cdots + |q_n-p_n| $$

![Manhattan distance](https://miro.medium.com/max/564/1*nSBd4Q8nA9zo_8iVUHa69w.png)

### Minkowski distance

The Minkowski distance between two points p and q is given by the following equation:

$$ d(p,q) = \sqrt[p]{\sum_{i=1}^n |q_i-p_i|^p} $$

![Minkowski distance](https://miro.medium.com/max/564/1*bg4hk_F5NVscDFNwSy1cWw.png)

### Hamming distance

The Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different. The Hamming distance is given by the following equation:

$$ d(p,q) = \sum_{i=1}^n I(p_i \neq q_i) $$

![Hamming distance](https://miro.medium.com/max/564/1*J27IH7DmKuf71YP4qKo2kw.png)

### Chebyshev distance

The Chebyshev distance between two points p and q is the maximum of their differences along any coordinate dimension. The Chebyshev distance is given by the following equation:

$$ d(p,q) = \max_{i=1}^n |q_i-p_i| $$

![Chebyshev distance](https://miro.medium.com/max/562/1*v7_aLyWp8fFuIZ9H0WWfjQ.png)

## Advantages of kNN

1. Simple to implement
2. Versatile: Can be used for classification, regression, and search
3. Training is trivial
4. Works with any number of classes
5. Effective if the training data is large

## Disadvantages of kNN

1. Computationally expensive: Since the algorithm stores all of the training data, prediction stage might be slow with a large training dataset.
2. High memory requirement
3. Prediction stage might be slow with a large testing dataset
4. Does not work well with high dimensional data
5. Categorical features do not have any order between them. Therefore, one-hot encoding is required for categorical features before using this algorithm.

## Applications of kNN

1. Recommender systems
2. Handwriting recognition
3. Image classification
4. Object recognition
5. Gene expression analysis
6. Bioinformatics
7. Text categorization
8. Search engines
9. Medical diagnosis
10. Face recognition
11. Speech recognition
12. Time series prediction
