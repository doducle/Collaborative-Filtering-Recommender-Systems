# Collaborative Filtering Recommender Systems

### Overview

This notebook implements a simple collaborative filtering recommender system using gradient descent and tensorflow. It is based on an exercise from the Coursera course: Machine Learning Specialization from Andrew Ng.



### Movie ratings dataset

The data set is derived from the [MovieLens "ml-latest-small"](https://grouplens.org/datasets/movielens/latest/) dataset.   
[F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>]

The original dataset has  9000 movies rated by 600 users. The dataset has been reduced in size to focus on movies from the years since 2000. This dataset consists of ratings on a scale of 0.5 to 5 in 0.5 step increments. The reduced dataset has 443 users, and 4778 movies. 



### Collaborative Filtering

We will implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings.
To train the model we need data that contains a set of movies and a set of users who have rated those movies. This can look like this:


|               | u<sub>1</sub> | u<sub>2</sub> | u<sub>3</sub> | u<sub>4</sub> |
| :------------ | ------------- | ------------- | ------------- | ------------- |
| m<sub>1</sub> | 5             | 4             |               | 2             |
| m<sub>2</sub> |               | 4             | 5             |               |
| m<sub>3</sub> | 1             | 2             |               | 4             |
| m<sub>4</sub> |               |               | 3             |               |
| m<sub>5</sub> | 5             | 3             | 3             | 5             |

The cells can be empty, as users only rate a few movies. In the above example, User 1 gave movie 1 a rating of 5, but did not rate movie 2.

The goal of a collaborative filtering recommender system is to learn two vectors: For each user, a 'parameter vector' $w^{user}$ that embodies the movie tastes of a user. For each movie, a feature vector $x_{movie}$ of the same size which embodies some description of the movie. The dot product of the two vectors plus the bias term should produce an estimate of the rating the user might give to that movie. So one training example from the table above would be: 
<p style="text-align: center;">$w^{(1)} \cdot x^{(3)} + b^{(1)} = 1$.</p>
The vectors $w^{(i)}, x^{(j)}$ must have the same length and it is a parameter of the model that is chosen before training. Let's say we choose the length as the number of different movie genres. One possible interpretation then could be, that each entry in the user vector $w^{user}$ represents how much the user likes that genre and the respective entry in the movie vector $x^{movie}$ describes how much that movie fits that genre.



### Implementation

The collaborative filtering cost function is given by
$$J({\mathbf{x}^{(0)},...,\mathbf{x}^{(n_m-1)},\mathbf{w}^{(0)},b^{(0)},...,\mathbf{w}^{(n_u-1)},b^{(n_u-1)}})= \left[ \frac{1}{2}\sum_{j=0}^{n_u-1} \sum_{i=0}^{n_m-1}r(i,j)*(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)})^2 \right]
+ \underbrace{\left[
\frac{\lambda}{2}
\sum_{j=0}^{n_u-1}\sum_{k=0}^{n-1}(\mathbf{w}^{(j)}_k)^2
+ \frac{\lambda}{2}\sum_{i=0}^{n_m-1}\sum_{k=0}^{n-1}(\mathbf{x}_k^{(i)})^2
\right].}_{regularization}
\tag{1}$$

The parameters  $\mathbf{X}$, $\mathbf{W}$, and $\mathbf{b}$ will be learned by a simple Gradient Descent.



### Recommendation

To predict the rating of movie $i$ for user $j$, you compute $\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}$. This can be computed for all ratings using matrix multiplication. As recommendations you can then output e.g. the movies with best predicted ratings for that user.