Linear regression is the process of fitting the best linear equation to a set of input data
It has a closed solution which means that it can be computed in one pass and does not need iterative approximation

There are three types
Singular linear regression       - Fitting a singular input to a singular output y = mx + c
Multivariable linear regression  - Fitting multiple inputs to a singular output y = m_1*x_1 + m_2*x_2 + ... + m_n*x_n + c
Multivariate Linear regression - Fitting multiple inputs (n) to multiple outputs (p)
y_1 = m_11 x_1 + m_12 + ... + m_1n +c_1
y_2 =  m_21 x_1 + m_22 + ... + m_2n +c_2
y_p =  m_p1 x_1 + m_p2 + ... + m_pn +c_p

The general algorithm to do this all at once is to construct a matrix of inputs (features) and outputs
Call X the input features that has shape (nxm) where n is the number of samples and m are the features
Call Y the output matrix that has shape (nxp) where n is the number of samples and p are the outputs

We first create the augmentation of X which adding an extra column to x where all the values are initialized with 1
X_aug = (nx(m+1))

The reason why we add the ones is because in the resultant matrix the values of the 1's will be the bias. By adding the ones we can calcualte the bias along with the weights instead of seperatly calculating it afterwards

The weights is calculated by the formula:

W_aug = (X_aug @ X_aug^T)^-1 @ X_aug^T @ Y

(X_aug @ X_aug^T)^-1 @ X_aug^T - this is known as the moore-penrose pseudoinverse and is denoted by X_aug^{+}

Note that in the situation where X_aug @ X_aug^T is invertible the regular inverse matrix can be used but the main reason of using the pseudoinverse is that it outputs the regular inverse for the situations where the input is invertible but also is defined for matrices that are not invertible (and was designed to find an optimal solution to least squares)

This is where we get deep into mathematical and computation theory.

To calculate this inverse we need to do a process called SVD - single value decomposition
This is a method of splitting a matrix into three seperate matrices which are simplier and easy to work with. This all sounds great until i found out it is really hard to implement and understand all of the maths.

SVD is primarrily based on computing eigen values and from those calculating the left and right eigen vectors. Which this is quite hard to implement since you need to first implement things such as Guasien-elimination (row reduction), eigen value calculation

And SVD has many different implementation methods which vary in complexity and ease of implementation

THIS IS A BAD EXPLANATION

The general output of SVD are U \Sigma V^T
The best description i found was that
U is a representation of each row
\Sigma is a representation of how important each column is
V is a representation of how rows are used in the original data

U are the left eigen vectors
\Sigma is the singlar values in a diagonal matrix (sqrt of the eigen values)
Vare the right eigen vectors
