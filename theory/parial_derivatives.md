The most important aspect of back propagation is partial derivatives

In the framework we create an auto differential engine which handles the recursive logic for going back through the computation tree

For this tree to be created we must first calculate the derivatives for operations which we will use

$$$
z = x + y
\frac{\partial z}{\partial x} = 1
\frac{\partial z}{\partial y} = 1
$$$

$$$
z = x - y
\frac{\partial z}{\partial x} = 1
\frac{\partial z}{\partial y} = -1
$$$

$$$
z = x * y
\frac{\partial z}{\partial x} = y
\frac{\partial z}{\partial y} = x
$$$

$$$
z = x / y
\frac{\partial z}{\partial x} = 1/y  
\frac{\partial z}{\partial y} = \frac{-x}{y^2}
$$$
