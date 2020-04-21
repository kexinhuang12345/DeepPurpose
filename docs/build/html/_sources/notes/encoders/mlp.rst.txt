MLP
======================================================


.. code-block:: python

	class DeepPurpose.models.MLP(nn.Sequential)


Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network. 
An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. 

**constructor** create  an instance of MLP

.. code-block:: python

	__init__(self, input_dim, hidden_dim, hidden_dims)

* **input_dim** (int) - dimension of input feature. 
* **hidden_dim** (int) - dimension of hidden layer. 


**Calling functions** implement the feedforward procedure of MLP. 


.. code-block:: python

	forward(self, v)

* **v** (torch.Tensor) - input feature of MLP. 



