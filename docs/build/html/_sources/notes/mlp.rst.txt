MLP
======================================================


.. code-block:: python

	class DeepPurpose.models.MLP(nn.Sequential)


Multi-Layer Perceptron

**constructor**

.. code-block:: python

	__init__(self, input_dim, hidden_dim, hidden_dims)

* **input_dim** (int) - dimension of input feature. 
* **hidden_dim** (int) - dimension of hidden layer. 


**Calling functions** implement the feedforward procedure of MLP. 


.. code-block:: python

	forward(self, v)

* **v**



