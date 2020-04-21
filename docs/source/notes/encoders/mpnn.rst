Message Passing Neural Network (MPNN)
======================================================





.. code-block:: python

	class DeepPurpose.models.MPNN(nn.Sequential)

`Message Passing Neural Network (MPNN) <https://www.biorxiv.org/content/10.1101/684662v3>`_ encode drug in its graph representation. 


**constructor** create MPNN class. 

.. code-block:: python

	__init__(self, mpnn_hidden_size, mpnn_depth) 



* **mpnn_hidden_size** (int) - specify dimension of hidden layer in MPNN, e.g,  mpnn_hidden_size = 256.
* **mpnn_depth** (int) - specify depth of MPNN, e.g.,  mpnn_depth = 3. 


**Calling functions** implement the feedforward procedure of MPNN. 


.. code-block:: python

	forward(self, feature)

* **feature** (tuple of length 5)
	* **todo**  
	* 






