Drug/Target Encoder 
========================




.. code-block:: python

 DeepPurpose.models.transformer(nn.Sequential)

`Transformer <https://arxiv.org/pdf/1908.06760.pdf>`_ can be used to encode both drug and protein on `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_. 

name of function: **constructor** create Transformer. 

.. code-block:: python

	__init__(self, encoding, **config)


* **encoding** (string, "drug" or "protein") - specify input type of the model, "drug" or "protein". 

* **config** (kwargs, keyword arguments) - specify the parameter of transformer. The keys include 
	* transformer_dropout_rate (float) - dropout rate of transformer. 
	* input_dim_drug (int) - input dimension when encoding drug. 
	* transformer_emb_size_drug (int) - dimension of embedding in input layer when encoding drug.
	* transformer_n_layer_drug (int) - number of layers in transformer when encoding drug.
	* **todo** 



**Calling functions** implement the feedforward procedure of MPNN. 

.. code-block:: python

	forward(self, v)

* **v** (tuple of length 2) - input feature of transformer. v[0] (np.array) is index of atoms. v[1] (np.array) is the corresponding mask. 









~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~









.. code-block:: python

	class DeepPurpose.models.CNN(nn.Sequential)


`CNN (Convolutional Neural Network) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ can be used to encode both drug and protein on `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_. 
 


**constructor** create CNN. 

.. code-block:: python

	__init__(self, encoding, **config)


* **encoding** (string, "drug" or "protein") - specify input type of model, "drug" or "protein". 

* **config** (kwargs, keyword arguments) - specify the parameter of CNN. The keys include 
	* cnn_drug_filters (list, each element is int) - specify the size of filter when encoding drug, e.g., cnn_drug_filters = [32,64,96]. 
	* cnn_drug_kernels (list, each element is int) - specify the size of kernel when encoding drug, e.g., cnn_drug_kernels = [4,6,8]. 
	* hidden_dim_drug (int) - specify the hidden dimension when encoding drug, e.g., hidden_dim_drug = 256. 
	* cnn_target_filters (list, each element is int) - specify the size of filter when encoding protein, e.g, cnn_target_filters = [32,64,96].
	* cnn_target_kernels (list, each element is int) - specify the size of kernel when encoding protein, e.g, cnn_target_kernels = [4,8,12].
	* hidden_dim_protein (int) - specify the hidden dimension when encoding protein, e.g., hidden_dim_protein = 256. 



**Calling functions** implement the feedforward procedure of CNN. 

.. code-block:: python

	forward(self, v)


* **v** (torch.Tensor) - input feature of CNN. 


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

	class DeepPurpose.models.CNN_RNN(nn.Sequential)

CNN_RNN means a GRU/LSTM on top of a CNN on `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_. 



**constructor** create CNN_RNN

.. code-block:: python

	__init__(self, encoding, **config)


* **encoding** (string, "drug" or "protein") - specify input type, "drug" or "protein". 
* **config** (kwargs, keyword arguments) - specify the parameter of transformer. The keys include 
	* cnn_drug_filters (list, each element is int) - specify the size of filter when encoding drug, e.g., cnn_drug_filters = [32,64,96]. 
	* cnn_drug_kernels (list, each element is int) - specify the size of kernel when encoding drug, e.g., cnn_drug_kernels = [4,6,8]. 
	* rnn_drug_hid_dim (int) - specify the hidden dimension of RNN when encoding drug, e.g., rnn_drug_hid_dim = 64.
	* rnn_drug_n_layers (int) - specify number of layer in RNN when encoding drug, .e.g, rnn_drug_n_layers = 2.
	* rnn_drug_bidirectional (bool) - specify if RNN is bidirectional when encoding drug, .e.g, rnn_drug_bidirectional = True.
	* hidden_dim_drug (int) - specify the hidden dimension when encoding drug, e.g., hidden_dim_drug = 256. 
	* cnn_target_filters (list, each element is int) - specify the size of filter when encoding protein, e.g, cnn_target_filters = [32,64,96].
	* cnn_target_kernels (list, each element is int) - specify the size of kernel when encoding protein, e.g, cnn_target_kernels = [4,8,12].
	* hidden_dim_protein (int) - specify the hidden dimension when encoding protein, e.g., hidden_dim_protein = 256. 
	* rnn_target_hid_dim (int) - specify hidden dimension of RNN when encoding protein, e.g., rnn_target_hid_dim = 64.  
	* rnn_target_n_layers (int) - specify the number of layer in RNN when encoding protein, e.g., rnn_target_n_layers = 2. 
	* rnn_target_bidirectional (bool) - specify if RNN is bidirectional when encoding protein, e.g., rnn_target_bidirectional = True


**Calling functions** implement the feedforward procedure of CNN_RNN. 


.. code-block:: python

	forward(self, v)


* **v** (torch.Tensor) - input feature of CNN_RNN. 


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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




