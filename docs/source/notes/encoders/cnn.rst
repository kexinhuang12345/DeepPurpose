CNN
===========================



.. code-block:: python

	class DeepPurpose.models.CNN(nn.Sequential)


`CNN (Convolutional Neural Network) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ (`Source <https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/models.py#L62>`_) can be used to encode both drug and protein on `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_. 
 


**constructor** create an instance of CNN. 

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



