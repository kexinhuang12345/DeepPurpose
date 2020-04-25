CNN+RNN
===========================

.. code-block:: python

	class DeepPurpose.models.CNN_RNN(nn.Sequential)

CNN_RNN (`Source <https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/models.py#L109>`_) means a GRU/LSTM on top of a CNN on `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_. 



**constructor** create  an instance of CNN_RNN

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



