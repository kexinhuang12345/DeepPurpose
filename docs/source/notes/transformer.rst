Transformer
================================================


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





