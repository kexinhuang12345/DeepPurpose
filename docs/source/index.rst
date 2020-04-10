.. doct documentation master file, created by


DeepPurpose documentation!
================================
Welcome! This is the documentation for DeepPurpose, a PyTorch-based deep learning library for Drug Target Interaction.
The Github repository is located `here <https://github.com/kexinhuang12345/DeepPurpose>`_.


1 How to Start
--------------



1.1 Download
^^^^^^^^^^^^

.. code-block:: bash

   $ git clone git@github.com:kexinhuang12345/DeepPurpose.git
   $ ###  Download code repository 
   $
   $
   $ cd DeepPurpose
   $ ### Change directory to DeepPurpose 


1.2 Installation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   $ conda env create -f env.yml  
   $ ## Build virtual environment with all packages installed using conda
   $ 
   $ conda activate DeepPurpose
   $ ##  Activate conda environment
   $
   $
   $ conda deactivate ### exit




2 Run
--------------------------






3 Documentation
--------------------------



3.1 Encoder Models
^^^^^^^^^^^^^^^^^^^^



**environment** 

.. code-block:: python

	import torch
	from torch.autograd import Variable
	import torch.nn.functional as F
	from torch.utils import data
	from torch.utils.data import SequentialSampler
	from torch import nn 

	from tqdm import tqdm
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	from time import time
	from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score
	from lifelines.utils import concordance_index
	from scipy.stats import pearsonr
	import pickle 
	import copy
	from prettytable import PrettyTable
	import scikitplot as skplt


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
	
	DeepPurpose.models.transformer(nn.Sequential)

`Transformer <https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/>`_ can be used to encode both drug and protein. 


.. code-block:: python

	__init__(self, encoding, **config)

**constructor**

* **encoding** (string, "drug" or "protein") - specify input type, "drug" or "protein". 

* **config** (kwargs, keyword arguments) - specify the parameter of transformer. 





**Calling functions** 

.. code-block:: python

	forward(self, v)

* **v** (torch.Tensor) - input feature of transformer. 









~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~









.. code-block:: python

	class DeepPurpose.models.CNN(nn.Sequential)


`CNN (Convolutional Neural Network) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ can be used to encode drug. 


.. code-block:: python

	__init__(self, encoding, **config)

**constructor**

* **encoding** (string, "drug" or "protein") - specify input type, "drug" or "protein". 

* **config** (kwargs, keyword arguments) - specify the parameter of transformer. 


**Calling functions** 

.. code-block:: python

	forward(self, v)


* **v** (torch.Tensor) - input feature of CNN. 


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

	class DeepPurpose.models.CNN_RNN(nn.Sequential)

CNN+RNN is 


**constructor**

.. code-block:: python

	__init__(self, encoding, **config)


* **encoding** (string, "drug" or "protein") - specify input type, "drug" or "protein". 
* **config** (kwargs, keyword arguments) - specify the parameter of transformer. 

**Calling functions** 


.. code-block:: python

	forward(self, v)


* **v**


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	class DeepPurpose.models.MLP(nn.Sequential)


Multi-Layer Perceptron

**constructor**

.. code-block:: python

	__init__(self, input_dim, hidden_dim, hidden_dims)

* **input_dim** (int) - dimension of input feature. 
* **hidden_dim** (int) - dimension of hidden layer. 


**Calling functions** 


.. code-block:: python

	forward(self, v)

* **v**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code-block:: python

	class DeepPurpose.models.MPNN(nn.Sequential)

Message Passing Neural Network 

**constructor**

.. code-block:: python

	__init__(self, mpnn_hidden_size, mpnn_depth) 



* **mpnn_hidden_size** (int) - dimension of hidden layer in MPNN. 
* **mpnn_depth** (int) - depth of MPNN. 


**Calling functions** 


.. code-block:: python

	forward(self, feature)

* **feature** (tuple of length 5)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.2 Classifier 
^^^^^^^^^^^^^^^^^^^^



.. code-block:: python

	class DeepPurpose.models.Classifier(nn.Sequential)

Classifier 


**constructor**

.. code-block:: python

	__init__(self, model_drug, model_protein, **config) 


* **model_drug** (DeepPurpose.models.XX) - Encoder model for drug. XX can be "transformer", "MPNN", "CNN", "CNN_RNN" ..., 
* **model_protein** (DeepPurpose.models.XX) - Encoder model for protein. XX can be "transformer", "CNN", "CNN_RNN" ..., 
* **config** (kwargs, keyword arguments) - specify the parameter of classifier.  



**Calling functions** 


.. code-block:: python

	forward(self, v_D, v_P)


* **v_D** 
* **v_P** 



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.3 Drug Target Binding Affinity 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	class DeepPurpose.models.DBTA

Drug Target Binding Affinity (DBTA) model include all learning procedure. 


name of function: **constructor** create DBTA model. 

.. code-block:: python

	__init__(self, **config)


* **config** (kwargs, keyword arguments) - specify the parameter of classifier.  



name of function: **test_** include all the inference procedure. 

.. code-block:: python


	test_(self, data_generator, model, repurposing_mode = False, test = False):

* **data_generator** - 
* **model** - 
* **repurposing_mode** (bool) - 
* **test** (bool) - 


name of function: **train** include all the training procedure. 

.. code-block:: python

	train(self, train, val, test = None, verbose = True)

* **train** () - 
* **val** () - 
* **test** () - 
* **verbose** (bool) - 


name of function: **predict** 

.. code-block:: python

	 predict(self, df_data)

* **df_data** () - 


name of function: **save_model** save the well-trained model to specific directory. 

.. code-block:: python

	save_model(self, path_dir) 

* **path_dir** (str, a directory) - the path where model is saved. 



name of function: **load_pretrained** load the well-trained model so that we are able to make inference directly. 

.. code-block:: python

	load_pretrained(self, path)

* **path** (str, a directory) - the path where model is loaded. 
















































