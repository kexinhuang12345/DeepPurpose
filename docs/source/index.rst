.. doct documentation master file, created by

.. image:: logo_deeppurpose_horizontal.png


DeepPurpose documentation!
================================
Welcome! This is the documentation for DeepPurpose, a PyTorch-based deep learning library for Drug Target Interaction.
The Github repository is located `here <https://github.com/kexinhuang12345/DeepPurpose>`_.




1 How to Start
--------------



1.1 Download
^^^^^^^^^^^^^^^^^^^^

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



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.2 Classifier 
^^^^^^^^^^^^^^^^^^^^



.. code-block:: python

	class DeepPurpose.models.Classifier(nn.Sequential)

Classifier is make the prediction for DBTA, it serve as a basic component of class DBTA. 


**constructor**

.. code-block:: python

	__init__(self, model_drug, model_protein, **config) 


* **model_drug** (DeepPurpose.models.XX) - Encoder model for drug. XX can be "transformer", "MPNN", "CNN", "CNN_RNN" ..., 
* **model_protein** (DeepPurpose.models.XX) - Encoder model for protein. XX can be "transformer", "CNN", "CNN_RNN" ..., 
* **config** (kwargs, keyword arguments) - specify the parameter of classifier.  



**Calling functions** implement the feedforward procedure of Classifier.


.. code-block:: python

	forward(self, v_D, v_P)


* **v_D** (many types) - input feature for drug encoder model, like "DeepPurpose.models.transformer", "DeepPurpose.models.CNN", "DeepPurpose.models.CNN_RNN", "DeepPurpose.models.MPNN". 
* **v_P** (many types) - input feature for protein encoder model, like "DeepPurpose.models.transformer", "DeepPurpose.models.CNN", "DeepPurpose.models.CNN_RNN".  



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.3 Drug Target Binding Affinity 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	class DeepPurpose.models.DBTA

Drug Target Binding Affinity (DBTA) model include all learning procedure. 


name of function: **constructor** create DBTA model. 

.. code-block:: python

	__init__(self, **config)


* **config** (kwargs, keyword arguments) - specify the parameter of DBTA.  
	* **drug_encoding** (str) - Encoder mode for drug. It can be "transformer", "MPNN", "CNN", "CNN_RNN" ...,
	* **target_encoding** (str) - Encoder mode for protein. It can be "transformer", "CNN", "CNN_RNN" ..., 
	* **result_folder** (str) - directory that store the learning log/results. 
	* **concrete parameter for encoder model** (repeated)


name of function: **test_** include all the inference procedure. 

.. code-block:: python


	test_(self, data_generator, model, repurposing_mode = False, test = False):

* **data_generator** (iterator) - iterator of torch.utils.data.DataLoader. It can be test data or validation data. 
* **model** (DeepPurpose.models.Classifier) - model of DBTA. 
* **repurposing_mode** (bool) - If repurposing_mode is True, then do repurposing. Otherwise, do compute the accuracy (including AUC score). 
* **test** (bool) - If test is True, plot ROC-AUC and PR-AUC curve. Otherwise, pass. 


name of function: **train** include all the training procedure. 

.. code-block:: python

	train(self, train, val, test = None, verbose = True)

* **train** () - Train data generator
* **val** () - Valid data generator
* **test** () - Test data generator
* **verbose** (bool) - If verbose is True, then print training record every 100 iterations. 


name of function: **predict** 

.. code-block:: python

	 predict(self, df_data)

* **df_data** (pd.DataFrame) - specify data that we need to predict. 


name of function: **save_model** save the well-trained model to specific directory. 

.. code-block:: python

	save_model(self, path_dir) 

* **path_dir** (str, a directory) - the path where model is saved. 



name of function: **load_pretrained** load the well-trained model so that we are able to make inference directly. 

.. code-block:: python

	load_pretrained(self, path)

* **path** (str, a directory) - the path where model is loaded. 



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


name of function: **read_file_training_dataset_bioassay** load bioarray dataset, with one target sequence and multiple drugs and their interaction score with the target. 

.. code-block:: python

	dataset.read_file_training_dataset_bioassay(path)


* **path** (str, a directory) - the path of bioassay dataset file. We have requirement on format of file. First line is target sequence. From 2nd line to n-th line, each line a SMILES and interaction score with target sequence. Example: ./toy_data/AID1706.txt 




name of function: **read_file_training_dataset_drug_target_pairs** load drug target pairs dataset. We have requirement on format of file. Each line contains a drug SMILES and target sequence and their interaction score. Example: ./toy_data/dti.txt 


.. code-block:: python

	dataset.read_file_training_dataset_drug_target_pairs(path)

* **path** (str, a directory) - the path of drug target pairs dataset file. We have requirement on format of file. First line is target sequence. From 2nd line to n-th line, each line a SMILES and interaction score with target sequence.  Example: ./toy_data/AID1706.txt 






name of function: **read_file_virtual_screening_drug_target_pairs** load virtual screening drug target pairs dataset. 
We have requirement on format of file. 
Each line contains a drug SMILES and target sequence. 
Example: ./toy_data/dti.txt 

.. code-block:: python

	dataset.read_file_virtual_screening_drug_target_pairs(path)


* **path** (str, a directory) - the path of virtual screening drug target pairs dataset file. 



name of function: **read_file_repurposing_library** load drug repurposing dataset. 
We have requirement on format of file. 
Each line contains a drug SMILES and its name.  
Example: ./toy_data/??


.. code-block:: python

	dataset.read_file_repurposing_library(path)


* **path** (str, a directory) - the path of drug repurposing dataset file. 





name of function: **read_file_target_sequence** load drug repurposing dataset. 
We have requirement on format of file.
The file only have one line. 
The line contains target name and target sequence. 
Example: ./toy_data/??

.. code-block:: python

	dataset.read_file_target_sequence(path)

* **path** (str, a directory) - the path of target sequence dataset file. 


































