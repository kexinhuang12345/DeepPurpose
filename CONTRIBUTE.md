## Instructions on how to include a new encoder

Thank you for your interest in DeepPurpose! As more and more models are coming up, we want to include as much as the models and their pretrained models in our framework. Here we provide step-by-step instructions to do that:


### Step 1: modify the ``utils.py`` file for data and parameter.

For any dataset, we expect each drug is associated with SMILES and each protein with amino acid sequence. However, as different encoders expect different input to the model (e.g., MPNN expects mol graph), we need to first transform it to the expected format. To do that, in the ``utils.py`` file, define a new function ``smiles2xxx`` or ``target2xxx`` which taks a input SMILES/sequence and outputs the encoding format for that single input. 

Then, in the ``encode_drug`` or ``encode_protein`` functions, include a ``elif`` statement to transform all of the data points in the input dataframe using just defined ``smiles2xxx`` or ``target2xxx``. 

For special input formats such as further transformation on the fly, please add a ``elif`` statement to the ``data_process_loader``, ``data_process_DDI_loader``, ``data_process_PPI_loader``, ``data_process_loader_Protein_Prediction``, ``data_process_loader_Protein_Prediction``. You can refer to the examples for CNN in these functions.

Now, in the ``generate_config`` file, add an ``elif`` statement to include all important encoder parameters (e.g. input dimension, model dim and etc.). If your encoder has new parameters that you want the users to specify in the ``model_initialize`` function, you should also add in the function parameter space. If so, please specify the default values.

### Step 2: modify the ``encoders.py`` for model definition

In the ```encoders.py```, define the encoder models. The input of the ``__init__`` in default should contain ``encoding``, which is either 'drug' or 'protein', and ``**config``, which includes all the model parameters defined by users. For the ``forward`` function, we expect to input one feature matrix and output the hidden embedding.

### Step 3: modify the training scripts ``DTI.py, DDI.py, PPI.py, CompoundPred.py, ProteinPred.py``

Finally, we need to modify the training wrappers. Every file has similar structures so we will talk about one file and the rest should follow. In the main class ``__init__`` function, include an ``elif`` statement to define the model based on the definitions in ``encoders.py``.

That's it! You have successfully included your model in DeepPurpose!

### Test and Write in README file

Before you create a pull request, please also test it locally and send kexinhuang@hsph.harvard.edu a test case. Then, you are good to go!
