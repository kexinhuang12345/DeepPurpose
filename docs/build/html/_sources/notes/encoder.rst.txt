Drug/Target Encoder 
================================================


Drug encoding 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================   ===================================================
    Drug Encodings                         Description 
======================   ===================================================
      Morgan                    Extended-Connectivity Fingerprints                    
      Pubchem                 Pubchem Substructure-based Fingerprints                   
      Daylight                      Daylight-type fingerprints 
 rdkit_2d_normalized                Normalized Descriptastorus                   
        CNN                    Convolutional Neural Network on SMILES                    
      CNN_RNN                     A GRU/LSTM on top of a CNN on SMILES                   
     Transformer                    Transformer Encoder on ESPF
       MPNN                    	     Message-passing neural network 
======================   ===================================================


Target encoding 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

====================   ===================================================
  Target Encodings                        Description   
====================   ===================================================
      AAC                       Amino acid composition up to 3-mers      
    PseudoAAC                    Pseudo amino acid composition            
   Conjoint_triad                    Conjoint triad features                  
    Quasi-seq                    Quasi-sequence order descriptor              
      CNN                    Convolutional Neural Network on target seq
    CNN_RNN                   A GRU/LSTM on top of a CNN on target seq             
   Transformer                     Transformer Encoder on ESPF
====================   ===================================================


Encoder Model 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

====================   ===================================================================
  Encoder Model                         Description   
====================   ===================================================================
      CNN                    Convolutional Neural Network on SMILES
    CNN_RNN                   A GRU/LSTM on top of a CNN on SMILES            
   Transformer                     Transformer Encoder on SMILES
     MPNN                     Message Passing Neural Network on Molecular Graph           
      MLP                       MultiLayer Perceptron on fix-dim feature vector                   
====================   ===================================================================









Technical Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


First, we describe the common modules we import in DeepPurpose.



.. code-block:: python

	import torch
	from torch.autograd import Variable
	import torch.nn.functional as F
	from torch import nn 
	import numpy as np
	import pandas as pd	






.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Links of details of various encoders 

   encoders/transformer
   encoders/mpnn
   encoders/cnnrnn
   encoders/cnn 
   encoders/mlp 























