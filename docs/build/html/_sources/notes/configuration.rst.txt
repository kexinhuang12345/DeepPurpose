Configuration
========================



**generate_config** generate all the configuration that can be used in learning and inference. 


.. code-block:: python

 utils.generate_config(
 	drug_encoding, 
 	target_encoding, 
	result_folder = "./result/",
	input_dim_drug = 1024, 
	input_dim_protein = 8420,
	hidden_dim_drug = 256, 
	hidden_dim_protein = 256,
	cls_hidden_dims = [1024, 1024, 512],
	mlp_hidden_dims_drug = [1024, 256, 64],
	mlp_hidden_dims_target = [1024, 256, 64],
	batch_size = 256,
	train_epoch = 10,
	test_every_X_epoch = 20,
	LR = 1e-4,
	transformer_emb_size_drug = 128,
	transformer_intermediate_size_drug = 512,
	transformer_num_attention_heads_drug = 8,
	transformer_n_layer_drug = 8,
	transformer_emb_size_target = 128,
	transformer_intermediate_size_target = 512,
	transformer_num_attention_heads_target = 8,
	transformer_n_layer_target = 4,
	transformer_dropout_rate = 0.1,
	transformer_attention_probs_dropout = 0.1,
	transformer_hidden_dropout_rate = 0.1,
	mpnn_hidden_size = 50,
	mpnn_depth = 3,
	cnn_drug_filters = [32,64,96],
	cnn_drug_kernels = [4,6,8],
	cnn_target_filters = [32,64,96],
	cnn_target_kernels = [4,8,12],
	rnn_Use_GRU_LSTM_drug = 'GRU',
	rnn_drug_hid_dim = 64,
	rnn_drug_n_layers = 2,
	rnn_drug_bidirectional = True,
	rnn_Use_GRU_LSTM_target = 'GRU',
	rnn_target_hid_dim = 64,
	rnn_target_n_layers = 2,
	rnn_target_bidirectional = True
	)


* **drug_encoding** (str) - Encoder mode for drug. It can be "transformer", "MPNN", "CNN", "CNN_RNN" ...,
* **target_encoding** (str) - Encoder mode for protein. It can be "transformer", "CNN", "CNN_RNN" ..., 
* **input_dim_drug** (int) - Dimension of input drug feature. 
* **input_dim_protein** (int) - Dimension of input protein feature. 
* **hidden_dim_drug** (int) - Dimension of hidden layer of drug feature. 
* **hidden_dim_protein** (int) - Dimension of hidden layer of protein feature. 
* **batch_size** (int) - batch size
* **train_epoch** (int) - training epoch
* **test_every_X_epoch** (int) - test every X epochs
* **LR** (float) - Learning rate. 
* **cls_hidden_dims** (list of int) - hidden dimensions of classifier. 
* **mlp_hidden_dims_drug** (list of int) - hidden dimension of MLP when encoding drug. 
* **mlp_hidden_dims_target** (list of int) - hidden dimension of MLP when encoding protein. 
* **transformer_emb_size_drug** (int) - embedding size of transformer when encoding drug. 
* **transformer_intermediate_size_drug** (int) - 
* **transformer_num_attention_heads_drug** (int) - 
* **transformer_n_layer_drug** (int) - 
* **transformer_emb_size_target** (int) - 
* **transformer_intermediate_size_target** (int) - 
* **transformer_num_attention_heads_target** (int) - 
* **transformer_n_layer_target** (int) - 
* **transformer_dropout_rate** (float) - 





