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
   $ cd DeepPurpose


1.2 Installation
^^^^^^^^^^^^

.. code-block:: bash

   $ conda env create -f env.yml  
   $ conda activate DeepPurpose

   $ conda deactivate ### exit




2 Run
--------------






3 Documentation
--------------



3.1 Encoder Models
^^^^^^^^^^^^

`Transformer <https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/>`_ can be used to encode drug represented as SMILES.
 

.. code-block:: python

	class transformer(nn.Sequential):
		def __init__(self, encoding, **config):
			super(transformer, self).__init__()
			if encoding == 'drug':
				self.emb = Embeddings(config['input_dim_drug'], config['transformer_emb_size_drug'], 50, config['transformer_dropout_rate'])
				self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_drug'], 
														config['transformer_emb_size_drug'], 
														config['transformer_intermediate_size_drug'], 
														config['transformer_num_attention_heads_drug'],
														config['transformer_attention_probs_dropout'],
														config['transformer_hidden_dropout_rate'])
			elif encoding == 'protein':
				self.emb = Embeddings(config['input_dim_protein'], config['transformer_emb_size_target'], 545, config['transformer_dropout_rate'])
				self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_target'], 
														config['transformer_emb_size_target'], 
														config['transformer_intermediate_size_target'], 
														config['transformer_num_attention_heads_target'],
														config['transformer_attention_probs_dropout'],
														config['transformer_hidden_dropout_rate'])

		def forward(self, v):
			e = v[0].long().to(device)
			e_mask = v[1].long().to(device)
			ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
			ex_e_mask = (1.0 - ex_e_mask) * -10000.0

			emb = self.emb(e)
			encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
			return encoded_layers[:,0]


`CNN (Convolutional Neural Network) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ can be used to encode drug. 


.. code-block:: python

	class CNN(nn.Sequential):
		def __init__(self, encoding, **config):
			super(CNN, self).__init__()
			if encoding == 'drug':
				in_ch = [63] + config['cnn_drug_filters']
				kernels = config['cnn_drug_kernels']
				layer_size = len(config['cnn_drug_filters'])
				self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
														out_channels = in_ch[i+1], 
														kernel_size = kernels[i]) for i in range(layer_size)])
				self.conv = self.conv.double()
				n_size_d = self._get_conv_output((63, 100))
				#n_size_d = 1000
				self.fc1 = nn.Linear(n_size_d, config['hidden_dim_drug'])

			if encoding == 'protein':
				in_ch = [26] + config['cnn_target_filters']
				kernels = config['cnn_target_kernels']
				layer_size = len(config['cnn_target_filters'])
				self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
														out_channels = in_ch[i+1], 
														kernel_size = kernels[i]) for i in range(layer_size)])
				self.conv = self.conv.double()
				n_size_p = self._get_conv_output((26, 1000))

				self.fc1 = nn.Linear(n_size_p, config['hidden_dim_protein'])

		def _get_conv_output(self, shape):
			bs = 1
			input = Variable(torch.rand(bs, *shape))
			output_feat = self._forward_features(input.double())
			n_size = output_feat.data.view(bs, -1).size(1)
			return n_size

		def _forward_features(self, x):
			for l in self.conv:
				x = F.relu(l(x))
			x = F.adaptive_max_pool1d(x, output_size=1)
			return x

		def forward(self, v):
			v = self._forward_features(v.double())
			v = v.view(v.size(0), -1)
			v = self.fc1(v.float())
			return v

::














