DeepPurpose.oneliner.virtual_screening
================================================


text:todo 

.. code-block:: python


	def virtual_screening(
			target, 
			X_repurpose = None,
			target_name = None, 
			drug_names = None,
			train_drug = None, 
			train_target = None, 
			train_y = None, 
			save_dir = './save_folder',               
			pretrained_dir = None,
			finetune_epochs = 10,
			finetune_LR = 0.01,
			finetune_batch_size = 32,
			convert_y = True,
			subsample_frac = 1,
			pretrained = True,
			split = 'random',
			frac = [0.7,0.1,0.2],
			agg = 'agg_mean_max',
			output_len = 30):












