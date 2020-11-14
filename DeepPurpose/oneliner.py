from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from prettytable import PrettyTable
import wget
from zipfile import ZipFile 

def repurpose(target, target_name = None, 
					X_repurpose = None, 
					drug_names = None,
					train_drug = None, 
					train_target = None, 
					train_y = None, 
					save_dir = './save_folder',               
					pretrained_dir = None,
					finetune_epochs = 10,
					finetune_LR = 0.001,
					finetune_batch_size = 32,
					convert_y = True,
					subsample_frac = 1,
					pretrained = True,
					split = 'random',
					frac = [0.7,0.1,0.2],
					agg = 'agg_mean_max',
					output_len = 30):
           
	if not os.path.exists(save_dir):
		print('Save path not found or given and set to default: \'./save_folder/\'. ')
		os.mkdir('save_folder')
		save_dir = './save_folder'

	if target_name is None:
		target_name = 'New Target'

	if X_repurpose is not None:
		if drug_names is None:
			drug_names = ['Drug ' + str(i) for i in list(range(len(X_repurpose)))]
		print("Loading customized repurposing dataset...")            
	else:
		if not os.path.exists(os.path.join(save_dir, 'data')):
			os.mkdir(os.path.join(save_dir, 'data'))
		
		data_path = os.path.join(save_dir, 'data')
		X_repurpose, _, drug_names = load_broad_repurposing_hub(data_path)
		# default repurposing hub dataset is broad repurposing hub

	pretrained_model_names = [['MPNN', 'CNN'], ['CNN','CNN'], ['Morgan', 'CNN'], ['Morgan', 'AAC'], ['Daylight', 'AAC']]

	y_preds_models = []

	if (pretrained_dir is None) & pretrained:
		# load 6 pretrained model
		print('Beginning Downloading Pretrained Model...')
		print('Note: if you have already download the pretrained model before, please stop the program and set the input parameter \'pretrained_dir\' to the path')
		pretrained_dir = download_pretrained_model('pretrained_models')
	elif pretrained == False:
		print('Beginning Downloading Configs Files for training from scratch...')
		pretrained_dir = download_pretrained_model('models_configs')		    
	else:
		print('Checking if pretrained directory is valid...')
		if not os.path.exists(pretrained_dir):
			print('The directory to pretrained model is not found. Please double check, or download it again by setting the input parameter \'pretrained_dir\' to be \'None\'')
		else:
			print('Beginning to load the pretrained models...')

	if train_drug is None:
		
		print('Using pretrained model and making predictions...')

		for idx, model_name in enumerate(pretrained_model_names):
			model_path = os.path.join(pretrained_dir, 'model_' + model_name[0] + '_' + model_name[1])
			model = models.model_pretrained(model_path)
			result_folder_path = os.path.join(save_dir, 'results_'+model_name[0] + '_' + model_name[1])

			if not os.path.exists(result_folder_path):
				os.mkdir(result_folder_path)

			y_pred = models.repurpose(X_repurpose, target, model, drug_names, target_name, convert_y = convert_y, result_folder = result_folder_path, verbose = False)
			y_preds_models.append(y_pred)
			print('Predictions from model ' + str(idx + 1) + ' with drug encoding ' + model_name[0] + ' and target encoding ' + model_name[1] + ' are done...')
			print('-------------')
	else:
		# customized training data
		print('Training on your own customized data...') 
		if not os.path.exists(os.path.join(save_dir, 'new_trained_models')):
			os.mkdir(os.path.join(save_dir, 'new_trained_models'))
		new_trained_models_dir = os.path.join(save_dir, 'new_trained_models')            
		if isinstance(train_target, str):
			train_target = [train_target] 
		for idx, model_name in enumerate(pretrained_model_names):
			drug_encoding = model_name[0]
			target_encoding = model_name[1]
			train, val, test = data_process(train_drug, train_target, train_y, 
                                drug_encoding, target_encoding, 
                                split_method=split,frac=frac, sample_frac = subsample_frac)
			model_path = os.path.join(pretrained_dir, 'model_' + model_name[0] + '_' + model_name[1])
			if pretrained:
				model = models.model_pretrained(model_path)
				print('Use pretrained model...')
			else:
				config = load_dict(model_path)
				model = models.model_initialize(**config)
				print('Training from scrtach...')
			print('Begin to train model ' + str(idx) + ' with drug encoding ' + drug_encoding + ' and target encoding ' + target_encoding)
			model.config['train_epoch'] = finetune_epochs
			model.config['LR'] = finetune_LR
			model.config['batch_size'] = finetune_batch_size

			result_folder_path = os.path.join(save_dir, 'results_'+model_name[0] + '_' + model_name[1])

			if not os.path.exists(result_folder_path):
				os.mkdir(result_folder_path)

			model.config['result_folder'] = result_folder_path
			model.train(train, val, test)

			print('model training finished, now repurposing')
			y_pred = models.repurpose(X_repurpose, target, model, drug_names, target_name, convert_y = convert_y, result_folder = result_folder_path, verbose = False)
			y_preds_models.append(y_pred)
			print('Predictions from model ' + str(idx) + ' with drug encoding ' + model_name[0] + ' and target encoding ' + model_name[1] + ' are done...')
			model.save_model(os.path.join(new_trained_models_dir, 'model_' + model_name[0] + '_' + model_name[1]))
	result_folder_path = os.path.join(save_dir, 'results_aggregation')

	if not os.path.exists(result_folder_path):
		os.mkdir(result_folder_path)

	print('models prediction finished...')
	print('aggregating results...')

	if agg == 'mean':
		y_pred = np.mean(y_preds_models, axis = 0)
	elif agg == 'max_effect':
		if convert_y:        
			y_pred = np.min(y_preds_models, axis = 0)
		else:
			y_pred = np.max(y_preds_models, axis = 0)
	elif agg == 'agg_mean_max':
		if convert_y:        
			y_pred = (np.min(y_preds_models, axis = 0) + np.mean(y_preds_models, axis = 0))/2
		else:
			y_pred = (np.max(y_preds_models, axis = 0) + np.mean(y_preds_models, axis = 0))/2          
            
	fo = os.path.join(result_folder_path, "repurposing.txt")
	print_list = []
	with open(fo, 'w') as fout:		
		
		print('---------------')
		if target_name is not None:
			print('Drug Repurposing Result for '+target_name)
		if model.binary:
			table_header = ["Rank", "Drug Name", "Target Name", "Interaction", "Probability"]
		else:
			### regression 
			table_header = ["Rank", "Drug Name", "Target Name", "Binding Score"]
		table = PrettyTable(table_header)

		if drug_names is not None:
			f_d = max([len(o) for o in drug_names]) + 1
			for i in range(len(y_pred)):
				if model.binary:
					if y_pred[i] > 0.5:
						string_lst = [drug_names[i], target_name, "YES", "{0:.2f}".format(y_pred[i])]
						
					else:
						string_lst = [drug_names[i], target_name, "NO", "{0:.2f}".format(y_pred[i])]
				else:
					#### regression 
					#### Rank, Drug Name, Target Name, binding score 
					string_lst = [drug_names[i], target_name, "{0:.2f}".format(y_pred[i])]
					string = 'Drug ' + '{:<{f_d}}'.format(drug_names[i], f_d =f_d) + \
						' predicted to have binding affinity score ' + "{0:.2f}".format(y_pred[i])
					#print_list.append((string, y_pred[i]))
				print_list.append((string_lst, y_pred[i]))
		if convert_y:
			print_list.sort(key = lambda x:x[1])
		else:
			print_list.sort(key = lambda x:x[1], reverse = True)
		print_list = [i[0] for i in print_list]
		for idx, lst in enumerate(print_list):
			lst = [str(idx + 1)] + lst 
			table.add_row(lst)
		fout.write(table.get_string())
	with open(fo, 'r') as fin:
		lines = fin.readlines()
		for idx, line in enumerate(lines):
			if idx < output_len + 3:
				print(line, end = '')
			else:
				print('checkout ' + fo + ' for the whole list')
				break
		print()
	with open(os.path.join(result_folder_path, 'output_list.pkl'), 'wb') as f:
		pickle.dump(print_list, f, pickle.HIGHEST_PROTOCOL)
        
        
def virtual_screening(target, X_repurpose = None,
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
           
	if not os.path.exists(save_dir):
		print('Save path not found or given and set to default: \'./save_folder/\'. ')
		os.mkdir('save_folder')
		save_dir = './save_folder'

	if target_name is None:
		target_name = ['Target ' + str(i) for i in list(range(len(X_repurpose)))]

	if X_repurpose is not None:
		if drug_names is None:
			drug_names = ['Drug ' + str(i) for i in list(range(len(X_repurpose)))]
		print("Loading customized repurposing dataset...")            
	else:
		print("Virtual Screening requires drug candidates input (a list of SMILESs)")

	pretrained_model_names = [['MPNN', 'CNN'], ['CNN','CNN'], ['Morgan', 'CNN'], ['Morgan', 'AAC'], ['Daylight', 'AAC']]

	y_preds_models = []

	if (pretrained_dir is None) & pretrained:
		# load 6 pretrained model
		print('Beginning Downloading Pretrained Model...')
		print('Note: if you have already download the pretrained model before, please stop the program and set the input parameter \'pretrained_dir\' to the path')
		pretrained_dir = download_pretrained_model('pretrained_models')
	elif pretrained == False:
		print('Beginning Downloading Configs Files for training from scratch...')
		pretrained_dir = download_pretrained_model('models_configs')		    
	else:
		print('Checking if pretrained directory is valid...')
		if not os.path.exists(pretrained_dir):
			print('The directory to pretrained model is not found. Please double check, or download it again by setting the input parameter \'pretrained_dir\' to be \'None\'')
		else:
			print('Beginning to load the pretrained models...')
			
	if train_drug is None:
		
		print('Using pretrained model and making predictions...')

		for idx, model_name in enumerate(pretrained_model_names):
			model_path = os.path.join(pretrained_dir, 'model_' + model_name[0] + '_' + model_name[1])
			model = models.model_pretrained(model_path)
			result_folder_path = os.path.join(save_dir, 'results_'+model_name[0] + '_' + model_name[1])

			if not os.path.exists(result_folder_path):
				os.mkdir(result_folder_path)

			y_pred = models.virtual_screening(X_repurpose, target, model, drug_names, target_name, convert_y = convert_y, result_folder = result_folder_path, verbose = False)
			y_preds_models.append(y_pred)
			print('Predictions from model ' + str(idx + 1) + ' with drug encoding ' + model_name[0] + ' and target encoding ' + model_name[1] + ' are done...')
			print('-------------')
	else:
		# customized training data
		print('Training on your own customized data...') 
		if not os.path.exists(os.path.join(save_dir, 'new_trained_models')):
			os.mkdir(os.path.join(save_dir, 'new_trained_models'))
		new_trained_models_dir = os.path.join(save_dir, 'new_trained_models')            
		if isinstance(train_target, str):
			train_target = [train_target] 
		for idx, model_name in enumerate(pretrained_model_names):
			drug_encoding = model_name[0]
			target_encoding = model_name[1]
			train, val, test = data_process(train_drug, train_target, train_y, 
                                drug_encoding, target_encoding, 
                                split_method=split,frac=frac, sample_frac = subsample_frac)
			model_path = os.path.join(pretrained_dir, 'model_' + model_name[0] + '_' + model_name[1])
			if pretrained:
				model = models.model_pretrained(model_path)
				print('Use pretrained model...')
			else:
				config = load_dict(model_path)
				model = models.model_initialize(**config)
				print('Training from scrtach...')
			print('Begin to train model ' + str(idx) + ' with drug encoding ' + drug_encoding + ' and target encoding ' + target_encoding)
			model.config['train_epoch'] = finetune_epochs
			model.config['LR'] = finetune_LR
			model.config['batch_size'] = finetune_batch_size

			result_folder_path = os.path.join(save_dir, 'results_'+model_name[0] + '_' + model_name[1])

			if not os.path.exists(result_folder_path):
				os.mkdir(result_folder_path)

			model.config['result_folder'] = result_folder_path
			model.train(train, val, test)

			print('model training finished, now doing virtual screening')
			y_pred = models.virtual_screening(X_repurpose, target, model, drug_names, target_name, convert_y = convert_y, result_folder = result_folder_path, verbose = False)
			y_preds_models.append(y_pred)
			print('Predictions from model ' + str(idx) + ' with drug encoding ' + model_name[0] + ' and target encoding ' + model_name[1] + ' are done...')
			model.save_model(os.path.join(new_trained_models_dir, 'model_' + model_name[0] + '_' + model_name[1]))
	result_folder_path = os.path.join(save_dir, 'results_aggregation')

	if not os.path.exists(result_folder_path):
		os.mkdir(result_folder_path)

	print('models prediction finished...')
	print('aggregating results...')

	if agg == 'mean':
		y_pred = np.mean(y_preds_models, axis = 0)
	elif agg == 'max_effect':
		if convert_y:        
			y_pred = np.min(y_preds_models, axis = 0)
		else:
			y_pred = np.max(y_preds_models, axis = 0)
	elif agg == 'agg_mean_max':
		if convert_y:        
			y_pred = (np.min(y_preds_models, axis = 0) + np.mean(y_preds_models, axis = 0))/2
		else:
			y_pred = (np.max(y_preds_models, axis = 0) + np.mean(y_preds_models, axis = 0))/2 
            
	with open(os.path.join(result_folder_path, 'logits_VS_mean.pkl'), 'wb') as f:
		pickle.dump(np.mean(y_preds_models, axis = 0), f, pickle.HIGHEST_PROTOCOL) 
	with open(os.path.join(result_folder_path, 'logits_VS_max.pkl'), 'wb') as f:
		pickle.dump(np.min(y_preds_models, axis = 0), f, pickle.HIGHEST_PROTOCOL) 
	with open(os.path.join(result_folder_path, 'logits_VS_mean_max.pkl'), 'wb') as f:
		pickle.dump((np.min(y_preds_models, axis = 0) + np.mean(y_preds_models, axis = 0))/2, f, pickle.HIGHEST_PROTOCOL) 
        
	fo = os.path.join(result_folder_path, "virtual_screening.txt")
	print_list = []

	if model.binary:
		table_header = ["Rank", "Drug Name", "Target Name", "Interaction", "Probability"]
	else:
		### regression 
		table_header = ["Rank", "Drug Name", "Target Name", "Binding Score"]
	table = PrettyTable(table_header)

	with open(fo,'w') as fout:
		print('virtual screening...')		
		print('---------------')
		if drug_names is not None and target_name is not None:
			print('Virtual Screening Result')
			f_d = max([len(o) for o in drug_names]) + 1
			f_p = max([len(o) for o in target_name]) + 1
			for i in range(target.shape[0]):
				if model.binary:
					if y_pred[i] > 0.5:
						string_lst = [drug_names[i], target_name[i], "YES", "{0:.2f}".format(y_pred[i])]
					else:
						string_lst = [drug_names[i], target_name[i], "NO", "{0:.2f}".format(y_pred[i])]
						
				else:
					### regression 
					string_lst = [drug_names[i], target_name[i], "{0:.2f}".format(y_pred[i])]
					
				print_list.append((string_lst, y_pred[i]))
		if convert_y:
			print_list.sort(key = lambda x:x[1])
		else:
			print_list.sort(key = lambda x:x[1], reverse = True)
		print_list = [i[0] for i in print_list]
		for idx, lst in enumerate(print_list):
			lst = [str(idx+1)] + lst
			table.add_row(lst)
		fout.write(table.get_string())
	with open(fo, 'r') as fin:
		lines = fin.readlines()
		for idx, line in enumerate(lines):
			if idx < output_len + 3:
				print(line, end = '')
			else:
				print('checkout ' + fo + ' for the whole list')
				break
	print()
	with open(os.path.join(result_folder_path, 'output_list_VS.pkl'), 'wb') as f:
		pickle.dump(print_list, f, pickle.HIGHEST_PROTOCOL)