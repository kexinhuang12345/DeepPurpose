Features of DeepPurpose
=====================================

DeepPurpose is a Deep Learning Based Drug Repurposing and Virtual Screening Toolkit (using PyTorch). 
It allows very easy usage (only one line of code!) for non-computational domain researchers to be able to obtain a list of potential drugs using deep learning while facilitating deep learning method research in this topic by providing a flexible framework (less than 10 lines of codes!) and baselines. 
The Github repository is located `here <https://github.com/kexinhuang12345/DeepPurpose>`_.

Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* For non-computational researchers, ONE line of code from raw data to output drug repurposing/virtual screening result, aiming to allow wet-lab biochemists to leverage the power of deep learning. The result is ensembled from five pretrained deep learning models!

* For computational researchers, 15+ powerful encodings for drugs and proteins, ranging from deep neural network on classic cheminformatics fingerprints, CNN, transformers to message passing graph neural network, with 50+ combined models! Most of the combinations of the encodings are not yet in existing works. All of these under 10 lines but with lots of flexibility! Switching encoding is as simple as changing the encoding names!

* Realistic and user-friendly design:

  * automatic identification to do drug target binding affinity (regression) or drug target interaction prediction (binary) task.
  * support cold target, cold drug settings for robust model evaluations and support single-target high throughput sequencing assay data setup.
  * many dataset loading/downloading/unzipping scripts to ease the tedious preprocessing, including antiviral, COVID19 targets, BindingDB, DAVIS, KIBA, ...
  * many pretrained checkpoints.
  * easy monitoring of training process with detailed training metrics output such as test set figures (AUCs) and tables, also support early stopping.
  * detailed output records such as rank list for repurposing result.
  * various evaluation metrics: ROC-AUC, PR-AUC, F1 for binary task, MSE, R-squared, Concordance Index for regression task.
  * label unit conversion for skewed label distribution such as Kd.
  * time reference for computational expensive encoding. 
  * PyTorch based, support CPU, GPU, Multi-GPUs.  




