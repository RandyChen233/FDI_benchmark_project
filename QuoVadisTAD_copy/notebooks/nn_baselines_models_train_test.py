#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import sys
from pathlib import Path
module_path = str(Path.cwd().parents[0])
if module_path not in sys.path:
    sys.path.append(module_path)


# In[3]:


import pandas as pd
from quovadis_tad.dataset_utils.dataset_reader import datasets
from quovadis_tad.dataset_utils.data_utils import preprocess_data, normalise_scores
from quovadis_tad.evaluation.single_series_evaluation import evaluate_ts
from quovadis_tad.model_utils.model_def import test_embedder
pd.set_option('display.precision', 3)


# In[4]:


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0: 
   tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


# # Train NN-Baselines Models

# #### We provide four model configurations to train corresponding to the introduced four NN-Baselines. See the "src/model_configs" folder. To train these on a dataset, go to project root and run from the console the following command by providing the dataset name. This will train all model configs on the dataset. 

# `CUDA_VISIBLE_DEVICES=0 python ./src/run_all_configs.py wadi_112`
# 

# #### or pass a specific config name to train with --config-to-run argument as below

# `CUDA_VISIBLE_DEVICES=0 python ./src/run_all_configs.py wadi_112 --config-to-run gcn_lstm_model_seq_5.yaml`
# 
# 

# #### See the file run_all_configs for the input arguments options. The trained model checkpoints will be saved by default to model_configs folder.

# # Trained Model Inference & Evaluation

# ### We include our trained model checkpoints for SWAT and WADI datasets in the model_checkpoints folder. These or the ones you train can be tested as below.

# ## Model Inference

# In[7]:


pred, gt = test_embedder(module_path,
                                 'ourBench',                           # Dataset name one of e.g 'swat', 'wadi_127', 'wadi_112', 'smd', see dataset_reader enum
                                 dataset_trace=None,
                                 model_name='1_Layer_MLP',    # one of the NN-Baselines '1_Layer_MLP', 'Single_block_MLPMixer', 'Single_Transformer_block', '1_Layer_GCN_LSTM'
                                 load_weights=True,
                                 training=False,
                                 subset='test'
                                )


# ## Evaluate the model prediction

# ### Evaluate the prediction under Point-Wise metrics

# In[ ]:


res, df = evaluate_ts(normalise_scores(pred).max(1),
                   gt.max(1),
                   eval_method='point_wise',
                   verbose=True)


# In[ ]:





# ### Evaluate the prediction under Range-Wise metrics

# In[ ]:


res, df = evaluate_ts(normalise_scores(pred).max(1),
                   gt.max(1),
                   eval_method='range_wise',
                   verbose=True)


# In[ ]:




