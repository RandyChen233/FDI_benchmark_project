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

import numpy as np

from quovadis_tad.eval_simple_baselines import evaluate_simple_baselines_on_all_paper_datasets


# # Evaluate the simple baselines
# In this notebook we run all our simple baselines on the datasets appearing in the paper. The main purpose of the notebook is reproducibility. If you want to use your own datasets, own methods and even combine them with ours, please look in the `simple_baselines_example_usage.ipynb` notebook for inspiration. 

# In[13]:


def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')
print(module_path)


# ## Evaluate on a single dataset
# Full evaluation on all datasets can take some time. One can first try running on a single dataset of interest to get some fast results. Below we have an example of evaluating the methods only on SWAT. On the first part, we apply it using the optimal score normalization and on the second part we return all such scores to check the impact of different normalization options.

# #### Optimal normalization

# In[12]:


df_std = evaluate_simple_baselines_on_all_paper_datasets(
    root_path=module_path,
    dataset_names=['ourBench'],  # provide one or more dataset names e.g ['swat', 'wadi_127', 'wadi_112', 'smd', 'ucr_IB'], see dataset_reader enum.
    data_normalization="0-1",              
    eval_method='point_wise',
    score_normalization='optimal',  # Will only return the scores for the optimal score normalization method.
    verbose=False
)


# In[ ]:


(
    df_std
    .style
    .format(precision=3)
    .apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
)


# In[ ]:





# #### See the impact of different score normalisations

# In[ ]:

'''
df_std = evaluate_simple_baselines_on_all_paper_datasets(
    root_path=module_path,
    dataset_names=['smd'],
    data_normalization="0-1",              
    eval_method='point_wise',
    score_normalization='all',  # Will return scores for all score normalization methods for baselines which return multiple outputs. In this case only PCA_Error.
    verbose=False
)
'''


'''
# In[ ]:


(
    df_std
    .style
    .format(precision=3)
    .apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
)


# ## Evaluate on all datasets - Point-Wise metrics
# Here we reproduce the point-wise F1 score of all simple baselines on all datasets.

# In[ ]:


df_point_wise = evaluate_simple_baselines_on_all_paper_datasets(
    root_path=module_path,
    dataset_names=None,
    data_normalization="0-1",              
    eval_method='point_wise',  # 
    score_normalization='optimal',
    verbose=True
)


# In[ ]:


(
    df_point_wise
    .style
    .format(precision=3)
    .apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
)


# In[ ]:


(
    df_point_wise
    .drop(['P', 'R','AUPRC'], axis=1, level=1)
    .style
    .format(precision=3)
    .apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
)
'''

'''
# ## Evaluate all datasets - Range-Wise metrics
# Here we reproduce the range-wise scores of all simple baselines on all datasets.

# In[ ]:


df_range_wise = evaluate_simple_baselines_on_all_paper_datasets(
    root_path=module_path,
    dataset_names=None,
    data_normalization="0-1",
    eval_method='range_wise',
    score_normalization='optimal',
    verbose=True
)


# In[ ]:


(
    df_range_wise
    .style
    .format(precision=3)
    .apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
)


# In[ ]:


(
    df_range_wise
    .drop(['P', 'R','AUPRC'], axis=1, level=1)
    .style
    .format(precision=3)
    .apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
)


'''


