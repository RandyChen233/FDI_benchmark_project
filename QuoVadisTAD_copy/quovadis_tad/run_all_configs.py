import os
from pathlib import Path
import glob
import numpy as np
import typer
import sys
import os

import config_data3D
# Add the directory to PYTHONPATH
'''
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from quovadis_tad.dataset_utils.dataset_reader import datasets
from quovadis_tad.model_utils.model_def import train_embedder

module_path = str(Path.cwd())
'''
# Determine the script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Change the current working directory to project_root
os.chdir(project_root)

# Append project_root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)

# Append 'quovadis_tad' directory to sys.path to ensure it's included
quovadis_tad_path = os.path.join(project_root, 'quovadis_tad')
if quovadis_tad_path not in sys.path:
    sys.path.append(quovadis_tad_path)

from quovadis_tad.dataset_utils.dataset_reader import datasets
from quovadis_tad.model_utils.model_def import train_embedder

module_path = str(Path.cwd())

print(f"Module path: {module_path}")
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")

def run_configs(dataset_name: str,
                dataset_trace: int = None,
                overwrite: bool = False,
                load_weights: bool = False,
                classification: bool = True,
                config_to_run: str = None,
                ):
    
    print(f'[INFO]: Module_path = {module_path}')
    configs = glob.glob(os.path.join(module_path, 'quovadis_tad', 'model_configs', '*.yaml'))
    print(f'Found {len(configs)} configs to run')
    for config_path in configs:
        # check if already trained this config
        config_name = os.path.basename(config_path)
        if dataset_trace is None:
            checkpoint = Path(module_path, 'resources', 'model_checkpoints', dataset_name, config_name.split('.')[0])
        else:
            checkpoint = Path(module_path, 'resources', 'model_checkpoints', dataset_name, str(dataset_trace), config_name.split('.')[0])
            
        if config_to_run is not None:
            if config_name != config_to_run:
                print(f'Skipping....: Trained Model on {dataset_name} already exist for configuration {config_name}. To retrain specify overwrite True or provide config name with --config-to-run param.')
                continue
            else:
                overwrite = True
        else:
            if os.path.exists(checkpoint):
                if not overwrite and not load_weights:
                    print(f'Skipping....: Trained Model on {dataset_name} already exist for configuration {config_name}. To retrain specify overwrite True.')
                    continue
        
        
        print(f'[INFO]: Training {config_name} on {dataset_name}')
        _ = train_embedder(module_path,
                           dataset_name,
                           dataset_trace=dataset_trace,
                           config_path=config_path,
                           load_weights=load_weights,
                           classification=classification,
                          )
        print(f'[INFO]: Finished training {config_name} on {dataset_name}')


def run_configs_trace(dataset_name: str,                
                overwrite: bool = False,
                classification: bool = True,
                load_weights: bool = False,
                config_to_run: str = None):
    
    trainset, _, l_ = datasets[dataset_name](module_path)
    if type(trainset) is list and not config_data3D.data3D:
        for i in range(len(trainset)):
            dataset_trace = i
            run_configs(dataset_name=dataset_name,
                    dataset_trace=i,
                    overwrite= overwrite,
                    load_weights=load_weights,
                    config_to_run=config_to_run,
                    classification=classification)
            
    else:
        run_configs(dataset_name=dataset_name,
                    dataset_trace=None,
                    overwrite= overwrite,
                    load_weights=load_weights,
                    config_to_run=config_to_run,
                    classification=classification)
        
if __name__ == "__main__":
    typer.run(run_configs_trace)