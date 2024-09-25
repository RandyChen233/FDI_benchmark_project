import sys
import os
# Add the parent directory of 'quovadis_tad' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quovadis_tad.evaluation.evaluation import evaluate_methods_on_datasets
from quovadis_tad.dataset_utils.dataset_reader import datasets
from quovadis_tad.baselines import simple_baselines, simple_baselines_3D


def evaluate_simple_baselines_on_all_paper_datasets(
    root_path,
    data_normalization='0-1',
    eval_method='point_wise',
    score_normalization='optimal',
    dataset_names=None,
    verbose=True
):

    if dataset_names is None:
        dataset_names = [
            'swat',
            'wadi_127',
            'wadi_112',
            'smd',
            'msl',
            'smap',
            'ucr_IB'
        ]

    if verbose:
        print(f'[INFO]: Evaluating {len(dataset_names)} datasets')

    datasets_dict = {
        dataset_name.upper(): {'train': data[0], 'test': data[1], 'labels': data[2]}
        for dataset_name in dataset_names
        for data in [datasets[dataset_name](root_path)]  # Trick to get the single value as a variable 'data'.
    }

    '''
    methods_dict = {
        'Sensor Range Deviation': simple_baselines.SensorRangeDeviation(sensor_range=(0, 1)),
        'Simple L2_norm': simple_baselines.LNorm(ord=2),
        '1-NN Distance': simple_baselines.NNDistance(distance='euclidean'),
        'PCA_Error': simple_baselines.PCAError(pca_dim='auto', svd_solver='full'),
        'Random': simple_baselines.Random(seed=9001)
    }
    
    methods_dict = {
        '1-NN Distance': simple_baselines.NNDistance(distance='euclidean')
    }
    '''
    methods_dict = {
        '1-NN Distance': simple_baselines_3D.NNDistance_3D(distance='euclidean'),
        'Simple L2_norm': simple_baselines_3D.LNorm_3D(ord=2),
        'Random': simple_baselines_3D.Random_3D(seed=9001)
    }


    return evaluate_methods_on_datasets(
        methods=methods_dict,
        datasets=datasets_dict,
        data_normalization=data_normalization,
        score_normalization=score_normalization,
        eval_method=eval_method,
        verbose=verbose
    )


if __name__ == '__main__':
    root_path = '../../QuoVadisTAD'
    df = evaluate_simple_baselines_on_all_paper_datasets(
        root_path,
        data_normalization='0-1',
        eval_method='point_wise',
        score_normalization='optimal',
        dataset_names=['swat'],
    )
    print(df.to_string(index=False))
