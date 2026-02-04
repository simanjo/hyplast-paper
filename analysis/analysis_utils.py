import itertools
import pickle
from pathlib import Path

import pandas as pd

from training.train_utils import DataSettings, ModelSettings
from training.train_utils import model_name, model_path

MODEL_SETTING_VALS = {
    'model': ['rf', 'gp', 'pls', 'krr', 'mlp'],
    'embedding': ['mol2vec', 'maccs', 'pubchem', 'rdkit', 'abrahams'],
    'target': ['kf', 'n']
}


def _get_setting_iterator(**kwargs):
    if (data_setting:=kwargs.get('data_setting', None)) is None:
        data_kwargs = {}
        data_setting_keys = set(DataSettings.__dataclass_fields__.keys())
        for missing_key in data_setting_keys.difference(kwargs):
            if missing_key == 'r2_cutoff':
                raise ValueError('Please specify the r2_cutoff manually.')
            elif missing_key == 'sorbent':
                data_kwargs[missing_key] = ('name', 'comp')
            elif missing_key == 'salinity':
                data_kwargs[missing_key] = ('freshwater', 'seawater', 'all_water', 'all_vals')
            else:
                data_kwargs[missing_key] = (True, False)
        for key in data_setting_keys.intersection(kwargs):
            data_kwargs[key] = (kwargs[key], )
        flattened_kwargs = [{}]
        for key, vals in data_kwargs.items():
            flattened_kwargs = [
                single_args | {key: val} for single_args in flattened_kwargs
                for val in vals
            ]
        data_settings = (DataSettings(**arg) for arg in flattened_kwargs)
    else:
        data_settings = (data_setting, )

    if (model_setting:=kwargs.get('model_setting', None)) is None:
        model_kwargs = {}
        model_setting_keys = set(ModelSettings.__dataclass_fields__.keys())
        for missing_key in model_setting_keys.difference(kwargs):
            model_kwargs[missing_key] = MODEL_SETTING_VALS[missing_key]
        for key in model_setting_keys.intersection(kwargs):
            model_kwargs[key] = (kwargs[key], )
        flattened_kwargs = [{}]
        for key, vals in model_kwargs.items():
            flattened_kwargs = [
                single_args | {key: val} for single_args in flattened_kwargs
                for val in vals
            ]
        model_settings = (ModelSettings(**mdl) for mdl in flattened_kwargs)
    else:
        model_settings = (model_setting, )
    return itertools.product(data_settings, model_settings)


def load_results(dir, score_only=True, mean_scores=False, flatten=False, explode_setting=True, **kwargs):
    if not (dir:=Path(dir)).is_dir():
        raise ValueError('Please specify a valid directory, got ', dir)

    results = {}
    for data_setting, model_setting in _get_setting_iterator(**kwargs):
        if not (data_setting, model_setting.model) in results.keys():
            results[(data_setting, model_setting.model)] = {}

        path = model_path(data_setting, dir) / model_name(model_setting)
        if path.is_file():
            with open(path, 'rb') as fh:
                full_spec = pickle.load(fh)
            res = full_spec[-1]['test_score']
        else:
            print(f'Missing model {model_setting} with {data_setting}.')
            print('should be at ', path)
            res = None
            full_spec = None
        if score_only:
            if mean_scores and res is not None:
                results[(data_setting, model_setting.model)]\
                    [(model_setting.embedding, model_setting.target)] = (res.mean(), res.std())
            else:
                results[(data_setting, model_setting.model)]\
                    [(model_setting.embedding, model_setting.target)] = res
        else:
            results[(data_setting, model_setting.model)]\
                [(model_setting.embedding, model_setting.target)] = full_spec
    results = pd.DataFrame(results)
    if flatten:
        score = []
        model = []
        embedding = []
        target = []
        setting = []

        for row in results.itertuples():
            for i, name in enumerate(results.columns):
                score.append(row[i+1])
                model.append(name[1])
                setting.append(name[0])
                target.append(row[0][1])
                embedding.append(row[0][0])
        results = pd.DataFrame({
            'score': score,
            'model': model,
            'embedding': embedding,
            'target': target,
            'setting': setting
        })
        if explode_setting:
            results.loc[:, 'conc_range'] = results['setting'].apply(lambda s: s.conc_range)
            results.loc[:, 'temperature'] = results['setting'].apply(lambda s: s.temperature)
            results.loc[:, 'sorbent'] = results['setting'].apply(lambda s: s.sorbent)
            results.loc[:, 'salinity'] = results['setting'].apply(lambda s: s.salinity)
    return results
