from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

from preprocessing.utils import calculate_sorbent_stats, unify_sorbent, label_modifications


@dataclass(frozen=True)
class DataSettings():
    r2_cutoff: float
    conc_range: bool
    salinity: Literal['freshwater', 'seawater', 'all_water', 'all_vals']
    temperature: bool
    sorbent: Literal['name', 'comp', 'mod']


def model_path(setting: DataSettings, path_prefix: Path):
    model_dir = f'{setting.salinity}_'
    model_dir += 'w_conc_' if setting.conc_range else ''
    model_dir += 'w_temp_' if setting.temperature else ''
    model_dir = model_dir[:-1]

    return path_prefix / f'sorbent_{setting.sorbent}' / model_dir


@dataclass(frozen=True)
class ModelSettings():
    model: Literal['rf', 'pls', 'krr', 'mlp', 'gp']
    embedding: Literal['mol2vec', 'maccs', 'pubchem', 'rdkit', 'abrahams']
    target: Literal['kf', 'n']


def model_name(setting: ModelSettings):
    return f'{setting.model}_{setting.embedding}_freundlich_{setting.target}.pkl'


def get_setting_from_model_name(name):
    model, emb, _, target = name.split('_')
    target = target.split('.')[0]
    return ModelSettings(model, emb, target) # type: ignore


def get_estimator_from_model(setting: ModelSettings):
    if setting.model == 'rf':
        return RandomForestRegressor()
    elif setting.model == 'krr':
        return KernelRidge()
    elif setting.model == 'pls':
        return PLSRegression()
    elif setting.model == 'gp':
        return GaussianProcessRegressor()
    elif setting.model == 'mlp':
        return MLPRegressor()


def get_data(data_setting, model_setting, data_dir):
    raw_data = pd.read_csv(Path(data_dir)/ f'hyplast_{model_setting.embedding}.csv')

    assert 0 < data_setting.r2_cutoff and data_setting.r2_cutoff < 1
    raw_data = raw_data[raw_data['r2'] >= data_setting.r2_cutoff]
    raw_data = raw_data[~raw_data['SSA (m^2/g)'].isna()]

    data_cols = ['ph', 'SLR (kg/L)', 'SSA (m^2/g)']
    # scale SSA and SLR
    raw_data['SLR (kg/L)'] = np.log(raw_data['SLR (kg/L)'])
    raw_data['SSA (m^2/g)'] = np.log(raw_data['SSA (m^2/g)'])
    # impute salinity with 0.1
    raw_data.loc[raw_data['salinity'].isna(), 'salinity'] = 0.1
    # impute pH with 8 if salinity > 30 else with 7
    raw_data.loc[raw_data['ph'].isna() & (raw_data['salinity'] > 30), 'ph'] = 8
    raw_data.loc[raw_data['ph'].isna(), 'ph'] = 7
    # convert room temperature to 25
    raw_data.loc[raw_data['T'] == 'room temperature', 'T'] = 25
    raw_data = raw_data.astype({'T': float})
    # temperature
    if data_setting.temperature:
        data_cols.append('T')
    else:
        raw_data = raw_data[
            (raw_data['T'] >= 20)
            & (raw_data['T'] <= 25)
        ]
    # concentration range
    if data_setting.conc_range:
        data_cols += [
            'low conc_range (µg/L)',
            'high conc_range (µg/L)'
        ]
        raw_data['low conc_range (µg/L)'] = np.log10(raw_data['low conc_range (µg/L)'])
        raw_data['high conc_range (µg/L)'] = np.log10(raw_data['high conc_range (µg/L)'])
    # salinity
    if data_setting.salinity == 'freshwater':
        raw_data = raw_data[
            raw_data['salinity'] < 5
        ]
    elif data_setting.salinity == 'seawater':
        raw_data = raw_data[
            raw_data['salinity'] > 30
        ]
    elif data_setting.salinity == 'all_water':
        # also include the few less saline saltwater entries
        raw_data['seawater'] = raw_data['salinity'] > 25
        data_cols.append('seawater')
    elif data_setting.salinity == 'all_vals':
        # use all data and have salinity as numeric value
        data_cols.append('salinity')

    # sorbent cols
    data = raw_data.copy(deep=True)
    data['cleaned_sorbent'] = data['sorbent_name'].apply(unify_sorbent)
    if data_setting.sorbent == 'mod':
        data['modification_label'] = label_modifications(data)
        data = pd.get_dummies(
            data,
            columns=['modification_label'], # type: ignore
            prefix='mod',
            drop_first=True
        )
        data_cols += [
            col for col in data.columns if col.startswith('mod_')
        ] # type: ignore
    else:
        data = data[~data['cleaned_sorbent'].str.startswith('modified')]
        if data_setting.sorbent == 'name':
            data = pd.get_dummies(
                data,
                columns=['cleaned_sorbent'], # type: ignore
                prefix='sorb',
                drop_first=True
            )
            data_cols += [
                col for col in data.columns if col.startswith('sorb_')
            ] # type: ignore
        else:
            assert data_setting.sorbent == 'comp'
            data[['c_weight_percent', 'hc_ratio', 'oc_ratio']] \
                = data['cleaned_sorbent'].apply(calculate_sorbent_stats).to_list()
            # only add the stats if they are not constant
            for col in ('c_weight_percent', 'hc_ratio', 'oc_ratio'):
                if data[col].unique().shape[0] > 1:
                    data_cols.append(col)

    # embedding cols
    data_cols += [
        col for col in data.columns
        if col.startswith(model_setting.embedding)
        and data[col].unique().shape[0] > 1
    ]
    X = data.loc[:, data_cols]

    if model_setting.target == 'n':
        y = data['scaled_n']#.apply(lambda n: 1/(n+1))
    else:
        assert model_setting.target == 'kf', model_setting
        y = np.log10(data['scaled_kf'])

    return X, y
