#%%
import itertools
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from joblib import Memory

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline

from training.train_utils import get_data, get_estimator_from_model, model_name, model_path
from training.train_utils import DataSettings, ModelSettings
from training.grids import get_full_grid

def run_and_save(
        data_setting, model_setting, pipe, grid,
        save_path, X, y,
        n_splits=7, random_state=1357911
    ):

    save_path = model_path(data_setting, save_path)
    os.makedirs(save_path, exist_ok=True)
    name = model_name(model_setting)
    if (save_path / name).is_file():
        print('  Skipped fitting ' + name +' as a similarly named model binary already exists.')
        return
    print('  Start CV for ' + name)
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state+1)
    clf = GridSearchCV(
        estimator=pipe, param_grid=grid,
        cv=inner_cv, n_jobs=-1, error_score='raise',
        scoring='neg_root_mean_squared_error',
        return_train_score=True
    )
    score = cross_validate(
        clf, X=X, y=y, cv=outer_cv,
        scoring='neg_root_mean_squared_error',
        return_train_score=True,
        return_estimator=True,
        return_indices=True # type: ignore
    )
    with open(save_path / name, 'wb') as fh:
        pickle.dump(((X, y), clf, grid, (inner_cv, outer_cv), score), fh)
    print('  Successfully pickled model binary for ' + name)
    print(f'  With score: {score['test_score'].mean()}')

EMBEDDINGS = ['mol2vec', 'abrahams', 'maccs', 'rdkit', 'pubchem']
MODEL_CLASSES = ['pls', 'krr', 'mlp', 'rf', 'gp']
R2_CUTOFF = 0.85
SAVE_PATH = Path(os.getcwd()) / 'models'
DATA_PATH = Path(os.getcwd()) / 'data'


if __name__=='__main__':
    data_settings = (
        DataSettings(
            r2_cutoff=R2_CUTOFF,
            conc_range=crange,
            salinity=sal,
            temperature=temp,
            sorbent=sorb
        ) for sal, crange, temp, sorb in itertools.product(
            ['freshwater', 'all_water', 'seawater', 'all_vals'], [True, False], [True, False], ['name', 'comp']
        ))
    model_settings = (
        ModelSettings(model, emb, tar) # type: ignore
        for model, emb, tar in itertools.product(
            MODEL_CLASSES, EMBEDDINGS, ['kf', 'n']
        )
    )
    pd.options.mode.copy_on_write = True
    data_path = DATA_PATH
    target_cache = {}
    current_data = None
    for data_setting, model_setting in itertools.product(data_settings, model_settings):
        model_dir = model_path(
            data_setting,
            path_prefix=SAVE_PATH
            )
        if data_setting == current_data:
            print(f' Starting {model_setting}')
        else:
            print(f'Starting hpt tuning for {data_setting} and {model_setting}')
            print(f'Storing pickles in directory {model_dir}')
            current_data = data_setting

        # skip calculation of data if a similarly named binary exists
        # instead of preparing data and skipping later on
        if (model_path(data_setting, SAVE_PATH) / model_name(model_setting)).is_file():
            print('  Skipped preparing dataset for ' + model_name(model_setting) +' as a similarly named model binary already exists.')
            continue
        X, y = target_cache.setdefault(
            (model_setting.embedding, model_setting.target, data_setting),
            get_data(data_setting, model_setting, data_path)
        )
        # HACK: need to provide cols as index and not name
        emb_cols = None \
            if model_setting.embedding == 'abrahams' \
            else [
                i for i, col in enumerate(X.columns)
                if col.startswith(model_setting.embedding)
            ]
        cachedir = tempfile.mkdtemp()
        mem = Memory(location=cachedir, verbose=0)
        pipe = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('dim_reduction', 'passthrough'),
            ('classifier', get_estimator_from_model(model_setting))
            ],
            memory=mem)
        run_and_save(
            data_setting=data_setting,
            model_setting=model_setting,
            pipe=pipe,
            grid=get_full_grid(
                model_setting.model,
                is_multimodel=False,
                embedding_cols=emb_cols
                ),
            X=X, y=np.asarray(y),
            save_path=SAVE_PATH,
        )
        shutil.rmtree(cachedir, ignore_errors=True)

# %%
