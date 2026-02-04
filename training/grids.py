import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, DotProduct

# REGRESSION
# Kernel Ridge Regression
KRR_GRID = [{
        'alpha': np.logspace(-4, 4, 9),
        'kernel': ['linear'],
    }, {
        'alpha': np.logspace(-4, 4, 9),
        'kernel': ['polynomial'],
        'degree': [2, 3, 4, 5],
    }, {
        'alpha': np.logspace(-4, 4, 9),
        'kernel': ['rbf', 'laplacian', 'sigmoid'],
        'gamma': np.logspace(-4, 4, 9),
    }]

# Gaussian Process
GP_GRID = {
    'random_state': [1357911],
    'kernel': [
        Matern(length_scale=1.0, length_scale_bounds=(1e-20, 1e3), nu=1.5),
        Matern(length_scale=1.0, length_scale_bounds=(1e-20, 1e3), nu=2.5),
        Matern(length_scale=1.0, length_scale_bounds=(1e-20, 1e3), nu=0.5),
        Matern(length_scale=1.0, length_scale_bounds=(1e-20, 1e3), nu=np.inf),
        ConstantKernel(0.1, (1e-10, 1e3)) * (
            DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-10, 1e3)) ** 2
        ),
    ],
    'normalize_y': [True, False],
    'n_restarts_optimizer': [5, 10],
    'alpha': [0.25, 0.5, 1, 1.5]
}
# add choice of whitenoise kernel as sum to all kernel choices
GP_GRID['kernel'] += [
    ker + WhiteKernel(noise_level=1, noise_level_bounds=(1e-40, 1e1))
    for ker in GP_GRID['kernel']]

MLP_GRID = {
    'hidden_layer_sizes': [(100,), (100,100), (50,), (50,50)],
    'alpha': 10.0 ** -np.arange(1, 7),
    'max_iter': [200, 2_000],
    'random_state': [1357911]
}

# Partial Least Squares Regression
PLS_GRID = {
    'n_components': np.arange(1, 7),
    'max_iter': [5_000],
}

# Random Forest
RF_GRID = {
    'n_estimators': [100, 500],
    'criterion': ['absolute_error', 'friedman_mse'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 5],
    'max_leaf_nodes': [50],
    'bootstrap': [True, False],
    'n_jobs': [1],
}


def _get_grid_from_model(model):
    if model == 'rf':
        return [RF_GRID]
    elif model == 'pls':
        return [PLS_GRID]
    elif model == 'krr':
        return KRR_GRID
    elif model == 'gp':
        return [GP_GRID]
    elif model == 'mlp':
        return [MLP_GRID]
    else:
        return []


def get_full_grid(model, is_multimodel, embedding_cols):
    # add classifier prefix
    pref = 'classifier__base_estimator__' if is_multimodel else 'classifier__'
    grids = [
        {pref+k: v for k,v in grid.items()}
        for grid in _get_grid_from_model(model)
    ]
    # add choice of dimensionality reduction if not w/o embedding
    if embedding_cols is None:
        return grids

    grids = [
        gd | {
            'dim_reduction': [
                ColumnTransformer(
                    [('reduce_dim', 'passthrough', embedding_cols)],
                    remainder='passthrough'
                )
            ],
            'dim_reduction__reduce_dim': [
                GaussianRandomProjection()
            ],
            'dim_reduction__reduce_dim__n_components': [
                5, 10
            ],
            'dim_reduction__reduce_dim__random_state': [1357911]
        }
        for gd in grids
    ] + [ # TODO: reason about KernelPCA()
        gd | {
            'dim_reduction': [
                ColumnTransformer(
                    [('reduce_dim', 'passthrough', embedding_cols)],
                    remainder='passthrough'
                )
            ],
            'dim_reduction__reduce_dim': [
                PCA()
            ],
            'dim_reduction__reduce_dim__n_components': [
                5, 10
            ],
            'dim_reduction__reduce_dim__random_state': [1357911]
        }
        for gd in grids
    ] + [
        gd | {
            'dim_reduction': [
                ColumnTransformer(
                    [('reduce_dim', 'passthrough', embedding_cols)],
                    remainder='passthrough'
                )
            ],
            'dim_reduction__reduce_dim': [
                SelectKBest()
            ],
            'dim_reduction__reduce_dim__score_func': [
                f_regression, mutual_info_regression
            ],
            'dim_reduction__reduce_dim__k': [5, 10]
        }
        for gd in grids

    ] + [
        gd | {
            'dim_reduction': [
                ColumnTransformer(
                    [('reduce_dim', 'passthrough', embedding_cols)],
                    remainder='passthrough'
                )
            ],
            'dim_reduction__reduce_dim': [
                SelectFromModel(est) for est in (
                    ExtraTreesRegressor(n_estimators=500),
                    GradientBoostingRegressor(n_estimators=500)
                )
            ],
            'dim_reduction__reduce_dim__max_features': [5, 10]
        }
        for gd in grids
    ]
    return grids
