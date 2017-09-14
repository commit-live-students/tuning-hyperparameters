from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


def tuneRF(X, y, grid_tuning_params, random_tuning_params, cv=5, seed=7, **kwargs):
    pass