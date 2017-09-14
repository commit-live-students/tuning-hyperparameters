from unittest import TestCase
from scipy.stats import randint as sp_randint
from build import tuneRF
from sklearn.datasets import load_boston

class TestTuneRF(TestCase):
    def test_tuneRF(self):
        data = load_boston()
        X = data.data
        y = data.target
        cv = 4
        seed = 80
        grid_param = {"n_estimators": [100, 120, 150],
                      "max_features": [6, 9]
                      }

        random_param = {"min_samples_split": sp_randint(3, 12),
                        "min_samples_leaf": sp_randint(1, 5)
                        }
        gs_model_features1, rs_model_features1 = tuneRF(X, y, grid_tuning_params=grid_param,
                                                        random_tuning_params=random_param,
                                                        n_iter=5, cv=cv, seed=seed
                                                        )
        self.assertAlmostEqual(gs_model_features1.best_score_, 0.5900030706782381)