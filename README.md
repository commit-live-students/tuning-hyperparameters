# Tuning the Parameters of your first Random Forest

Write a function called `tuneRF()` which takes the following parameters:

1. `X`: features matrix
2. `y`: the target array
3. `cv` : cross-validation fold (int)
3. `seed` : random seed (integer)
5. `grid_tuning_parameters` : parameter values to be tuned for grid search (dict)
6. `random_tuning_parameters` : parameter values to be tuned for random search (dict)
7. (optional): **kwargs for `n_iter` argument for random search api

The function should fit a random forest model to the dataset (cross-validation + hyper parameter tuning) and return the 2 best trained models obtained from hyper-parameter tuning through both grid search and random search as output.

The function should

* perform hyper-paramter tuning using `Grid Search`
* perform hyper-paramter tuning using `Random Search`

The function should retrun
* Trained gridsearchCV object
* Trained randomserachCV object

### Hint:    

Potential hyperparameters that can be tuned are {[Read more here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)}
* n_estimators
* max_depth
- max_features
- min_samples_split
- min_samples_leaf

You can use following parametrs to check your function
