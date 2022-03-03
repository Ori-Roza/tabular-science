import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from extended_model import get_features_correlations, ExtendedModel, DATASET_NAME


def load_data(dataset):
    # return dataset
    data = pd.DataFrame(dataset['data'], columns=dataset.feature_names)
    target = dataset['target']

    X = data
    y = target
    y = np.array(y)
    # preprocessing
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return dataset.feature_names, X, y


if __name__ == '__main__':
    # Load dataset
    columns, X, y = load_data(datasets.load_wine())

    get_features_correlations(X, y, columns)
    # train
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start=1, stop=15, num=15)]
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
    min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
    bootstrap = [True, False]

    hyperparams_pool = {"n_estimators": n_estimators,
                        "max_features": max_features,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "bootstrap": bootstrap}

    clf = ExtendedModel(X,
                        y,
                        RandomForestClassifier,
                        hyperparams_pool,
                        dataset=DATASET_NAME,
                        send_to_email=False)
    clf.train(columns)
