import os
import mlflow
import smtplib
import numpy as np
import pandas as pd
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from extended_model.config import *

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def send_email(_from, to, text, *files):
    try:
        msg = MIMEMultipart('related')
        msg["From"] = _from
        msg["Subject"] = "MLOPS RUN"
        msg["To"] = to
        text_ = MIMEText(text)
        msg.attach(text_)

        for f in files:
            image = MIMEImage(open(f, "rb").read(), name=os.path.basename(f))
            msg.attach(image)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL, PASSWORD)
            server.send_message(msg)
            return True
    except Exception as e:
        print("Failed to send alert email.\nException: " + str(e))


class ExtendedModel:
    def __init__(self, X, y, model, hyperparams, dataset=None, send_to_email=False):
        self.model = model
        self.hyperparams = hyperparams
        self._clf = None
        self._rs = None
        self.rs_df = None
        self.dataset = dataset
        self._X = X
        self._y = y
        self.send_to_email = send_to_email
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(X, y,
                                                                                            test_size=0.3
                                                                                            )
        pca = PCA(n_components=X.shape[1])
        pca.fit(self.X_train)
        self.X_train = pca.transform(self.X_train)

        self.X_validation, self.X_test, self.y_validation, self.y_test = train_test_split(self.X_validation,
                                                                                          self.y_validation,
                                                                                          test_size=0.5)

    def get_best_hyper_params(self):
        rfc = self.model()
        self._rs = RandomizedSearchCV(rfc,
                                      self.hyperparams,
                                      n_iter=200,
                                      cv=3,
                                      verbose=1,
                                      n_jobs=-1,
                                      random_state=0)
        self._rs.fit(self.X_train, self.y_train)

        return self._rs.best_params_

    def arrange_results(self):
        rs_df = pd.DataFrame(self._rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
        self.rs_df = rs_df.drop([
            'mean_fit_time',
            'std_fit_time',
            'mean_score_time',
            'std_score_time',
            'params',
            'split0_test_score',
            'split1_test_score',
            'split2_test_score',
            'std_test_score'],
            axis=1)

        for row_dict in self.rs_df.to_dict(orient="records"):
            with mlflow.start_run():
                for key, value in row_dict.items():
                    if key == "mean_test_score":
                        mlflow.log_metric(key, value)
                    else:
                        mlflow.log_param(key, value)

    def params_to_score(self):
        cols = len(self.hyperparams.keys()) // 2
        fig, axs = plt.subplots(ncols=cols, nrows=2)
        sns.set(style="whitegrid", color_codes=True, font_scale=2)
        fig.set_size_inches(30, 25)
        i = 0
        j = 0
        for param in self.hyperparams.keys():
            if j == cols:
                j = 0
                i += 1
            sns.barplot(x=f"param_{param}", y='mean_test_score', data=self.rs_df, ax=axs[i, j], color='lightgrey')
            axs[i, j].set_title(label=f"{param}", size=30, weight='bold')
            j += 1
        plt.savefig(PARAMS_TO_SCORE_PATH)

    def trendline(self, data, order=1):
        coeffs = np.polyfit(list(range(data.shape[0])), list(data), order)
        slope = coeffs[-2]
        return float(slope)

    def get_insights(self):
        d = ""
        threshold = 0.01
        for col in self.rs_df:
            if col in ['rank_test_score', 'mean_test_score', "param_max_features", "param_bootstrap"]:
                continue
            t = self.rs_df.groupby([col], as_index=False).mean().groupby([col])["mean_test_score"].mean()
            df = t.sort_values()

            resultent = self.trendline(t)
            # if sequence is increasing (then the higher the better)
            if threshold < resultent:
                text = f"\nWe can see that higher values are better for {col}, {df.keys()[-1]}\n"
            # if sequence is decreasing (then the lowe the better)
            elif resultent < -threshold:
                text = f"\nWe can see that lower values are better for {col}: {df.keys()[-1]}\n"
            else:
                text = f"\nThere is no trend for {col} - the best value is {df.keys()[-1]}\n"

            d += text
            if not self.send_to_email:
                print(d)
        return d

    def get_feature_importance(self, features):
        feats = {}
        for feature, importance in zip(features, self._clf.feature_importances_):
            feats[feature] = importance
        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
        importances = importances.sort_values(by='Gini-Importance', ascending=False)
        importances = importances.reset_index()
        importances = importances.rename(columns={'index': 'Features'})

        sns.set(font_scale=5)
        sns.set(style="whitegrid", color_codes=True, font_scale=1.7)
        fig, ax = plt.subplots()
        fig.set_size_inches(30, 15)
        sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
        plt.xlabel('Importance', fontsize=25, weight='bold')
        plt.ylabel('Features', fontsize=25, weight='bold')
        plt.title('Feature Importance', fontsize=25, weight='bold')

        plt.savefig(FEATURE_IMPORTANCE_PATH)

    def train(self, columns):
        best_conf = self.get_best_hyper_params()

        self._clf = self.model(**best_conf)
        self._clf.fit(self._X, self._y)

        self.arrange_results()

        self.params_to_score()
        text = self.get_insights()

        self.get_feature_importance(columns)

        mlflow.set_tag("results", "artifacts")
        for f in [FEATURE_CORR_PATH, PARAMS_TO_SCORE_PATH, FEATURE_IMPORTANCE_PATH]:
            mlflow.log_artifact(f)
        # log best model
        mlflow.sklearn.log_model(self._clf, artifact_path=f"{str(self.model)}_{self.dataset}")

        if self.send_to_email:
            send_email(EMAIL,  # from
                       EMAIL,  # to
                       f"model {str(self.model)} for {self.dataset}.\n\n{text}",
                       FEATURE_CORR_PATH,
                       PARAMS_TO_SCORE_PATH,
                       FEATURE_IMPORTANCE_PATH)

    def predict(self, X_test, model=None):
        clf = model if model else self._clf
        if clf:
            return clf.predict(X_test)
        raise ValueError("model is not provided")


def get_features_correlations(X, y, columns):
    _norm_ = pd.DataFrame(np.hstack((X, y[:, np.newaxis])),
                          columns=list(columns) + ['class'])
    cor = sns.pairplot(_norm_, hue='class', diag_kind='hist')
    cor.savefig(FEATURE_CORR_PATH)
