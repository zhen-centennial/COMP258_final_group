import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib


def change_STAY_STATUS(row):
    if 'Left College' in row:
        return 'Left College'
    if 'Completed' in row:
        return 'Completed'
    if 'Graduated' in row:
        return 'Graduated'

class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if 'ADMIT TERM CODE' in X.columns:
            X.drop(columns=['ADMIT TERM CODE'], inplace=True)
        if 'ADMIT YEAR' in X.columns:
            X.drop(columns=['ADMIT YEAR'], inplace=True)
        if 'ID 2' in X.columns:
            X.drop(columns=['ID 2'], inplace=True)
        if 'FUTURE TERM ENROL' in X.columns:
            X.drop(columns=['FUTURE TERM ENROL'], inplace=True)

        return X[self.cols]

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['MAILING COUNTRY NAME'].fillna('Canada', inplace=True)
        X['PREV EDU CRED LEVEL NAME'].fillna(X['PREV EDU CRED LEVEL NAME'].mode()[0], inplace=True)
        X['APPLICANT CATEGORY NAME'].fillna(X['APPLICANT CATEGORY NAME'].mode()[0], inplace=True)
        X['AGE GROUP LONG NAME'].fillna(X['AGE GROUP LONG NAME'].mode()[0], inplace=True)

        X['CURRENT STAY STATUS'] = X['CURRENT STAY STATUS'].apply(change_STAY_STATUS)

        return X[self.cols]