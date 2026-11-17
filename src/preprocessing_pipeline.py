# This preprocessing pipeline is to be fitted using the training data only, then used to transform the training, validation, and test data.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

numerical_features = ['duration', 'amount', 'age']

nominal_features = [
    'status', 'credit_history', 'purpose', 'savings', 'personal_status_sex',
    'other_debtors', 'other_installment_plans', 'housing', 'people_liable',
    'telephone', 'foreign_worker',
]

ordinal_features = [
    'employment_duration', 'installment_rate', 'present_residence',
    'property', 'number_credits', 'job',
]

TARGET = 'credit_risk'
ordinal_categories = {
    'employment_duration': [
        'unemployed', 
        '< 1 yr', 
        '1 <= ... < 4 yrs', 
        '4 <= ... < 7 yrs', 
        '>= 7 yrs',
    ],
    
    'installment_rate': [
        '>= 35', 
        '25 <= ... < 35', 
        '20 <= ... < 25', 
        '< 20'],
    
    'present_residence': [
        '< 1 yr', 
        '1 <= ... < 4 yrs', 
        '4 <= ... < 7 yrs', 
        '>= 7 yrs'],
    
    'property': [
        'unknown / no property', 
        'car or other',
        'building soc. savings agr./life insurance', 
        'real estate',
    ],
    
    'number_credits': [
        '1', 
        '2-3', 
        '4-5', 
        '>= 6'
    ],
    
    'job': [
        'unemployed/unskilled - non-resident', 
        'unskilled - resident',
        'skilled employee/official', 
        'manager/self-empl./highly qualif. employee',
    ],
}

continuous_features = numerical_features + ['amount_per_month']


def engineer_features(X):
    """Return a copy with fixed features; accepts decoded future observations.

    amount_per_month is a transformed-amount/month proxy, not a currency payment.
    """
    result = X.copy()
    duration = pd.to_numeric(result['duration'], errors='raise')
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError('duration must be finite and strictly positive')
    result['amount_per_month'] = result['amount'] / duration
    return result


class Winsorizer(TransformerMixin, BaseEstimator):
    """Learn per-column quantile limits only from rows supplied to fit."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError('Require 0 <= lower < upper <= 1')
        values = validate_data(self, X, dtype=float, reset=True)
        self.low_ = np.quantile(values, self.lower, axis=0)
        self.high_ = np.quantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ['low_', 'high_'])
        values = validate_data(self, X, dtype=float, reset=False)
        return np.clip(values, self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'n_features_in_')
        names = getattr(self, 'feature_names_in_', None)
        if input_features is None:
            return names.copy() if names is not None else np.asarray(
                [f'x{i}' for i in range(self.n_features_in_)], dtype=object)
        names_in = np.asarray(input_features, dtype=object)
        if len(names_in) != self.n_features_in_:
            raise ValueError('input_features must match the fitted feature count')
        if names is not None and not np.array_equal(names, names_in):
            raise ValueError('input_features must match feature_names_in_')
        return names_in


def build_continuous_pipeline():
    return Pipeline([
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('winsorize', Winsorizer(lower=0.01, upper=0.99)),
        ('scale', StandardScaler()),
    ])


def _build_encoder(continuous):
    ordinal_pipeline = Pipeline([
        ('encode', OrdinalEncoder(
            categories=[ordinal_categories[f] for f in ordinal_features],
            handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scale', StandardScaler()),
    ])
    return ColumnTransformer([
        ('continuous', continuous, continuous_features),
        ('ordinal', ordinal_pipeline, ordinal_features),
        ('nominal', OneHotEncoder(handle_unknown='ignore', drop='if_binary',
                                  sparse_output=False), nominal_features),
    ], remainder='drop', verbose_feature_names_out=False).set_output(transform='pandas')


def build_preprocessor():
    """Fresh unfitted transformer for no sampling, class weights, or duplication."""
    return _build_encoder(build_continuous_pipeline())
