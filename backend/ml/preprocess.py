"""Shared, leak-free preprocessing for all models.

Design rules (see docs/model_details.md):
- Every transformer here is used INSIDE a sklearn Pipeline, so it is fitted on
  training folds only and then applied unchanged to validation/test data.
- Potability (the label) is never a feature.
- Outlier handling is clipping to 1.5*IQR fences *learned from the training
  portion of each split/fold*, not from the full dataset.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MedianImputer(BaseEstimator, TransformerMixin):
    """Median-impute NaNs using medians learned from fit data only."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.medians_ = X.median(numeric_only=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.fillna(self.medians_)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] from the fit data only."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1)


def make_feature_frame(df: pd.DataFrame, feature_cols):
    """Return X (features only) and ensure the label never sneaks in."""
    X = df[feature_cols].copy()
    if "Potability" in X.columns:
        raise AssertionError("Leakage guard: Potability must never be in features")
    return X


def build_classifier_pipeline(estimator, with_smote: bool = False, smote_kwargs: dict | None = None):
    """Pipeline: impute -> clip outliers -> (optional SMOTE) -> estimator.

    SMOTE is applied only inside CV folds via imblearn's Pipeline; class_weight
    is the simpler alternative for plain fits. smote_kwargs lets callers tune
    e.g. k_neighbors for tiny datasets (tests/smoke runs).
    """
    smote_kwargs = smote_kwargs or {}
    if with_smote:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        return ImbPipeline(
            steps=[
                ("impute", MedianImputer()),
                ("clip", IQRClipper()),
                ("scale", StandardScalerSafe()),
                ("smote", SMOTE(random_state=42, **smote_kwargs)),
                ("model", estimator),
            ]
        )
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("impute", MedianImputer()),
            ("clip", IQRClipper()),
            ("scale", StandardScalerSafe()),
            ("model", estimator),
        ]
    )


def build_regressor_pipeline(estimator):
    """pH regressor pipeline: impute -> clip -> scale -> estimator."""
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("impute", MedianImputer()),
            ("clip", IQRClipper()),
            ("scale", StandardScalerSafe()),
            ("model", estimator),
        ]
    )


def StandardScalerSafe():
    from sklearn.preprocessing import StandardScaler

    return StandardScaler()
