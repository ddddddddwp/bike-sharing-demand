import pandas as pd

from src.features.build import FeatureBuilder


def test_feature_builder_no_leakage_columns():
    df = pd.read_parquet("data/processed/train.parquet").head(2000)
    b = FeatureBuilder().fit(df)
    x = b.transform(df.head(100))
    assert "count" not in x.columns
    assert "casual" not in x.columns
    assert "registered" not in x.columns
    assert "datetime" not in x.columns
    assert len(x) == 100
