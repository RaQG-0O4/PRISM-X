import numpy as np
import pandas as pd
import pytest

from prismx.ml.risk_prediction import train_risk_model


def test_risk_model_explains_single_class_training_data():
    features = pd.DataFrame(
        np.random.default_rng(3).normal(size=(150, 3)),
        columns=["a", "b", "c"],
    )
    target = pd.Series(np.zeros(150, dtype=int))

    with pytest.raises(ValueError, match="severe-loss events: 0; non-events:"):
        train_risk_model(features, target)
