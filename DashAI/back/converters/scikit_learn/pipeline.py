from DashAI.back.converters.base_converter import BaseConverter
from DashAI.back.converters.scikit_learn.sklearn_like_converter import (
    SklearnLikeConverter,
)


class Pipeline(BaseConverter, SklearnLikeConverter):
    """Pipeline of transforms with a final estimator."""

    DESCRIPTION = (
        "A Pipeline applies a sequence of converters to preprocess "
        "data, passing the output of one converter to the next, with "
        "its scope defined by the first converter."
    )

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for step in self.steps:
            step.fit(X, y)
        return self

    def transform(self, X, y=None):
        for step in self.steps:
            X = step.transform(X, y)
        return X
