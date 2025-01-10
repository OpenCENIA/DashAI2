from sklearn.preprocessing import MinMaxScaler as MinMaxScalerOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class MinMaxScalerSchema(BaseSchema):
    # feature_range: schema_field(
    #     tuple, # tuple (min, max),
    #     (0, 1),
    #     "Desired range of transformed data.",
    # )  # type: ignore
    copy: schema_field(
        bool_field(),
        True,
        "Set to False to perform inplace row normalization.",
    )  # type: ignore
    clip: schema_field(
        bool_field(),
        False,
        "Set to True to clip the data to the feature range.",
    )  # type: ignore


class MinMaxScaler(SklearnWrapper, MinMaxScalerOperation):
    """Scikit-learn's MinMaxScaler wrapper for DashAI."""

    SCHEMA = MinMaxScalerSchema
    DESCRIPTION = "Transform features by scaling each feature to a given range."
