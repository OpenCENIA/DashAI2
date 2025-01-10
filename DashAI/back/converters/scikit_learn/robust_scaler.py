from sklearn.preprocessing import RobustScaler as RobustScalerOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class RobustScalerSchema(BaseSchema):
    with_centering: schema_field(
        bool_field(),
        True,
        "If True, center the data before scaling.",
    )  # type: ignore
    with_scaling: schema_field(
        bool_field(),
        True,
        "If True, scale the data to the IQR.",
    )  # type: ignore
    # quantile_range: schema_field(
    #     float, # tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
    #     25.0,
    #     "The IQR range used to scale the data.",
    # )  # type: ignore
    copy: schema_field(
        bool_field(),
        True,
        "Set to False to perform inplace scaling.",
    )  # type: ignore
    unit_variance: schema_field(
        bool_field(),
        False,
        "If True, scale the data to unit variance.",
    )  # type: ignore


class RobustScaler(SklearnWrapper, RobustScalerOperation):
    """Scikit-learn's RobustScaler wrapper for DashAI."""

    SCHEMA = RobustScalerSchema
    DESCRIPTION = "Scale features using statistics that are robust to outliers."
