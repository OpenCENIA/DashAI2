from sklearn.cross_decomposition import PLSRegression as PLSRegressionOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    float_field,
    int_field,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class PLSRegressionSchema(BaseSchema):
    n_components: schema_field(
        int_field(ge=1),
        2,
        "Number of components to keep.",
    )  # type: ignore
    scale: schema_field(
        bool_field(),
        True,
        "Whether to scale the data.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(ge=1),
        500,
        "Maximum number of iterations to perform.",
    )  # type: ignore
    tol: schema_field(
        float_field(ge=0.0),
        1e-6,
        "Tolerance for the stopping condition.",
    )  # type: ignore
    copy: schema_field(
        bool_field(),
        True,
        "Whether to copy X and Y or perform in-place normalization.",
    )  # type: ignore


class PLSRegression(SklearnWrapper, PLSRegressionOperation):
    """Scikit-learn's PLSRegression wrapper for DashAI."""

    SCHEMA = PLSRegressionSchema
    DESCRIPTION = "Partial Least Squares Regression."
