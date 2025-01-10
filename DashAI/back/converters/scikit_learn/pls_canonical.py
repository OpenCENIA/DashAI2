from sklearn.cross_decomposition import PLSCanonical as PLSCanonicalOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    enum_field,
    float_field,
    int_field,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class PLSCanonicalSchema(BaseSchema):
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
    algorithm: schema_field(
        enum_field(["nipals", "svd"]),
        "nipals",
        "Algorithm to use.",
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
        "Whether to copy X and Y in fit before applying centering, and potentially scaling.",
    )  # type: ignore


class PLSCanonical(SklearnWrapper, PLSCanonicalOperation):
    """Scikit-learn"s PLSCanonical wrapper for DashAI."""

    SCHEMA = PLSCanonicalSchema
    DESCRIPTION = "Partial Least Squares transformer and regressor."
