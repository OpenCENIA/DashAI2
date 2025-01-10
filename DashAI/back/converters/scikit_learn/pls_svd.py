from sklearn.cross_decomposition import PLSSVD as PLSSVDOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    int_field,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class PLSSVDSchema(BaseSchema):
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
    copy: schema_field(
        bool_field(),
        True,
        "Whether to copy X and Y in fit before applying centering, and potentially scaling.",
    )  # type: ignore


class PLSSVD(SklearnWrapper, PLSSVDOperation):
    """Scikit-learn's PLSSVD wrapper for DashAI."""

    SCHEMA = PLSSVDSchema
    DESCRIPTION = "Partial Least Squares SVD."
