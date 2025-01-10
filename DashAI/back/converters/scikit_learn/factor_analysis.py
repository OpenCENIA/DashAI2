from sklearn.decomposition import FactorAnalysis as FactorAnalysisOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class FactorAnalysisSchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of components to keep.",
    )  # type: ignore
    tol: schema_field(
        float_field(ge=0.0),
        1e-2,
        "Stopping tolerance.",
    )  # type: ignore
    copy: schema_field(
        bool_field(),
        True,
        "Whether to copy X.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(ge=1),
        1000,
        "Maximum number of iterations.",
    )  # type: ignore
    # noise_variance_init: schema_field(
    #     none_type(float_field(ge=0.0)), # array-like of shape (n_features,)
    #     None,
    #     "Initial guess of the noise variance for each feature.",
    # )  # type: ignore
    svd_method: schema_field(
        enum_field(["lapack", "randomized"]),
        "randomized",
        "Method for singular value decomposition.",
    )  # type: ignore
    iterated_power: schema_field(
        int_field(ge=1),
        3,
        "Number of iterations for the power method.",
    )  # type: ignore
    rotation: schema_field(
        none_type(enum_field(["varimax", "quartimax"])),
        None,
        "Method for rotation.",
    )  # type: ignore
    random_state: schema_field(
        int_field(),  # int or RandomState instance
        0,
        "Only used when svd_method equals ‘randomized’. "
        "Pass an int for reproducible results across multiple function calls.",
    )  # type: ignore


class FactorAnalysis(SklearnWrapper, FactorAnalysisOperation):
    """Scikit-learn's FactorAnalysis wrapper for DashAI."""

    SCHEMA = FactorAnalysisSchema
    DESCRIPTION = "A simple linear generative model with Gaussian latent variables."
