from sklearn.decomposition import NMF as NMFOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
    union_type,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class NMFSchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of components to extract.",
    )  # type: ignore
    init: schema_field(
        none_type(enum_field(["random", "nndsvd", "nndsvda", "nndsvdar", "custom"])),
        None,
        "Method used to initialize the procedure.",
    )  # type: ignore
    solver: schema_field(
        enum_field(["cd", "mu"]),
        "cd",
        "Numerical solver to use.",
    )  # type: ignore
    beta_loss: schema_field(
        union_type(
            float_field(),
            enum_field(["frobenius", "kullback-leibler", "itakura-saito"]),
        ),
        "frobenius",
        "String must be 'frobenius', 'kullback-leibler', or 'itakura-saito'.",
    )  # type: ignore
    tol: schema_field(
        float_field(gt=0.0),
        1e-4,
        "Tolerance of the stopping condition.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(gt=0),
        200,
        "Maximum number of iterations before timing out.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Seed used by the random number generator.",
    )  # type: ignore
    alpha_W: schema_field(
        float_field(gt=0.0),
        0.0,
        "Constant that multiplies the regularization terms of W.",
    )  # type: ignore
    alpha_H: schema_field(
        float_field(gt=0.0),
        0.0,
        "Constant that multiplies the regularization terms of H.",
    )  # type: ignore
    l1_ratio: schema_field(
        float_field(ge=0.0, le=1.0),
        0.0,
        "The regularization mixing parameter.",
    )  # type: ignore
    verbose: schema_field(
        int_field(ge=0),
        0,
        "Level of verbosity.",
    )  # type: ignore
    shuffle: schema_field(
        bool_field(),
        False,
        "If True, randomize the order of coordinates in the CD solver.",
    )  # type: ignore


class NMF(SklearnWrapper, NMFOperation):
    """Scikit-learn's NMF wrapper for DashAI."""

    SCHEMA = NMFSchema
    DESCRIPTION = "Find two non-negative matrices, i.e. matrices with all non-negative elements, (W, H) whose product approximates the non-negative matrix X. This factorization can be used for example for dimensionality reduction, source separation or topic extraction."
