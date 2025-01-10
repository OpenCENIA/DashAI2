from sklearn.decomposition import MiniBatchNMF as MiniBatchNMFOperation

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


class MiniBatchNMFSchema(BaseSchema):
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
    batch_size: schema_field(
        int_field(gt=0),
        1024,
        "Number of samples in each mini-batch.",
    )  # type: ignore
    beta_loss: schema_field(
        union_type(
            float_field(),
            enum_field(["frobenius", "kullback-leibler", "itakura-saito"]),
        ),
        "frobenius",
        "Beta divergence to be minimized, measuring the distance between X and the dot product WH.",
    )  # type: ignore
    tol: schema_field(
        float_field(gt=0.0),
        1e-4,
        "Tolerance of the stopping condition.",
    )  # type: ignore
    max_no_improvement: schema_field(
        int_field(gt=0),
        10,
        "Number of iterations with no improvement to wait before stopping the optimization.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(gt=0),
        200,
        "Maximum number of iterations before timing out.",
    )  # type: ignore
    alpha_W: schema_field(
        float_field(gt=0.0),
        0.0,
        "Regularization parameter for W.",
    )  # type: ignore
    alpha_H: schema_field(
        union_type(float_field(gt=0.0), enum_field(["same"])),
        "same",
        "Regularization parameter for H.",
    )  # type: ignore
    l1_ratio: schema_field(
        float_field(ge=0.0, le=1.0),
        0.0,
        "The regularization mixing parameter.",
    )  # type: ignore
    forget_factor: schema_field(
        float_field(gt=0.0),
        0.7,
        "Forget factor to be used in the online learning mechanism.",
    )  # type: ignore
    fresh_restarts: schema_field(
        bool_field(),
        False,
        "Whether to use fresh restarts in the online learning mechanism.",
    )  # type: ignore
    fresh_restarts_max_iter: schema_field(
        int_field(gt=0),
        30,
        "Maximum number of iterations when solving for W at each step. Only used when doing fresh restarts.",
    )  # type: ignore
    transform_max_iter: schema_field(
        none_type(int_field(gt=0)),
        None,
        "Maximum number of iterations when solving for W at transform time.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Used for initialisation (when init == ‘nndsvdar’ or ‘random’), and in Coordinate Descent. Pass an int for reproducible results across multiple function calls.",
    )  # type: ignore
    verbose: schema_field(
        bool_field(),
        False,
        "Whether to be verbose.",
    )  # type: ignore


class MiniBatchNMF(SklearnWrapper, MiniBatchNMFOperation):
    """Scikit-learn's MiniBatchNMF wrapper for DashAI."""

    SCHEMA = MiniBatchNMFSchema
    DESCRIPTION = "Find two non-negative matrices, i.e. matrices with all non-negative elements, (W, H) whose product approximates the non-negative matrix X. This factorization can be used for example for dimensionality reduction, source separation or topic extraction."
