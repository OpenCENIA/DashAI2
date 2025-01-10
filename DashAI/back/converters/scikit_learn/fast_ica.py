from sklearn.decomposition import FastICA as FastICAOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class FastICASchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of components to extract.",
    )  # type: ignore
    algorithm: schema_field(
        enum_field(["parallel", "deflation"]),
        "parallel",
        "Apply parallel or deflational algorithm for FastICA.",
    )  # type: ignore
    # Deprecated since version 1.1
    # whiten: schema_field(
    #     none_type(
    #         union_type(
    #             enum_field(["arbitrary-variance", "unit-variance"]), bool_field()
    #         )
    #     ),
    #     "unit-variance",
    #     "If True, the data is whitened.",
    # )  # type: ignore
    fun: schema_field(
        enum_field(
            ["logcosh", "exp", "cube"]
        ),  # {‘logcosh’, ‘exp’, ‘cube’} or callable
        "logcosh",
        "The functional form of the G function used in the approximation to neg-entropy.",
    )  # type: ignore
    # fun_args: schema_field(
    #     dict,
    #     {"logcosh": 1.0, "exp": 1.0, "cube": 1.0},
    #     "Arguments to the G function.",
    # )  # type: ignore
    max_iter: schema_field(
        int_field(ge=1),
        200,
        "Maximum number of iterations to perform.",
    )  # type: ignore
    tol: schema_field(
        float_field(ge=0.0),
        1e-04,
        "Tolerance on update at each iteration.",
    )  # type: ignore
    # w_init: schema_field(
    #     none_type(float_field()), # array-like of shape (n_components, n_components)
    #     None,
    #     "Initial guess for the unmixing matrix.",
    # )  # type: ignore
    whiten_solver: schema_field(
        enum_field(["eigh", "svd"]),
        "svd",
        "The solver to use for whitening.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Used to initialize w_init when not specified, with a normal distribution. "
        "Pass an int, for reproducible results across multiple function calls.",
    )  # type: ignore


class FastICA(SklearnWrapper, FastICAOperation):
    """Scikit-learn's FastICA wrapper for DashAI."""

    SCHEMA = FastICASchema
    DESCRIPTION = "FastICA: a fast algorithm for Independent Component Analysis."
