from sklearn.decomposition import SparsePCA as SparsePCAOperation

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


class SparsePCASchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of sparse atoms to extract.",
    )  # type: ignore
    alpha: schema_field(
        float_field(gt=0),
        1.0,
        "Sparsity controlling parameter.",
    )  # type: ignore
    ridge_alpha: schema_field(
        float_field(gt=0),
        0.01,
        "Amount of ridge shrinkage to apply in order to improve conditioning.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(gt=0),
        1000,
        "Maximum number of iterations to perform.",
    )  # type: ignore
    tol: schema_field(
        float_field(gt=0),
        1e-8,
        "Tolerance for the stopping condition.",
    )  # type: ignore
    method: schema_field(
        enum_field(["lars", "cd"]),
        "lars",
        "Method used to solve the optimization problem.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "Number of parallel jobs to run.",
    )  # type: ignore
    # U_init: schema_field(
    #     none_type(float_field()), # ndarrat of shape (n_samples, n_components)
    #     None,
    #     "Initial values for the loadings for warm restart scenarios.",
    # )  # type: ignore
    # V_init: schema_field(
    #     none_type(float_field()), # ndarrat of shape (n_features, n_components)
    #     None,
    #     "Initial values for the components for warm restart scenarios.",
    # )  # type: ignore
    verbose: schema_field(
        union_type(int_field(ge=0), bool_field()),
        False,
        "Level of verbosity.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Used during dictionary learning. Pass an int for reproducible results across multiple function calls.",
    )  # type: ignore


class SparsePCA(SklearnWrapper, SparsePCAOperation):
    """Scikit-learn's SparsePCA wrapper for DashAI."""

    SCHEMA = SparsePCASchema
    DESCRIPTION = "Finds the set of sparse components that can optimally reconstruct the data. The amount of sparseness is controllable by the coefficient of the L1 penalty, given by the parameter alpha."
