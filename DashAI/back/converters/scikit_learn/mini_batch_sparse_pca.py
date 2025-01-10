from sklearn.decomposition import MiniBatchSparsePCA as MiniBatchSparsePCAOperation

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


class MiniBatchSparsePCASchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of sparse atoms to extract.",
    )  # type: ignore
    alpha: schema_field(
        int_field(gt=0.0),
        1,
        "Sparsity controlling parameter.",
    )  # type: ignore
    ridge_alpha: schema_field(
        float_field(gt=0.0),
        0.01,
        "Amount of ridge shrinkage to apply in order to improve conditioning.",
    )  # type: ignore
    max_iter: schema_field(
        none_type(int_field(gt=0)),
        None,
        "Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.",
    )  # type: ignore
    # callback: schema_field(
    #     none_type(), # callable
    #     None,
    #     "If not None, callback function to call after each iteration.",
    # )  # type: ignore
    batch_size: schema_field(
        int_field(gt=0),
        3,
        "The number of samples to take in each batch.",
    )  # type: ignore
    verbose: schema_field(
        union_type(int_field(ge=0), bool_field()),
        False,
        "Level of verbosity.",
    )  # type: ignore
    shuffle: schema_field(
        bool_field(),
        True,
        "If True, the order of the samples is shuffled.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "Number of jobs to run in parallel.",
    )  # type: ignore
    method: schema_field(
        enum_field(["lars", "cd"]),
        "lars",
        "Algorithm used to solve the lasso problem.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Used for random shuffling when shuffle is set to True, during online dictionary learning. "
        "Pass an int for reproducible results across multiple function calls.",
    )  # type: ignore
    tol: schema_field(
        float_field(gt=0.0),
        1e-3,
        "Tolerance for the stopping condition.",
    )  # type: ignore
    max_no_improvement: schema_field(
        none_type(int_field(gt=0)),
        10,
        "Maximum number of iterations with no improvement to wait before early stopping.",
    )  # type: ignore


class MiniBatchSparsePCA(SklearnWrapper, MiniBatchSparsePCAOperation):
    """Scikit-learn's MiniBatchSparsePCA wrapper for DashAI."""

    SCHEMA = MiniBatchSparsePCASchema
    DESCRIPTION = "Finds the set of sparse components that can optimally reconstruct the data. The amount of sparseness is controllable by the coefficient of the L1 penalty, given by the parameter alpha."
