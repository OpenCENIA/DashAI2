from sklearn.decomposition import (
    MiniBatchDictionaryLearning as MiniBatchDictionaryLearningOperation,
)

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


class MiniBatchDictionaryLearningSchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of dictionary elements to extract.",
    )  # type: ignore
    alpha: schema_field(
        float_field(gt=0.0),
        1.0,
        "Sparsity controlling parameter.",
    )  # type: ignore
    max_iter: schema_field(
        none_type(int_field(gt=0)),
        None,
        "Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.",
    )  # type: ignore
    fit_algorithm: schema_field(
        enum_field(["lars", "cd"]),
        "lars",
        "Algorithm used to solve the lasso problem.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "Number of jobs to run in parallel.",
    )  # type: ignore
    batch_size: schema_field(
        int_field(gt=0),
        3,
        "The number of samples to take in each batch.",
    )  # type: ignore
    shuffle: schema_field(
        bool_field(),
        True,
        "If True, the order of the samples is shuffled.",
    )  # type: ignore
    # dict_init: schema_field(
    #     none_type(), # ndarray of shape (n_components, n_features)
    #     None,
    #     "Dictionary initialization method.",
    # )  # type: ignore
    transform_algorithm: schema_field(
        enum_field(["lasso_lars", "lasso_cd", "lars", "omp", "threshold"]),
        "omp",
        "Algorithm used to transform the data.",
    )  # type: ignore
    transform_n_nonzero_coefs: schema_field(
        none_type(int_field(gt=0)),
        None,
        "Number of nonzero coefficients to target in transform step.",
    )  # type: ignore
    transform_alpha: schema_field(
        none_type(float_field(gt=0.0)),
        None,
        "Sparsity controlling parameter for transform step.",
    )  # type: ignore
    verbose: schema_field(
        int_field(),
        0,
        "Verbosity level.",
    )  # type: ignore
    split_sign: schema_field(
        bool_field(),
        False,
        "If True, the dictionary is split in atoms with positive and negative values.",
    )  # type: ignore
    random_state: schema_field(
        int_field(),  # int, RandomState instance or None
        None,
        "Seed for the random number generator.",
    )  # type: ignore
    positive_code: schema_field(
        bool_field(),
        False,
        "If True, the code vectors are positive.",
    )  # type: ignore
    positive_dict: schema_field(
        bool_field(),
        False,
        "If True, the dictionary is positive.",
    )  # type: ignore
    transform_max_iter: schema_field(
        int_field(gt=0),
        1000,
        "Maximum number of iterations for the transform step.",
    )  # type: ignore
    # callback: schema_field(
    #     none_type(object), # callable
    #     None,
    #     "Callable that gets invoked every five iterations.",
    # )  # type: ignore
    tol: schema_field(
        float_field(gt=0.0),
        1e-3,
        "Tolerance for the stopping condition.",
    )  # type: ignore
    max_no_improvement: schema_field(
        int_field(gt=0),
        10,
        "Maximum number of iterations with no improvement in the fit.",
    )  # type: ignore


class MiniBatchDictionaryLearning(SklearnWrapper, MiniBatchDictionaryLearningOperation):
    """Scikit-learn's MiniBatchDictionaryLearning wrapper for DashAI."""

    SCHEMA = MiniBatchDictionaryLearningSchema
    DESCRIPTION = "Finds a dictionary (a set of atoms) that performs well at sparsely encoding the fitted data."
