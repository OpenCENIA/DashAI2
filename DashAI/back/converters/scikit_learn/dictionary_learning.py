from sklearn.decomposition import DictionaryLearning as DictionaryLearningOperation

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


class DictionaryLearningSchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of dictionary elements to extract.",
    )  # type: ignore
    alpha: schema_field(
        float_field(ge=0.0),
        1.0,
        "Sparsity controlling parameter.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(ge=1),
        1000,
        "Maximum number of iterations to perform.",
    )  # type: ignore
    tol: schema_field(
        float_field(ge=0.0),
        1e-8,
        "Tolerance for the stopping condition.",
    )  # type: ignore
    fit_algorithm: schema_field(
        enum_field(["lars", "cd"]),
        "lars",
        "lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) "
        "cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso).",
    )  # type: ignore
    transform_algorithm: schema_field(
        enum_field(["lasso_lars", "lasso_cd", "lars", "omp", "threshold"]),
        "omp",
        (
            "Algorithm used to transform the data. "
            "lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) "
            "cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso) "
            "lasso_lars: uses Lars to compute the Lasso solution at fixed sparsity lasso_cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso) "
            "omp: uses Orthogonal Matching Pursuit to estimate the sparse coefficients (linear_model.OrthogonalMatchingPursuit) "
            "threshold: enforces sparsity by thresholding the coefficients of the full least square solution"
        ),
    )  # type: ignore
    transform_n_nonzero_coefs: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of non-zero atoms to target in pursuit algorithms.",
    )  # type: ignore
    transform_alpha: schema_field(
        none_type(float_field(ge=0.0)),
        None,
        "Sparsity controlling parameter for pursuit.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "Number of parallel jobs to run.",
    )  # type: ignore
    # code_init: schema_field(
    #     none_type(), # ndarray of shape (n_samples, n_components)
    #     None,
    #     "Initial value for the code, for warm restart. "
    #     "Only used if code_init and dict_init are not None",
    # )  # type: ignore
    # dict_init: schema_field(
    #     none_type(),  # ndarray of shape (n_components, n_features)
    #     None,
    #     "Initial value for the dictionary for warm restart. "
    #     "Only used if code_init and dict_init are not None",
    # )  # type: ignore
    verbose: schema_field(
        bool_field(),
        False,
        "Verbosity level.",
    )  # type: ignore
    split_sign: schema_field(
        bool_field(),
        False,
        "Whether to split the sparse feature vector into the sign component and the absolute values component.",
    )  # type: ignore
    random_state: schema_field(
        none_type(
            int_field(),  # int, RandomState instance or None
        ),
        None,
        "Used for initializing the dictionary when dict_init is not specified, "
        "randomly shuffling the data when shuffle is set to True, and updating the dictionary. "
        "Pass an int for reproducible results across multiple function calls.",
    )  # type: ignore
    positive_code: schema_field(
        bool_field(),
        False,
        "If positive_code is True, the code coefficients are positive.",
    )  # type: ignore
    positive_dict: schema_field(
        bool_field(),
        False,
        "If positive_dict is True, the dictionary components are positive.",
    )  # type: ignore
    transform_max_iter: schema_field(
        int_field(gt=0),
        1000,
        "Maximum number of iterations to perform for pursuit.",
    )  # type: ignore


class DictionaryLearning(SklearnWrapper, DictionaryLearningOperation):
    """Scikit-learn's DictionaryLearning wrapper for DashAI."""

    SCHEMA = DictionaryLearningSchema
    DESCRIPTION = "Finds a dictionary (a set of atoms) that performs well at sparsely encoding the fitted data."
