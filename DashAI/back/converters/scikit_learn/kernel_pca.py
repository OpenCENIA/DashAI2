from sklearn.decomposition import KernelPCA as KernelPCAOperation

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


class KernelPCASchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of components to extract.",
    )  # type: ignore
    kernel: schema_field(
        enum_field(
            ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"]
        ),  # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’} or callable
        "linear",
        "Kernel to be used.",
    )  # type: ignore
    gamma: schema_field(
        none_type(float_field(gt=0.0)),
        None,
        (
            "Kernel coefficient for rbf, poly and sigmoid kernels. "
            "If None, then 1/n_features will be used."
        ),
    )  # type: ignore
    degree: schema_field(
        int_field(gt=0),
        3,
        "Degree of the polynomial kernel. Ignored by other kernels.",
    )  # type: ignore
    coef0: schema_field(
        float_field(),
        1.0,
        "Independent term in poly and sigmoid kernels. Ignored by other kernels.",
    )  # type: ignore
    # kernel_params: schema_field(
    #     none_type(dict),
    #     None,
    #     "Parameters (keyword arguments) and values for kernel passed as callable object.",
    # )  # type: ignore
    alpha: schema_field(
        float_field(ge=0.0),
        1.0,
        "Hyperparameter of the ridge regression that learns the inverse transform.",
    )  # type: ignore
    fit_inverse_transform: schema_field(
        bool_field(),
        False,
        "Learn the inverse transform for non-precomputed kernels.",
    )  # type: ignore
    eigen_solver: schema_field(
        enum_field(["auto", "dense", "arpack", "randomized"]),
        "auto",
        (
            "Select eigensolver to use. If n_components is much less than the number of samples, "
            "arpack may be more efficient than the dense eigensolver."
        ),
    )  # type: ignore
    tol: schema_field(
        float_field(ge=0.0),
        0.0,
        "Tolerance for the eigenvalue decomposition.",
    )  # type: ignore
    max_iter: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Maximum number of iterations for the eigenvalue decomposition.",
    )  # type: ignore
    iterated_power: schema_field(
        union_type(int_field(ge=0), enum_field(["auto"])),
        "auto",
        "Number of iterations for the power method. Only used by arpack.",
    )  # type: ignore
    remove_zero_eig: schema_field(
        bool_field(),
        False,
        "If True, then all components with zero eigenvalues are removed, "
        "so that the number of components in the output may be less than n_components.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Used for random shuffling when using arpack or randomized. "
        "Pass an int for reproducible results across multiple function calls.",
    )  # type: ignore
    copy_X: schema_field(
        bool_field(),
        True,
        "If True, input X will be copied for computation.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "Number of parallel jobs to run.",
    )  # type: ignore


class KernelPCA(SklearnWrapper, KernelPCAOperation):
    """Scikit-learn's KernelPCA wrapper for DashAI."""

    SCHEMA = KernelPCASchema
    DESCRIPTION = "Non-linear dimensionality reduction through the use of kernels."
