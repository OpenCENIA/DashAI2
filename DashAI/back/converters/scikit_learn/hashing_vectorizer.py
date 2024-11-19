from sklearn.feature_extraction.text import (
    HashingVectorizer as HashingVectorizerOperation,
)

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    enum_field,
    int_field,
    none_type,
    schema_field,
    string_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class HashingVectorizerSchema(BaseSchema):
    input: schema_field(
        enum_field(["filename", "file", "content"]),
        "content",
        "Whether the input is a filename or the content itself.",
    )  # type: ignore
    encoding: schema_field(
        string_field(),
        "utf-8",
        "The encoding of the text.",
    )  # type: ignore
    decode_error: schema_field(
        enum_field(["strict", "ignore", "replace"]),
        "strict",
        "The error handling scheme to use for decoding errors.",
    )  # type: ignore
    strip_accents: schema_field(
        none_type(enum_field(["ascii", "unicode"])),  # {‘ascii’, ‘unicode’} or callable
        None,
        "Remove accents and perform other character normalization during the preprocessing step.",
    )  # type: ignore
    lowercase: schema_field(
        bool_field(),
        True,
        "Convert all characters to lowercase before tokenizing.",
    )  # type: ignore
    # preprocessor: schema_field(
    #     none_type(string_field()), # callable
    #     None,
    #     "Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.",
    # )  # type: ignore
    # tokenizer: schema_field(
    #     none_type(string_field()),  # callable
    #     None,
    #     "Override the string tokenization step while preserving the preprocessing and n-grams generation steps.",
    # )  # type: ignore
    stop_words: schema_field(
        none_type(enum_field(["english"])),  # {‘english’} or list
        None,
        "If ‘english’, a built-in stop word list for English is used.",
    )  # type: ignore
    token_pattern: schema_field(
        none_type(string_field()),
        r"(?u)\b\w\w+\b",
        "Regular expression denoting what constitutes a ‘token’.",
    )  # type: ignore
    # ngram_range: schema_field(
    #     string_field(), # tuple
    #     "1,1",
    #     "The lower and upper boundary of the range of n-values for different n-grams to be extracted.",
    # )  # type: ignore
    analyzer: schema_field(
        enum_field(
            ["word", "char", "char_wb"]
        ),  # {‘word’, ‘char’, ‘char_wb’} or callable
        "word",
        "Whether the feature should be made of word or character n-grams.",
    )  # type: ignore
    n_features: schema_field(
        int_field(ge=1),
        2**20,
        "The number of features (columns) in the output matrix.",
    )  # type: ignore
    binary: schema_field(
        bool_field(),
        False,
        "If True, all non-zero counts are set to 1.",
    )  # type: ignore
    norm: schema_field(
        none_type(enum_field(["l1", "l2"])),
        "l2",
        "The norm used to normalize term vectors.",
    )  # type: ignore
    alternate_sign: schema_field(
        bool_field(),
        True,
        "Whether to alternate the sign of the hash value for each feature.",
    )  # type: ignore
    # dtype: schema_field(
    #     none_type(enum_field(["np.float32", "np.float64"])),  # dtype
    #     "np.float64",
    #     "Type of the matrix returned.",
    # )  # type: ignore


class HashingVectorizer(SklearnWrapper, HashingVectorizerOperation):
    """Scikit-learn's HashingVectorizer wrapper for DashAI."""

    SCHEMA = HashingVectorizerSchema
    DESCRIPTION = (
        "Convert a collection of text documents to a matrix of token occurrences."
    )
