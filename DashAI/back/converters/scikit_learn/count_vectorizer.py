from sklearn.feature_extraction.text import CountVectorizer as CountVectorizerOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
    string_field,
    union_type,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class CountVectorizerSchema(BaseSchema):
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
    max_df: schema_field(
        union_type(
            float_field(ge=0.0, le=1.0),
            int_field(ge=1),
        ),
        1.0,
        "When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold.",
    )  # type: ignore
    min_df: schema_field(
        union_type(
            float_field(ge=0.0, le=1.0),
            int_field(ge=1),
        ),
        1,
        "When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.",
    )  # type: ignore
    max_features: schema_field(
        none_type(int_field(ge=1)),
        None,
        "If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.",
    )  # type: ignore
    # vocabulary: schema_field(
    #     none_type(string_field()), # Mapping or iterable
    #     None,
    #     "Mapping of terms to feature indices.",
    # )  # type: ignore
    binary: schema_field(
        bool_field(),
        False,
        "If True, all non-zero counts are set to 1.",
    )  # type: ignore
    # dtype: schema_field(
    #     none_type(enum_field(["np.float32", "np.float64"])), # dtype
    #     np.int64,
    #     "Type of the matrix returned.",
    # )  # type: ignore


class CountVectorizer(SklearnWrapper, CountVectorizerOperation):
    """Scikit-learn's CountVectorizer wrapper for DashAI."""

    SCHEMA = CountVectorizerSchema
    DESCRIPTION = "Convert a collection of text documents to a matrix of token counts."
