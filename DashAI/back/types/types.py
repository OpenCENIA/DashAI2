from dataclasses import dataclass

from base_value_type import BaseValue
from datasets import Value


@dataclass
class Integer(BaseValue):
    size: int = 64
    unsigned: bool = False

    def __post_init__(self):
        if self.size not in {8, 16, 32, 64}:
            self.size = 64

        self.dtype = f"uint{self.size}" if self.unsigned else f"int{self.size}"
        super().__post_init__()

    @staticmethod
    def from_value(value: Value):
        if not value.dtype.startswith(("int", "uint")):
            raise ValueError(f"dtype {value.dtype} is not an integer")

        unsigned: bool = value.dtype.startswith("uint")
        size: int = int(value.dtype[4:]) if unsigned else int(value.dtype[3:])

        return Integer(size=size, unsigned=unsigned)


@dataclass
class Float(BaseValue):
    size: int = 64

    def __post_init__(self):
        if self.size not in {16, 32, 64}:
            self.size = 64

        self.dtype = f"float{self.size}"
        super().__post_init__()

    @staticmethod
    def from_value(value: Value):
        if not value.dtype.startswith("float"):
            raise ValueError(f"dtype {value.dtype} is not a float")

        size: int = int(value.dtype[5:])
        return Float(size=size)


@dataclass
class Text(BaseValue):
    string_type: str = "string"

    def __post_init__(self):
        if self.string_type not in ("large_string", "string"):
            self.string_type = "string"

        self.dtype = self.string_type
        super().__post_init__()

    @staticmethod
    def from_value(value: Value):
        if value.dtype not in ("string", "large_string"):
            raise ValueError(f"dtype {value.dtype} is not a string")

        return Text(string_type=value.dtype)


if __name__ == "__main__":
    int_val = Integer()
    text_val = Text()
    float_val = Float()

    print(int_val)
    print(text_val)
    print(float_val)
