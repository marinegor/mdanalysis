import numpy as np
from typing import Callable, Iterator
import re
import warnings
from collections import defaultdict

# from . import base
# from ..lib import util
# from ..lib.util import store_init_arguments

Outline = dict[str, tuple[str, int]]


def compile_ascii(pattern):
    return re.compile(pattern, re.A)


TYPE_REGEX = {"R": r"\d+\.\d*", "I": r"\d+", "S": r"\w+"}
CONCRETE_TYPES = {"R": float, "I": int, "S": str}
MANDATORY_FIELDS = ("Properties", "Lattice")
FIELDS_RE = compile_ascii(r"(?P<name>[a-zA-Z]+)=(?P<value>\".*?\"|\S+)")
PROPERTIES_RE = compile_ascii(r"((?P<name>\w+):(?P<type>[RIS]):(?P<n_cols>\d+))+")


def regex_piece(name: str, typename: str, n_cols: int) -> str:
    col_regex = TYPE_REGEX.get(typename)
    if not col_regex:
        raise ValueError(f"Wrong type name: {typename}, should be 'R', 'S' or 'I'")

    if n_cols > 1:
        return f"(?P<{name}>({col_regex}\s*){{{n_cols}}})"
    else:
        return f"(?P<{name}>{col_regex})\s*"


def create_mapper(typename: str, n_cols: int) -> Callable:
    if n_cols == 1:
        return lambda s: CONCRETE_TYPES[typename](s)
    else:
        return lambda s: list(map(CONCRETE_TYPES[typename], s.split()))


class Parser:

    CHUNKSIZE = 1024

    def __init__(self, header: str, chunksize: int = None):
        self.chunksize = chunksize or self.CHUNKSIZE

        self.fields: dict[str, str] = dict()
        for name, value in FIELDS_RE.findall(header):
            self.fields[name] = value

        for p in MANDATORY_FIELDS:
            if p not in self.fields:
                raise ValueError(f"Property {p} not in comment line {header}")

        self.properties_record: str = self.fields.pop("Properties")
        self.properties: Outline = {
            name: (typename, int(n_cols))
            for _, name, typename, n_cols in PROPERTIES_RE.findall(
                self.properties_record
            )
        }
        self.line_regex: re.Pattern = self.create_line_regex(self.properties)
        self.line_mapper: Callable = self.create_line_mapper(self.properties)

        lattice_record: str = self.fields.pop("Lattice")
        self.lattice: np.ndarray = self.create_lattice(lattice_record)

    def create_lattice(self, lattice_record: str) -> np.ndarray:
        # Lattice="R1x R1y R1z R2x R2y R2z R3x R3y R3z"
        return (
            np.array([float(i) for i in lattice_record.strip('"').split()])
            .reshape(3, 3)
            .T
        )

    def create_line_regex(self, properties: Outline) -> re.Pattern:
        regex = "".join(
            (
                regex_piece(name, typename, n_cols)
                for name, (typename, n_cols) in properties.items()
            )
        )
        return compile_ascii(regex)

    def create_line_mapper(self, properties: Outline) -> dict[str, Callable]:
        mapper = {}
        for name, (typename, n_cols) in properties.items():
            mapper[name] = create_mapper(typename, n_cols)
        return mapper

    def matches(self, chunk: str) -> Iterator:
        for match_ in self.line_regex.finditer(chunk):
            for key, value in match_.groupdict().items():
                yield key, self.line_mapper[key](value)

    def read_chunk_into(self, chunk: str, storage: dict):
        for match_ in self.line_regex.finditer(chunk):
            for key, value in match_.groupdict().items():
                storage[key].append(self.line_mapper[key](value))


# class EXTXYZReader(base.ReaderBase):
#     format = "EXTXYZ"
#     units = {"time": None, "length": "Angstrom"}
#
#     @store_init_arguments
#     def __init__(self, filename, **kwargs):
#         super().__init__(filename, **kwargs)
#
#         self.n_atoms = None
#
#         with util.openany(filename) as input_file:
#             # parse header
#             self.n_atoms = int(input_file.readline().strip())
#
#             # parse comment line
#             parser = self._parse_header(input_file.readline().strip())
#
#             self._data = parser.create_data_layout()
#
#             for chunk in input_file.readlines(1024):
#                 parser.read_chunk_into()
#
#     def _parse_header(self, header: str) -> Parser:
#         return Parser(header)
#

parser = Parser(
    'Lattice="5.44 0.0 0.0 0.0 5.44 0.0 0.0 0.0 5.44" Properties=species:S:1:pos:R:3 Time=0.0'
)

block = """Si        0.00000000      0.00000000      0.00000000
Si        1.36000000      1.36000000      1.36000000
Si        2.72000000      2.72000000      0.00000000
Si        4.08000000      4.08000000      1.36000000
Si        2.72000000      0.00000000      2.72000000
Si        4.08000000      1.36000000      4.08000000
Si        0.00000000      2.72000000      2.72000000
Si        1.36000000      4.08000000      4.08000000
"""


from time import time


from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


@timing
def test_parser(block):
    rv = parser.line_regex.findall(block)
    storage = defaultdict(list)
    parser.read_chunk_into(block, storage)
    return rv


@timing
def test_naive(block):
    splits = [line.split() for line in block.split("\n")]
    rv = {
        "species": [elem[0] for elem in splits if elem],
        "pos": [list(map(float, elem[1:])) for elem in splits if elem],
    }
    return rv

big_block = block * 1_000_000

test_parser(big_block)
test_naive(big_block)
