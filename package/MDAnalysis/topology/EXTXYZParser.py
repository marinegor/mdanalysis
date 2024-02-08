"""
EXTXYZ Topology Parser
===================

.. versionadded:: 2.8.0

Reads an extended xyz file and pulls the atom information from it.  Because
xyz only has atom name information, all information about residues
and segments won't be populated. For the reference about EXTXYZ file format,
please refer to the ASE [documentation](https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#extxyz).

Classes
-------

.. autoclass:: EXTXYZParser
   :members:

"""

import numpy as np

from . import guessers
from ..lib.util import openany
from .base import TopologyReaderBase
from ..core.topology import Topology
from ..core.topologyattrs import (
    Atomnames,
    Atomids,
    Atomtypes,
    Masses,
    Resids,
    Resnums,
    Segids,
    Elements,
    Charges,
)


from dataclasses import dataclass, field


@dataclass
class Property:
    name: str
    typename: str
    n_cols: int
    loc: slice | int

    def __str__(self):
        return f"{self.name}:{self.typename}:{self.n_cols}"


@dataclass
class EXTXYZLayout:
    record: str
    properties: list[Property] = field(init=False)

    def __post_init__(self):
        properties = []
        values = self.record.split(":")
        fieldnames, typenames, cols = values[0::3], values[1::3], values[2::3]

        for idx, (name, typename, n_cols) in enumerate(
            zip(fieldnames, typenames, cols)
        ):
            start, stop = idx, idx + int(n_cols)
            loc = slice(start, stop) if stop > start+1 else start
            properties.append(Property(name, typename, n_cols, loc))
        self.properties = properties

    def get_loc(self, name: str, default=None) -> slice:
        for p in self.properties:
            if p.name == name:
                return p.loc
            return default

    def __str__(self):
        return ":".join((str(p) for p in self.properties))


class EXTXYZParser(TopologyReaderBase):
    """Parse a list of atoms from an XYZ file.

    Creates the following attributes:
     - Atomnames

    Guesses the following attributes:
     - Atomtypes
     - Masses

    .. versionadded:: 2.8.0

    .. versionchanged: 1.0.0
       Store elements attribute, based on XYZ atom names
    """

    format = "EXTXYZ"

    def parse(self, **kwargs):
        """Read the file and return the structure.

        Returns
        -------
        MDAnalysis Topology object
        """
        with openany(self.filename) as inf:
            natoms = int(inf.readline().strip())
            layout = EXTXYZLayout(
                [
                    line.replace('Properties=', '')
                    for line in inf.readline().strip().split()
                    if line.startswith("Properties")
                ][0]
            )

            # perhaps should try parsing them?
            names = np.zeros(natoms, dtype=object)
            mapper = {
                name: (layout.get_loc(name), np.zeros(natoms, dtype=float))
                for name in (
                    "mass",
                    "charge",
                )
                if layout.get_loc(name)
            }
            mapper.update(
                {"name": (layout.get_loc("species"), names)}
            )

            for i in range(natoms):
                split = inf.readline().split()

                for _, (loc, arr) in mapper.items():
                    arr[i] = split[loc]

        # Guessing types
        atomtypes = guessers.guess_types(names)

        # Perhaps we can assign masses and charges
        masses = (
            guessers.guess_masses(names) if "mass" not in mapper else mapper["mass"][1]
        )
        charges = (
            np.zeros(natoms, dtype=float)
            if "charge" not in mapper
            else mapper["charge"]
        )

        attrs = [
            Atomnames(names),
            Atomids(np.arange(natoms) + 1),
            Atomtypes(atomtypes, guessed=True),
            Masses(masses, guessed="mass" not in mapper),
            Resids(np.array([1])),
            Resnums(np.array([1])),
            Segids(np.array(["SYSTEM"], dtype=object)),
            Elements(names),
            Charges(charges),
        ]

        top = Topology(natoms, 1, 1, attrs=attrs)

        return top
