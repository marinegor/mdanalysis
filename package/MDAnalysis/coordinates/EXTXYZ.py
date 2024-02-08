import itertools
import os
import errno
import numpy as np
import warnings
import logging

logger = logging.getLogger("MDAnalysis.coordinates.EXTXYZ")

from .. import Universe, AtomGroup
from . import base
from .timestep import Timestep
from ..lib import util
from ..lib.util import cached, store_init_arguments
from ..exceptions import NoDataError
from ..version import __version__


class EXTXYZWriter(base.WriterBase):
    format = "EXTXYZ"
    multiframe = False

    # these are assumed!
    units = {"time": "ps", "length": "Angstrom"}

    def __init__(
        self,
        filename: str,
        n_atoms: int = None,
        convert_units: bool = True,
        remark=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.filename = filename
        self.remark = remark
        self.n_atoms = n_atoms
        self.convert_units = convert_units

        # can also be gz, bz2
        self._xyz = util.anyopen(self.filename, "wt")

    def _get_atoms_elements_or_names(self, atoms: AtomGroup):
        """Return a list of atom elements (if present) or fallback to atom names"""
        try:
            return atoms.atoms.elements
        except (AttributeError, NoDataError):
            try:
                return atoms.atoms.names
            except (AttributeError, NoDataError):
                wmsg = (
                    "Input AtomGroup or Universe does not have atom "
                    "elements or names attributes, writer will default "
                    "atom names to 'X'"
                )
                warnings.warn(wmsg)
                return itertools.cycle(("X",))

    def close(self):
        """Close the trajectory file and finalize the writing"""
        if self._xyz is not None:
            self._xyz.write("\n")
            self._xyz.close()
        self._xyz = None

    def write(self, obj: Universe | AtomGroup):
        if not isinstance(obj, (Universe, AtomGroup)):
            raise TypeError("Input obj is neither a Universe nor an AtomGroup")

        atoms = obj.atoms

        if hasattr(obj, "universe"):
            ts_full = obj.universe.trajectory.ts
            if ts_full.n_atoms == atoms.n_atoms:
                ts = ts_full
            else:
                ts = ts_full.copy_slice(atoms.indices)
        elif hasattr(obj, "trajectory"):
            ts = obj.trajectory.ts

        self.atomnames = self._get_atoms_elements_or_names(atoms)
        self._write_next_frame(ts)
