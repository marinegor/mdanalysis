"""Analysis backends --- :mod:`MDAnalysis.analysis.backends`

Module introduces :class:`BackendBase` base class to implement custom
backends for :meth:`MDAnalysis.analysis.base.AnalysisBase.run()` and its
subclasses. Also, it introduces 2 built-in backend classes:
:class:`BackendMultiprocessing` that supports parallelization via standard
python ``multiprocessing`` module, and :class:`BackendDask`, that uses the same
process-based parallelization as :class:`BackendMultiprocessing`, but different
serialization algorithm.
==============================================================
"""
import warnings
from typing import Callable
from MDAnalysis.lib.util import is_installed


class BackendBase:
    """Base class for backend implementation. Initializes an instance and performs
    checks for its validity, such as n_workers and possibly other ones.

    Parameters
    ----------
    n_workers : int
        number of workers (usually, processes) over which the work is split

    Examples
    --------
    .. code-block:: python
        # implement a thread-based backend
        from MDAnalysis.analysis.backends import BackendBase
        class ThreadsBackend(BackendBase):
            def apply(self, func, computations):
                from multiprocessing.dummy import Pool

                with Pool(processes=self.n_workers) as pool:
                    results = pool.map(func, computations)
                return results
        from MDAnalysis.analysis.rms import RMSD
        R = RMSD(...) # setup the run
        n_workers = 2
        backend = ThreadsBackend(n_workers=n_workers)
        R.run(backend=backend)

    .. versionadded:: 2.8.0
    """

    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self._validate()

    def _get_checks(self):
        """Get dictionary with `condition: error_message` pairs that ensure the
        validity of the backend instance

        Returns
        -------
        dict
            dictionary with `condition: error_message` pairs that will get
            checked during _validate() run

        .. versionadded:: 2.8.0
        """
        return {
            isinstance(self.n_workers, int) and self.n_workers > 0:
            f"n_workers should be positive integer, got {self.n_workers=}",
        }

    def _get_warnings(self):
        """Get dictionary with `condition: warning_message` pairs that ensure
        the good usage of the backend instance

        Returns
        -------
        dict
            dictionary with `condition: warning_message` pairs that will get
            checked during _validate() run

        .. versionadded:: 2.8.0
        """
        return dict()

    def _validate(self):
        """Check correctness (e.g. `dask` is installed if using `backend='dask'`)
        and good usage (e.g. `n_workers=1` if backend is serial) of the backend

        Raises
        ------
        ValueError
            if one of the conditions in :meth:`self._get_checks()` is True

        .. versionadded:: 2.8.0
        """
        for check, msg in self._get_checks().items():
            if not check:
                raise ValueError(msg)
        for check, msg in self._get_warnings().items():
            if not check:
                warnings.warn(msg)

    def apply(self, func: Callable, computations: list) -> list:
        """Main function that will get called when using an instance of an object,
        mapping function to all tasks in the `computations` list. Should effectively
        be equivalent to running `[func(item) for item in computations]`
        while using the parallel backend capabilities.

        Parameters
        ----------
        func : Callable
            function to be called on each of the tasks in computations list
        computations : list
            computation tasks to apply function to

        Returns
        -------
        list
            list of results of the function

        .. versionadded:: 2.8.0
        """
        raise NotImplementedError


class BackendSerial(BackendBase):
    """A built-in backend that does serial execution of the function, without any
    parallelization

    .. versionadded:: 2.8.0
    """

    def _get_warnigns(self):
        return {
            self.n_workers > 1,
            "n_workers is ignored when executing with backend='serial'"
        }

    def apply(self, func: Callable, computations: list) -> list:
        return [func(task) for task in computations]


class BackendMultiprocessing(BackendBase):
    """A built-in backend that executes a given function using
    multiprocessing.Pool.map method

    .. versionadded:: 2.8.0
    """

    def apply(self, func: Callable, computations: list) -> list:
        from multiprocessing import Pool

        with Pool(processes=self.n_workers) as pool:
            results = pool.map(func, computations)
        return results


class BackendDask(BackendBase):
    """A built-in backend that executes a given function using dask.delayed.compute
    method with `scheduler='processes'` and `chunksize=1` (this ensures uniform
    distribution of tasks among processes). Requires `dask` module to be installed;
    see [documentation](https://docs.dask.org/en/stable/install.html)

    .. versionadded:: 2.8.0
    """

    def apply(self, func: Callable, computations: list) -> list:
        from dask.delayed import delayed
        import dask

        computations = [delayed(func)(task) for task in computations]
        results = dask.compute(computations,
                               scheduler="processes",
                               chunksize=1,
                               n_workers=self.n_workers)[0]
        return results

    def _get_checks(self):
        base_checks = super()._get_checks()
        checks = {
            is_installed("dask"):
            ("module 'dask' should be installed:"
             "https://docs.dask.org/en/stable/install.html")
        }
        return base_checks | checks
