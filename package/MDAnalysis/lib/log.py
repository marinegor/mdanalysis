# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#


"""Setting up logging --- :mod:`MDAnalysis.lib.log`
====================================================

Configure logging for MDAnalysis. Import this module if logging is
desired in application code.

Logging to a file and the console is set up by default as described
under `logging to multiple destinations`_.

The top level logger of the library is named *MDAnalysis* by
convention; a simple logger that writes to the console and logfile can
be created with the :func:`create` function. This only has to be done
*once*. For convenience, the default MDAnalysis logger can be created
with :func:`MDAnalysis.start_logging`::

 import MDAnalysis
 MDAnalysis.start_logging()

Once this has been done, MDAnalysis will write messages to the logfile
(named `MDAnalysis.log` by default but this can be changed with the
optional argument to :func:`~MDAnalysis.start_logging`).

Any code can log to the MDAnalysis logger by using ::

 import logging
 logger = logging.getLogger('MDAnalysis.MODULENAME')

 # use the logger, for example at info level:
 logger.info("Starting task ...")

The important point is that the name of the logger begins with
"MDAnalysis.".

.. _logging to multiple destinations:
   http://docs.python.org/library/logging.html?#logging-to-multiple-destinations

Note
----
The :mod:`logging` module in the standard library contains in depth
documentation about using logging.


Convenience functions
---------------------

Two convenience functions at the top level make it easy to start and
stop the default *MDAnalysis* logger.

.. autofunction:: MDAnalysis.start_logging
.. autofunction:: MDAnalysis.stop_logging


Other functions and classes for logging purposes
------------------------------------------------


.. versionchanged:: 2.0.0
   Deprecated :class:`MDAnalysis.lib.log.ProgressMeter` has now been removed.

.. autogenerated, see Online Docs

"""

import logging

from tqdm.auto import tqdm

from .. import version


def start_logging(logfile="MDAnalysis.log", version=version.__version__):
    """Start logging of messages to file and console.

    The default logfile is named `MDAnalysis.log` and messages are
    logged with the tag *MDAnalysis*.
    """
    create("MDAnalysis", logfile=logfile)
    logging.getLogger("MDAnalysis").info(
        "MDAnalysis %s STARTED logging to %r", version, logfile
    )


def stop_logging():
    """Stop logging to logfile and console."""
    logger = logging.getLogger("MDAnalysis")
    logger.info("MDAnalysis STOPPED logging")
    clear_handlers(logger)  # this _should_ do the job...


def create(logger_name="MDAnalysis", logfile="MDAnalysis.log"):
    """Create a top level logger.

    - The file logger logs everything (including DEBUG).
    - The console logger only logs INFO and above.

    Logging to a file and the console as described under `logging to
    multiple destinations`_.

    The top level logger of MDAnalysis is named *MDAnalysis*.  Note
    that we are configuring this logger with console output. If a root
    logger also does this then we will get two output lines to the
    console.

    .. _logging to multiple destinations:
       http://docs.python.org/library/logging.html?#logging-to-multiple-destinations
    """

    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG)

    # handler that writes to logfile
    logfile_handler = logging.FileHandler(logfile)
    logfile_formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    )
    logfile_handler.setFormatter(logfile_formatter)
    logger.addHandler(logfile_handler)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def clear_handlers(logger):
    """clean out handlers in the library top level logger

    (only important for reload/debug cycles...)

    """
    for h in logger.handlers:
        logger.removeHandler(h)


class NullHandler(logging.Handler):
    """Silent Handler.

    Useful as a default::

      h = NullHandler()
      logging.getLogger("MDAnalysis").addHandler(h)
      del h

    see the advice on logging and libraries in
    http://docs.python.org/library/logging.html?#configuring-logging-for-a-library
    """

    def emit(self, record):
        pass


class ProgressBar(tqdm):
    r"""Display a visual progress bar and time estimate.

    The :class:`ProgressBar` decorates an iterable object, returning an
    iterator which acts exactly like the original iterable, but prints a
    dynamically updating progressbar every time a value is requested. See the
    example below for how to use it when iterating over the frames of a
    trajectory.


    Parameters
    ----------
    iterable  : iterable, optional
        Iterable to decorate with a progressbar.
        Leave blank to manually manage the updates.
    verbose : bool, optional
        If ``True`` (the default) then show the progress bar, *unless* the
        `disable` keyword is set to ``True`` (`disable` takes precedence over
        `verbose`). If `verbose` is set to ``None`` then the progress bar is
        displayed (like ``True``), *unless* this is a non-TTY output device
        (see `disable`).
    desc  : str, optional
        Prefix for the progressbar.
    total  : int or float, optional
        The number of expected iterations. If unspecified,
        ``len(iterable)`` is used if possible. If ``float("inf")`` or as a last
        resort, only basic progress statistics are displayed
        (no ETA, no progressbar).
    leave  : bool, optional
        If [default: ``True``], keeps all traces of the progressbar
        upon termination of iteration.
        If ``None``, will leave only if `position` is 0.
    file  : :class:`io.TextIOWrapper` or :class:`io.StringIO`, optional
        Specifies where to output the progress messages (default:
        :data:`sys.stderr`). Uses :meth:`file.write` and :meth:`file.flush`
        methods.  For encoding, see `write_bytes`.
    ncols  : int, optional
        The width of the entire output message. If specified,
        dynamically resizes the progressbar to stay within this bound.
        If unspecified, attempts to use environment width. The
        fallback is a meter width of 10 and no limit for the counter and
        statistics. If 0, will not print any meter (only stats).
    mininterval  : float, optional
        Minimum progress display update interval [default: 0.1] seconds.
    maxinterval  : float, optional
        Maximum progress display update interval [default: 10] seconds.
        Automatically adjusts `miniters` to correspond to `mininterval`
        after long display update lag. Only works if `dynamic_miniters`
        or monitor thread is enabled.
    miniters  : int or float, optional
        Minimum progress display update interval, in iterations.
        If 0 and `dynamic_miniters`, will automatically adjust to equal
        `mininterval` (more CPU efficient, good for tight loops).
        If > 0, will skip display of specified number of iterations.
        Tweak this and `mininterval` to get very efficient loops.
        If your progress is erratic with both fast and slow iterations
        (network, skipping items, etc) you should set miniters=1.
    ascii  : bool or str, optional
        If unspecified or ``False``, use unicode (smooth blocks) to fill
        the meter. The fallback is to use ASCII characters " 123456789#".
    disable  : bool, optional
        Whether to disable the entire progressbar wrapper
        [default: ``False``]. If set to None, disable on non-TTY.
    unit  : str, optional
        String that will be used to define the unit of each iteration
        [default: it].
    unit_scale  : bool or int or float, optional
        If 1 or True, the number of iterations will be reduced/scaled
        automatically and a metric prefix following the
        International System of Units standard will be added
        (kilo, mega, etc.) [default: ``False``]. If any other non-zero
        number, will scale `total` and `n`.
    dynamic_ncols  : bool, optional
        If set, constantly alters `ncols` and `nrows` to the
        environment (allowing for window resizes) [default: ``False``].
    smoothing  : float, optional
        Exponential moving average smoothing factor for speed estimates
        (ignored in GUI mode). Ranges from 0 (average speed) to 1
        (current/instantaneous speed) [default: 0.3].
    bar_format  : str, optional
        Specify a custom bar string formatting. May impact performance.
        [default: ``'{l_bar}{bar}{r_bar}'``], where ``l_bar='{desc}:
        {percentage:3.0f}%|'`` and ``r_bar='| {n_fmt}/{total_fmt}
        [{elapsed}<{remaining}, {rate_fmt}{postfix}]'``

        Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
        percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
        rate, rate_fmt, rate_noinv, rate_noinv_fmt,
        rate_inv, rate_inv_fmt, postfix, unit_divisor,
        remaining, remaining_s.

        Note that a trailing ": " is automatically removed after {desc}
        if the latter is empty.
    initial  : int or float, optional
        The initial counter value. Useful when restarting a progress bar
        [default: 0]. If using :class:`float`, consider specifying ``{n:.3f}``
        or similar in `bar_format`, or specifying `unit_scale`.
    position  : int, optional
        Specify the line offset to print this bar (starting from 0)
        Automatic if unspecified.
        Useful to manage multiple bars at once (e.g., from threads).
    postfix  : dict or \*, optional
        Specify additional stats to display at the end of the bar.
        Calls ``set_postfix(**postfix)`` if possible (:class:`dict`).
    unit_divisor  : float, optional
        [default: 1000], ignored unless `unit_scale` is ``True``.
    write_bytes  : bool, optional
        If (default: ``None``) and `file` is unspecified,
        bytes will be written in Python 2. If `True` will also write
        bytes. In all other cases will default to unicode.
    lock_args  : tuple, optional
        Passed to `refresh` for intermediate output
        (initialisation, iterating, and updating).
    nrows  : int, optional
        The screen height. If specified, hides nested bars outside this
        bound. If unspecified, attempts to use environment height.
        The fallback is 20.

    Returns
    -------
    out  : decorated iterator.

    Example
    -------
    To get a progress bar when analyzing a trajectory::

      from MDAnalysis.lib.log import ProgressBar

      ...

      for ts in ProgressBar(u.trajectory):
         # perform analysis


    will produce something similar to ::

       30%|███████████                       | 3/10 [00:13<00:30,  4.42s/it]

    in a terminal or Jupyter notebook.


    See Also
    --------
    The :class:`ProgressBar` is derived from :class:`tqdm.auto.tqdm`; see the
    `tqdm documentation`_ for further details on how to use it.



    .. _`tqdm documentation`: https://tqdm.github.io/

    """

    def __init__(self, *args, **kwargs):
        """"""
        # ^^^^ keep the empty doc string to avoid Sphinx doc errors with the
        # original doc string from tqdm.auto.tqdm
        verbose = kwargs.pop("verbose", True)
        # disable: Whether to disable the entire progressbar wrapper [default: False].
        # If set to None, disable on non-TTY.
        # disable should be the opposite of verbose unless it's None
        disable = verbose if verbose is None else not verbose
        # disable should take precedence over verbose if both are set
        kwargs["disable"] = kwargs.pop("disable", disable)
        super(ProgressBar, self).__init__(*args, **kwargs)
