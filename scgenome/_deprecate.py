""" Helpers for renaming public functions and arguments.

Renames here are additive: the new name carries the implementation and the old
name keeps working while warning. Nothing is ever removed on a schedule, which
matches how the rest of the package has handled renames.

Only the Python surface is renamed. Keys inside an ``AnnData`` --
``obs['cell_order']``, ``obs['cluster_id']``, ``uns['cell_order']`` -- are
written into every saved ``.h5ad``, so renaming those would invalidate files on
disk. They keep their names regardless of what the arguments are called.
"""

import functools
import warnings


def warn_renamed(old_name, new_name, stacklevel=3):
    """ Warn that ``old_name`` is deprecated in favour of ``new_name``. """
    warnings.warn(
        f'{old_name} is deprecated, use {new_name}',
        DeprecationWarning, stacklevel=stacklevel)


def renamed_arguments(**renames):
    """ Accept the old spelling of renamed keyword arguments.

    Each keyword maps an old argument name to its new name. A call using the
    old name still works and warns; a call passing both raises, since there is
    no way to know which one was meant.

    The wrapped function declares only the new names, so its signature and
    docstring describe the current API.

    Parameters
    ----------
    **renames : str
        old argument name mapped to new argument name

    Returns
    -------
    callable
        decorator

    Examples
    --------
    >>> @renamed_arguments(cell_order='obs_order')
    ... def plot(adata, obs_order=None):
    ...     return obs_order
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old_name, new_name in renames.items():
                if old_name not in kwargs:
                    continue
                if new_name in kwargs:
                    raise TypeError(
                        f'{func.__name__}() received both {new_name} and '
                        f'{old_name}; {old_name} is a deprecated alias for '
                        f'{new_name}, pass only {new_name}')
                warn_renamed(old_name, new_name)
                kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)

        wrapper._renamed_arguments = dict(renames)
        return wrapper
    return decorator
