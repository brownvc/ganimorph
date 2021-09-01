# -*- coding: utf-8 -*-
# File: serialize.py

import sys
import os
from .develop import create_dummy_func
from . import logger

__all__ = ['loads', 'dumps']


def dumps_msgpack(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object.
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads_msgpack(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return msgpack.loads(buf, raw=False)


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object.
        May not be compatible across different versions of pyarrow.
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


try:
    # import pyarrow has a lot of side effect: https://github.com/apache/arrow/pull/2329
    # So we need an option to disable it.
    if os.environ.get('TENSORPACK_SERIALIZE', 'pyarrow') == 'pyarrow':
        if 'horovod' in sys.modules:
            logger.warn("Horovod and pyarrow may conflict due to pyarrow bugs. "
                        "Uninstall pyarrow and use msgpack instead.")
        import pyarrow as pa
    else:
        pa = None
except ImportError:
    pa = None
    dumps_pyarrow = create_dummy_func('dumps_pyarrow', ['pyarrow'])  # noqa
    loads_pyarrow = create_dummy_func('loads_pyarrow', ['pyarrow'])  # noqa

try:
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()
except ImportError:
    loads_msgpack = create_dummy_func(  # noqa
        'loads_msgpack', ['msgpack', 'msgpack_numpy'])
    dumps_msgpack = create_dummy_func(  # noqa
        'dumps_msgpack', ['msgpack', 'msgpack_numpy'])

if pa is None or os.environ.get('TENSORPACK_SERIALIZE', None) == 'msgpack':
    loads = loads_msgpack
    dumps = dumps_msgpack
else:
    loads = loads_pyarrow
    dumps = dumps_pyarrow
