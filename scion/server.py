"""
Compatibility shim — re-exports ScionSession as ScionServer.

In Rootstock the user-process side is called ``RootstockServer`` (because
it binds the Unix socket and accepts the worker's connection). The Scion
analog is implemented in ``session.py`` as ``ScionSession`` since it owns
much more than just the socket (subprocess lifecycle, RPC framing, multi
-method dispatch). The alias here keeps the Rootstock parallel for users
who expect a ``Server`` class.
"""

from .session import ScionSession as ScionServer
from .session import decode_result

__all__ = ["ScionServer", "decode_result"]
