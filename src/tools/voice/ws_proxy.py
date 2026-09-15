"""Proxy policy for outbound Volcengine WebSocket connections.

``websockets`` 14 introduced a ``proxy`` argument that defaults to ``True``,
which makes every connection silently consult ``HTTPS_PROXY`` / ``ALL_PROXY``.
For the Volcengine speech endpoints that default is actively harmful: they are
domestic endpoints, so routing them through an overseas proxy adds a full
round trip to the TLS handshake. Measured on this host, the same BigASR stream
took 1.51s to connect through the proxy versus 0.10s on the physical
interface.

So the default here is "ignore the ambient proxy variables" rather than
"inherit them", and ``VOLC_WS_PROXY`` exists for the rare deployment that
really does need an explicit hop.

This only governs the environment-variable layer. A transparent TUN device
(Clash's ``tun.enable`` with ``auto-route``) intercepts traffic at the IP
routing layer, below anything a Python client can see or opt out of. Bypassing
that requires a routing or rule change outside this process.
"""
from __future__ import annotations

import os

import websockets

_PROXY_ENV_VAR = "VOLC_WS_PROXY"

# Values that mean "restore the library default and read the ambient proxy
# environment variables", for the deployment that genuinely wants that.
_INHERIT_ENVIRONMENT_VALUES = frozenset({"system", "env", "environ", "inherit"})

# Values that spell out the default explicitly, so an operator can pin the
# behaviour in ``.env`` instead of relying on an unset variable.
_FORCE_DIRECT_VALUES = frozenset({"", "direct", "none", "off", "no", "false", "0"})


def _websockets_supports_proxy_argument() -> bool:
    """Whether the installed ``websockets`` accepts ``connect(proxy=...)``.

    ``requirements.txt`` permits ``websockets>=12.0,<17.0``, and the argument
    only exists on the asyncio client shipped from 14 onward. On 12 and 13 the
    legacy client forwards unknown keyword arguments to the protocol, where
    they raise ``TypeError``, so the argument has to be withheld entirely.
    """
    raw_version = getattr(websockets, "__version__", "")
    leading_digits = ""
    for character in raw_version:
        if not character.isdigit():
            break
        leading_digits += character
    if not leading_digits:
        return False
    return int(leading_digits) >= 14


_SUPPORTS_PROXY_ARGUMENT = _websockets_supports_proxy_argument()


def resolve_proxy_setting() -> object | None:
    """Return the value to pass as ``websockets.connect(proxy=...)``.

    ``None`` disables proxy lookup, ``True`` restores the library default of
    reading the environment, and a string is used as the proxy address.
    """
    configured_value = os.getenv(_PROXY_ENV_VAR, "").strip()
    if configured_value.lower() in _INHERIT_ENVIRONMENT_VALUES:
        return True
    if configured_value.lower() in _FORCE_DIRECT_VALUES:
        return None
    return configured_value


def websocket_proxy_kwargs() -> dict[str, object]:
    """Keyword arguments pinning the proxy policy for one ``connect`` call.

    Returns an empty mapping when the installed ``websockets`` predates the
    argument, so callers can splat the result unconditionally.
    """
    if not _SUPPORTS_PROXY_ARGUMENT:
        return {}
    return {"proxy": resolve_proxy_setting()}


def describe_proxy_policy() -> str:
    """One-line summary for startup logs, safe to print (no credentials)."""
    if not _SUPPORTS_PROXY_ARGUMENT:
        return (
            f"websockets={getattr(websockets, '__version__', '?')} "
            "(无 proxy 参数，沿用库默认行为)"
        )
    setting = resolve_proxy_setting()
    if setting is None:
        return "direct (忽略 HTTPS_PROXY/ALL_PROXY)"
    if setting is True:
        return "system (读取 HTTPS_PROXY/ALL_PROXY)"
    return f"explicit ({setting})"
