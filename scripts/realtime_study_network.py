"""Process-local direct IPv4 routing for an isolated benchmark, never system settings."""
import ipaddress
import os
import socket
from urllib.parse import urlparse


def enable_direct(source_ip="192.168.5.4"):
    ipaddress.IPv4Address(source_ip)
    for key in list(os.environ):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "ws_proxy", "wss_proxy"}:
            os.environ.pop(key)
    os.environ["NO_PROXY"] = "*"
    if getattr(socket.socket, "_study_direct", False):
        return
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def bind_external(sock, address):
        if sock.family != socket.AF_INET or not isinstance(address, tuple):
            return
        ip = ipaddress.ip_address(socket.gethostbyname(address[0]))
        if ip.is_loopback or ip.is_unspecified or ip.is_private:
            return
        if sock.getsockname()[1] == 0:
            sock.bind((source_ip, 0))

    def connect(sock, address):
        bind_external(sock, address)
        return original_connect(sock, address)

    def connect_ex(sock, address):
        bind_external(sock, address)
        return original_connect_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket._study_direct = True
    # uvloop can connect C sockets without Python socket.connect.
    import inspect
    import websockets
    original_ws = websockets.connect
    supports_proxy = 'proxy' in inspect.signature(original_ws).parameters

    def direct_websocket(uri, *args, **kwargs):
        if urlparse(uri).hostname not in {'localhost', '127.0.0.1', '::1'}:
            kwargs.setdefault('local_addr', (source_ip, 0))
            if supports_proxy:
                kwargs.setdefault('proxy', None)
        return original_ws(uri, *args, **kwargs)

    websockets.connect = direct_websocket
