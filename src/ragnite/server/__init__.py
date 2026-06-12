__all__ = ["create_app", "create_mcp_server"]


def __getattr__(name: str):
    if name == "create_app":
        from ragnite.server.app import create_app

        return create_app
    if name == "create_mcp_server":
        from ragnite.server.mcp import create_mcp_server

        return create_mcp_server
    raise AttributeError(name)
