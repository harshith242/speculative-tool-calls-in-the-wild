"""A tiny MCP stdio server for tests: search returns ids, get returns a record, boom fails."""
import time

try:
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server

app = Server("fake")


@app.tool()
def search(q: str) -> list:
    time.sleep(0.05)
    return [{"id": f"{q}-1"}, {"id": f"{q}-2"}]


@app.tool()
def get(id: str) -> dict:
    time.sleep(0.05)
    return {"id": id, "title": f"Title of {id}"}


@app.tool()
def pid() -> str:
    import os
    return str(os.getpid())


@app.tool()
def boom() -> str:
    raise ValueError("nope")


if __name__ == "__main__":
    app.run()
