#!/usr/bin/env bash
# Installs the 3 FastLane v2 MCP servers from MCP-Bench (OKX Exchange, Game Trends, NixOS), pinned and without install scripts.
# The other servers were dropped: rate limits (Wikipedia, DexPaprika, Paper Search, Met Museum bursts), serial handling
# (Call for Papers, FruityVice), a prompt-injection CVE (Context7) or failures (OpenAPI Explorer); see the v2 spec.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d mcp_servers/.git ]; then
  git clone --filter=blob:none --no-checkout --depth 1 https://github.com/Accenture/mcp-bench.git mcp_servers
  git -C mcp_servers sparse-checkout set --no-cone /mcp_servers/okx-mcp /mcp_servers/game-trends-mcp /mcp_servers/mcp-nixos /tasks
  git -C mcp_servers checkout HEAD
fi
S=mcp_servers/mcp_servers
# OKX: axios pinned to 1.20.0 (the locked 1.10 has 32 CVEs; 1.14.1 was malware), then build with its own TypeScript
(cd $S/okx-mcp && npm install --ignore-scripts --no-audit --no-fund axios@1.20.0 --save-exact \
   && npm install --ignore-scripts --no-audit --no-fund && ./node_modules/.bin/tsc)
# Game Trends: mcp v2 / fastmcp env
uv venv .venv-mcp --python 3.12
UV_EXCLUDE_NEWER=2026-09-24T00:00:00Z uv pip install --python .venv-mcp/bin/python fastmcp requests beautifulsoup4 lxml aiohttp
# NixOS: written for the v1 MCP SDK, so it gets its own env
uv venv .venv-mcp1 --python 3.12
UV_EXCLUDE_NEWER=2026-09-24T00:00:00Z uv pip install --python .venv-mcp1/bin/python "mcp<2" requests beautifulsoup4
UV_EXCLUDE_NEWER=2026-09-24T00:00:00Z uv pip install --python .venv-mcp1/bin/python --no-deps -e $S/mcp-nixos
echo "== checks"
find .venv-mcp .venv-mcp1 -name '*.pth' | grep -v -E '_virtualenv|_editable_impl_' || true
grep -l "plain-crypto-js" $S/okx-mcp/package-lock.json || echo "no plain-crypto-js"
python3 scripts/osv_check.py $S/okx-mcp/package-lock.json
