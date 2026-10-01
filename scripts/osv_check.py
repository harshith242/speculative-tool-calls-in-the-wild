"""OSV batch check of npm lockfiles: prints vulnerable packages and fails on any malware record."""
import json
import sys
import urllib.request


def query(pkgs):
    body = json.dumps({"queries": [{"package": {"name": n, "ecosystem": "npm"}, "version": v} for n, v in pkgs]}).encode()
    req = urllib.request.Request("https://api.osv.dev/v1/querybatch", body,
                                 {"Content-Type": "application/json", "User-Agent": "fastlane-research/0.1"})
    return json.load(urllib.request.urlopen(req, timeout=60))["results"]


malware = False
for path in sys.argv[1:]:
    pk = json.load(open(path)).get("packages", {})
    pkgs = sorted({(k.split("node_modules/")[-1], v["version"]) for k, v in pk.items() if k and "version" in v})
    for i in range(0, len(pkgs), 500):
        for (name, ver), res in zip(pkgs[i:i + 500], query(pkgs[i:i + 500])):
            ids = [v["id"] for v in res.get("vulns", [])]
            if ids:
                print(f"{path.split('/')[-2]}: {name}@{ver} {ids}")
                malware |= any(x.startswith("MAL-") for x in ids)
sys.exit(1 if malware else 0)
