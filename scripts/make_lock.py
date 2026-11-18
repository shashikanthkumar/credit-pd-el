"""Regenerate requirements.lock = transitive closure of requirements.txt in the current environment."""
import importlib.metadata as m, platform, re
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
direct = [l.split("==")[0] for l in open("requirements.txt") if "==" in l and not l.startswith("#")]
seen = {}
def walk(name):
    if norm(name) in seen:
        return
    try:
        d = m.distribution(name)
    except m.PackageNotFoundError:
        return
    seen[norm(name)] = (d.metadata["Name"], d.version)
    for r in d.requires or []:
        if "extra ==" not in r:
            walk(re.split(r"[ ;<>=!~\[(]", r.strip())[0])
for p in direct:
    walk(p)
hdr = ["# Full resolved dependency closure of requirements.txt (no extras).", f"# Python {platform.python_version()}."]
open("requirements.lock", "w").write("\n".join(hdr + [f"{n}=={v}" for n, v in sorted(seen.values(), key=lambda x: x[0].lower())]) + "\n")
