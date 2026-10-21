"""Rename the package: python scripts/rename.py NEWNAME  (lowercase, letters/digits/underscore). Rewrites the source folder,
imports, command name and docs. Run from the repository root, then reinstall."""
import pathlib
import sys

old = "seedproof"
new = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
assert new.isidentifier() and new == new.lower(), "use a lowercase Python identifier"
root = pathlib.Path(__file__).resolve().parents[1]
for p in root.rglob("*"):
    if p.is_file() and p.suffix in {".py", ".md", ".toml", ".cff", ".yml", ".yaml", ".txt"} and ".git/" not in str(p) and p.name != "rename.py":
        t = p.read_text(encoding="utf8")
        if old in t:
            p.write_text(t.replace(old, new), encoding="utf8")
src = root / "src" / old
if src.exists():
    src.rename(root / "src" / new)
print(f"renamed {old} -> {new}; run: pip install -e .")
