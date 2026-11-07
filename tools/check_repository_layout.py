"""Static repository checks; does not access clinical data."""
from pathlib import Path
import json
import re
import sys

root = Path(__file__).resolve().parents[1]
move_map = (root / "src" / "MOVE_MAP.md").read_text(encoding="utf-8")
pairs = []
for line in move_map.splitlines():
    if line.startswith('| `') and '` | [`' in line:
        old = line.split('`')[1]
        new = line.split('`')[3]
        pairs.append((old, new))
errors = []
if not pairs:
    errors.append("No moved paths parsed from src/MOVE_MAP.md")
for old, new in pairs:
    if not (root / new).is_file():
        errors.append(f"Missing destination: {new}")
    if (root / old).exists():
        errors.append(f"Unexpected duplicate at original path: {old}")
for item in json.loads((root / "CODE_INVENTORY.json").read_text(encoding="utf-8"))["files"]:
    if not (root / item["path"]).is_file():
        errors.append(f"Missing inventory entry: {item['path']}")
for error in errors:
    print("ERROR:", error)
print(f"Checked {len(pairs)} moves and inventory; {len(errors)} errors")
sys.exit(bool(errors))
