"""
Convert a grants CSV (see grants_template.csv for the expected columns) into
the grants.json format the rest of the app reads.

Usage:
    python csv_to_json.py my_grants.csv grants.json

The 'focus_areas' column should be semicolon-separated, e.g. "health;AI;nonprofit".
"""
import csv
import json
import sys
from pathlib import Path


def convert(csv_path: str, json_path: str) -> None:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["focus_areas"] = [
                s.strip() for s in row.get("focus_areas", "").split(";") if s.strip()
            ]
            row["criteria"] = [
                s.strip() for s in row.get("criteria", "").split(";") if s.strip()
            ]
            rows.append(row)

    payload = {
        "_meta": {
            "description": "Grants database generated from a CSV by csv_to_json.py. Verify each entry's "
            "details on the funder's site before relying on amounts/deadlines.",
            "schema_version": 1,
            "fields": [
                "id", "name", "funder", "focus_areas", "eligibility", "criteria",
                "typical_amount", "geography", "link", "notes", "last_verified",
            ],
        },
        "grants": rows,
    }

    Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} grants to {json_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_json.py <input.csv> <output.json>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
