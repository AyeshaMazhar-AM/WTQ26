"""Compare the KESC_0008 row of a level1.csv with the correct JSON from section 5.7 of the guide.

Usage (from the source folder):
    python compare.py ../train_out/level1.csv
"""
import csv
import json
import sys

EXPECTED = {
    "provider": "KE", "tariff": "A1-R", "sanctioned_load_kw": 3, "bill_month": "2026-04",
    "reading_date": "2026-04-03", "issue_date": "2026-04-07", "due_date": "2026-04-21",
    "previous_reading": 8819, "current_reading": 8970, "units_consumed": 151,
    "charges": [
        {"type": "fixed", "amount": 900}, {"type": "energy", "amount": 1054},
        {"type": "energy", "amount": 663.51}, {"type": "quarterly_adjustment", "amount": 52.91},
        {"type": "fpa", "amount": 125.27}, {"type": "surcharge", "amount": 64.93}],
    "total_charges": 2860.62,
    "taxes": [
        {"type": "electricity_duty", "amount": 29.41}, {"type": "gst", "amount": 520.21},
        {"type": "municipal_tax", "amount": 20}],
    "total_taxes": 569.62, "current_bill": 3430.24, "arrears": -0.53,
    "payable_within_due_date": 3430, "payable_after_due_date": 3716,
}


def same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1
    return a == b


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../train_out/level1.csv"
    with open(path, newline="", encoding="utf-8") as f:
        row = next((r for r in csv.DictReader(f) if r["bill_id"] == "KESC_0008"), None)
    if row is None:
        sys.exit("KESC_0008 not found in " + path)
    got = json.loads(row["json"])
    ok = 0
    for key, want in EXPECTED.items():
        have = got.get(key)
        if key in ("charges", "taxes"):
            w = sorted((i["type"], round(i["amount"], 2)) for i in want)
            h = sorted((i.get("type"), round(i.get("amount", 0), 2)) for i in (have or []))
            good = w == h
        else:
            good = same(have, want)
        ok += good
        print(("OK   " if good else "DIFF ") + key)
        if not good:
            print("       expected:", want)
            print("       got     :", have)
    print(f"\n{ok}/{len(EXPECTED)} fields match")


if __name__ == "__main__":
    main()
