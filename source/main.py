"""Utility Bill Decoder - WTQ Build Track.

Level 1: bill image -> structured JSON (vision model, JSON output).
Level 2: bill image + extracted JSON + questions -> plain-English answers
         (model with a calculator tool so arithmetic is done in code).

Providers (set MODEL_PROVIDER in .env): "google" (Gemini) or "anthropic" (Claude).

Usage:
    python main.py --bills test/bills --csv test/csv --out output
"""
import argparse
import ast
import base64
import csv
import io
import json
import logging
import operator
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

log = logging.getLogger("bill-decoder")

MAX_IMAGE_BYTES = 4_500_000  # API limit is 5 MB per image

# Level 1 schema + prompt
NUM = {"type": ["number", "null"]}
STR = {"type": ["string", "null"]}

CHARGE_TYPES = ["energy", "fixed", "fpa", "quarterly_adjustment", "surcharge",
                "meter_rent", "subsidy", "other"]
TAX_TYPES = ["gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax"]


def _line_items(types):
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"type": {"type": "string", "enum": types}, "amount": {"type": "number"}},
            "required": ["type", "amount"],
        },
    }


BILL_SCHEMA = {
    "type": "object",
    "properties": {
        "provider": {"type": ["string", "null"], "enum": ["KE", "LESCO", "IESCO", None]},
        "tariff": STR,
        "sanctioned_load_kw": NUM,
        "bill_month": STR,
        "reading_date": STR,
        "issue_date": STR,
        "due_date": STR,
        "previous_reading": NUM,
        "current_reading": NUM,
        "units_consumed": NUM,
        "charges": _line_items(CHARGE_TYPES),
        "total_charges": NUM,
        "taxes": _line_items(TAX_TYPES),
        "total_taxes": NUM,
        "current_bill": NUM,
        "arrears": NUM,
        "payable_within_due_date": NUM,
        "payable_after_due_date": NUM,
    },
    "required": ["provider", "tariff", "sanctioned_load_kw", "bill_month", "reading_date",
                 "issue_date", "due_date", "previous_reading", "current_reading",
                 "units_consumed", "charges", "total_charges", "taxes", "total_taxes",
                 "current_bill", "arrears", "payable_within_due_date",
                 "payable_after_due_date"],
}

L1_SYSTEM = """You extract data from Pakistani electricity bills (K-Electric, LESCO, IESCO).
The image may be tilted, partly in Urdu, and personal details are hidden by grey boxes.
Call the record_bill tool exactly once. Rules:
- Extract values AS PRINTED. Never recalculate, round or correct amounts.
- Amounts are plain numbers in PKR: no 'Rs.', no commas. 'Rs. 19,964' -> 19964.
- Credits/subsidies are NEGATIVE ('38209CR' -> -38209). Subsidy lines are negative.
- Dates: YYYY-MM-DD. bill_month: YYYY-MM (e.g. 'MAR 2026' -> '2026-03').
- If a value is not printed, blank, covered by a grey box or unreadable, use null. Do not guess.
- provider is KE for K-Electric (even if the file says KESC), LESCO, or IESCO.
- tariff: exactly as printed (e.g. 'A1-R', 'A-1a(01)').
- sanctioned_load_kw: labelled Sanc Load / S.Load / San Load.
- units_consumed: units billed for the CURRENT month as printed.
- charges: one entry per line in the charges section; skip blank lines and lines printed as 0.
  Keep separate entries when the same type appears twice (e.g. two energy slabs).
  Types: energy (cost of units/variable slabs/cost of electricity), fixed, fpa (FPA and FCA),
  quarterly_adjustment (QTA / uniform quarterly adjustment), surcharge (e.g. PHL, F.C surcharge),
  meter_rent (meter/service rent), subsidy (negative), other.
- taxes: one entry per line in the tax/government section. Classify by SECTION, not name.
  Types: gst (incl. GST on FPA), electricity_duty (ED), income_tax,
  municipal_tax (e.g. KE MUCT/KMC), other_tax (extra/further tax, TV fee...).
  If only one combined tax amount is printed, return taxes=[] and put it in total_taxes.
- total_charges / total_taxes: the printed subtotals, or null if none printed.
- current_bill: printed amount for this month's bill.
- arrears: previous dues carried forward, keeping the printed sign; null if blank.
- payable_within_due_date: amount payable by the due date (last date without late surcharge).
- payable_after_due_date: if several late amounts are shown, use the HIGHEST.
- NEVER include names, addresses, CNIC, account/reference/consumer/meter numbers."""

# Level 2 tools + prompt
CALC_TOOL = {
    "name": "calculate",
    "description": "Evaluate an arithmetic expression exactly (+ - * / ** % parentheses, "
                   "and functions sum, round, min, max, abs, len). Use for ALL math.",
    "input_schema": {"type": "object",
                     "properties": {"expression": {"type": "string"}},
                     "required": ["expression"]},
}

SUBMIT_TOOL = {
    "name": "submit_answers",
    "description": "Submit the final answer for every question. Call once, when done.",
    "input_schema": {
        "type": "object",
        "properties": {"answers": {"type": "array", "items": {
            "type": "object",
            "properties": {"question_id": {"type": "string"}, "answer": {"type": "string"}},
            "required": ["question_id", "answer"]}}},
        "required": ["answers"],
    },
}

L2_SYSTEM = """You are a helpful assistant explaining an electricity bill to a customer in plain English.
You get the bill image, a JSON extraction of it, and numbered questions.
Rules:
- Use ONLY information on the bill. No outside tariff or rate knowledge. If the bill does not
  show something needed, say so clearly instead of guessing.
- Use the calculate tool for every average, difference, percentage, count or estimate.
  Do not do arithmetic in your head. Check the image for things the JSON lacks
  (usage history, billing history, message board, bar charts).
- Label estimates as estimates and state how you computed them.
- If a question can be read in more than one way, say which meaning you used, or answer both.
- Be specific: quote the actual numbers (PKR amounts, units, months) from the bill.
- Never include names, addresses, CNIC, account/consumer/meter numbers.
- Write in English, even if parts of the bill are Urdu. 2-5 sentences per answer."""

L2_END_TOOL = "\n- Finish by calling submit_answers once with an answer for EVERY question id."
L2_END_JSON = ('\n- When finished, reply with ONLY a JSON object mapping every question id to its '
               'answer string, e.g. {"Q1": "...", "Q2": "..."}. No other text, no markdown.')

L1_JSON_HINT = """

Return ONE JSON object (no markdown) with exactly these keys:
provider, tariff, sanctioned_load_kw, bill_month, reading_date, issue_date, due_date,
previous_reading, current_reading, units_consumed, charges, total_charges, taxes, total_taxes,
current_bill, arrears, payable_within_due_date, payable_after_due_date.
charges is a list like [{"type": "energy", "amount": 1054.0}] using only the charge types above;
taxes is a list like [{"type": "gst", "amount": 520.21}] using only the tax types above.
Use null for unknown values and [] for empty lists. Numbers must be JSON numbers."""

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv}
_FUNCS = {"sum": sum, "round": round, "min": min, "max": max, "abs": abs, "len": len}


def safe_eval(expr: str):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(e) for e in n.elts]
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in _FUNCS and not n.keywords):
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr.strip(), mode="eval").body)


# Image helpers
def image_bytes(path: Path):
    """Return (raw_bytes, media_type), shrinking the file if it is too big."""
    raw = path.read_bytes()
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    if len(raw) > MAX_IMAGE_BYTES:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        scale, quality = 1.0, 90
        while True:
            w, h = int(img.width * scale), int(img.height * scale)
            buf = io.BytesIO()
            img.resize((w, h)).save(buf, "JPEG", quality=quality)
            if buf.tell() <= MAX_IMAGE_BYTES:
                break
            scale *= 0.85
        raw, media = buf.getvalue(), "image/jpeg"
        log.info("Resized %s to %dx%d", path.name, w, h)
    return raw, media


def image_block(path: Path) -> dict:
    """Anthropic image content block."""
    raw, media = image_bytes(path)
    data = base64.standard_b64encode(raw).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def find_image(bills_dir: Path, bill_id: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        p = bills_dir / f"{bill_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image for {bill_id} in {bills_dir}")


# Level 1
def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        s = str(v).upper().replace("RS.", "").replace(",", "").strip()
        neg = s.endswith("CR")
        s = s.replace("CR", "").strip()
        x = float(s)
        return -abs(x) if neg else x
    except ValueError:
        return None


def clean_bill(d: dict) -> dict:
    """Guarantee every field exists with the right type."""
    out = OrderedDict()
    for key in BILL_SCHEMA["required"]:
        out[key] = d.get(key)
    for k in ("sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed",
              "total_charges", "total_taxes", "current_bill", "arrears",
              "payable_within_due_date", "payable_after_due_date"):
        out[k] = _num(out[k])
    for k, allowed, default in (("charges", CHARGE_TYPES, "other"), ("taxes", TAX_TYPES, "other_tax")):
        items = []
        for it in out[k] or []:
            amt = _num(it.get("amount")) if isinstance(it, dict) else None
            if amt is None or amt == 0:
                continue
            t = it.get("type")
            items.append({"type": t if t in allowed else default, "amount": amt})
        out[k] = items
    for k in ("subsidy",):  # subsidies must be negative
        for it in out["charges"]:
            if it["type"] == k and it["amount"] > 0:
                it["amount"] = -it["amount"]
    if out["provider"] not in ("KE", "LESCO", "IESCO"):
        out["provider"] = None
    return dict(out)


def extract_bill(client, model, path: Path) -> dict:
    resp = client.messages.create(
        model=model, max_tokens=4000, temperature=0, system=L1_SYSTEM,
        tools=[{"name": "record_bill", "description": "Record the extracted bill.",
                "input_schema": BILL_SCHEMA}],
        tool_choice={"type": "tool", "name": "record_bill"},
        messages=[{"role": "user", "content": [
            image_block(path),
            {"type": "text", "text": "Extract this bill."}]}],
    )
    for b in resp.content:
        if b.type == "tool_use" and b.name == "record_bill":
            return clean_bill(b.input)
    raise RuntimeError("Model did not return a record_bill call")


# Level 2
def answer_questions(client, model, path: Path, bill: dict, questions: list) -> dict:
    qtext = "\n".join(f"{qid}: {q}" for qid, q in questions)
    prompt = (f"Extracted JSON of this bill (may contain small errors; the image is the source "
              f"of truth):\n{json.dumps(bill, ensure_ascii=False)}\n\nQuestions:\n{qtext}")
    messages = [{"role": "user", "content": [image_block(path), {"type": "text", "text": prompt}]}]
    answers = {}
    for _ in range(10):
        resp = client.messages.create(
            model=model, max_tokens=6000, temperature=0, system=L2_SYSTEM + L2_END_TOOL,
            tools=[CALC_TOOL, SUBMIT_TOOL], messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        results, done = [], False
        for b in resp.content:
            if b.type != "tool_use":
                continue
            if b.name == "calculate":
                try:
                    out = str(safe_eval(b.input.get("expression", "")))
                except Exception as e:  # noqa: BLE001
                    out = f"error: {e}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            elif b.name == "submit_answers":
                for a in b.input.get("answers", []):
                    answers[str(a.get("question_id"))] = str(a.get("answer", "")).strip()
                done = True
        if done:
            return answers
        if results:
            messages.append({"role": "user", "content": results})
        else:
            messages.append({"role": "user", "content": "Please call submit_answers now."})
    return answers


# Gemini (Google AI Studio) implementation
def parse_json_text(text: str):
    """Parse JSON from model text, tolerating ```json fences and stray prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            return json.loads(text[a:b + 1])
        raise


def gemini_call(fn, tries=6):
    """Run fn(); on rate-limit/server errors wait and retry (free tier is 15 requests/min)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            retryable = any(c in msg for c in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if not retryable or i == tries - 1:
                raise
            wait = 10 * (i + 1)
            log.warning("Gemini busy (%s...), retrying in %ds", msg[:60], wait)
            time.sleep(wait)


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Supports + - * / ** % ( ) and the functions
    sum, round, min, max, abs, len. Use this for ALL math (averages, percentages, differences)."""
    try:
        return str(safe_eval(expression))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def extract_bill_gemini(client, model, path: Path) -> dict:
    from google.genai import types
    raw, media = image_bytes(path)
    cfg = types.GenerateContentConfig(
        system_instruction=L1_SYSTEM + L1_JSON_HINT, temperature=0,
        response_mime_type="application/json")
    resp = gemini_call(lambda: client.models.generate_content(
        model=model, config=cfg,
        contents=[types.Part.from_bytes(data=raw, mime_type=media), "Extract this bill."]))
    return clean_bill(parse_json_text(resp.text))


def answer_questions_gemini(client, model, path: Path, bill: dict, questions: list) -> dict:
    from google.genai import types
    raw, media = image_bytes(path)
    qtext = "\n".join(f"{qid}: {q}" for qid, q in questions)
    prompt = (f"Extracted JSON of this bill (may contain small errors; the image is the source "
              f"of truth):\n{json.dumps(bill, ensure_ascii=False)}\n\nQuestions:\n{qtext}")
    # Passing a Python function as a tool lets the SDK run the calculate() loop automatically.
    cfg = types.GenerateContentConfig(
        system_instruction=L2_SYSTEM + L2_END_JSON, temperature=0, tools=[calculate])
    resp = gemini_call(lambda: client.models.generate_content(
        model=model, config=cfg,
        contents=[types.Part.from_bytes(data=raw, mime_type=media), prompt]))
    data = parse_json_text(resp.text)
    return {str(k): str(v).strip() for k, v in data.items() if v}


# CSV driver
def read_csv(p: Path):
    with p.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_csv(p: Path, fields, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--bills", required=True, help="folder of bill images")
    ap.add_argument("--csv", required=True, help="folder containing level1.csv and level2.csv templates")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cache", default=".cache", help="per-bill JSON cache (avoids paying twice)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    provider = os.getenv("MODEL_PROVIDER", "google").lower()
    key = os.getenv("API_KEY")
    if not key or key == "your-api-key-here":
        sys.exit("API_KEY missing. Copy .env.example to .env and put your real key in it.")
    if provider == "google":
        from google import genai
        model = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
        client = genai.Client(api_key=key)
        extract_fn, answer_fn = extract_bill_gemini, answer_questions_gemini
        pause = 5  # seconds between bills, keeps us under the free 15 requests/min limit
    elif provider == "anthropic":
        import anthropic
        model = os.getenv("MODEL_NAME", "claude-haiku-4-5")
        client = anthropic.Anthropic(api_key=key, max_retries=5)
        extract_fn, answer_fn = extract_bill, answer_questions
        pause = 0
    else:
        sys.exit("MODEL_PROVIDER must be 'google' or 'anthropic'.")
    log.info("Using %s / %s", provider, model)

    bills_dir, cache = Path(args.bills), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # ---- Level 1 ----
    f1, rows1 = read_csv(Path(args.csv) / "level1.csv")
    bills = {}
    for row in rows1:
        bid = row["bill_id"]
        cf = cache / f"{bid}.l1.json"
        try:
            if cf.exists():
                bill = json.loads(cf.read_text(encoding="utf-8"))
            else:
                log.info("Level 1: %s", bid)
                bill = extract_fn(client, model, find_image(bills_dir, bid))
                cf.write_text(json.dumps(bill, ensure_ascii=False), encoding="utf-8")
                time.sleep(pause)
        except Exception as e:  # noqa: BLE001
            log.error("Level 1 failed for %s: %s", bid, e)
            bill = clean_bill({})
        bills[bid] = bill
        row["json"] = json.dumps(bill, ensure_ascii=False, separators=(",", ":"))
    write_csv(Path(args.out) / "level1.csv", f1, rows1)

    # ---- Level 2 ----
    f2, rows2 = read_csv(Path(args.csv) / "level2.csv")
    by_bill = OrderedDict()
    for r in rows2:
        by_bill.setdefault(r["bill_id"], []).append(r)
    for bid, rows in by_bill.items():
        cf = cache / f"{bid}.l2.json"
        answers = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else {}
        missing = [(r["question_id"], r["question"]) for r in rows if not answers.get(r["question_id"])]
        for attempt in range(2):  # second pass re-asks only the missing questions
            if not missing:
                break
            try:
                log.info("Level 2: %s (%d questions, attempt %d)", bid, len(missing), attempt + 1)
                answers.update({k: v for k, v in answer_fn(
                    client, model, find_image(bills_dir, bid), bills.get(bid, {}), missing).items() if v})
                time.sleep(pause)
            except Exception as e:  # noqa: BLE001
                log.error("Level 2 failed for %s: %s", bid, e)
            missing = [(r["question_id"], r["question"]) for r in rows if not answers.get(r["question_id"])]
        cf.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
        for r in rows:
            r["answer"] = answers.get(r["question_id"]) or (
                "I could not determine this from the bill.")
    write_csv(Path(args.out) / "level2.csv", f2, rows2)
    log.info("Done. Wrote %s/level1.csv and %s/level2.csv", args.out, args.out)


if __name__ == "__main__":
    main()
