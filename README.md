# Utility Bill Decoder (WTQ Build Track)

Reads K-Electric / LESCO / IESCO bill images, extracts structured JSON (Level 1) and answers
customer questions in plain English (Level 2).

**Runtime:** Python 3.13

## Setup
```bash
cd source
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
```

## Run
```bash
python main.py --bills ../test/bills --csv ../test/csv --out ../output
```
Produces `output/level1.csv` (10 rows) and `output/level2.csv` (120 rows).
Per-bill results are cached in `.cache/` so a re-run does not call the model again
(delete the `.cache` folder to force a fresh run). The run takes a few minutes because it
pauses between calls to stay inside the free Gemini rate limit.

Optional: `python compare.py ../train_out/level1.csv` checks the training bill KESC_0008
against the correct JSON given in the participant guide.

## Configuration (.env)
| Variable | Value used | Meaning |
|---|---|---|
| MODEL_PROVIDER | `google` | `google` (Gemini) or `anthropic` (Claude) |
| MODEL_NAME | `gemini-3.5-flash-lite` | model ID |
| API_KEY | your key | secret, never submitted |

## How it works
1. **Level 1** - one vision call per bill. The prompt contains the extraction rules from the guide
   and the model returns a JSON object. Python then normalises it: numeric coercion, enum checks
   for charge/tax types, `CR` credits made negative, zero lines dropped, all fields always present.
2. **Level 2** - one call per bill containing the image, the Level 1 JSON and all 12 questions.
   The model has a `calculate` tool (safe AST-based evaluator) so every average, percentage and
   estimate is computed in code. Missing answers are re-asked once; any remaining failure falls
   back to a clear "could not determine" message so no cell is empty.
3. Rate limits are handled with a pause between bills and retry/back-off on 429/5xx errors.
4. CSVs are written with Python's `csv` module (correct quoting, UTF-8, original row order).

## AI usage
- **Models used at runtime:** Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) via Google AI Studio.
- **Other services/APIs:** none (no separate OCR, no invoice-extraction services)
- **AI tools to help and debug:** Claude (claude.ai)
