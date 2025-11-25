# MainpipeNS English Pretraining Dataset

## 1. Overview

This dataset is an English-language corpus prepared for LLM pretraining.  
The pipeline performs:

- Document-level **deduplication** (exact and near-duplicate)
- **HTML stripping** and boilerplate removal
- **Language filtering** to keep only English
- **Code-heavy document filtering**
- Length-based filtering (remove ultra-short and ultra-long docs)
- Text **normalization** (whitespace, control chars, line breaks)
- **Tokenization** with a GPT-2–style BPE + special tokens
- **Packing** into fixed-length 2048-token blocks
- **Sharding** into train / val / test splits

All intermediate steps are implemented in Python under `src/` and orchestrated via notebooks in `notebooks/`.

---

## 2. Source Data

- Input format: JSONL
- Keys: `{"text": ..., "url": ...}` (only `text` is used)
- Total raw lines: ~269k
- Raw file: `data/raw/mainpipe_data_v1.jsonl`

---

## 3. Cleaning & Normalisation Pipeline

Cleaning is implemented in `src/cleaning/clean_pipe.py`:

Steps:

1. **Exact deduplication**  
   - Hash: SHA-256 of full `text`  
   - Function: `dedup_exact(input_path, output_path)`  
   - Output: `data/dedup/mainpipe_data_deduplicated.jsonl`

2. **HTML detection & stripping**  
   - HTML detection via regexes (`has_html`)  
   - Stripping with `strip_html` to drop tags, scripts, boilerplate.

3. **Language filtering**  
   - Library: `lingua` / `LanguageDetector`  
   - Keep only documents detected as **EN** after HTML stripping.

4. **Code-heavy filtering**  
   - Strong code heuristics via `is_strong_code_line` patterns (Python/JS/C++/Rust/Java).  
   - Compute `code_fraction(text)` over non-empty lines.  
   - Drop documents with `code_fraction > 0.40`.

5. **Length-based filtering**  
   - Drop docs with `len(text) < 200` characters  
   - Drop docs with `len(text) > 50_000` characters  

6. **Text normalization** (`normalize_text`)  
   - Remove zero-width and BOM chars
   - Normalize non-breaking spaces
   - Collapse internal whitespace
   - Preserve sentence punctuation and semantic structure

**Cleaning summary (example):**

- HTML stripped       : 53,294 docs  
- Non-English removed : 21,200 docs  
- Code-heavy removed  : 72,472 docs  
- Too short removed   : 19,622 docs  
- Too long removed    : 282 docs  
- **Kept for tokenization**: 107,320 docs (~39% of deduplicated corpus)

(Adjust to your final numbers.)

---

## 4. Tokenization & Special Tokens

Tokenizer based on **GPT-2 BPE** (via `tiktoken`):

- Base encoding: `tiktoken.get_encoding("gpt2")`
- Extended with 4 special tokens:

  - `<|bos|>` – beginning of sequence  
  - `<|eos|>` – end of sequence  
  - `<|pad|>` – padding  
  - `<|unk|>` – unknown (rarely used for GPT-2 BPE, but reserved)

Total vocab size:

- Base GPT-2: 50,257  
- Extended: 50,261

Tokenization:

- For each cleaned document:  
  `ids = [BOS] + enc.encode(text) + [EOS]`  
- Sequences truncated to `max_seq_len = 2048`.

Output format: JSONL

```json
{"input_ids": [50257, 123, 456, ..., 50258], "length": 187}
