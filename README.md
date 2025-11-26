# MainpipeNS — English Pretraining Dataset Pipeline

## 1. Overview

MainpipeNS is a full end-to-end data pipeline for preparing a high-quality English-language corpus suitable for LLM pretraining.

It performs:

- Raw dataset inspection + reporting
- Document-level deduplication
- Cleaning and normalization (HTML, boilerplate, non-English, code-heavy)
- Tokenization using an extended GPT-2 BPE tokenizer
- Packing into fixed-length 2048-token blocks
- Sharding into train / val / test subsets
- Quality report (toxicity, PII, perplexity, language coverage)
- Metadata generation (meta.json)

The entire pipeline is orchestrated via:

```bash
python main.py --raw path/to/raw.jsonl
```

## 2. Repository Structure

```
MainpipeNS/
│
├── main.py                         # Main CLI pipeline
│
├── data/
│   ├── raw/                        # Raw input files
│   ├── dedup/                      # After deduplication
│   ├── clean/                      # After full cleaning
│   └── final/
│       ├── tokenized.jsonl
│       ├── packed_blocks.jsonl
│       ├── sharded_dataset/
│       └── meta.json
│
├── figures/                        # Auto-generated plots
├── reports/                        # JSON summary reports
│
├── src/
│   ├── cleaning/                   # Dedup + cleaning modules
│   ├── detectors/                  # HTML, code, language
│   ├── reporting/                  # Stats, quality report, meta-writer
│   ├── tokenization/               # Tokenizers, packers, sharders
│   └── utils/                      # IO utilities
```

## 3. Source Data

- **Format:** JSON Lines (JSONL)
- **Minimum expected key:** `"text"`
- **Only `"text"` is processed;** other metadata keys are ignored.

Example raw dataset:

```
data/raw/mainpipe_data_v1.jsonl   (~269k lines)
```

## 4. Cleaning & Normalisation

Implemented in `src/cleaning/clean_pipe.py`.

### 4.1 Cleaning Steps

#### Exact deduplication
- **Function:** `dedup_exact()`
- SHA-256 fingerprint
- Removes duplicate text entries

#### HTML filtering & stripping
- `has_html()` detection
- Remove HTML tags, scripts, inline styling, boilerplate

#### Language filtering (English only)
- Uses `detect_lang()` (lingua-based)
- Removes non-EN documents

#### Code-heavy filtering
- Regex-based detection of Python/JS/C++/Java/Rust patterns
- Compute `code_fraction_strong()`
- Drop docs with >40% code lines

#### Length filtering
- Drop docs <200 chars
- Drop docs >50,000 chars

#### Normalization (`normalize_text`)
- Remove zero-width & BOM chars
- Normalize whitespace (collapse multi-spaces)
- Remove accidental line breaks
- Keep natural punctuation and structure

### 4.2 Example Cleaning Summary

```
HTML_STRIPPED   : 53,294
NON_ENGLISH     : 21,200
CODE_HEAVY      : 72,472
TOO_SHORT       : 19,622
TOO_LONG        : 282
KEPT            : 107,320   (39.1%)
```

## 5. Tokenization (GPT-2 Extended)

Tokenizer implemented in: `src/tokenization/tokenizers.py`

### 5.1 Base encoding

GPT-2 BPE (tiktoken.get_encoding("gpt2"))

### 5.2 Added special tokens

```
<|bos|>   # begin sequence
<|eos|>   # end sequence
<|pad|>   # padding (used during packing)
<|unk|>   # unknown
```

### 5.3 Procedure

For each cleaned document:

```
ids = [BOS] + enc_ext.encode(text) + [EOS]
```

Sequences longer than 2048 are truncated and forced to end with EOS.

### 5.4 Tokenized Output Format

```json
{"input_ids": [50257, 123, 456, ..., 50258], "length": 187}
```

## 6. 2048-Token Packing

Implemented in: `src/tokenization/packers.py`

### Goal

Transform variable-length tokenized documents into fixed 2048-token blocks, suitable for transformer training.

### Method

- Documents are concatenated sequentially
- When a block would overflow, it is padded with `<|pad|>`
- Final block also padded to exactly 2048 tokens
- Output: `data/final/packed_blocks.jsonl`

Each line:

```json
{"input_ids": [...2048 tokens...], "length": 2048}
```

## 7. Sharding (Train / Val / Test)

Implemented in: `src/tokenization/sharders.py`

### Current split

- **train:** 98%
- **val:** 1%
- **test:** 1%

Shards are written to: `data/final/sharded_dataset/`

Each shard contains up to 50,000 packed examples.

## 8. Quality Report

Implemented in: `src/reporting/quality_reporter.py`

Runs on cleaned data.

### Metrics

#### 8.1 PII Detection

- Email
- Phone numbers
- Credit-card–like patterns

#### 8.2 Toxicity (Detoxify)

- Outputs avg + max toxicity.

#### 8.3 Perplexity Proxy (GPT-2 small)

- A small LM is used to estimate corpus difficulty.

#### 8.4 Language Distribution

- Uses `detect_lang()`.

#### Output example

```json
{
  "samples_analyzed": 1500,
  "pii_hits": {"email": 27, "phone": 57, "credit_card": 11},
  "toxicity": {"avg": 0.0085, "max": 0.8266},
  "perplexity": {"avg": 44.8, "median": 34.2},
  "language_distribution": {"EN": 1500}
}
```

Saved to: `reports/quality_report.json`

## 9. Metadata (meta.json)

Created by: `src/reporting/meta_writer.py`

### Contains

- Pipeline version
- Timestamp
- Tokenizer info (vocab size, special tokens)
- Cleaning summary
- Total packed blocks
- Shard information

Saved at: `data/final/meta.json`

## 10. Running the Pipeline via CLI

### 10.1 Basic usage

```bash
python main.py --raw data/raw/mainpipe_data_v1.jsonl
```

### 10.2 What happens when you run it

The pipeline automatically performs:

- Raw inspection + summary reports
- Deduplication
- Cleaning
- Cleaned dataset inspection
- Token length stats
- Tokenization (BOS/EOS)
- Packing into 2048-token blocks
- Train/Val/Test sharding
- Quality report
- Metadata export

All outputs saved under: `reports/`, `figures/`, `data/dedup/`, `data/clean/`, `data/final/`

## 11. Outputs Produced by the Pipeline

| File / Folder | Description |
|---|---|
| `reports/raw_doc_stats.json` | Raw key/length distribution |
| `reports/raw_category_pct.json` | Category distribution (raw) |
| `reports/clean_category_pct.json` | Category distribution (cleaned) |
| `reports/token_length_stats.json` | Token length statistics |
| `reports/quality_report.json` | PII, toxicity, perplexity, language |
| `figures/*.pdf` | Histograms and category plots |
| `data/dedup/*.jsonl` | Deduplicated dataset |
| `data/clean/*.jsonl` | Fully cleaned dataset |
| `data/final/tokenized.jsonl` | Tokenized documents |
| `data/final/packed_blocks.jsonl` | 2048-token fixed blocks |
| `data/final/sharded_dataset/` | Train/Val/Test shards |
| `data/final/meta.json` | Pipeline metadata |

## 12. Contact / Notes

- This pipeline is part of a pretraining dataset preparation project for building an indigenous Australian LLM foundation model.
- Designed for reproducibility, transparency, and fault-tolerance.