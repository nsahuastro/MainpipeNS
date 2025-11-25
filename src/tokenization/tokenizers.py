import json 
import tiktoken
import numpy as np

base_enc = tiktoken.get_encoding("gpt2")
start_id = base_enc.n_vocab

# for gpt2
special_tokens = {
    "<|bos|>": start_id,
    "<|eos|>": start_id + 1,
    "<|pad|>": start_id + 2,
    "<|unk|>": start_id + 3,
}

# extended tokenizer
enc_ext = tiktoken.Encoding(
    name="gpt2_extended",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens=special_tokens
)

# special IDs for use in packing/padding
BOS_ID = enc_ext.encode("<|bos|>", allowed_special="all")[0]
EOS_ID = enc_ext.encode("<|eos|>", allowed_special="all")[0]
PAD_ID = enc_ext.encode("<|pad|>", allowed_special="all")[0]
UNK_ID = enc_ext.encode("<|unk|>", allowed_special="all")[0]

# Basic tokenization function
def tokenize_to_jsonl(input_path, output_path, encoder, max_seq_len=2048, limit=None):
    """
    Reads cleaned JSONL with {"text": ...} and writes tokenized JSONL as:
        {"input_ids": [...], "length": N}

    Args:
        input_path   : path to cleaned dataset
        output_path  : where tokenized jsonl is written
        encoder      : tiktoken Encoding instance (e.g. base_enc or extended_enc)
        max_seq_len  : truncate if longer
        limit        : optional max number of docs to process
    """

    count_in = 0
    count_out = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if limit is not None and count_in >= limit:
                break

            row = json.loads(line)
            text = row.get("text", "").strip()
            if not text:
                continue

            count_in += 1

            # encode using provided encoder
            token_ids = encoder.encode(text, allowed_special="all")

            # truncate for now
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]

            fout.write(json.dumps({
                "input_ids": token_ids,
                "length": len(token_ids)
            }) + "\n")

            count_out += 1

    print(f"Read docs : {count_in}")
    print(f"Wrote docs: {count_out}")

# Extended tokenizer function
def tokenize_ext_to_jsonl(input_path, output_path, encoder,
                          bos_token="<|bos|>", eos_token="<|eos|>",
                          max_seq_len=2048, limit=None):
    """
    Tokenize text using an extended tokenizer with BOS/EOS tokens.

    Args:
        encoder      : tiktoken Encoding with special tokens added
        bos_token    : BOS token string (must exist in encoder)
        eos_token    : EOS token string
    """

    # get IDs safely
    BOS = encoder.encode(bos_token, allowed_special="all")[0]
    EOS = encoder.encode(eos_token, allowed_special="all")[0]

    count_in, count_out = 0, 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if limit is not None and count_in >= limit:
                break

            row = json.loads(line)
            text = row.get("text", "").strip()
            if not text:
                continue

            count_in += 1

            # encode
            ids = encoder.encode(text, allowed_special="all")

            # add BOS and EOS
            ids = [BOS] + ids + [EOS]

            # truncate if needed
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                ids[-1] = EOS  # ensure ends with EOS

            fout.write(json.dumps({
                "input_ids": ids,
                "length": len(ids)
            }) + "\n")

            count_out += 1

    print(f"Read docs : {count_in}")
    print(f"Wrote docs: {count_out}")


def token_length_stats(path, encoder, max_docs=None):
    """
    Compute token-length statistics for a dataset before packing.

    Args:
        path      : cleaned dataset JSONL with {"text": ...}
        encoder   : tiktoken Encoding instance (base or extended)
        max_docs  : optional limit
    """
    
    lengths = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break

            row = json.loads(line)
            text = row.get("text", "")
            if not text.strip():
                continue

            tokens = encoder.encode(text, allowed_special="all")
            lengths.append(len(tokens))

    arr = np.array(lengths)
    print(f"Docs counted: {len(arr)}")
    print(f"Avg tokens   : {arr.mean():.2f}")
    print(f"Median tokens: {np.median(arr):.2f}")
    print(f"95th pct     : {np.percentile(arr, 95):.2f}")
    print(f"99th pct     : {np.percentile(arr, 99):.2f}")
    print(f"Max tokens   : {arr.max()}")

    return arr

def token_length_stats2(path, encoder, max_docs=None):
    lengths = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break

            row = json.loads(line)
            text = row.get("text", "")
            if not text.strip():
                continue

            tokens = encoder.encode(text, allowed_special="all")
            lengths.append(len(tokens))

    arr = np.array(lengths)

    stats = {
        "docs_counted": int(len(arr)),
        "avg_tokens": float(arr.mean()),
        "median_tokens": float(np.median(arr)),
        "p95_tokens": float(np.percentile(arr, 95)),
        "p99_tokens": float(np.percentile(arr, 99)),
        "max_tokens": int(arr.max())
    }

    print(stats)
    return arr, stats
