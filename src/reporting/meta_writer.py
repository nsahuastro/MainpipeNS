import json
import time
from pathlib import Path

def write_meta(
    output_dir,
    *,
    tokenizer_name,
    vocab_size,
    special_tokens,
    block_size,
    total_blocks,
    cleaning_summary,
    shard_info=None,
    cli_args=None, 
    pipeline_version="1.0"
):
    """
    Writes a dataset metadata file (meta.json) to output_dir.

    Args:
        output_dir        : folder where meta.json will be saved
        tokenizer_name    : name of tokenizer (e.g. "gpt2_extended")
        vocab_size        : size of vocabulary
        special_tokens    : dict mapping special token to ID
        block_size        : fixed packed length (e.g. 2048)
        total_blocks      : number of packed blocks
        cleaning_summary  : dict returned by clean_dataset()
        shard_info        : dict, optional (num_shards, shard_size, split ratios)
        pipeline_version  : version tag for your pipeline
    """

    meta = {
        "pipeline_version": pipeline_version,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
         "cli_args": cli_args or {},  
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": vocab_size,
            "special_tokens": special_tokens
        },
        "data": {
            "total_blocks": total_blocks,
            "block_size": block_size,
            "cleaning_summary": cleaning_summary
        },
        "shards": shard_info or {}
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print(f"[meta_writer] Saved metadata to: {output_dir / 'meta.json'}")
