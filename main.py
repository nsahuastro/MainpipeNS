import os
import argparse
import logging
from datetime import datetime
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random
import json 
#import numpy as np
#import tiktoken

#from src.utils.io_utils import ensure_dir, read_jsonl, sample_jsonl, sample_docs

from src.reporting.explore_stats_sumry import quick_stats_report, summarize_dataset_exclusive
from src.reporting.meta_writer import write_meta
from src.reporting.viz_plots import plot_summary_percentage, plot_cleaning_report

#from src.detectors.html_detect import show_html_examples
#from src.detectors.code_ASCII_detect import detect_non_ascii
#from src.detectors.language_detect import sample_language_distribution

from src.cleaning.deduplication_pipe import dedup_exact
from src.cleaning.clean_pipe import clean_dataset, print_cleaning_summary
from src.reporting.quality_reporter import quality_report

from src.tokenization.tokenizers import base_enc, enc_ext, tokenize_ext_to_jsonl, token_length_stats2
from src.tokenization.tokenizers import BOS_ID, EOS_ID, PAD_ID, special_tokens
from src.tokenization.packers import pack_to_fixed_blocks, diagnose_packed_lengths
from src.tokenization.sharders import shard_packed_dataset


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger(__name__)

def count_blocks(path):
    return sum(1 for _ in open(path))



def run_pipeline(args):
    logger = setup_logging()
    start_time = datetime.now()

    logger.info("=*= Starting MainpipeNS pipeline =*=")

    raw_path   = args.raw
    dedup_path = "data/dedup/dedup.jsonl"
    clean_path = "data/clean/clean.jsonl"
    tok_path   = "data/final/tokenized.jsonl"
    pack_path = "data/final/packed_blocks.jsonl"
    shard_dir  = "data/final/sharded_dataset"

    os.makedirs("reports", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # inspection
    logger.info("Inspecting raw file...")
    stats = quick_stats_report(raw_path, sample_size=25000, 
                            save_json_path="reports/raw_doc_stats.json",
                            save_fig_path="figures/raw_doc_length_hist.pdf")
    logger.info("Raw quick_stats:")
    logger.info(json.dumps(stats, indent=2))

    # Exclusive category distribution
    logger.info("category percentages in raw input file:")
    summary1, summary_pct1 = summarize_dataset_exclusive(raw_path, sample_size=25000)
    
    logger.info(json.dumps(summary_pct1, indent=2))
    pct_json_path = "reports/raw_category_pct.json"
    with open(pct_json_path, "w") as f:
        json.dump(summary_pct1, f, indent=2)

    logger.info(f"Saved raw category percentages to {pct_json_path}")
    print(f"[INFO] Saved raw category percentages to {pct_json_path}")

    fig = plot_summary_percentage(summary_pct1)
    fig_path = "figures/raw_category_pct.pdf"
    fig.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved raw category percentage plot to {fig_path}")
    print(f"[INFO] Saved category histogram to {fig_path}")

   
    # exact dedup
    logger.info("Deduplication...")
    dedup_exact(raw_path, dedup_path)
    logger.info(f"Deduplicated data saved to {dedup_path}")

    
    # cleaning
    logger.info("Cleaning dataset...")
    counters = clean_dataset(dedup_path, clean_path)
    logger.info(f"Cleaned data saved to {clean_path}")
    logger.info(f"Cleaning summary: {dict(counters)}")

    fig=plot_cleaning_report(counters)
    fig.savefig("figures/clean_data_hist.pdf", format="pdf", dpi=300, bbox_inches="tight")
    logger.info("Saved cleaning report figure to figures/clean_data_hist.pdf")

    # clean file check
    logger.info("Inspecting cleaned dataset...")

    sumry_clean, sumry_pct_clean = summarize_dataset_exclusive(clean_path, sample_size=25000)
    logger.info("Clean dataset category percentages:")
    logger.info(json.dumps(sumry_pct_clean, indent=2))

    clean_pct_json_path = "reports/clean_category_pct.json"
    with open(clean_pct_json_path, "w") as f:
        json.dump(sumry_pct_clean, f, indent=2)

    logger.info(f"Saved clean category percentages to {clean_pct_json_path}")

    fig = plot_summary_percentage(sumry_pct_clean)
    fig_path = "figures/clean_category_pct.pdf"
    fig.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved clean category percentage plot to {fig_path}")

    # clean data quality report
    logger.info("Running quality report on cleaned dataset...")
    quality = quality_report(clean_path,
                            sample_size=1500,
                            save_path="reports/quality_report.json"
            )
    
    logger.info(json.dumps(quality, indent=2))
    print("\n=*= Quality Report =*=")
    for k, v in quality.items():
        print(f"{k:20}: {v}")

    logger.info("Quality report:")
    logger.info(json.dumps(quality, indent=2))
    print("[INFO] Saved quality report to reports/quality_report.json")

    #token length stats
    logger.info("Computing token length stats…")
    _, token_stats = token_length_stats2(clean_path, encoder=enc_ext, max_docs=None)

    stats_path = "reports/token_length_stats.json"
    with open(stats_path, "w") as f:
        json.dump(token_stats, f, indent=2)

    logger.info(f"Saved token-length statistics to {stats_path}")
    print(f"[INFO] Saved token-length stats to {stats_path}")


    # tokenization
    logger.info("Tokenization...")
    tokenize_ext_to_jsonl(clean_path, tok_path, encoder=enc_ext, max_seq_len=2048)
    logger.info(f"Tokenized data saved to {tok_path}")
    
    # packing blocks
    logger.info("Packing to 2048-token blocks...")
    total_blocks= pack_to_fixed_blocks(tok_path, pack_path,
                                        encoder=enc_ext,
                                        block_size=2048,
                                        pad_token="<|pad|>"
                                        )   
    logger.info(f"Packed blocks saved to {pack_path}")

    logger.info("Diagnosing packed block lengths...")
    diagnose_packed_lengths(tok_path, pack_path, block_size=2048)

    # sharding
    logger.info("Train/Val/Test sharding...")
    shard_packed_dataset(pack_path, shard_dir,
                        train_ratio=0.98,
                        val_ratio=0.01,
                        test_ratio=0.01,
                        shard_size=50000
                        )
    logger.info(f"Sharded dataset saved to {shard_dir}")


    # metadata
    logger.info("Writing meta.json...")
    write_meta(output_dir="data/final",
                tokenizer_name=enc_ext.name,
                vocab_size=enc_ext.n_vocab,
                special_tokens=special_tokens,
                block_size=2048,
                total_blocks=total_blocks,
                cleaning_summary=dict(counters),
                shard_info={"train_ratio": 0.98,
                            "val_ratio": 0.01,
                            "test_ratio": 0.01,
                            "shard_output_dir": shard_dir
                            }, 
                cli_args=vars(args)   
            )

    logger.info("=*= Pipeline completed successfully =*=")
    print("[INFO] Pipeline completed successfully")

    elapsed = datetime.now() - start_time
    logger.info(f"Total pipeline time: {str(elapsed).split('.')[0]}")  

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MainpipeNS Data Pipeline")
    parser.add_argument("--raw", required=True, help="Path to raw JSONL file")

    args = parser.parse_args()
    run_pipeline(args)
