import json
from utils.hash_utils import hash_text, simhash_text

def dedup_exact(input_path, output_path):
    seen = set()
    kept = 0
    dropped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            try:
                row = json.loads(line)
            except:
                continue

            text = row.get("text", "")
            h = hash_text(text)

            if h in seen:
                dropped += 1
                continue

            seen.add(h)
            kept += 1
            fout.write(json.dumps(row) + "\n")

    print(f"Exact dedup done: kept={kept:,}, dropped={dropped:,}")


def dedup_near(input_path, output_path, hamming_threshold=3):
    signatures = []
    kept = 0
    dropped = 0

    with open(input_path, "r") as fin, \
         open(output_path, "w") as fout:

        for line in fin:
            row = json.loads(line)
            text = row.get("text", "")

            sig = simhash_text(text) 

            is_duplicate = False
            for prev_sig in signatures:
                if bin(sig ^ prev_sig).count("1") <= hamming_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                dropped += 1
                continue
            
            signatures.append(sig)
            kept += 1
            fout.write(json.dumps(row) + "\n")

    print(f"Near-dedup: kept={kept:,}, dropped={dropped:,}")
