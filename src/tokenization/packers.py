import json 

def pack_to_variable_blocks(
    tokenized_path,
    output_path,
    encoder,
    block_size=2048,
    pad_token="<|pad|>"
):
    """
    Pack multiple tokenized samples into blocks of <= block_size.
    Only the final block is padded.
    """

    PAD = encoder.encode(pad_token, allowed_special="all")[0]
    block = []
    block_len = 0
    total_blocks = 0

    with open(tokenized_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            ids = row["input_ids"]

            # If adding this sample overflows: flush current block
            if block_len + len(ids) > block_size:
                # truncate if slightly over
                block = block[:block_size]

                fout.write(json.dumps({
                    "input_ids": block,
                    "length": len(block)
                }) + "\n")

                total_blocks += 1
                block = []
                block_len = 0

            # Add tokens to current block
            block.extend(ids)
            block_len += len(ids)

        # Final block (pad to block_size)
        if block:
            pad_len = block_size - len(block)
            block.extend([PAD] * pad_len)

            fout.write(json.dumps({
                "input_ids": block,
                "length": len(block)
            }) + "\n")

            total_blocks += 1

    print(f"Total variable-size blocks (padded last only): {total_blocks}")



def pack_to_fixed_blocks(
    tokenized_path,
    output_path,
    encoder,
    block_size=2048,
    pad_token="<|pad|>"
):
    """
    Pack data so that *every* output block is exactly block_size tokens.
    """

    PAD = encoder.encode(pad_token, allowed_special="all")[0]
    block = []
    block_len = 0
    total_blocks = 0

    with open(tokenized_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            ids = row["input_ids"]

            # If full, flush block
            if block_len + len(ids) > block_size:
                pad_len = block_size - block_len
                block.extend([PAD] * pad_len)

                fout.write(json.dumps({
                    "input_ids": block,
                    "length": block_size
                }) + "\n")

                total_blocks += 1
                block = []
                block_len = 0

            # Add current document tokens
            block.extend(ids)
            block_len += len(ids)

        # Final block (pad it)
        if block:
            pad_len = block_size - block_len
            block.extend([PAD] * pad_len)

            fout.write(json.dumps({
                "input_ids": block,
                "length": block_size
            }) + "\n")

            total_blocks += 1

    print(f"Total fixed-length blocks written: {total_blocks}")



def diagnose_packed_lengths(
    tokenized_path,
    packed_path,
    block_size=2048
):
    print("\n=== Checking ORIGINAL token lengths (FULL) ===")
    orig_lengths = []

    with open(tokenized_path, "r") as f:
        for line in f:
            row = json.loads(line)
            orig_lengths.append(len(row["input_ids"]))

    print(f"Total original docs: {len(orig_lengths)}")
    print(f"Min original length: {min(orig_lengths)}")
    print(f"Max original length: {max(orig_lengths)}")
    print(f"Avg original length: {sum(orig_lengths)/len(orig_lengths):.2f}")

    print("\n=== Checking PACKED blocks (FULL) ===")
    packed_lengths = []
    with open(packed_path, "r") as f:
        for line in f:
            row = json.loads(line)
            packed_lengths.append(len(row["input_ids"]))

    print(f"Total packed blocks: {len(packed_lengths)}")
    unique_lengths = set(packed_lengths)
    print(f"Unique packed block lengths: {unique_lengths}")

    if unique_lengths == {block_size}:
        print(f"All packed blocks are EXACTLY {block_size} tokens.")
    else:
        print("ERROR: Some blocks do NOT match the required length.")
        bad = [x for x in packed_lengths if x != block_size]
        print(f"Examples of incorrect lengths (first 10): {bad[:10]}")
