import os

def shard_packed_dataset(
    packed_path,
    out_dir,
    train_ratio=0.98,
    val_ratio=0.01,
    test_ratio=0.01,
    shard_size=50000      # number of blocks per shard file
):
    """Shard packed 2048-token blocks into train/val/test splits."""

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    # Create output directories
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(out_dir, s), exist_ok=True)

    # Count total examples
    with open(packed_path, "r") as f:
        total = sum(1 for _ in f)

    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)
    n_test  = total - n_train - n_val

    print(f"\nTotal blocks: {total}")
    print(f"Train blocks ({train_ratio*100:.1f}%): {n_train}")
    print(f"Val blocks   ({val_ratio*100:.1f}%): {n_val}")
    print(f"Test blocks  ({test_ratio*100:.1f}%): {n_test}\n")

    # Helper to open shard
    def open_shard(split, idx):
        shard_path = os.path.join(out_dir, split, f"shard_{idx:05d}.jsonl")
        return open(shard_path, "w")

    # Initialize writers
    shard_idx = {"train": 1, "val": 1, "test": 1}
    counters  = {"train": 0, "val": 0, "test": 0}

    writers = {
        "train": open_shard("train", 1),
        "val":   open_shard("val", 1),
        "test":  open_shard("test", 1),
    }

    def rotate(split):
        """Start new shard when shard_size is hit."""
        writers[split].close()
        shard_idx[split] += 1
        writers[split] = open_shard(split, shard_idx[split])

    # Start reading and splitting
    with open(packed_path, "r") as f:
        for line in f:
            # Decide split based on counters (percentage logic)
            if counters["train"] < n_train:
                split = "train"
            elif counters["val"] < n_val:
                split = "val"
            else:
                split = "test"

            # Write line
            writers[split].write(line)
            counters[split] += 1

            # Rotate shard if needed
            if counters[split] % shard_size == 0:
                rotate(split)

    # Close final shards
    for w in writers.values():
        w.close()

    print("Sharding completed.")
    print(f"Train shards: {shard_idx['train']}")
    print(f"Val shards:   {shard_idx['val']}")
    print(f"Test shards:  {shard_idx['test']}")
