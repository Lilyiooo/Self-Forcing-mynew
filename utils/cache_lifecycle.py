def queue_aged_blocks(
    recent_blocks,
    clean_block,
    chunk_idx,
    current_start_frame,
    nsink,
    nrecent,
    extra=None,
):
    """Append a clean block and return blocks that aged out of recent."""
    block = {
        "latent": clean_block,
        "chunk_idx": chunk_idx,
        "start": current_start_frame,
        "end": current_start_frame + clean_block.shape[1],
    }
    if extra:
        block.update(extra)

    recent_blocks.append(block)
    cutoff_frame = block["end"] - nrecent

    aged_blocks = []
    while recent_blocks and recent_blocks[0]["end"] <= cutoff_frame:
        aged = recent_blocks.pop(0)
        if aged["start"] < nsink:
            continue
        aged_blocks.append(aged)

    return aged_blocks
