import math


def chunk_latency(chunk_s, rtf, rtt_s):
    """Calculates latency: audio chunk collection + compute + network transmission."""
    return chunk_s + rtf * chunk_s + rtt_s


def largest_chunk_under(budget_s, rtf, rtt_s, step=0.05):
    """Finds the largest multiple of `step` whose chunk latency does not exceed budget_s.

    Returns the chunk size rounded to 2 decimals, or None if step exceeds budget.
    """
    # chunk_latency = chunk_s * (1 + rtf) + rtt_s <= budget_s
    # chunk_s <= (budget_s - rtt_s) / (1 + rtf)
    max_chunk = (budget_s - rtt_s) / (1 + rtf)

    if max_chunk < step:
        return None

    # Find the largest multiple of step <= max_chunk
    # An epsilon handles floating-point precision issues like 0.29999999999999994
    num_steps = math.floor((max_chunk + 1e-9) / step)
    return round(num_steps * step, 2)


def concurrent_streams(rtf, gpu_share=1.0):
    """Calculates how many live concurrent streams can be served."""
    if rtf <= 0:
        raise ValueError("rtf must be positive")
    return math.floor(gpu_share / rtf)


def end_of_speech_latency(hangover_s, chunk_s, rtf, rtt_s):
    """Total time from speech end to transcript receipt including VAD silence hangover."""
    return hangover_s + chunk_latency(chunk_s, rtf, rtt_s)


def budget_breakdown(chunk_s, rtf, rtt_s):
    """Returns the fractional breakdown of latency across wait, model, and network."""
    total = chunk_latency(chunk_s, rtf, rtt_s)
    if total == 0:
        return {"wait": 0.0, "model": 0.0, "network": 0.0}

    wait_s = chunk_s
    model_s = rtf * chunk_s
    network_s = rtt_s

    return {
        "wait": round(wait_s / total, 4),
        "model": round(model_s / total, 4),
        "network": round(network_s / total, 4),
    }