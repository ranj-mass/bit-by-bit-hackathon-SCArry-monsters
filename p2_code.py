import argparse
import os
import re
import sys
from typing import Tuple
import numpy as np
import pandas as pd

INV_SBOX = np.array([
    0x52,0x09,0x6A,0xD5,0x30,0x36,0xA5,0x38,0xBF,0x40,0xA3,0x9E,0x81,0xF3,0xD7,0xFB,
    0x7C,0xE3,0x39,0x82,0x9B,0x2F,0xFF,0x87,0x34,0x8E,0x43,0x44,0xC4,0xDE,0xE9,0xCB,
    0x54,0x7B,0x94,0x32,0xA6,0xC2,0x23,0x3D,0xEE,0x4C,0x95,0x0B,0x42,0xFA,0xC3,0x4E,
    0x08,0x2E,0xA1,0x66,0x28,0xD9,0x24,0xB2,0x76,0x5B,0xA2,0x49,0x6D,0x8B,0xD1,0x25,
    0x72,0xF8,0xF6,0x64,0x86,0x68,0x98,0x16,0xD4,0xA4,0x5C,0xCC,0x5D,0x65,0xB6,0x92,
    0x6C,0x70,0x48,0x50,0xFD,0xED,0xB9,0xDA,0x5E,0x15,0x46,0x57,0xA7,0x8D,0x9D,0x84,
    0x90,0xD8,0xAB,0x00,0x8C,0xBC,0xD3,0x0A,0xF7,0xE4,0x58,0x05,0xB8,0xB3,0x45,0x06,
    0xD0,0x2C,0x1E,0x8F,0xCA,0x3F,0x0F,0x02,0xC1,0xAF,0xBD,0x03,0x01,0x13,0x8A,0x6B,
    0x3A,0x91,0x11,0x41,0x4F,0x67,0xDC,0xEA,0x97,0xF2,0xCF,0xCE,0xF0,0xB4,0xE6,0x73,
    0x96,0xAC,0x74,0x22,0xE7,0xAD,0x35,0x85,0xE2,0xF9,0x37,0xE8,0x1C,0x75,0xDF,0x6E,
    0x47,0xF1,0x1A,0x71,0x1D,0x29,0xC5,0x89,0x6F,0xB7,0x62,0x0E,0xAA,0x18,0xBE,0x1B,
    0xFC,0x56,0x3E,0x4B,0xC6,0xD2,0x79,0x20,0x9A,0xDB,0xC0,0xFE,0x78,0xCD,0x5A,0xF4,
    0x1F,0xDD,0xA8,0x33,0x88,0x07,0xC7,0x31,0xB1,0x12,0x10,0x59,0x27,0x80,0xEC,0x5F,
    0x60,0x51,0x7F,0xA9,0x19,0xB5,0x4A,0x0D,0x2D,0xE5,0x7A,0x9F,0x93,0xC9,0x9C,0xEF,
    0xA0,0xE0,0x3B,0x4D,0xAE,0x2A,0xF5,0xB0,0xC8,0xEB,0xBB,0x3C,0x83,0x53,0x99,0x61,
    0x17,0x2B,0x04,0x7E,0xBA,0x77,0xD6,0x26,0xE1,0x69,0x14,0x63,0x55,0x21,0x0C,0x7D
], dtype=np.uint8)

HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

HEXBYTE_RE = re.compile(r"(?i)\\b([0-9a-f]{2})\\b")

def parse_ciphertext(cell: str) -> np.ndarray:
    """Return uint8[16] from a variety of common ciphertext string formats."""
    if cell is None:
        raise ValueError("Ciphertext cell is None")
    s = str(cell).strip()
    # Case 1: contiguous 32 hex chars
    hexonly = re.fullmatch(r"(?i)[0-9a-f]{32}", s.replace("0x",""))
    if hexonly:
        s2 = s.lower().replace("0x","")
        return np.frombuffer(bytes.fromhex(s2), dtype=np.uint8)
    # Case 2: find 16 hex bytes anywhere in string
    parts = HEXBYTE_RE.findall(s)
    if len(parts) >= 16:
        parts = parts[:16]
        return np.array([int(p,16) for p in parts], dtype=np.uint8)
    raise ValueError(f"Could not parse ciphertext format: '{s[:64]}...'")

def center_and_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center columns and return (Xc, mean, std). Adds tiny eps to std to avoid division by zero."""
    mu = X.mean(axis=0, dtype=np.float64)
    Xc = X - mu
    std = Xc.std(axis=0, ddof=1)
    std[std == 0] = 1e-12
    return Xc, mu, std

def cpa_last_round(ciphertexts: np.ndarray, traces: np.ndarray, poi: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform CPA for each key byte independently.
    Returns:
      scores: (16, 256) max |corr| per key guess
      best_idx: (16,) sample index at which the max occurs (for debugging/reporting)
    """
    N, T = traces.shape
    # Optional variance-based prefilter (POI) to reduce T
    if poi is not None and poi < T:
        variances = traces.var(axis=0)
        idx = np.argsort(variances)[-poi:]
        idx.sort()
        traces = traces[:, idx]
        reduced_indices = idx
    else:
        reduced_indices = None

    Xc, _, Xstd = center_and_std(traces)

    scores = np.zeros((16, 256), dtype=np.float64)
    best_idx = np.zeros((16,), dtype=np.int64)

    for b in range(16):
        ct_b = ciphertexts[:, b]
        # Build hypotheses matrix H: (N, 256)
        # hyp(k) = HW( INV_SBOX[ ct_b ^ k ] )
        # Vectorized computation:
        ct_tile = np.tile(ct_b.reshape(-1,1), (1,256)).astype(np.uint8)
        ks = np.arange(256, dtype=np.uint8).reshape(1,256)
        sbox_out = INV_SBOX[ct_tile ^ ks]
        H = HW[sbox_out].astype(np.float64)

        # Center each hyp column
        Hc = H - H.mean(axis=0, dtype=np.float64)
        Hstd = Hc.std(axis=0, ddof=1)
        Hstd[Hstd == 0] = 1e-12

        # Correlation: for every k, corr(X[:,t], H[:,k]) across t
        # corr = (Xc^T @ Hc[:,k]) / ((N-1) * Xstd * Hstd[k])
        # Compute dot products for all k in one go: (T, N) @ (N, 256) = (T,256)
        dots = Xc.T @ Hc   # shape (T,256)
        denom = (N - 1) * (Xstd.reshape(-1,1) * Hstd.reshape(1,-1))
        corr = np.abs(dots / denom)  # (T,256)
        # Take max over time
        max_corr = corr.max(axis=0)  # (256,)
        argmax_t = corr.argmax(axis=0)  # (256,)
        scores[b, :] = max_corr
        # Keep the time index for the best k (for reporting; we record index of top-1 k)
        k_best = int(np.argmax(max_corr))
        best_t = int(argmax_t[k_best])
        if reduced_indices is not None:
            best_idx[b] = int(reduced_indices[best_t])
        else:
            best_idx[b] = best_t

    return scores, best_idx

def main():
    ap = argparse.ArgumentParser(description="Problem 1 CPA (last-round, ciphertext-only)")
    ap.add_argument("--input", required=True, help="Path to CSV: col0=ciphertext, cols1..=trace samples")
    ap.add_argument("--outdir", required=True, help="Output directory for key.txt and byte_XX.txt files")
    ap.add_argument("--header", action="store_true", help="Set if the CSV has a header row to skip")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
    ap.add_argument("--poi", type=int, default=None, help="Optional: keep only top-variance POI columns (integer)")
    ap.add_argument("--hexout", action="store_true", help="Write ranks and key in hex (default is decimal)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load CSV with pandas for robustness
    try:
        df = pd.read_csv(args.input, header=0 if args.header else None, delimiter=args.delimiter, dtype=str)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if df.shape[1] < 2:
        print("CSV must have at least 2 columns: [ciphertext] + [trace samples...]", file=sys.stderr)
        sys.exit(1)

    # Parse ciphertext column to uint8[16]
    try:
        cts = np.vstack([parse_ciphertext(v) for v in df.iloc[:,0].values])
    except Exception as e:
        print(f"Error parsing ciphertext column: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert remaining columns to float traces
    try:
        # Attempt numeric conversion; coerce errors to NaN then drop rows with NaNs
        traces = df.iloc[:,1:].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    except Exception as e:
        print(f"Error converting trace samples to float: {e}", file=sys.stderr)
        sys.exit(1)

    # Drop any rows with NaNs in traces or ciphertext parsing failure
    mask = ~np.isnan(traces).any(axis=1)
    cts = cts[mask]
    traces = traces[mask]
    if cts.shape[0] < 10:
        print("Too few valid traces after cleaning (<10). Check your CSV format.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {cts.shape[0]} traces with {traces.shape[1]} samples each. Running CPA...")

    scores, best_idx = cpa_last_round(cts, traces, poi=args.poi)

    # Write ranked lists and assemble key
    key_bytes = []
    for b in range(16):
        order = np.argsort(-scores[b])  # descending
        if args.hexout:
            ranks = [f"{k:02x}" for k in order]
        else:
            ranks = [str(int(k)) for k in order]
        with open(os.path.join(args.outdir, f"byte_{b:02d}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ranks) + "\n")
        key_bytes.append(order[0])

    # Save final key (top-1 of each)
    if args.hexout:
        key_str = "".join(f"{int(k):02x}" for k in key_bytes)
    else:
        key_str = " ".join(str(int(k)) for k in key_bytes)
    with open(os.path.join(args.outdir, "key.txt"), "w", encoding="utf-8") as f:
        f.write(key_str + "\n")

    # Also save debug info: best sample index per byte
    with open(os.path.join(args.outdir, "debug_best_sample_idx.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(str(int(t)) for t in best_idx) + "\n")

    print("Done.")
    print("Final key (top-1 per byte):", key_str)
    print("Best sample index per byte saved to debug_best_sample_idx.txt")

if __name__ == "__main__":
    main()
