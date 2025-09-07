import argparse, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import correlate, savgol_filter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# AES inverse S-box and HW table
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

HEXBYTE_RE = re.compile(r'(?i)\b([0-9a-f]{2})\b')
def parse_hex_bytes(cell):
    s = str(cell).strip()
    s2 = s.lower().replace("0x","")
    if len(s2) == 32 and all(c in "0123456789abcdef" for c in s2):
        return np.frombuffer(bytes.fromhex(s2), dtype=np.uint8)
    parts = HEXBYTE_RE.findall(s)
    if len(parts) >= 16:
        return np.array([int(p,16) for p in parts[:16]], dtype=np.uint8)
    raise ValueError("Can't parse hex bytes from: "+s[:60])

def center_std(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    s = Xc.std(axis=0, ddof=1)
    s[s==0] = 1e-12
    return Xc, mu, s

def cpa_scores_lastround(cts, traces):
    N, T = traces.shape
    Xc, _, Xstd = center_std(traces)
    scores = np.zeros((16,256))
    for b in range(16):
        ct_b = cts[:, b]
        ct_tile = np.tile(ct_b.reshape(-1,1), (1,256)).astype(np.uint8)
        ks = np.arange(256, dtype=np.uint8).reshape(1,256)
        sbox_out = INV_SBOX[ct_tile ^ ks]
        H = HW[sbox_out].astype(np.float64)
        Hc = H - H.mean(axis=0)
        Hstd = Hc.std(axis=0, ddof=1); Hstd[Hstd==0]=1e-12
        dots = Xc.T @ Hc
        denom = (N-1) * (Xstd.reshape(-1,1) * Hstd.reshape(1,-1))
        corr = np.abs(dots / denom)
        scores[b,:] = corr.max(axis=0)
    return scores

def mutual_info_discrete(x_int, y_cont, bins=20):
    N = len(x_int)
    edges = np.unique(np.percentile(y_cont, np.linspace(0,100,bins+1)))
    if len(edges) <= 2:
        return 0.0
    y_d = np.digitize(y_cont, edges, right=False) - 1
    y_d[y_d < 0] = 0
    y_d[y_d >= len(edges)-1] = len(edges)-2
    Ky = y_d.max()+1
    Kx = int(x_int.max())+1
    idx = x_int.astype(np.int32) * Ky + y_d.astype(np.int32)
    counts = np.bincount(idx, minlength=Kx*Ky).astype(np.float64)
    joint = counts.reshape((Kx, Ky)) / N
    px = joint.sum(axis=1); py = joint.sum(axis=0)
    nz = joint > 0
    if not nz.any(): return 0.0
    return float((joint[nz] * np.log(joint[nz] / (px[:,None] * py[None,:])[nz])).sum())

def mia_scores_lastround(cts, traces, bins=20):
    N, T = traces.shape
    scores = np.zeros((16,256))
    # pre-discretize Y per time
    y_discrete = []
    for t in range(T):
        edges = np.unique(np.percentile(traces[:,t], np.linspace(0,100,bins+1)))
        if len(edges) <= 1:
            edges = np.histogram_bin_edges(traces[:,t], bins=min(10, max(2, N//10)))
        y_d = np.digitize(traces[:,t], edges, right=False)-1
        y_d[y_d<0]=0
        y_d[y_d>=len(edges)-1]=len(edges)-2
        y_discrete.append(y_d)
    y_discrete = np.array(y_discrete)  # T x N

    for b in range(16):
        ct_b = cts[:, b]
        ct_tile = np.tile(ct_b.reshape(-1,1), (1,256)).astype(np.uint8)
        sbox = INV_SBOX[ct_tile ^ np.arange(256, dtype=np.uint8).reshape(1,256)]
        H_all = HW[sbox]  # N x 256
        for k in range(256):
            x = H_all[:,k]
            best_mi = 0.0
            for t in range(T):
                y_d = y_discrete[t]
                Ky = y_d.max()+1
                Kx = int(x.max())+1
                idx = x.astype(np.int32)*Ky + y_d.astype(np.int32)
                counts = np.bincount(idx, minlength=Kx*Ky).astype(np.float64)
                joint = counts.reshape((Kx,Ky))/N
                px = joint.sum(axis=1); py = joint.sum(axis=0)
                nz = joint > 0
                if nz.any():
                    mi = (joint[nz] * np.log(joint[nz] / (px[:,None] * py[None,:])[nz])).sum()
                    if mi > best_mi: best_mi = mi
            scores[b,k] = best_mi
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--filter", action="store_true")
    ap.add_argument("--pca", type=int, default=0)
    ap.add_argument("--poi", type=int, default=200)
    ap.add_argument("--hexout", action="store_true")
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, header=None)
    if df.shape[1] < 3:
        print("Expect at least plaintext,ciphertext,trace samples")
        sys.exit(1)

    # parse PT & CT
    plain = df.iloc[:,0].astype(str).values
    cipher = df.iloc[:,1].astype(str).values
    traces = df.iloc[:,2:].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    mask = ~np.isnan(traces).any(axis=1)
    plain = plain[mask]; cipher = cipher[mask]; traces = traces[mask]
    N, T = traces.shape
    print(f"Loaded {N} traces x {T} samples")

    # parse ciphertext into byte arrays
    cts = np.vstack([parse_hex_bytes(c) for c in cipher])

    # baseline CPA
    print("Running baseline CPA (no alignment)...")
    base_cpa = cpa_scores_lastround(cts, traces)
    base_key = "".join(f"{int(np.argmax(base_cpa[b])):02x}" for b in range(16))
    print("Baseline key (top-1 per byte):", base_key)

    # alignment step
    if args.align:
        print("Aligning traces to median template...")
        template = np.median(traces, axis=0)
        aligned = np.zeros_like(traces)
        max_shift = min(1000, T//10)
        for i in range(N):
            corr = correlate(traces[i], template, mode='full')
            shift = corr.argmax() - (T-1)
            if shift > max_shift: shift = max_shift
            if shift < -max_shift: shift = -max_shift
            aligned[i] = np.roll(traces[i], -shift)
        traces = aligned
        plt.figure(figsize=(6,3))
        for i in range(min(30, N)):
            plt.plot(traces[i] + i*0.0, alpha=0.5)
        plt.title("Aligned traces (preview)")
        plt.savefig(os.path.join(args.outdir, "aligned_preview.png"))
        plt.close()

    # filtering
    if args.filter:
        print("Applying Savitzky-Golay filter (window=11, poly=3)...")
        if T > 11:
            for i in range(N):
                traces[i] = savgol_filter(traces[i], 11, 3, mode='nearest')

    # crop to POI window
    var = traces.var(axis=0)
    top_idxs = np.argsort(var)[-args.poi:]
    wmin = max(0, top_idxs.min()-10)
    wmax = min(T, top_idxs.max()+10)
    print(f"Cropping to window {wmin}:{wmax} (size {wmax-wmin})")
    traces_crop = traces[:, wmin:wmax]

    # PCA features
    if args.pca and args.pca > 0:
        print(f"Running PCA -> {args.pca} components")
        pca = PCA(n_components=min(args.pca, max(1, traces_crop.shape[1]-1)))
        PC = pca.fit_transform(traces_crop)
        traces_feat = PC
    else:
        traces_feat = traces_crop

    # CPA & MIA (may be slow)
    print("Computing CPA on features...")
    cpa_scores = cpa_scores_lastround(cts, traces_feat)
    print("Computing MIA on features (this can be slow)...")
    mia_scores = mia_scores_lastround(cts, traces_feat, bins=args.bins)

    # normalize and ensemble
    cpa_norm = (cpa_scores - cpa_scores.min(axis=1, keepdims=True)) / (cpa_scores.ptp(axis=1, keepdims=True) + 1e-12)
    mia_norm = (mia_scores - mia_scores.min(axis=1, keepdims=True)) / (mia_scores.ptp(axis=1, keepdims=True) + 1e-12)
    ensemble = cpa_norm + mia_norm

    # write ranks and final key
    for b in range(16):
        order = np.argsort(-ensemble[b])
        with open(os.path.join(args.outdir, f"byte_{b:02d}.txt"), "w") as f:
            for k in order:
                if args.hexout: f.write(f"{int(k):02x}\n")
                else: f.write(f"{int(k)}\n")
    key_bytes = [int(np.argsort(-ensemble[b])[0]) for b in range(16)]
    key_hex = "".join(f"{k:02x}" for k in key_bytes)
    with open(os.path.join(args.outdir, "key.txt"), "w") as f:
        f.write(key_hex + "\n")

    # save diagnostics
    pd.DataFrame(cpa_scores).to_csv(os.path.join(args.outdir, "CPA_scores.csv"), index=False, header=False)
    pd.DataFrame(mia_scores).to_csv(os.path.join(args.outdir, "MIA_scores.csv"), index=False, header=False)
    np.savetxt(os.path.join(args.outdir, "var_per_time.txt"), var)
    print("Done. Outputs in:", args.outdir)
    print("Ensemble key (top-1 per byte):", key_hex)

if __name__ == "__main__":
    main()
