import numpy as np
import torch
import wfdb


def load_record(name, duration_sec=600):
    rec = wfdb.rdrecord(name, pn_dir="mitdb")
    ann = wfdb.rdann(name, "atr", pn_dir="mitdb")

    fs = rec.fs
    sig = rec.p_signal[:int(fs * duration_sec)]

    valid = ann.sample < int(fs * duration_sec)
    ann.sample = ann.sample[valid]
    ann.symbol = np.array(ann.symbol)[valid].tolist()

    return sig, ann, fs

def segment_beats(signal, ann, fs, window_sec=0.8):
    half = int(fs * window_sec / 2)
    beats = []

    for sample, sym in zip(ann.sample, ann.symbol):
        if sym != "N":
            continue

        start = sample - half
        end = sample + half

        if start < 0 or end > len(signal):
            continue

        seg = signal[start:end, 0]

        seg = seg - seg.mean()
        seg = seg / (seg.std() + 1e-8)

        seg = np.clip(seg, -5, 5) / 5.0

        beats.append(seg.astype(np.float32))

    if not beats:
        beat_len = int(fs * window_sec)
        return np.empty((0, beat_len), dtype=np.float32)

    return np.stack(beats)


def build_dataset(records, window_sec=0.8):
    xs = []

    for r in records:
        sig, ann, fs = load_record(r)
        beats = segment_beats(sig, ann, fs, window_sec)

        if beats.size == 0:
            continue

        xs.append(beats[:, None, :])

    if not xs:
        raise RuntimeError("No beats found.")

    X = np.concatenate(xs, axis=0)
    np.random.shuffle(X)

    print(f"Total Normal beats: {len(X)}")
    return X


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
