import torch
from dataclasses import dataclass

@dataclass
class CFG:
    seed = 42
    records = [
        "100","101","102","103","104","105","106","107","108","109",
        "111","112","113","114","115","116","117","118","119","121",
        "122","123","124","200","201","202","203","205","207","208",
        "209","210","212","213","214","215","217","219","220","221",
        "222","223","228","230","231","232","233","234"
    ]
    window_sec = 2.0
    crop_len = 720
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    split_mode = "record"
    batch_size = 64
    epochs = 100
    z_dim = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"



cfg = CFG()