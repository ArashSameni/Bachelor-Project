import gc
import itertools
import os
import pickle
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

# Modulation types
MODULATIONS = {
    "QPSK": 0,
    "8PSK": 1,
    "AM-DSB": 2,
    "QAM16": 3,
    "GFSK": 4,
    "QAM64": 5,
    "PAM4": 6,
    "CPFSK": 7,
    "BPSK": 8,
    "WBFM": 9,
}

# Signal-to-Noise Ratios
SNRS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]


class RadioML2016(torch.utils.data.Dataset):
    URL = "."
    modulations = MODULATIONS
    snrs = SNRS

    def __init__(self, data_dir: str = ".", file_name: str = "RML2016.10a_dict.dat"):
        self.file_name = file_name
        self.data_dir = data_dir
        self.n_classes = len(self.modulations)
        self.X, self.y = self.load_data()
        gc.collect()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from file"""
        print("Loading dataset from file...")
        with open(os.path.join(self.data_dir, self.file_name), "rb") as f:
            data = pickle.load(f, encoding="latin1")

        X, y = [], []
        print("Processing dataset")
        for mod, snr in tqdm(list(itertools.product(self.modulations, self.snrs))):
            X.append(data[(mod, snr)])

            for i in range(data[(mod, snr)].shape[0]):
                y.append((mod, snr))

        X = np.vstack(X)

        return X, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a batch of input and labels"""
        x, (mod, snr) = self.X[idx], self.y[idx]
        y = self.modulations[mod]
        x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = x.to(torch.float).unsqueeze(0)
        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]
