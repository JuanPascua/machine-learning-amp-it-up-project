import sys
from math import ceil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html

# The settings for this are the same when we trained it because they have to be for consistencty
BEST_MODEL = "best_model_ElectroHarmonix-MetalMuff_V100_G000_D000_T050_M050_B050_h64.pt"
HIDDEN = 64 # Like the hidden layers
SEGMENTATION, WARM = 8192, 512
SR = 48000
INPUT_PEAK = 0.5
ROOT = Path(__file__).resolve().parent.parent 

dev = "cuda" if torch.cuda.is_available() else "cpu"

class AmpLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return x + self.out(h), state

FILE_PATH = ROOT / "src" / "audio" / sys.argv[1]
path = Path(FILE_PATH) # Used in terminal when running command (we could prob change this later)
x, sr = sf.read(str(path), dtype="float32")
if x.ndim > 1:
    x = x.mean(axis=1)

if sr != SR:
    g = np.gcd(sr, SR)
    x = resample_poly(x, SR // g, sr // g).astype("float32")

# 1e-9 avoids division by 0

x = x / (np.abs(x).max() + 1e-9) * INPUT_PEAK

model = AmpLSTM().to(dev)
model.load_state_dict(torch.load(BEST_MODEL, map_location=dev))
model.eval()


step = SEGMENTATION - WARM
n_blocks = ceil(len(x) / step)
xp = np.concatenate([np.zeros(WARM, np.float32), x, np.zeros(SEGMENTATION, np.float32)])


blocks = np.stack([xp[i * step: i * step + SEGMENTATION] for i in range(n_blocks)])

out = []
with torch.no_grad():
    for i in range(0, n_blocks, 64):
        # In [..., None], the ... means “all the existing dimensions,” and None adds one new dimension at the end. It turns a (batch, 8192) array into (batch, 8192, 1), because the LSTM wants a trailing “channels” dimension.
        b = torch.from_numpy(blocks[i: i + 64])[..., None].to(dev)
        out.append(model(b)[0][:, WARM:, 0].cpu().numpy())

y = np.concatenate(out).reshape(-1)[:len(x)]

print(f"Peak of output: {np.abs(y).max():.2f}    (1.0 would clip)")

y = y / max(1.0, np.abs(y).max())
sf.write(str(path.with_name(path.stem + "_processed.wav")), y, SR)

print("wrote", path.stem + "_processed.wav")