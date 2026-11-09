import copy
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

"""
You would have train and test
"""
# Split into dry and wet
DEVICE = "LandlordBrewers-DroopChorus"
SETTINGS = "L050_D100_S050" # This the pedal

ROOT = Path(__file__).resolve().parent.parent   # project root (one level above src/)
DRY_DIR = ROOT / "data"  / "dry" 
WET_DIR = ROOT / "data" / "wet" / DEVICE



SEGMENTATION = 8192 # length of each training window (samples)
WARM = 512 # ignore the first samples of each window (LSTM warm-up)
BATCH = 32 # windows per training step
STEPS = 20000 # number of training steps
LEARNING_RATE = 3e-3 # Learning Rate
HIDDEN = 64 # LSTM memory size
TRIM_S = 2 # seconds cut from each end to remove the sync markers

dev = "cuda" if torch.cuda.is_available() else "cpu" # checks if cuda is available on GPU

if SETTINGS is None:
    SETTINGS = available[0]            # nothing chosen: use the first one
print("using setting:", SETTINGS)
 
def find_sources(split):
    """Look at the wet file names to find which recordings exist.
    A wet file is called <SETTING>.<source>.target.wav, so the source is the 2nd piece."""
    folder = WET_DIR / split / SETTINGS
    return sorted(f.name.split(".")[1] for f in folder.glob("*.target.wav"))
 
TRAIN_SOURCES = find_sources("trainval")
TEST_SOURCES = find_sources("test")
print("train sources:", TRAIN_SOURCES)
print("test sources: ", TEST_SOURCES)


def load(split, source):
    x, sr = sf.read(str(Path(DRY_DIR) / split / f"{source}.input.wav"), dtype="float32")

    folder = Path(WET_DIR) / split / SETTINGS
    matches = list(folder.glob(f"*.{source}.target.wav"))     # any prefix, same source
    if not matches:
        raise FileNotFoundError(f"No wet file for '{source}' in {folder}")
    y, _ = sf.read(str(matches[0]), dtype="float32")

    if x.ndim > 1: x = x.mean(axis=1)
    if y.ndim > 1: y = y.mean(axis=1)
    n, t = min(len(x), len(y)), int(TRIM_S * sr)
    return x[t:n - t], y[t:n - t], sr

train, val = [], []
for s in TRAIN_SOURCES:
    x, y, sr = load("trainval", s)
    cut = int(len(x) * 0.9)
    train.append((x[:cut - SEGMENTATION], y[:cut - SEGMENTATION]))
    val.append((x[cut:], y[cut:]))

    print(f"loaded {len(train)} training files, sample rate {sr}")

class AmpLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return x + self.out(h), state  # predict the CHANGE from dry to wet

def esr(pred, target):
    return ((target - pred) ** 2).sum() / (target ** 2).sum()

def pre_emph(a): # boosts high frequencies a little
    return a[:, 1:] - 0.95 * a[:, :-1]

def train_loss(pred, target):
    return esr(pre_emph(pred[:, WARM:]), pre_emph(target[:, WARM:]))

@torch.no_grad()
def score(model, pairs, gain=None):
    """ESR over a list of (dry, wet) files. model=None -> gain-only baseline."""
    err = energy = 0.0
    for x, y in pairs:
        n = (len(x) // SEGMENTATION) * SEGMENTATION
        xs = torch.from_numpy(x[:n]).view(-1, SEGMENTATION, 1).to(dev)
        ys = torch.from_numpy(y[:n]).view(-1, SEGMENTATION, 1).to(dev)
        for i in range(0, len(xs), 64):
            xb, yb = xs[i:i + 64], ys[i:i + 64]
            pred = gain * xb if model is None else model(xb)[0]
            err += ((yb - pred)[:, WARM:] ** 2).sum().item()
            energy += (yb[:, WARM:] ** 2).sum().item()

    return err / energy

def get_batch():
    xs, ys = [], []
    for _ in range(BATCH):
        x, y = random.choice(train)
        i = random.randint(0, len(x) - SEGMENTATION)
        xs.append(x[i: i + SEGMENTATION])
        ys.append(y[i:i + SEGMENTATION])

    f = lambda a : torch.from_numpy(np.stack(a))[..., None].to(dev)

    return f(xs), f(ys)


# baseline: just make the dry signal as loud as the wet one
gain = float(np.sqrt(sum((y ** 2).sum() for _, y in train) /
    sum((x ** 2).sum() for x, _ in train
)))

print(f"baseline validation ESR (dry x {gain:.2f}): {score(None, val, gain):.4f}")

model = AmpLSTM().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
best, best_state = 1e9, None
for step in range(1, STEPS  + 1):
    xb, yb = get_batch()
    loss = train_loss(model(xb)[0], yb)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 250 == 0:
        v = score(model, val)
        print(f"step {step:5d}   train loss {loss.item():.4f}   validation ESR {v:.4f}")
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
torch.save(best_state, "best_model.pt")
test = [load("test", s)[:2] for s in TEST_SOURCES]

print(f"\nTEST  baseline ESR: {score(None, test, gain):.4f}   LSTM ESR: {score(model, test):.4f}")

x, y = test[0]
out, state = [], None
RENDER_BLOCK = 8192
with torch.no_grad():
    for i in range(0, len(x), RENDER_BLOCK): # carry the LSTM memory between blocks
        chunk = np.ascontiguousarray(x[i:i + RENDER_BLOCK])
        block = torch.from_numpy(chunk)[None, :, None].to(dev)
        pred, state = model(block, state)
        out.append(pred[0, :, 0].cpu().numpy())

NAME = f"{DEVICE}_{SETTINGS}_h{HIDDEN}"
torch.save(best_state, f"best_model_{NAME}.pt")
sf.write(f"model_output_{NAME}.wav", np.concatenate(out), sr)
sf.write(f"real_pedal_{NAME}.wav", y, sr)
print("wrote model_output.wav (your model) and real_pedal.wav (the real pedal)")