import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# At some point we can try to change this via a button press for now we default to overdrive
MODEL_FILE = "./trained_effects/best_model_MultidrivePedalPro-808-Scream_O000_L080_T100_h64.pt" 
HIDDEN = 64 # needs to be the same as the one we trained it on
SR = 48000
BLOCK = 256 # this is for less delay but WAY harder to keep up

INPUT_GAIN = 1.0 # Volume knob (kinda) 
OUTPUT_GAIN = 0.5 # low so we dont lose out hearing

torch.set_num_threads(1)

class AmpLSTM(nn.Module):
    # Basically the same as the model in training

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)    
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return x + self.out(h), state


def load_model():
    model = AmpLSTM()
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    return model

def process_block(model, block, state):
    x = torch.from_numpy(block)[None, :, None]
    with torch.no_grad():
        y, state = model(x, state)

    return y[0, :, 0].numpy(), state


def benchmark():
    """Time the model on fake audio and compare it with the real-time deadline."""
    model = load_model()
    block = (np.random.randn(BLOCK) * 0.1).astype("float32")

    state = None

    for _ in range(20):
        _, state = process_block(model, block, state)

    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        _, state = process_block(model, block, state)
        times.append((time.perf_counter() - t0) * 1000) # time here is in miliseconds

    deadline = BLOCK / SR * 1000 # time available per chunk
    worst = max(times)

    # Debug statements
    print(f"block size {BLOCK} samples -> deadline per block: {deadline:.2f} ms")
    print(f"processing time: average {np.mean(times):.2f} ms, worst {worst:.2f} ms")
    if worst < deadline * 0.5:
            print("OK: comfortably fast enough.")
    elif worst < deadline:
        print("TIGHT: it fits, but glitches are possible. Try a larger BLOCK.")
    else:
        print("TOO SLOW: increase BLOCK, or use a smaller HIDDEN model.")

def check(dry_path, wet_path):
    model= load_model()
    x, sr = sf.read(dry_path, dtype="float32")
    y, _ = sf.read(wet_path, dtype="float32")

    assert sr == SR, f"file is {sr} Hz, Model expects {SR} Hz"

    n = min(len(x), len(y))

    x, y = x[2 * SR : n - 2 * SR][: 30 * SR], y[2 * SR : n - 2 * SR][: 30 * SR]

    state, out = None, []

    for i in range(0, len(x), BLOCK):
        pred, state = process_block(model, x[i: i + BLOCK], state)
        out.append(pred)

    pred = np.concatenate(out)

    esr = np.sum((y - pred) ** 2) / np.sum(y ** 2)

    # Debug

    print(f"streaming ESR over {len(x) / SR:.0f} s: {esr:.4f}")
    print("Compare with your windowed test ESR. Similar = streaming works fine.")


def live():
    import sounddevice as sd

    model = load_model()
    memory = {"state": None, "in_peak": 0.0, "out_peak": 0.0, "glitches": 0}

    def callback(indata, outdata, frames, time_info, status):
        if status:
            memory["glitches"] += 1

        block = (indata[:, 0] * INPUT_GAIN).astype("float32")
        pred, memory["state"] = process_block(model, block, memory["state"])
        pred = np.clip(pred * OUTPUT_GAIN, -1.0, 1.0)

        outdata[:, 0] = pred
        if outdata.shape[1] > 1:
            outdata[:, 1] = pred

        memory["in_peak"] = max(memory["in_peak"], float(np.abs(block).max()))
        memory["out_peak"] = max(memory["out_peak"], float(np.abs(pred).max()))

    print(sd.query_devices())
    print("\nStarting. Press Ctrl+C to stop. HEADPHONES RECOMMENDED\n") # Will change later to be more dynamic

    with sd.Stream(samplerate=SR, blocksize=BLOCK, channels=(1, 2), dtype="float32", latency="low", callback=callback):

        try:
            while True:
                time.sleep(0.5)
                print(f"input peak {memory['in_peak']:.2f}   output peak "
                      f"{memory['out_peak']:.2f}   glitches {memory['glitches']}")
                memory["in_peak"] = memory["out_peak"] = 0.0

        except KeyboardInterrupt:
            print("Stopped")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"

    if mode == "benchmark":
        benchmark()

    elif mode == "check":
        check(sys.argv[2], sys.argv[3])
    else:
        live()