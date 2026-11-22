import copy
import random
from pathlib import Path
import shutil
from typing import TypedDict
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from datetime import datetime
from src.baseline.models import accumulate_normal_equations, solve_ridge, linear_predict

class pedal_data(TypedDict):
  alpha: float
  steps: int
  batch: int
  hidden_size: int

SEED = 42 
ROOT = Path(__file__).resolve().parent.parent   # project root (one level above src/)
DRY_DIR = ROOT /"data/dry/DRY-with-markers" 
WET_DIR = ""

dev = "cuda" if torch.cuda.is_available() else "cpu" # checks if cuda is available on GPU

SEGMENTATION = 8192 # length of each training window (samples)
WARM = 512 # ignore the first samples of each window (LSTM warm-up)
TRIM_S = 2 # seconds cut from each end to remove the sync markers
#To tune per model
BATCH = 0
STEPS = 20000 # number of training steps
STEP = 0
LEARNING_RATE = 0
HIDDEN = 0
SETTING = ""
train,val = [], []
def train_models(device:str,device_metadata:dict[str,pedal_data]):
    # Split into dry and wet
    global BATCH
    global LEARNING_RATE
    global HIDDEN
    global SETTING
    global train,val
    global WET_DIR
    
    DEVICE =  device
    WET_DIR = ROOT /"data/wet"/DEVICE
    for SETTING in device_metadata.keys():
        print(SETTING)
        save_to_path =(ROOT/"models"/DEVICE/SETTING)
        save_to_path.mkdir(parents=True, exist_ok=True)
        (save_to_path/"models").mkdir(parents=True,exist_ok=True)
        (save_to_path/"audio").mkdir(parents=True,exist_ok=True)
        #To tune per model
        BATCHES = [16,32,64] # windows per training step
        LEARNING_RATES = [1e-3,3e-3,1e-2,3e-3,1e-1] # Learning Rate
        HIDDEN_SIZES = [16,32,64] # LSTM memory size

        if SETTING is None:
            raise KeyError("Setting for device not provided")
        print("using setting:", SETTING)
        
        TRAIN_SOURCES = find_sources("trainval")
        TEST_SOURCES = find_sources("test")
        print("train sources:", TRAIN_SOURCES)
        print("test sources: ", TEST_SOURCES)

        train, val = [], []
        for s in TRAIN_SOURCES:
            x, y, sr = load("trainval", s)
            cut = int(len(x) * 0.9)
            train.append((x[:cut - SEGMENTATION], y[:cut - SEGMENTATION]))
            val.append((x[cut:], y[cut:]))
            print(f"loaded {len(train)} training files, sample rate {sr}")
        test = [load("test", s)[:2] for s in TEST_SOURCES]

        # baseline: just make the dry signal as loud as the wet one
        gain = float(np.sqrt(sum((y ** 2).sum() for _, y in train) /
            sum((x ** 2).sum() for x, _ in train
        )))
        #Used to determine whether to save the model or let it go
        overall_best_score = score(None, val, gain)[0]
        overall_best_state = None
        overall_best_parms = None

        print(f"baseline validation ESR (dry x {gain:.2f}): {overall_best_score:.4f}")
        SAVED_NAME = ""
        parameters = [["Model","Hidden_Size","Batch","Learning_Rate", "Best_Step", "Train_Time","Train_ESR","Train_MSE","Val_ESR","Val_MSE"]] 

        num_past_samples = [8,16,32,64]
        xtx, xty, count = accumulate_normal_equations([x for x, _ in train], [y for _, y in train], max(num_past_samples))
        best_ridge_esr = 1e9 
        best_ridge_alpha = None 
        best_ridge_num_sample= None
        best_ridge_weights = None
        for num_past_sample in num_past_samples:
            for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
                start_time = datetime.now()
                weights = solve_ridge(xtx, xty, count, num_past_sample, alpha)
                train_time = round((datetime.now() - start_time).total_seconds(),3)
                ridge_train_esr,ridge_train_mse = score(None, train, weights=weights)
                ridge_val_esr,ridge_val_mse = score(None, val, weights=weights)
                parameters.append([f"ridge_N{num_past_sample}_a{alpha}",0,0,0,0,train_time, ridge_train_esr,ridge_train_mse,ridge_val_esr,ridge_val_mse])
                if ridge_val_esr < best_ridge_esr:
                    best_ridge_esr = ridge_val_esr
                    best_ridge_alpha = alpha
                    best_ridge_num_sample = num_past_sample
                    best_ridge_weights = weights
        ridge_name = f"ridge_N{best_ridge_num_sample}_a{best_ridge_alpha}"
                                                                                                                            
        # record baselines in validation_results.csv
        for label, model_gain, model_weights in [("naive_dry", 1.0, None), ("naive_gain", gain, None)]:
            train_esr, train_mse = score(None, train, gain=model_gain, weights=model_weights)                           
            val_esr, val_mse = score(None, val, gain=model_gain, weights=model_weights)                                 
            parameters.append([label, 0, 0, 0, 0,0, train_esr, train_mse, val_esr, val_mse]) 

        for HIDDEN in HIDDEN_SIZES:
            for BATCH in BATCHES:
                for LEARNING_RATE in LEARNING_RATES:
                    random.seed(SEED)
                    np.random.seed(SEED)
                    torch.manual_seed(SEED)
                    model = AmpLSTM().to(dev)
                    STEP = 0
                    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    best, best_state = 1e9, copy.deepcopy(model.state_dict())
                    current_best_score = best
                    start_time = datetime.now()
                    for step in range(1, STEPS  + 1):
                        xb, yb = get_batch()
                        loss = train_loss(model(xb)[0], yb)
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        if step % 100 == 0:
                            v = score(model, val)[0]
                            print(f"step {step:5d}   train loss {loss.item():.4f}   validation ESR {v:.4f}")
                            if not np.isfinite(v):
                                break;
                            if v < best:
                                best, best_state = v, copy.deepcopy(model.state_dict())
                                STEP = step
                                current_best_score = v

                            #Stop training early if no improvements are made.
                            if step - STEP > 2000:
                                break
                    train_time = round((datetime.now() - start_time).total_seconds(),3)
                    model.load_state_dict(best_state)
                    NAME = f"{DEVICE}_{SETTING}_h{HIDDEN}_b{BATCH}_a{LEARNING_RATE}"
                    print(f"Finished: {NAME}")
                    train_esr, train_mse = score(model, train)
                    val_esr, val_mse = score(model, val)
                    parameters.append([NAME, HIDDEN, BATCH, LEARNING_RATE, STEP, train_time, train_esr, train_mse, val_esr, val_mse]) 
                    if current_best_score < overall_best_score:
                        overall_best_score = current_best_score
                        overall_best_state = copy.deepcopy(best_state)
                        overall_best_parms = (NAME, HIDDEN, BATCH, LEARNING_RATE, STEP)

              
        df = pd.DataFrame(data=parameters[1:],columns=parameters[0])
        df.to_csv(save_to_path / "validation_results.csv", index=False)

        # evaluate test set once on baselines, ridge, and best LSTM
        test_rows = [["Model", "Source", "Test_ESR", "Test_MSE"]]
        final_models = [("naive_dry", None, 1.0, None),("naive_gain", None, gain, None), (ridge_name, None, None, best_ridge_weights)]
                                                                                                                      
        if overall_best_parms is not None:
            SAVED_NAME, HIDDEN, BATCH, LEARNING_RATE, STEP = overall_best_parms
            best_model = AmpLSTM().to(dev)
            best_model.load_state_dict(overall_best_state)
            final_models.append((SAVED_NAME, best_model, None, None))
                                                
            for old_audio in (save_to_path/ "audio").iterdir():
              (old_audio).unlink(missing_ok=True)
            for old_models in (save_to_path/ "models").iterdir():
              (old_models).unlink(missing_ok=True)
            x, y = test[0]
            out, state = [], None
            RENDER_BLOCK = 8192
            with torch.no_grad():
              for i in range(0, len(x), RENDER_BLOCK): # carry the LSTM memory between blocks
                chunk = np.ascontiguousarray(x[i:i + RENDER_BLOCK])
                block = torch.from_numpy(chunk)[None, :, None].to(dev)
                pred, state = best_model(block, state)
                out.append(pred[0, :, 0].cpu().numpy())

            torch.save(overall_best_state, f"{save_to_path}/models/best_model_{SAVED_NAME}.pt")
            sf.write(f"{save_to_path}/audio/model_output_{SAVED_NAME}.wav", np.concatenate(out), sr)
            sf.write(f"{save_to_path}/audio/real_pedal_{SAVED_NAME}.wav", y, sr)
            print("wrote model_output.wav (your model) and real_pedal.wav (the real pedal)")
            device_metadata[SETTING] = {"alpha": LEARNING_RATE, "steps": STEP, "batch": BATCH, "hidden_size": HIDDEN}
        for label, eval_model, model_gain, model_weights in final_models:
            test_esr, test_mse = score(eval_model, test, gain=model_gain, weights=model_weights)
            print(f"TEST {label}: ESR={test_esr:.4f}  MSE={test_mse:.6f}")
            test_rows.append([label, "ALL", test_esr, test_mse])
            for source, pair in zip(TEST_SOURCES, test):
                source_esr, source_mse = score(eval_model, [pair], gain=model_gain, weights=model_weights)
                test_rows.append([label, source, source_esr, source_mse])
        pd.DataFrame(data=test_rows[1:], columns=test_rows[0]).to_csv(save_to_path / "test_results.csv", index=False) 


def find_sources(split):
    """Look at the wet file names to find which recordings exist.
    A wet file is called <SETTING>.<source>.target.wav, so the source is the 2nd piece."""
    folder = WET_DIR / split / SETTING
    return sorted(f.name.split(".")[1] for f in folder.glob("*.target.wav"))

def load(split, source):
    x, sr = sf.read(str(Path(DRY_DIR) / split / f"{source}.input.wav"), dtype="float32")

    folder = Path(WET_DIR) / split / SETTING
    matches = list(folder.glob(f"*.{source}.target.wav"))     # any prefix, same source
    if not matches:
        raise FileNotFoundError(f"No wet file for '{source}' in {folder}")
    y, _ = sf.read(str(matches[0]), dtype="float32")

    if x.ndim > 1: x = x.mean(axis=1)
    if y.ndim > 1: y = y.mean(axis=1)
    n, t = min(len(x), len(y)), int(TRIM_S * sr)
    return x[t:n - t], y[t:n - t], sr

class AmpLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return x + self.out(h), state  # predict the CHANGE from dry to wet

def esr(pred, target):
    return ((target - pred) ** 2).sum() / ((target ** 2).sum() + 1e-9)

def pre_emph(a): # boosts high frequencies a little
    return a[:, 1:] - 0.95 * a[:, :-1]

def train_loss(pred, target):
    return esr(pre_emph(pred[:, WARM:]), pre_emph(target[:, WARM:]))

@torch.no_grad()
def score(model, pairs, gain=None,weights=None):
    """ESR and MSE over a list of (dry, wet) files for the models."""
    err = energy = 0.0
    total_samples = 0 
    for x, y in pairs:
        n = (len(x) // SEGMENTATION) * SEGMENTATION
        xs = torch.from_numpy(x[:n]).view(-1, SEGMENTATION, 1).to(dev)
        ys = torch.from_numpy(y[:n]).view(-1, SEGMENTATION, 1).to(dev)
        for i in range(0, len(xs), 64):
            xb, yb = xs[i:i + 64], ys[i:i + 64]
            if weights is not None:
                ridge_pred = linear_predict(xb[:, :, 0].cpu().numpy(), weights)
                pred = torch.from_numpy(ridge_pred).to(dev, dtype=yb.dtype)[..., None]
            else:
                pred = gain * xb if model is None else model(xb)[0] 
            err += ((yb - pred)[:, WARM:] ** 2).sum().item()
            energy += (yb[:, WARM:] ** 2).sum().item()
            total_samples += yb[:, WARM:].numel()
    return err / energy, err/ total_samples

def get_batch():
    xs, ys = [], []
    for _ in range(BATCH):
        x, y = random.choice(train)
        i = random.randint(0, len(x) - SEGMENTATION)
        xs.append(x[i: i + SEGMENTATION])
        ys.append(y[i:i + SEGMENTATION])

    f = lambda a : torch.from_numpy(np.stack(a))[..., None].to(dev)

    return f(xs), f(ys)
