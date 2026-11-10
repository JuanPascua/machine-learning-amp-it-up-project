"""
TUTORIAL FOR UI
https://www.pythonguis.com/pyqt6-tutorial/
"""


from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (QApplication, QComboBox, QDial, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QVBoxLayout, QWidget)


import sys
import re
import torch.nn as nn
import torch
from pathlib import Path
import numpy as np

# Use if using command line args
# app = QApplication(sys.argv)

SR = 48000
BLOCK = 256
_here = Path(__file__).resolve().parent.parent
MODELS = next((p for p in (_here / "trained_effects", _here.parent / "trained_effects")
               if p.exists()), _here / "trained_effects")

torch.set_num_threads(1)

class AmpLSTM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(1,hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return x + self.out(h), state


# ============= EVERYTHING HERE IS THE SAME AS THE ONE WHERE IT RUNS IN TERMINAL =============== #
class Engine:

    """
    Purely for Audio, Does not interact with any widgets
    """

    def __init__(self):
        self.model = None
        self.pending = None
        self.state = None
        self.stream = None

        self. gain = 1.0
        self.volume = 0.5
        self.bypass = False

        self.in_peak = 0.0
        self.out_peak = 0.0

        self.glitches = 0

    def load(self, name):
        if not name:
            return
        state = torch.load(MODELS / name, map_location="cpu")
        hidden = state["lstm.weight_hh_l0"].shape[1]

    def callback(self, indata, outdata, frames, t, status):
        if status:
            self.glitches += 1

        if self.pending is not None:
            self.model = self.pending
            self.state = None
            self.pending = None

        x = indata[:, 0] * self.gain

        if self.bypass or self.model is None:
            y = x

        else:
            with torch.no_grad():
                out, self.state = self.model(torch.from_numpy(x)[None, :, None], self.state)

                y = out[0, :, 0].numpy()

            y = np.clip(y * self.volume, -1.0, 1.0)
            outdata[:, :] = y[:, None]

            self.in_peak = max(self.in_peak, float(np.abs(x).max()))
            self.out_peak = max(self.out_peak, float(np.abs(y).max()))
        
    def start(self):
        import sounddevice as sd
        if self.stream is None:
            self.stream = sd.Stream(samplerate=SR, blocksize=BLOCK, channels=(1, 2), dtype="float32", latency="low", callback=self.callback)
            self.stream.start()

    def stop(self):
        if self.stream is None:
            self.stream.stop()
            self.stream.close()
            self.stream = None





# ========== UI STUFF ==========
class AmpWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Amp It Up!")
        self.eng = Engine()
        layout = QVBoxLayout(self)

        self.combo = QComboBox()
        self.combo.addItems(sorted(p.name for p in MODELS.glob("*.pt")))
        self.combo.currentTextChanged.connect(self.eng.load)

        self.eng.load(self.combo.currentText()) # load selected

        layout.addWidget(self.combo)

        dials = QHBoxLayout()
        self.gain_label = QLabel("gain 1.00")
        self.vol_label = QLabel("volume 0.50")
        dials.addLayout(self.make_dial("INPUT GAIN", 10, 400, 100, self.on_gain, self.gain_label))
        dials.addLayout(self.make_dial("VOLUME", 0, 100, 50, self.on_volume, self.vol_label))
        layout.addLayout(dials)


        self.in_meter = QProgressBar()
        self.out_meter = QProgressBar()

        for name, bar in (("IN", self.in_meter), ("OUT", self.out_meter)):
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            row.addWidget(bar)
            layout.addLayout(row)

        row = QHBoxLayout()
        self.power = QPushButton("START")
        self.power.setCheckable(True)
        self.power.toggled.connect(self.on_power) # Kinda like an onclick call this function
        self.bypass = QPushButton("BYPASS")
        self.bypass.setCheckable(True)
        self.bypass.toggled.connect(lambda on: setattr(self.eng, "bypass", on))

        row.addWidget(self.power)
        row.addWidget(self.bypass)
        self.status = QLabel("Stopped. Headphones Reccomended")
        layout.addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_meters)
        self.timer.start(33)

        self.setStyleSheet( # Like CSS might move it to its own file later
            "QWidget{background: #2a2420; color: #eee}"
            "QPushButton{padding:8px;border:1px solid #6b5a3e;} "
            "QPushButton:checked{background:#ffb03b;color:#111;} "
            "QProgressBar{background:#111;border:none;} "
            "QProgressBar::chunk{background:#4caf50;}"
        ) 


    def make_dial(self, title, lo, hi, start, handler, value_label):
        box = QVBoxLayout()
        dial = QDial()
        dial.setRange(lo, hi)
        dial.setValue(start)
        dial.setNotchesVisible(True)
        dial.valueChanged.connect(handler)
        box.addWidget(QLabel(title))
        box.addWidget(dial)
        box.addWidget(value_label)
        return box


    
    def on_gain(self, v):
        self.eng.gain = v / 100
        self.gain_label.setText(f"gain {v / 100}")


    def on_volume(self, v):
        self.eng.volume = v / 100
        self.vol_label.setText(f"volume {v / 100}")

        
    def on_power(self, on):
        try:
            self.eng.start() if on else self.eng.stop()
            self.power.setText("STOP" if on else "START")
            self.status.setText("running" if on else "stopped")
        except Exception as e:
            self.power.setChecked(False)
            self.status.setText(f"audio error: {e}")

    def update_meters(self):
        self.in_meter.setValue(int(min(1.0, self.eng.in_peak) * 100))
        self.out_meter.setValue(int(min(1.0, self.eng.out_peak) * 100))
        self.eng.in_peak = self.eng.out_peak = 0.0
        if self.eng.stream is not None:
            self.status.setText(f"running glitches: {self.eng.glitches}")


    def closeEvent(self, event):
            self.eng.stop()
            super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AmpWindow()
    win.show()
    sys.exit(app.exec())

        
        