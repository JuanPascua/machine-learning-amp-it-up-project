# https://www.chciken.com/digital/signal/processing/2020/05/13/guitar-tuner.html

import sounddevice as sd
import scipy.fftpack
import numpy as np
import os

SAMPLE_FREQ = 44100 # Sample freq in Hz
WINDOW_SIZE = 44100 # window size of DFT (Discrete Fourier Transform)
WINDOW_STEP = 21050 # step Size of window
WINDOW_T_LENGTH = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
windowSamples = [0 for _ in range(WINDOW_SIZE)]


CONCERT_PITCH = 440 # 440 Hz for the A Note
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def find_closest_note(pitch):
    
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12) 
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)

    return closest_note, closest_pitch


def callback(indata, frames, time, status):
    global windowSamples
    if status:
        print(status)

    if any(indata):
        windowSamples = np.concatenate((windowSamples, indata[:, 0])) # append new samples
        windowSamples = windowSamples[len(indata[:, 0]) : ] # remove old samples
        maginitudeSpec = abs(scipy.fftpack.fft(windowSamples)[: len(windowSamples) // 2])

        for i in range(int(62 / (SAMPLE_FREQ / WINDOW_SIZE))):
            maginitudeSpec[i] = 0 # supress the humming

        maxInd = np.argmax(maginitudeSpec)
        maxFreq = maxInd * (SAMPLE_FREQ / WINDOW_SIZE)
        closestNote, closestPitch = find_closest_note(maxFreq)

        os.system('cls' if os.name=='nt' else 'clear')
        print(f"Closest Note: {closestNote} {maxFreq:.1f}/{closestPitch:.1f}")

    else:
        print("No Input")


try:
    with sd.InputStream(channels=1, callback=callback,
                        blocksize=WINDOW_STEP,
                        samplerate=SAMPLE_FREQ):
        while True:
            pass
except Exception as e:
    print(str(e))