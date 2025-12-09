import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class MFCCVisualizer:
    def __init__(self, threshold, mfcc_dim=40):
        self.threshold = threshold

        self.history = deque(maxlen=120)  # 120 windows stored

        self.fig, (self.ax_mfcc, self.ax_err) = plt.subplots(2, 1, figsize=(6,8))
        
        # MFCC Spectrogram view
        self.img = self.ax_mfcc.imshow(
            np.zeros((mfcc_dim, 120)),
            origin='lower',
            aspect='auto',
            cmap='plasma'
        )
        self.ax_mfcc.set_title("Live MFCC Stream")

        self.ax_err.set_title("Reconstruction Error")
        self.ax_err.set_xlim(0, 1)
        self.ax_err.set_ylim(0, threshold * 2)
        self.ax_err.axhline(threshold, color="cyan", linewidth=2)
        self.dot, = self.ax_err.plot([0.5], [0], 'wo', markersize=12)

        plt.tight_layout()
        plt.ion()
        plt.show()

    def update(self, mfcc, error):
        if mfcc.ndim == 2:
            col = np.mean(mfcc, axis=1)
        else:
            col = mfcc

        self.history.append(col)

        data = np.stack(self.history, axis=1)
        self.img.set_data(data)

        self.dot.set_ydata([error])

        if error > self.threshold:
            self.ax_err.set_facecolor("darkred")
        else:
            self.ax_err.set_facecolor("black")

        plt.pause(0.001)
