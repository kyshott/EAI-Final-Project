import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import tkinter as tk

interpreter = tf.lite.Interpreter(model_path="artifacts/autoencoder_int8.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# MSE Threshold
threshold = 32.873253

# Audio Parameters
sr = 22050
duration = 0.5   # Half a second windows
n_mfcc = 40

# Window for anomaly visualization
root = tk.Tk()
root.title("Fan Sound Monitor")
root.attributes("-fullscreen", True)  # full screen
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack()

# Audio capture processing
def audio_callback(indata, frames, time, status):
    audio = indata[:, 0]  # mono channel

    # MFCC extraction
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    W = mfcc.shape[1]
    pad_w = (4 - W % 4) % 4
    mfcc = np.pad(mfcc, ((0,0),(0,pad_w)), mode='constant')
    mfcc_input = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

    # Run TFLite inference
    interpreter.set_tensor(input_index, mfcc_input)
    interpreter.invoke()
    reconstructed = interpreter.get_tensor(output_index)

    # Compute reconstruction error
    error = np.mean((reconstructed - mfcc_input)**2)

    # Update background color
    color = "red" if error > threshold else "green"
    canvas.configure(bg=color)

# Actually start audio stream
stream = sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr*duration), callback=audio_callback)
stream.start()

# Run the gui loop
root.mainloop()

stream.stop()
stream.close()