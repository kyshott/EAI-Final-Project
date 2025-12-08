import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import tkinter as tk

interpreter = tf.lite.Interpreter(model_path="../artifacts/autoencoder_int8.tflite")
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
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack()

def extract_mfcc_fixed(audio, sr, n_mfcc=40, target_frames=64):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Pad if too short
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    # Truncate if too long
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

# Audio capture processing
def audio_callback(indata, frames, time, status):
    audio = indata[:, 0]  # mono channel

    mfcc_input = extract_mfcc_fixed(audio, sr, n_mfcc=40, target_frames=64)

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