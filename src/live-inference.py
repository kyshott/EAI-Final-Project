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
threshold_colab = 32.873253
threshold_actual = 35 # Actual threshold from environment testing

# Audio Parameters
sr = 16000
env_freq = 44100 # Change to frequency of microphone used
duration = 0.5   # Half a second windows
n_mfcc = 40

# Window/gui setup
root = tk.Tk()
root.title("Fan Sound Monitor")
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack()

def extract_mfcc_fixed(audio, sr, n_mfcc=40, target_frames=64):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

# Audio capture processing
def audio_callback(indata, frames, time, status):
    audio = indata[:, 0].astype(np.float32)  # mono channel
    audio = librosa.resample(audio, orig_sr=env_freq, target_sr=sr)

    mfcc_input = extract_mfcc_fixed(audio, sr, n_mfcc=40, target_frames=64)

    interpreter.set_tensor(input_index, mfcc_input)
    interpreter.invoke()
    reconstructed = interpreter.get_tensor(output_index)

    error = np.mean((reconstructed - mfcc_input)**2)

    color = "red" if error > threshold_actual else "green"
    canvas.configure(bg=color)

stream = sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr*duration), callback=audio_callback)
stream.start()

# Run the gui loop
root.mainloop()

stream.stop()
stream.close()