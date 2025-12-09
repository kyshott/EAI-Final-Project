# latency_test.py
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import time
import psutil
import os

"""
This script measure the average inference latency as well as on-device RAM usage.
"""

interpreter = tf.lite.Interpreter(model_path="../artifacts/autoencoder_int8.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

sr = 22050
env_freq = 44100  
duration = 0.5  
n_mfcc = 40
target_frames = 64
num_runs = 100   

def extract_mfcc_fixed(audio, sr, n_mfcc=40, target_frames=64):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

def record_window():
    audio = sd.rec(int(duration * env_freq), samplerate=env_freq, channels=1, dtype='float32')
    sd.wait()
    audio = audio[:, 0]  # mono
    audio = librosa.resample(audio, orig_sr=env_freq, target_sr=sr)
    return audio

latencies = []
mem_usages = []
cpu_usages = []

process = psutil.Process(os.getpid())

for i in range(num_runs):
    audio = record_window()
    mfcc_input = extract_mfcc_fixed(audio, sr, n_mfcc=n_mfcc, target_frames=target_frames)

    start_time = time.time()
    interpreter.set_tensor(input_index, mfcc_input)
    interpreter.invoke()
    reconstructed = interpreter.get_tensor(output_index)
    end_time = time.time()

    latencies.append(end_time - start_time)

    mem_info = process.memory_info()
    mem_usages.append(mem_info.rss / 1024**2)
    cpu_usages.append(process.cpu_percent(interval=None)) 

avg_mem_mb = np.mean(mem_usages)
avg_cpu_percent = np.mean(cpu_usages)
latencies_ms = np.array(latencies) * 1000
p50 = np.percentile(latencies_ms, 50)
p95 = np.percentile(latencies_ms, 95)

print(f"P50 latency: {p50:.2f} ms")
print(f"P95 latency: {p95:.2f} ms")
print(f"Average memory usage (RSS): {avg_mem_mb:.2f} MB")
print(f"Average CPU utilization: {avg_cpu_percent:.2f}%")
