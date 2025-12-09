import time
import tensorflow as tf
import numpy as np

start_load = time.time()
interpreter = tf.lite.Interpreter(model_path="../artifacts/autoencoder_int8.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
end_load = time.time()

load_time = end_load - start_load
print(f"Model load & allocation time: {load_time*1000:.2f} ms")

# Dummy input
dummy_input = np.zeros((1, 40, 64, 1), dtype=np.float32)

start_first = time.time()
interpreter.set_tensor(input_index, dummy_input)
interpreter.invoke()
reconstructed = interpreter.get_tensor(output_index)
end_first = time.time()

first_inference_time = end_first - start_first
print(f"Time to first inference: {first_inference_time*1000:.2f} ms")
