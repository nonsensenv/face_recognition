import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Check TensorFlow version and GPU availability
print("TensorFlow Version:", tf.__version__)
print("Physical GPU Devices:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("No GPU detected. Running on CPU.")

# Check CUDA and cuDNN versions
print("CUDA Version:", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN Version:", tf.sysconfig.get_build_info()["cudnn_version"])