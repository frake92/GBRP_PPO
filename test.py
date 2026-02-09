import torch
import tensorflow as tf
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("avail:", torch.cuda.is_available())
print("hip:", torch.version.hip)
print("device:", torch.cuda.get_device_name(0))