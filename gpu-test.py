#GPU Test Kodu:

import tensorflow as tf
print("GPU Kullanılıyor mu?", tf.test.is_built_with_cuda())
print("Mevcut GPU:", tf.config.list_physical_devices('GPU'))
print("TensorFlow versiyonu:", tf.__version__)

# GPU kullanımı kontrolü
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU Başarıyla Kullanılıyor.")
else:
    print("GPU Bulunamadı.")