#!/usr/bin/env python

import tensorflow as tf

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

print("It would appear that you can use GPU!")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
