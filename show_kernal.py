import tensorflow as tf
import json
import custom_function as cf
import model
import matplotlib.pyplot as plt
import numpy as np

# tf version check
tf_version = cf.get_tf_version()

# prevent GPU overflow
cf.tf_gpu_active_alloc()

# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

frame_size = config["frame_size"]
sampling_rate = config["sampling_rate"]

cut_off_freq = config["cut_off_freq"]
alpha = config["alpha"]
filter_order = config["filter_order"]

load_filter_checkpoint = config["load_filter_checkpoint"]

plot_name = config["plot_name"]

# make model
filter = model.Filter(frame_size, cut_off_freq, filter_order, sampling_rate)

x = tf.zeros([1, frame_size, 1])
filter(x)

if load_filter_checkpoint != "":
    full_path = cf.load_directory() + '/checkpoint/' + load_filter_checkpoint
    filter.load_weights(full_path + '/data.ckpt')


# weights, biases = filter.layers[0].get_weights()
# weights = tf.squeeze(tf.transpose(weights, [2, 1, 0]))

weights = filter.W
weights = tf.squeeze(tf.transpose(weights, [2, 1, 0])).numpy()
weights = weights[:,frame_size-1:]
biases = filter.b.numpy()

mag_weight = abs(np.fft.fft(weights))
plt.plot(np.transpose(mag_weight))
plt.show()

print(*biases)
