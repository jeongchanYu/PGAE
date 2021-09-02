import tensorflow as tf
import json
import os
import custom_function as cf
import model
import time
import datetime
import math
import make_dataset as md
import numpy as np
import wav


# tf version check
tf_version = cf.get_tf_version()

# prevent GPU overflow
cf.tf_gpu_active_alloc()

# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

frame_size = config["frame_size"]
shift_size = config["shift_size"]
window_type = config["window_type"]
sampling_rate = config["sampling_rate"]

cut_off_freq = config["cut_off_freq"]
alpha = config["alpha"]

filter_order = config["filter_order"]
batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]
default_float = config["default_float"]

test_source_path = config["test_source_path"]
test_target_path = config["test_target_path"]

load_checkpoint_name = config["load_checkpoint_name"]

if len(alpha) != len(cut_off_freq)+1:
    raise Exception("len(alpha) must be same with len(cut_off_freq)+1")

# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()


with strategy.scope():
    # make model
    encoder = model.Encoder()
    decoder = model.Decoder()

    def loss_object(y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)), axis=1)
        loss = tf.reduce_sum(loss, axis=0)
        return loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # load model
    if load_checkpoint_name != "":
        full_path = cf.load_directory() + '/checkpoint/' + load_checkpoint_name
        encoder.load_weights(full_path + '/encoder_data.ckpt')
        decoder.load_weights(full_path + '/decoder_data.ckpt')
        test_loss.reset_states()
    else:
        raise Exception("E: load_checkpoint_name is empty")


# test function
@tf.function
def test_step(dist_inputs):
    result_list = []
    def step_fn(inputs):
        index, x, y = inputs
        x = tf.reshape(x, [x.shape[0], x.shape[1], 1])
        y = tf.reshape(y, [y.shape[0], y.shape[1], 1])

        latent = encoder(x)
        y_pred = decoder(latent)

        mae = loss_object(y, y_pred)
        if y_pred.shape[0] != 0:
            batch_split_list = tf.split(y_pred, num_or_size_splits=y_pred.shape[0], axis=0)
            for i in range(len(batch_split_list)):
                result_list.append([index[i], tf.squeeze(batch_split_list[i])])
        return mae

    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=None)
    test_loss(mean_loss / batch_size)
    return result_list


# test_target_path is path or file?
test_source_path_isdir = os.path.isdir(test_source_path)
test_target_path_isdir = os.path.isdir(test_target_path)
if test_target_path_isdir != test_source_path_isdir:
    raise Exception("E: Target and source path is incorrect")
if test_target_path_isdir:
    if not cf.compare_path_list(test_target_path, test_source_path, 'wav'):
        raise Exception("E: Target and source file list is not same")
    test_source_file_list = cf.read_path_list(test_source_path, "wav")
    test_target_file_list = cf.read_path_list(test_target_path, "wav")
else:
    test_source_file_list = [test_source_path]
    test_target_file_list = [test_target_path]


# test run
with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    for l in range(len(test_source_file_list)):
        # make dataset
        test_source_cut_list, test_target_cut_list, _, test_number_of_total_frame, front_padding, rear_padding, padded_length, sample_rate_check = md.make_dataset_for_test(test_source_file_list[l], test_target_file_list[l], frame_size, shift_size, window_type, cut_off_freq, filter_order, sampling_rate)
        test_dataset = tf.data.Dataset.from_tensor_slices((list(range(test_number_of_total_frame)), test_source_cut_list, test_target_cut_list)).batch(batch_size).with_options(options)
        dist_dataset_test = strategy.experimental_distribute_dataset(dataset=test_dataset)

        if test_source_path_isdir:
            file_name = test_source_file_list[l].replace(test_source_path, '').lstrip('\\/')
        else:
            file_name = os.path.basename(test_source_path)


        output_sort = []
        output_list = np.zeros(padded_length)
        i = 0
        start = time.time()
        for inputs in dist_dataset_test:
            print("\rTest({}) : Iter {}/{}".format(file_name, i + 1, math.ceil(test_number_of_total_frame / batch_size)), end='')
            result_package = test_step(inputs)
            for index, value in result_package:
                output_sort.append([int(index.numpy()), value.numpy()])
            i += 1

        output_sort.sort()
        for index, value in output_sort:
            output_list[shift_size*index:shift_size*index+len(value)] += value

        # save wav file
        full_path = cf.load_directory() + '/test_result/' + load_checkpoint_name + "/" + file_name
        cf.createFolder(os.path.dirname(full_path))
        wav.write_wav(output_list[front_padding:len(output_list)-rear_padding], full_path, sample_rate_check)

        print(" | Loss : " + str(float(test_loss.result())) + "Processing time :", datetime.timedelta(seconds=time.time() - start))

        test_loss.reset_states()