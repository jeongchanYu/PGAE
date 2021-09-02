import tensorflow as tf
import json
import os
import custom_function as cf
import model
import time
import datetime
import math
import make_dataset as md


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

pg_step = config["pg_step"]

batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]
default_float = config["default_float"]

train_source_path = config["train_source_path"]
train_target_path = config["train_target_path"]

valid_source_path = config["valid_source_path"]
valid_target_path = config["valid_target_path"]

load_checkpoint_name = config["load_checkpoint_name"]
save_checkpoint_name = config["save_checkpoint_name"]
save_checkpoint_period = config["save_checkpoint_period"]
validation_test = config["validation_test"]
plot_name = config["plot_name"]


# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # make dataset
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_source_cut_list, train_target_cut_list, train_number_of_total_frame = md.make_dataset(train_source_path, train_target_path, frame_size, shift_size, window_type, sampling_rate)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_source_cut_list, train_target_cut_list)).shuffle(train_number_of_total_frame).batch(batch_size).with_options(options)
    dist_dataset_train = strategy.experimental_distribute_dataset(dataset=train_dataset)
    if validation_test:
        valid_source_cut_list, valid_target_cut_list, valid_number_of_total_frame = md.make_dataset(valid_source_path, valid_target_path, frame_size, shift_size, window_type, sampling_rate)
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_source_cut_list, valid_target_cut_list)).shuffle(valid_number_of_total_frame).batch(batch_size).with_options(options)
        dist_dataset_valid = strategy.experimental_distribute_dataset(dataset=valid_dataset)

    # make model
    latent_size = pow(2, pg_step)
    encoder = model.Encoder(frame_size, latent_size)
    decoder = model.Decoder(latent_size, channel_size=256)
    def loss_object(y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)), axis=1)
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_sum(loss)
        return loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    if validation_test:
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')


# train function
@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        x, y = inputs
        x = tf.reshape(x, [x.shape[0], x.shape[1]])
        y = tf.reshape(y, [y.shape[0], y.shape[1], 1])
        y = y[:, ::int(pow(2,9-pg_step))]
        with tf.GradientTape(persistent=True) as tape:
            latent = encoder(x)
            y_pred = decoder(latent, pg_step)

            mae = loss_object(y, y_pred)
            loss = mae * (1.0 / batch_size)

        encoder_gradients = tape.gradient(loss, encoder.trainable_variables)
        decoder_gradients = tape.gradient(loss, decoder.trainable_variables)
        optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
        optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))

        return loss

    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=None)
    train_loss(mean_loss)


# test function
@tf.function
def valid_step(dist_inputs):
    def step_fn(inputs):
        x, y = inputs
        x = tf.reshape(x, [x.shape[0], x.shape[1]])
        y = tf.reshape(y, [y.shape[0], y.shape[1], 1])
        y = y[:, ::int(pow(2,9-pg_step))]

        latent = encoder(x)
        y_pred = decoder(latent, pg_step)

        mae = loss_object(y, y_pred)
        loss = mae * (1.0 / batch_size)
        return loss

    if tf_version[1] > 2:
        per_example_losses = strategy.run(step_fn, args=(dist_inputs,))
    else:
        per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=None)
    valid_loss(mean_loss)


# train run
with strategy.scope():
    # load model
    if load_checkpoint_name != "":
        saved_epoch = int(load_checkpoint_name.split('_')[-1])
        if math.isnan(saved_epoch):
            saved_epoch = 0
        for inputs in dist_dataset_train:
            train_step(inputs)
            break
        full_path = cf.load_directory() + '/checkpoint/' + load_checkpoint_name
        encoder.load_weights(full_path + '/encoder_data.ckpt')
        decoder.load_weights(full_path + '/decoder_data.ckpt')
        cf.load_optimizer_state(optimizer, full_path + '/optimizer')

        train_loss.reset_states()
        if validation_test:
            valid_loss.reset_states()
    else:
        full_path = cf.load_directory() + '/plot/'
        cf.createFolder(full_path)
        cf.clear_plot_file(full_path + plot_name + '.plot')
        cf.clear_csv_file(full_path + plot_name + '.csv')
        if validation_test:
            cf.clear_plot_file(full_path + plot_name + '_valid.plot')
            cf.clear_csv_file(full_path + plot_name + '_valid.csv')
        saved_epoch = 0


    for epoch in range(saved_epoch, saved_epoch+epochs):
        i = 0
        start = time.time()
        for inputs in dist_dataset_train:
            print("\rTrain : epoch {}/{}, iter {}/{}".format(epoch + 1, saved_epoch+epochs, i + 1, math.ceil(train_number_of_total_frame / batch_size)), end='')
            train_step(inputs)
            i += 1

        loss_sum = str(float(train_loss.result())) + " | "
        print(" | loss : " + loss_sum + "Processing time :", datetime.timedelta(seconds=time.time() - start))

        if ((epoch + 1) % save_checkpoint_period == 0) or (epoch + 1 == 1):
            full_path = cf.load_directory() + '/checkpoint/' + save_checkpoint_name + '_' + str(epoch+1)
            cf.createFolder(full_path)
            encoder.save_weights(full_path + '/encoder_data.ckpt')
            decoder.save_weights(full_path + '/decoder_data.ckpt')
            cf.save_optimizer_state(optimizer, full_path + '/optimizer')

        if validation_test:
            i = 0
            start = time.time()
            for inputs in dist_dataset_valid:
                print("\rValid : epoch {}/{}, iter {}/{}".format(epoch + 1, saved_epoch + epochs, i + 1, math.ceil(valid_number_of_total_frame / batch_size)), end='')
                valid_step(inputs)
                i += 1

            loss_sum = str(float(valid_loss.result())) + " | "
            print(" | loss : " + loss_sum + "Processing time :", datetime.timedelta(seconds=time.time() - start))


        # write plot file
        full_path = cf.load_directory() + '/plot/'
        cf.createFolder(full_path)
        cf.write_plot_file(full_path + plot_name + '.plot', epoch+1, train_loss.result())
        cf.write_csv_file(full_path + plot_name + '.csv', epoch+1, train_loss.result())
        train_loss.reset_states()
        if validation_test:
            cf.write_plot_file(full_path + plot_name + '_valid.plot', epoch + 1, valid_loss.result())
            cf.write_csv_file(full_path + plot_name + '_valid.csv', epoch + 1, valid_loss.result())
            valid_loss.reset_states()