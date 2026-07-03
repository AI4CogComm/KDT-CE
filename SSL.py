import time
from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, Subtract, Flatten
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import tensorflow as tf
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置GPU配置以允许增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

N = 256  # BS antennas
snr = -10
P = 10 ** (snr / 10.0)
Nx = 16
Ny = 16
sigma_extra = 0.7
############## training set ##################
data_num_train = 100000
H_noisy_in_1 = np.zeros((data_num_train, Nx, Ny, 2), dtype=float)
H_noisy_in_2 = np.zeros((data_num_train, Nx, Ny, 2), dtype=float)
# H_noisy_2 = np.zeros((data_num_train, Nx, Ny, 2), dtype=float)
H_true_out = np.zeros((data_num_train, Nx, Ny, 2), dtype=float)
data1 = sio.loadmat('f1n5_256ANTS_1000by100')
channel = data1['Channel_mat']
sigma_real_imag = sigma_extra / np.sqrt(2)
for i in range(data_num_train):
    h = channel[i]
    H = np.reshape(h, (Nx, Ny))
    H_true_out[i, :, :, 0] = np.real(H)
    H_true_out[i, :, :, 1] = np.imag(H)
    noise = 1 / np.sqrt(2) * np.random.randn(Nx, Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx, Ny)
    H_noisy_1 = H + 1 / np.sqrt(P) * noise
    H_noisy_in_1[i, :, :, 0] = np.real(H_noisy_1)
    H_noisy_in_1[i, :, :, 1] = np.imag(H_noisy_1)
    oise_extra = sigma_real_imag * np.random.randn(Nx, Ny) + 1j * sigma_real_imag * np.random.randn(Nx, Ny)
    H_noisy_2 = H_noisy_1 + oise_extra

    # H_noisy_in[i, :, :, 0] = np.real(H_noisy)
    # H_noisy_in[i, :, :, 1] = np.imag(H_noisy)


    H_noisy_in_2[i, :, :, 0] = np.real(H_noisy_2)
    H_noisy_in_2[i, :, :, 1] = np.imag(H_noisy_2)

print(((H_noisy_in_2) ** 2).mean(),((H_noisy_in_1) ** 2).mean(), ((H_true_out) ** 2).mean())
print(H_noisy_in_2.shape,H_noisy_in_1.shape, H_true_out.shape)


def TransformerBlock(inputs, num_heads, ff_dim, rate=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(inputs, inputs)
    attn_output = tf.keras.layers.Dropout(rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return out2


def TransformerDecoderBlock(inputs, encoder_output, num_heads, ff_dim, rate=0.1):
    # 解码器第一层多头注意力
    attn1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(inputs, inputs)
    attn1 = tf.keras.layers.Dropout(rate)(attn1)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn1)

    # 解码器第二层多头注意力，与编码器输出交互
    attn2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(out1, encoder_output)
    attn2 = tf.keras.layers.Dropout(rate)(attn2)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + attn2)

    # 前馈网络
    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(out2)
    ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    out3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out2 + ffn_output)

    return out3


K = 3
input_dim = (Nx, Ny, 2)
output_dim = 2

inp = Input(shape=input_dim)
x = Conv2D(filters=32, kernel_size=(K, K), padding='Same', activation='relu')(inp)
x = BatchNormalization()(x)
x = Conv2D(filters=32, kernel_size=(K, K), padding='Same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=32, kernel_size=(K, K), padding='Same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(x)
x = BatchNormalization()(x)

# 展平和重新调整形状
x_flat = Flatten()(x)
x_reshape = tf.reshape(x_flat, (-1, Nx * Ny, 64))

# 添加 Transformer 编码器层
encoder_output = TransformerBlock(x_reshape, num_heads=4, ff_dim=128)
# encoder_output = TransformerBlock(encoder_output, num_heads=4, ff_dim=128)

# 添加 Transformer 解码器层
decoder_output = TransformerDecoderBlock(x_reshape, encoder_output, num_heads=4, ff_dim=128)
# decoder_output = TransformerDecoderBlock(decoder_output, encoder_output, num_heads=4, ff_dim=128)

# 重新调整形状
x_final = tf.reshape(decoder_output, (-1, Nx, Ny, 64))

# 后续卷积层
x_final = Conv2D(filters=output_dim, kernel_size=(K, K), padding='Same', activation='linear')(x_final)
x1 = Subtract()([inp, x_final])

model = Model(inputs=inp, outputs=x1)

# checkpoint
filepath = '-10dB+0.7-3_300ep_64b.hdf5'

adam = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='mse')
model.summary()

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
start_time = time.time()
model.fit(x=H_noisy_in_2, y=H_noisy_in_1, epochs=300, batch_size=64, callbacks=callbacks_list, verbose=2, shuffle=True, validation_split=0.1)

end_time = time.time()
training_time = end_time - start_time
print(f'Training time: {training_time} seconds')
############## testing set ##################
data_num_test = 2000
H_noisy_in_test = np.zeros((data_num_test, Nx, Ny, 2), dtype=float)
H_true_out_test = np.zeros((data_num_test, Nx, Ny, 2), dtype=float)
data1 = sio.loadmat('f1n5_256ANTS_10by200')
channel = data1['Channel_mat']

for i in range(data_num_test):
    h = channel[i]
    H = np.reshape(h, (Nx, Ny))
    H_true_out_test[i, :, :, 0] = np.real(H)
    H_true_out_test[i, :, :, 1] = np.imag(H)
    noise = 1 / np.sqrt(2) * np.random.randn(Nx, Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx, Ny)
    H_noisy = H + 1 / np.sqrt(P) * noise
    H_noisy_in_test[i, :, :, 0] = np.real(H_noisy)
    H_noisy_in_test[i, :, :, 1] = np.imag(H_noisy)

# load model
ResCNN2d = load_model(filepath)

decoded_channel = ResCNN2d.predict(H_noisy_in_test)
print(((H_noisy_in_test) ** 2).mean(), ((H_true_out_test) ** 2).mean(), ((decoded_channel) ** 2).mean())
nmse1 = np.zeros((data_num_test, 1), dtype=float)
nmse2 = np.zeros((data_num_test, 1), dtype=float)
for n in range(data_num_test):
    MSE1 = ((H_true_out_test[n, :, :, :] - H_noisy_in_test[n, :, :, :]) ** 2).sum()
    MSE2 = ((H_true_out_test[n, :, :, :] - decoded_channel[n, :, :, :]) ** 2).sum()
    norm_real = ((H_true_out_test[n, :, :, :]) ** 2).sum()
    nmse1[n] = MSE1 / norm_real
    nmse2[n] = MSE2 / norm_real
print(nmse1.sum() / data_num_test, nmse2.sum() / data_num_test)
