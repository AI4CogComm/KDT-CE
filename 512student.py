import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, Dropout, Subtract, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np
import scipy.io as sio
import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

# 设置GPU配置以允许增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def TransformerBlock(inputs, num_heads, ff_dim, rate=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(inputs, inputs)
    attn_output = tf.keras.layers.Dropout(rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")(out1 + ffn_output)

    return out2


def TransformerDecoderBlock(inputs, encoder_output, num_heads, ff_dim, rate=0.1):
    # 解码器第一层多头注意力
    # attn1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(inputs, inputs)
    attn1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(encoder_output,encoder_output)
    attn1 = tf.keras.layers.Dropout(rate)(attn1)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoder_output + attn1)

    # 解码器第二层多头注意力，与编码器输出交互
    attn2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(out1, encoder_output)
    attn2 = tf.keras.layers.Dropout(rate)(attn2)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + attn2)

    # 前馈网络
    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(out2)
    ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    out3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_layernorm_1")(out2 + ffn_output)

    return out3

def adaptation_layer(student_feat, teacher_feat):
    s_shape = tf.shape(student_feat)[-1]
    t_shape = tf.shape(teacher_feat)[-1]
    if s_shape != t_shape:
        return Conv2D(t_shape, kernel_size=1, padding='same')(student_feat)
    else:
        return student_feat

Nx, Ny = 32, 16

# 构建学生模型（示例）
def build_student_model(input_shape,K= 3):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(K, K), padding='Same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(K, K), padding='Same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(K, K), padding='Same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(x)
    x = BatchNormalization(name="bn_3")(x)
    x_flat = Flatten()(x)
    x_reshape = tf.reshape(x_flat, (-1, Nx * Ny, 64))

     # 添加 Transformer 编码器层
    encoder_output = TransformerBlock(x_reshape, num_heads=4, ff_dim=128)
    # encoder_output = TransformerBlock(encoder_output, num_heads=4, ff_dim=128)
    # 添加 Transformer 解码器层
    decoder_output = TransformerDecoderBlock(x_reshape, encoder_output, num_heads=4, ff_dim=128)
    # decoder_output = TransformerDecoderBlock(decoder_output, encoder_output, num_heads=4, ff_dim=128)
    x_final = tf.reshape(decoder_output, (-1, Nx, Ny, 64))
    # 后续卷积层
    x_final = Conv2D(filters=2, kernel_size=(K, K), padding='Same', activation='linear')(x_final)
    output = Subtract()([inputs, x_final])
    # model = tf.keras.Model(inputs=inputs, outputs=[output, conv_feat, decoder_output])
    return Model(inputs=inputs, outputs=output)

    # return model

student_model = build_student_model((Nx, Ny, 2))


def compute_hint_loss(student_feats, teacher_feats):
    total_hint_loss = 0
    for s_feat, t_feat in zip(student_feats, teacher_feats):
        if s_feat.shape[-1] != t_feat.shape[-1]:
            s_feat = tf.keras.layers.Conv2D(t_feat.shape[-1], 1, padding='same')(s_feat)

            # L1 Hint Loss
            total_hint_loss += tf.reduce_mean(tf.abs(s_feat - t_feat))
            # L2 Hint Loss（调试用）
            # total_hint_loss += tf.reduce_mean(tf.square(s_feat - t_feat))
    return total_hint_loss

def load_teacher_predictions(predictions_path):
    return np.load(predictions_path)

def build_student_with_intermediates(input_shape, student_model, student_layer_names):
    outputs = [student_model.get_layer(name).output for name in student_layer_names]
    return Model(inputs=student_model.input, outputs=[student_model.output] + outputs)

def build_teacher_with_intermediates(teacher_model, teacher_layer_names):
    teacher_model.trainable = False
    outputs = [teacher_model.get_layer(name).output for name in teacher_layer_names]
    return Model(inputs=teacher_model.input, outputs=outputs)


def base_loss(y_true, y_pred):
    student_mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return student_mse

def hint_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def attention_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
def load_teacher_predictions(predictions_path):
    return np.load(predictions_path)

def load_data(mat_path, num_samples, Nx, Ny, snr_db):
    data = sio.loadmat(mat_path)
    channel = data['Channel_mat']
    P = 10 ** (snr_db / 10.0)

    H_in = np.zeros((num_samples, Nx, Ny, 2), dtype='float32')
    H_out = np.zeros((num_samples, Nx, Ny, 2), dtype='float32')

    for i in range(num_samples):
        h = channel[i]
        H = np.reshape(h, (Nx, Ny))
        noise = 1/np.sqrt(2) * (np.random.randn(Nx, Ny) + 1j * np.random.randn(Nx, Ny))
        H_noisy = H + 1 / np.sqrt(P) * noise

        H_in[i, ..., 0] = np.real(H_noisy)
        H_in[i, ..., 1] = np.imag(H_noisy)
        H_out[i, ..., 0] = np.real(H)
        H_out[i, ..., 1] = np.imag(H)
    return H_in, H_out

def create_dataset(x, y, teacher_y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y, teacher_y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def cache_teacher_output(teacher_model, x_data, batch_size):
    return teacher_model.predict(x_data, batch_size=batch_size)

def train(student_model, train_ds, val_ds, epochs, optimizer, save_path,
          teacher_model, teacher_layer_names, student_layer_names,  tor_weight,atten_weight ):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    print("Student model layers:")
    for layer in student_model.layers:
        print(layer.name)
    student_with_hint = build_student_with_intermediates((Nx, Ny, 2), student_model, student_layer_names)
    teacher_with_hint = build_teacher_with_intermediates(teacher_model, teacher_layer_names)

    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = []

        for batch_x, batch_y, batch_teacher in train_ds:
            with tf.GradientTape() as tape:
                # student_out_and_feats = student_model(batch_x, training=True)
                student_out_and_feats = student_with_hint(batch_x, training=True)
                student_pred = student_out_and_feats[0]
                # student_conv_feats = student_out_and_feats[1]
                # student_atten_feats = student_out_and_feats[2]

                student_atten_feats = student_out_and_feats[1]
                # student_atten_feats_1 = student_out_and_feats[2]

                teacher_feats = teacher_with_hint(batch_x, training=False)
                # teacher_conv_feats = teacher_feats[0]
                # teacher_atten_feats = teacher_feats[1]
                teacher_atten_feats = teacher_feats[0]
                # teacher_atten_feats_1 = teacher_feats[1]

                # loss1 = base_loss(batch_y, student_pred)
                loss2 = base_loss(student_pred, batch_teacher)
                # loss3 = hint_loss(student_conv_feats, teacher_conv_feats)
                loss4 = attention_loss(student_atten_feats, teacher_atten_feats)
                # loss5 = attention_loss(student_atten_feats_1, teacher_atten_feats_1)
                # loss5 = attention_loss(student_atten_feats_1, teacher_atten_feats_1)
                # total_loss = base_weight * loss1 + tor_weight * loss2 + hint_weight * loss3 + atten_weight * loss4
                # total_loss = base_weight * loss1 + tor_weight * loss2 + hint_weight * loss3
                # total_loss =  tor_weight * loss2
                total_loss = tor_weight * loss2 + atten_weight * loss4
            #
            grads = tape.gradient(total_loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
            epoch_loss.append(total_loss.numpy())

        val_epoch_loss = []
        for val_x, val_y, val_teacher in val_ds:
            # student_out_and_feats = student_model(val_x, training=False)
            student_out_and_feats = student_with_hint(val_x, training=False)
            student_pred = student_out_and_feats[0]
            # student_conv_feats = student_out_and_feats[1]
            # student_atten_feats = student_out_and_feats[2]
            student_atten_feats = student_out_and_feats[1]

            # student_atten_feats_1 = student_out_and_feats[2]

            teacher_feats = teacher_with_hint(val_x, training=False)
            # teacher_conv_feats = teacher_feats[0]
            # teacher_atten_feats = teacher_feats[1]
            teacher_atten_feats = teacher_feats[0]
            # teacher_atten_feats_1 = teacher_feats[1]

            # loss1 = base_loss(val_y, student_pred)
            loss2 = base_loss(student_pred, val_teacher)
            # loss3 = hint_loss(student_conv_feats, teacher_conv_feats)
            loss4 = attention_loss(student_atten_feats, teacher_atten_feats)
            # loss5 = attention_loss(student_atten_feats_1, teacher_atten_feats_1)


            # loss5 = attention_loss(student_atten_feats_1, teacher_atten_feats_1)
            # total_loss = base_weight * loss1 + tor_weight * loss2 + hint_weight * loss3 + atten_weight * loss4
            # total_loss = base_weight * loss1 + tor_weight * loss2 + hint_weight * loss3
            # total_loss =  tor_weight * loss2
            total_loss = tor_weight * loss2 + atten_weight * loss4

            val_epoch_loss.append(total_loss.numpy())

        avg_train = np.mean(epoch_loss)
        avg_val = np.mean(val_epoch_loss)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            student_model.save(save_path)
            print(f">>> Saved best model at epoch {epoch+1} with val_loss: {avg_val:.5f}")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.5f}, Val Loss: {avg_val:.5f} | Time: {time.time() - t0:.2f}s")

    return train_losses, val_losses

def predict_and_evaluate(student_model, teacher_model, H_in_test, H_out_test, batch_size=128):
    teacher_preds = teacher_model.predict(H_in_test, batch_size=batch_size)
    student_preds = student_model.predict(H_in_test, batch_size=batch_size)

    def to_complex(x):
        return x[..., 0] + 1j * x[..., 1]

    y_true = to_complex(H_out_test)
    teacher_complex = to_complex(teacher_preds)
    student_complex = to_complex(student_preds)

    def nmse(y_true, y_pred):
        power = np.sum(np.abs(y_true) ** 2, axis=(1, 2))
        error = np.sum(np.abs(y_true - y_pred) ** 2, axis=(1, 2))
        return np.mean(error / power)

    student_nmse = nmse(y_true, student_complex)
    teacher_nmse = nmse(y_true, teacher_complex)

    print(f"\n🔍 Teacher NMSE: {teacher_nmse:.6f}")
    print(f"🎯 Student NMSE: {student_nmse:.6f}")

    return student_preds, teacher_preds, student_nmse, teacher_nmse

if __name__ == "__main__":
    Nx, Ny = 32, 16
    batch_size = 128
    epochs = 200
    val_split = 0.1
    snr = 15

    teacher_model = tf.keras.models.load_model("newMATCENet_M512_15dB.h5")
    teacher_model.trainable = False

    H_in, H_out = load_data('f1n5_512ANTS_1000by100.mat', 100000, Nx, Ny, snr)
    val_size = int(val_split * len(H_in))
    train_x, val_x = H_in[:-val_size], H_in[-val_size:]
    train_y, val_y = H_out[:-val_size], H_out[-val_size:]
    # 假设你读取的是一个 .npz 文件
    teacher_y = load_teacher_predictions("new512_teacher_pred_result_15dB_200ep.npy")

    teacher_train_y, teacher_val_y = teacher_y[:-val_size], teacher_y[-val_size:],
    # val_size = 10000
    teacher_train_y, teacher_val_y = teacher_y[:-val_size], teacher_y[-val_size:]

    train_ds = create_dataset(train_x, train_y, teacher_train_y, batch_size)
    val_ds = create_dataset(val_x, val_y, teacher_val_y, batch_size)

    student_model = build_student_model((Nx, Ny, 2))
    # optimizer = tf.keras.optimizers.Adam(1e-3)
    optimizer = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_save_path = "new512student_15dB.hdf5"

    # student_layers = ['bn_3','final_layernorm_1']
    # teacher_layers = ['batch_normalization_3', 'layer_normalization_5']
    # student_layers = ['bn_3']
    # teacher_layers = ['batch_normalization_3']
    student_layers = ['final_layernorm_1']
    # teacher_layers = ['layer_normalization_5']
    teacher_layers = ['layer_normalization_28']
    # student_layers = ['final_layernorm','final_layernorm_1']
    # teacher_layers = ['layer_normalization_2','layer_normalization_5']
    train_losses, val_losses = train(student_model, train_ds, val_ds, epochs, optimizer,  save_path=model_save_path,
                                     teacher_model=teacher_model,
                                     teacher_layer_names=teacher_layers,
                                     student_layer_names=student_layers,
                                     tor_weight=0.5,
                                     atten_weight=0.5)

    H_in_test, H_out_test = load_data('f1n5_512ANTS_10by200.mat', 2000, Nx, Ny, snr)

    best_student = tf.keras.models.load_model(model_save_path, compile=False)

    student_preds, teacher_preds, student_nmse, teacher_nmse = predict_and_evaluate(
        best_student, teacher_model, H_in_test, H_out_test, batch_size=batch_size
    )

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
