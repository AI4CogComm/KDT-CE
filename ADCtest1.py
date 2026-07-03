from numpy import *
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio
import time

# -------------------- 参数 --------------------
# rho1 = 0.3634  #1
rho1 = 0.1175   #2
# rho1 = 0.03454   #3
# rho1 = 0.009497 #量化噪声比例4
# rho1 =  0.00004151
alpha1 = 1 - rho1
N = 256  # BS 天线数
snr_db = 10
P = 10 ** (snr_db / 10.0)
sigma2_R = 1 / P
Nx = 16
Ny = 16
data_num_test = 2000

# -------------------- 加载信道 --------------------
H_noisy_in_test = np.zeros((data_num_test, Nx, Ny, 2), dtype=float)
H_true_out_test = np.zeros((data_num_test, Nx, Ny, 2), dtype=float)
data1 = sio.loadmat('f1n5_256ANTS_10by200.mat')
channel = data1['Channel_mat']

for i in range(data_num_test):
    h = channel[i]
    H = np.reshape(h, (Nx, Ny))
    H_true_out_test[i, :, :, 0] = np.real(H)
    H_true_out_test[i, :, :, 1] = np.imag(H)

    # 原始噪声 n_R
    noise = (np.random.randn(Nx, Ny) + 1j * np.random.randn(Nx, Ny)) / np.sqrt(2)
    H_noisy = H + np.sqrt(sigma2_R) * noise
    H_noisy_in_test[i, :, :, 0] = np.real(H_noisy)
    H_noisy_in_test[i, :, :, 1] = np.imag(H_noisy)

# -------------------- ADC 量化 --------------------
H_noisy_adc_test = np.zeros_like(H_noisy_in_test, dtype=np.float32)

for i in range(data_num_test):
    F = H_true_out_test[i, :, :, 0] + 1j * H_true_out_test[i, :, :, 1]  # Nx x Ny
    yR = H_noisy_in_test[i, :, :, 0] + 1j * H_noisy_in_test[i, :, :, 1]  # Nx x Ny

    # -------------------- 修正量化噪声 --------------------
    # element-wise 方差，保证每个接收天线每个元素独立量化
    var_nq1_matrix = alpha1 * rho1 * (np.abs(F) ** 2 + sigma2_R)  # Nx x Ny

    # 生成复高斯量化噪声
    nq1 = (np.random.randn(Nx, Ny) + 1j * np.random.randn(Nx, Ny)) / np.sqrt(2)
    nq1 = nq1 * np.sqrt(var_nq1_matrix)

    # 线性 AQNM
    yR_q = alpha1 * yR + nq1

    H_noisy_adc_test[i, :, :, 0] = np.real(yR_q)
    H_noisy_adc_test[i, :, :, 1] = np.imag(yR_q)

print("修正后的 ADC 量化完成，H_noisy_adc_test shape:", H_noisy_adc_test.shape)
ResCNN2d = tf.keras.models.load_model(r"D:\LISHUANGSHUANG\ssl_kd_mat\distillation_T200\useful\kd5newdecoder_10dB_16_64_e3lr.hdf5")
ResCNN2d.summary()
start_time = time.time()
decoded_channel = ResCNN2d.predict(H_noisy_adc_test)
end_time = time.time()
# 总耗时（秒）
total_time = end_time - start_time
# 每个样本的平均预测时间（毫秒）
avg_time_per_sample = total_time / data_num_test * 1000
print(((H_noisy_in_test)**2).mean(),((H_true_out_test)**2).mean(),((decoded_channel)**2).mean())
nmse1=zeros((data_num_test,1), dtype=float)
nmse2=zeros((data_num_test,1), dtype=float)
for n in range(data_num_test):
    MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
    MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
    norm_real=((H_true_out_test[n,:,:,:])**2).sum()
    nmse1[n] = MSE1 / norm_real
    nmse2[n]=MSE2/norm_real
print(nmse1.sum()/(data_num_test), nmse2.sum()/(data_num_test))

nmse1=zeros((data_num_test,1), dtype=float)
nmse2=zeros((data_num_test,1), dtype=float)
for n in range(data_num_test):
    MSE1 = (np.linalg.norm(H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:])) ** 2
    MSE2 = (np.linalg.norm(H_true_out_test[n,:,:,:] - decoded_channel[n,:,:,:])) ** 2
    norm_real = (np.linalg.norm(H_true_out_test[n,:,:,:])) ** 2
    nmse1[n] = MSE1 / norm_real
    nmse2[n] = MSE2/norm_real
print(nmse1.sum()/(data_num_test), nmse2.sum()/(data_num_test))

print(f"总测试时间：{total_time:.4f} 秒")
print(f"平均每个样本预测时间：{avg_time_per_sample:.4f} 毫秒")
