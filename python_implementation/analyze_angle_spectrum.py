"""
分析角度谱 - 查看目标分布
"""

import numpy as np
import matplotlib.pyplot as plt
from radar_data_loader import RadarDataLoader
from signal_processing import SignalProcessor

# 数据集的实际参数（从MATLAB代码）
Tc = 200 / 4e6  # 57e-6 秒
freq_slope = 60.006e12  # Hz/s
bandwidth = Tc * freq_slope  # 3.42 GHz
range_resolution = 3e8 / (2 * bandwidth)  # 0.0439 m

print("="*80)
print("角度谱分析 - Position1_(1).bin")
print("="*80)
print()
print("雷达参数（数据集实际值）:")
print(f"  带宽: {bandwidth/1e9:.3f} GHz")
print(f"  距离分辨率: {range_resolution*100:.2f} cm")
print()

# 加载数据 - 使用8个RX通道（与MATLAB一致）
radar_file = "../FMCW radar-based multi-person vital sign monitoring data/1_AsymmetricalPosition/1_Radar_Raw_Data/position_ (1)/adc_3GHZ_position1_ (1).bin"

loader = RadarDataLoader(num_rx=8, num_tx=1, num_adc_samples=200,
                        num_chirps_per_frame=1, num_frames=1200)
params = loader.get_radar_params(bandwidth_ghz=bandwidth/1e9)
radar_cube = loader.load_bin_file(radar_file)

print(f"数据形状: {radar_cube.shape}")
print()

# 处理数据
processor = SignalProcessor(params)

# 使用256点FFT（与MATLAB一致）
print("执行256点Range FFT...")
range_fft_256 = np.fft.fft(radar_cube, n=256, axis=3)

# 取平均帧
num_frames_avg = 100
range_fft_avg = np.mean(np.abs(range_fft_256[:num_frames_avg]), axis=0)

print(f"Range FFT形状: {range_fft_avg.shape}")
print()

# 计算距离轴（从1开始，与MATLAB一致）
range_bins_matlab = np.arange(1, 257) * range_resolution

# 找到最强的range bin
range_profile = np.mean(np.abs(range_fft_avg[0, :, :]), axis=0)
target_range_bin = np.argmax(range_profile[5:100]) + 5  # 忽略前5个bin

print(f"最强目标在range bin: {target_range_bin}")
print(f"对应距离: {range_bins_matlab[target_range_bin]:.2f} m")
print()

# 波束成形 - MVDR（与MATLAB完全一致）
print("执行波束成形（MVDR）...")

# 参数（与MATLAB一致）
num_rx = 8  # 使用8个RX通道
d = 0.025  # 天线间距
wavelength = 0.005  # 5mm（60GHz对应）
search_angle_range = 60  # ±60度
num_angle_bins = 121  # -60到+60，每度一个

# 角度轴
angle_axis = np.linspace(-search_angle_range, search_angle_range, num_angle_bins)

# 计算所有range bin的角度谱（与MATLAB一致）
print(f"计算Range-Angle谱...")
azimuSpectrogram = np.zeros((num_angle_bins, 256))

for r in range(256):
    # 提取该range bin的数据 [frames, rx]
    xt = range_fft_256[:num_frames_avg, 0, :, r].T  # [rx, frames]

    # 计算协方差矩阵（与MATLAB一致）
    Rx = xt @ xt.conj().T  # [8x8]

    # 伪逆（与MATLAB的pinv一致）
    Rxv = np.linalg.pinv(Rx)

    # 对每个角度计算MVDR功率
    for an, angle in enumerate(angle_axis):
        # 导向矢量（8个元素）
        phi = 2 * np.pi * np.sin(np.deg2rad(angle)) * d / wavelength
        aTheta = np.array([np.exp(-1j * k * phi) for k in range(8)])

        # MVDR功率
        azimuSpectrogram[an, r] = 1.0 / np.abs(aTheta.conj() @ Rxv @ aTheta)

# 归一化
azimuSpectrogram = azimuSpectrogram / np.max(azimuSpectrogram)

# 翻转（与MATLAB一致）
azimuSpectrogram = np.flipud(azimuSpectrogram)

# 计算角度维度的和（用于峰值检测）- 与MATLAB一致
# MATLAB使用 sum(azimuSpectrogram,2) 对所有range bin求和
maxAzimu = np.sum(azimuSpectrogram, axis=1)

# 峰值检测（与MATLAB一致）
from scipy.signal import find_peaks
threshold = np.mean(azimuSpectrogram)
# 增加distance参数避免检测到太多相邻峰值
# 增加prominence参数确保只检测显著的峰值
# 使用更严格的参数：distance=10（至少间隔10度），prominence更高
peaks, properties = find_peaks(maxAzimu, height=threshold, distance=10, prominence=threshold*2.0)

# 如果检测到的峰值太多，只保留最强的前几个
if len(peaks) > 5:
    # 按峰值高度排序，保留最强的5个
    peak_heights = maxAzimu[peaks]
    top_indices = np.argsort(peak_heights)[-5:]
    peaks = peaks[top_indices]
    peaks = np.sort(peaks)  # 按角度重新排序

# 反转索引（因为之前flipud了）
angle_spectrum = maxAzimu

# 归一化并转换为dB
angle_spectrum_db = 10 * np.log10(angle_spectrum / np.max(angle_spectrum))

# 找峰值 - 使用更合理的阈值
# 不需要重复find_peaks，使用上面已经找到的peaks即可

print(f"检测到 {len(peaks)} 个峰值:")
for i, peak in enumerate(peaks):
    print(f"  峰值{i+1}: 角度={angle_axis[peak]:+.1f}°, 功率={angle_spectrum_db[peak]:.1f} dB")
print()

# 绘图
plt.figure(figsize=(14, 10))

# 1. Range Profile
plt.subplot(3, 1, 1)
plt.plot(range_bins_matlab[:100], 20*np.log10(range_profile[:100]))
plt.axvline(range_bins_matlab[target_range_bin], color='r', linestyle='--',
            label=f'Selected bin={target_range_bin}, R={range_bins_matlab[target_range_bin]:.2f}m')
plt.xlabel('Distance (m)')
plt.ylabel('Power (dB)')
plt.title('Range Profile')
plt.grid(True)
plt.legend()

# 2. Angle Spectrum
plt.subplot(3, 1, 2)
plt.plot(angle_axis, angle_spectrum_db, 'b-', linewidth=2)
plt.plot(angle_axis[peaks], angle_spectrum_db[peaks], 'r^', markersize=10, label='Detected Peaks')
plt.xlabel('Angle (degrees)')
plt.ylabel('Power (dB)')
plt.title(f'MVDR Angle Spectrum at Range={range_bins_matlab[target_range_bin]:.2f}m')
plt.grid(True)
plt.legend()
plt.xlim([-60, 60])

# 3. Range-Angle Heatmap
plt.subplot(3, 1, 3)

# 计算多个range bin的角度谱
range_bins_to_plot = range(10, 80, 2)  # 每隔2个bin
angle_heatmap = np.zeros((len(range_bins_to_plot), num_angle_bins))

for idx, rb in enumerate(range_bins_to_plot):
    target_data_rb = range_fft_256[:num_frames_avg, 0, :, rb]
    Rxx_rb = (target_data_rb.T @ target_data_rb.conj()) / num_frames_avg
    # 使用pinv代替inv，与前面的MVDR实现一致
    Rxx_inv_rb = np.linalg.pinv(Rxx_rb)

    for i, angle in enumerate(angle_axis):
        phi = 2 * np.pi * np.sin(np.deg2rad(angle)) * d / wavelength
        a = np.exp(-1j * np.arange(num_rx) * phi)
        angle_heatmap[idx, i] = 1.0 / np.abs(a.conj() @ Rxx_inv_rb @ a)

# 绘制热图
extent = [-search_angle_range, search_angle_range,
          range_bins_matlab[range_bins_to_plot[0]],
          range_bins_matlab[range_bins_to_plot[-1]]]
plt.imshow(10*np.log10(angle_heatmap / np.max(angle_heatmap)),
           aspect='auto', origin='lower', extent=extent, cmap='jet')
plt.colorbar(label='Power (dB)')
plt.xlabel('Angle (degrees)')
plt.ylabel('Distance (m)')
plt.title('Range-Angle Heatmap (MVDR)')

plt.tight_layout()
plt.savefig('angle_spectrum_analysis.png', dpi=150, bbox_inches='tight')
print("图像已保存: angle_spectrum_analysis.png")
plt.show()

print()
print("="*80)
print("分析完成")
print("="*80)
