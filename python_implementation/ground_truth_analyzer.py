"""
Ground Truth分析工具
正确解析和分析ECG/PCG数据
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


class GroundTruthAnalyzer:
    """
    Ground Truth数据分析器
    """

    def __init__(self, sampling_rate=125):
        """
        Args:
            sampling_rate: 采样率 (Hz)，默认125Hz (8ms间隔)
        """
        self.sampling_rate = sampling_rate

    def read_and_analyze(self, csv_file, visualize=False):
        """
        读取并分析ground truth数据

        Args:
            csv_file: CSV文件路径
            visualize: 是否可视化

        Returns:
            results: 分析结果字典
        """
        # 读取CSV
        df = pd.read_csv(csv_file, skiprows=1)
        ecg = df.iloc[:, 0].values
        pcg = df.iloc[:, 1].values

        # 分析ECG（心率）
        heart_rate, heart_peaks = self._analyze_ecg(ecg)

        # 分析PCG（呼吸率）
        breathing_rate, breath_peaks = self._analyze_pcg(pcg)

        results = {
            'heart_rate_bpm': heart_rate,
            'breathing_rate_bpm': breathing_rate,
            'heart_peaks': heart_peaks,
            'breath_peaks': breath_peaks,
            'ecg_data': ecg,
            'pcg_data': pcg,
            'duration_sec': len(ecg) / self.sampling_rate
        }

        if visualize:
            self._visualize(results, csv_file)

        return results

    def _analyze_ecg(self, ecg):
        """
        分析ECG数据提取心率

        策略:
        1. 带通滤波 (0.5-40 Hz) 去除基线漂移和高频噪声
        2. 自适应阈值检测R波
        3. 计算心率
        """
        # 1. 带通滤波
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 40 / nyquist

        if high >= 1.0:
            high = 0.99

        b, a = butter(4, [low, high], btype='band')
        ecg_filtered = filtfilt(b, a, ecg)

        # 2. 自适应阈值检测
        # 使用滑动窗口计算局部阈值
        window_size = int(0.2 * self.sampling_rate)  # 200ms窗口
        ecg_abs = np.abs(ecg_filtered)
        local_max = uniform_filter1d(ecg_abs, size=window_size, mode='nearest')

        # 阈值设为局部最大值的60%
        threshold = 0.6 * local_max

        # 检测峰值
        # 最小间隔: 0.4秒 (对应150 BPM最大心率)
        min_distance = int(0.4 * self.sampling_rate)

        peaks, properties = find_peaks(
            ecg_filtered,
            height=threshold,
            distance=min_distance
        )

        # 3. 计算心率
        if len(peaks) > 1:
            duration = len(ecg) / self.sampling_rate
            heart_rate = (len(peaks) / duration) * 60
        else:
            heart_rate = 0.0

        return heart_rate, peaks

    def _analyze_pcg(self, pcg):
        """
        分析PCG数据提取呼吸率

        策略:
        1. 低通滤波 (0.05-2 Hz) 提取呼吸成分
        2. 平滑处理
        3. 峰值检测
        """
        # 1. 低通滤波
        nyquist = self.sampling_rate / 2
        low = 0.05 / nyquist
        high = 2.0 / nyquist

        if high >= 1.0:
            high = 0.99

        b, a = butter(4, [low, high], btype='band')
        pcg_filtered = filtfilt(b, a, pcg)

        # 2. 额外平滑
        pcg_smooth = uniform_filter1d(pcg_filtered, size=int(0.5 * self.sampling_rate))

        # 3. 峰值检测
        # 最小间隔: 2秒 (对应30 BPM最大呼吸率)
        min_distance = int(2.0 * self.sampling_rate)

        peaks, _ = find_peaks(
            pcg_smooth,
            distance=min_distance,
            prominence=np.std(pcg_smooth) * 0.5
        )

        # 4. 计算呼吸率
        if len(peaks) > 1:
            duration = len(pcg) / self.sampling_rate
            breathing_rate = (len(peaks) / duration) * 60
        else:
            breathing_rate = 0.0

        return breathing_rate, peaks

    def _visualize(self, results, title=""):
        """可视化分析结果"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        time_axis = np.arange(len(results['ecg_data'])) / self.sampling_rate

        # ECG
        axes[0].plot(time_axis, results['ecg_data'], 'b-', alpha=0.5, label='Raw ECG')
        if len(results['heart_peaks']) > 0:
            axes[0].plot(time_axis[results['heart_peaks']],
                        results['ecg_data'][results['heart_peaks']],
                        'r^', markersize=10, label=f'Heart Peaks ({len(results["heart_peaks"])})')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('ECG Amplitude')
        axes[0].set_title(f'ECG - Heart Rate: {results["heart_rate_bpm"]:.1f} BPM')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # PCG
        axes[1].plot(time_axis, results['pcg_data'], 'g-', alpha=0.5, label='Raw PCG')
        if len(results['breath_peaks']) > 0:
            axes[1].plot(time_axis[results['breath_peaks']],
                        results['pcg_data'][results['breath_peaks']],
                        'r^', markersize=10, label=f'Breath Peaks ({len(results["breath_peaks"])})')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('PCG Amplitude')
        axes[1].set_title(f'PCG - Breathing Rate: {results["breathing_rate_bpm"]:.1f} BPM')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        plt.show()


if __name__ == "__main__":
    # 测试
    analyzer = GroundTruthAnalyzer()

    log_file = "../FMCW radar-based multi-person vital sign monitoring data/1_AsymmetricalPosition/2_Log_data/Target1/position_ (1)/log_Target1_3GHZ_position1_ (1).csv"

    print("分析Ground Truth数据...")
    results = analyzer.read_and_analyze(log_file, visualize=True)

    print(f"\n结果:")
    print(f"  心率: {results['heart_rate_bpm']:.1f} BPM")
    print(f"  呼吸率: {results['breathing_rate_bpm']:.1f} BPM")
    print(f"  持续时间: {results['duration_sec']:.1f} 秒")
    print(f"  心跳峰值数: {len(results['heart_peaks'])}")
    print(f"  呼吸峰值数: {len(results['breath_peaks'])}")
