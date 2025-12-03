"""
通用多目标生命体征检测器
Universal Multi-Target Vital Signs Detector

核心改进:
1. 自适应波束成形（单/多chirp）
2. 鲁棒的目标检测（去除虚假目标）
3. 多目标分离和跟踪
4. 增强的生命体征提取
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings


@dataclass
class DetectorConfig:
    """检测器配置"""
    # 雷达参数
    num_rx: int = 4
    num_tx: int = 1
    num_adc_samples: int = 200
    num_chirps_per_frame: int = 1

    # 检测参数
    cfar_guard_len: int = 2
    cfar_noise_len: int = 8
    cfar_threshold_scale: float = 3.0
    min_target_distance_m: float = 0.5  # 最小目标间距
    max_targets: int = 4

    # 波束成形参数
    num_azimuth_bins: int = 64
    angle_range_deg: float = 60

    # 生命体征参数
    breath_freq_min: float = 0.15  # 9 BPM
    breath_freq_max: float = 0.5   # 30 BPM
    heart_freq_min: float = 0.8    # 48 BPM
    heart_freq_max: float = 4.0    # 240 BPM

    # 质量控制
    min_snr_db: float = 10.0
    min_confidence: float = 1.5


@dataclass
class Target:
    """目标信息"""
    id: int
    range_m: float
    range_bin: int
    azimuth_deg: float
    azimuth_bin: int
    power: float
    snr_db: float
    breathing_rate_bpm: float = 0.0
    heart_rate_bpm: float = 0.0
    confidence_breath: float = 0.0
    confidence_heart: float = 0.0
    is_valid: bool = True


class AdaptiveBeamformer:
    """
    自适应波束成形器
    根据chirp数量自动选择算法
    """

    def __init__(self, params, num_azimuth_bins=64, angle_range_deg=60):
        self.params = params
        self.num_rx = params['num_rx']
        self.wavelength = params['wavelength']
        self.num_azimuth_bins = num_azimuth_bins
        self.angle_range_deg = angle_range_deg

        # 生成导向矢量
        self.steering_vectors = self._generate_steering_vectors()

    def _generate_steering_vectors(self):
        """生成导向矢量"""
        angles_deg = np.linspace(-self.angle_range_deg, self.angle_range_deg,
                                self.num_azimuth_bins)
        angles_rad = np.deg2rad(angles_deg)

        d = self.wavelength / 2
        antenna_indices = np.arange(self.num_rx)

        steering_vectors = np.zeros((self.num_azimuth_bins, self.num_rx), dtype=complex)

        for idx, angle in enumerate(angles_rad):
            phase_shift = 2 * np.pi * d * np.sin(angle) / self.wavelength
            steering_vectors[idx, :] = np.exp(-1j * phase_shift * antenna_indices)

        return steering_vectors

    def process(self, range_data, range_idx, mode='auto'):
        """
        自适应波束成形

        Args:
            range_data: [num_chirps, num_rx, num_range_bins]
            range_idx: 要处理的距离bin
            mode: 'auto', 'conventional', 'capon'

        Returns:
            azimuth_spectrum: [num_azimuth_bins]
        """
        num_chirps = range_data.shape[0]

        # 自动选择模式
        if mode == 'auto':
            if num_chirps == 1:
                mode = 'conventional'
            else:
                mode = 'capon'

        if mode == 'conventional':
            return self._conventional_beamforming(range_data, range_idx)
        else:
            return self._capon_beamforming(range_data, range_idx)

    def _conventional_beamforming(self, range_data, range_idx):
        """常规波束成形（适用于单chirp）"""
        # 提取数据并对chirps求平均
        x = np.mean(range_data[:, :, range_idx], axis=0)  # [num_rx]

        azimuth_spectrum = np.zeros(self.num_azimuth_bins)

        for az_idx in range(self.num_azimuth_bins):
            a = self.steering_vectors[az_idx, :]
            azimuth_spectrum[az_idx] = np.abs(a.conj() @ x) ** 2

        return azimuth_spectrum

    def _capon_beamforming(self, range_data, range_idx, gamma=0.01):
        """Capon波束成形（适用于多chirp）"""
        x = range_data[:, :, range_idx]  # [num_chirps, num_rx]
        X = x.T  # [num_rx, num_chirps]
        num_chirps = X.shape[1]

        # 协方差矩阵
        R = (X @ X.conj().T) / num_chirps
        R = R + gamma * np.eye(self.num_rx)

        # 求逆
        try:
            R_inv = np.linalg.inv(R)
        except:
            R_inv = np.linalg.pinv(R)

        azimuth_spectrum = np.zeros(self.num_azimuth_bins)

        for az_idx in range(self.num_azimuth_bins):
            a = self.steering_vectors[az_idx, :]
            denominator = np.abs(a.conj() @ R_inv @ a)

            if denominator > 1e-10:
                azimuth_spectrum[az_idx] = 1.0 / denominator
            else:
                azimuth_spectrum[az_idx] = 0.0

        return azimuth_spectrum

    def get_azimuth_angles(self):
        """获取方位角数组"""
        return np.linspace(-self.angle_range_deg, self.angle_range_deg,
                          self.num_azimuth_bins)


class RobustTargetDetector:
    """
    鲁棒的目标检测器
    解决虚假目标问题
    """

    def __init__(self, config: DetectorConfig, params):
        self.config = config
        self.params = params
        self.range_resolution = params['range_resolution']

    def detect_targets(self, range_profile, range_bins) -> List[Target]:
        """
        检测目标

        Args:
            range_profile: 距离维功率谱
            range_bins: 距离bin数组

        Returns:
            targets: 检测到的目标列表
        """
        # 1. CFAR检测
        detected_bins = self._cfar_detection(range_profile)

        if len(detected_bins) == 0:
            return []

        # 2. 聚类相邻bins
        clusters = self._cluster_detections(detected_bins)

        # 3. 每个cluster选择最强的bin
        targets = []
        for cluster in clusters:
            powers = [range_profile[b] for b in cluster]
            max_idx = cluster[np.argmax(powers)]

            # 计算SNR
            noise_floor = np.median(range_profile)
            snr_db = 10 * np.log10(range_profile[max_idx] / noise_floor)

            # 质量检查
            if snr_db < self.config.min_snr_db:
                continue

            target = Target(
                id=len(targets),
                range_m=range_bins[max_idx],
                range_bin=max_idx,
                azimuth_deg=0.0,  # 稍后填充
                azimuth_bin=0,
                power=range_profile[max_idx],
                snr_db=snr_db
            )
            targets.append(target)

        # 4. 去除过近的目标（可能是虚假目标）
        targets = self._remove_close_targets(targets)

        # 5. 限制最大目标数
        if len(targets) > self.config.max_targets:
            # 按SNR排序，保留最强的
            targets.sort(key=lambda t: t.snr_db, reverse=True)
            targets = targets[:self.config.max_targets]

        # 重新分配ID
        for i, target in enumerate(targets):
            target.id = i

        return targets

    def _cfar_detection(self, range_profile):
        """CFAR检测"""
        guard_len = self.config.cfar_guard_len
        noise_len = self.config.cfar_noise_len
        threshold_scale = self.config.cfar_threshold_scale

        detected_bins = []

        for i in range(noise_len + guard_len, len(range_profile) - noise_len - guard_len):
            # 左侧噪声窗口
            left_start = i - guard_len - noise_len
            left_end = i - guard_len
            left_noise = range_profile[left_start:left_end]

            # 右侧噪声窗口
            right_start = i + guard_len + 1
            right_end = i + guard_len + noise_len + 1
            right_noise = range_profile[right_start:right_end]

            # 噪声估计
            noise_estimate = np.mean(np.concatenate([left_noise, right_noise]))
            threshold = noise_estimate * threshold_scale

            # 检测
            if range_profile[i] > threshold:
                detected_bins.append(i)

        return detected_bins

    def _cluster_detections(self, detected_bins, max_gap=3):
        """聚类相邻的检测"""
        if len(detected_bins) == 0:
            return []

        clusters = []
        current_cluster = [detected_bins[0]]

        for i in range(1, len(detected_bins)):
            if detected_bins[i] - detected_bins[i-1] <= max_gap:
                current_cluster.append(detected_bins[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [detected_bins[i]]

        clusters.append(current_cluster)
        return clusters

    def _remove_close_targets(self, targets):
        """去除过近的目标"""
        if len(targets) <= 1:
            return targets

        # 按距离排序
        targets.sort(key=lambda t: t.range_m)

        filtered = [targets[0]]

        for i in range(1, len(targets)):
            # 检查与已接受目标的距离
            too_close = False
            for accepted in filtered:
                if abs(targets[i].range_m - accepted.range_m) < self.config.min_target_distance_m:
                    too_close = True
                    break

            if not too_close:
                filtered.append(targets[i])

        return filtered


class UniversalVitalSignsDetector:
    """
    通用多目标生命体征检测器
    """

    def __init__(self, config: DetectorConfig, params):
        self.config = config
        self.params = params

        # 初始化子模块
        self.beamformer = AdaptiveBeamformer(
            params,
            config.num_azimuth_bins,
            config.angle_range_deg
        )
        self.target_detector = RobustTargetDetector(config, params)

        # 生命体征处理器（每个目标一个）
        self.vs_processors = {}

    def process_frame(self, range_fft_data):
        """
        处理单帧数据

        Args:
            range_fft_data: [num_chirps, num_rx, num_range_bins]

        Returns:
            targets: 检测到的目标列表
        """
        # 1. 计算距离维功率谱
        range_profile = np.mean(np.abs(range_fft_data) ** 2, axis=(0, 1))

        # 2. 目标检测
        from signal_processing import SignalProcessor
        processor = SignalProcessor(self.params)
        range_bins = processor.get_range_bins()

        targets = self.target_detector.detect_targets(range_profile, range_bins)

        # 3. 对每个目标进行角度估计
        for target in targets:
            azimuth_spectrum = self.beamformer.process(
                range_fft_data,
                target.range_bin,
                mode='auto'
            )

            # 找峰值角度
            peak_az_idx = np.argmax(azimuth_spectrum)
            target.azimuth_deg = self.beamformer.get_azimuth_angles()[peak_az_idx]
            target.azimuth_bin = peak_az_idx

        return targets

    def extract_vital_signs(self, target: Target, complex_signal):
        """
        提取单个目标的生命体征

        Args:
            target: 目标对象
            complex_signal: 复数信号
        """
        # 为每个目标创建独立的处理器
        if target.id not in self.vs_processors:
            from vital_signs_processing import VitalSignsProcessor
            self.vs_processors[target.id] = VitalSignsProcessor(self.params)

        vs_proc = self.vs_processors[target.id]
        result = vs_proc.process_frame(complex_signal)

        # 更新目标信息
        target.breathing_rate_bpm = result.get('breathing_rate_bpm', 0.0)
        target.heart_rate_bpm = result.get('heart_rate_bpm', 0.0)
        target.confidence_breath = result.get('confidence_breath', 0.0)
        target.confidence_heart = result.get('confidence_heart', 0.0)

        # 质量检查
        if (target.confidence_breath < self.config.min_confidence or
            target.confidence_heart < self.config.min_confidence):
            target.is_valid = False

    def finalize_vital_signs(self, target: Target):
        """
        完成生命体征估计

        Args:
            target: 目标对象
        """
        if target.id in self.vs_processors:
            vs_proc = self.vs_processors[target.id]
            result = vs_proc.estimate_vital_signs()

            target.breathing_rate_bpm = result['breathing_rate_bpm']
            target.heart_rate_bpm = result['heart_rate_bpm']
            target.confidence_breath = result['confidence_breath']
            target.confidence_heart = result['confidence_heart']


# 使用示例
if __name__ == "__main__":
    print("通用多目标生命体征检测器")
    print("="*60)
    print()
    print("核心改进:")
    print("  ✅ 自适应波束成形（单/多chirp）")
    print("  ✅ 鲁棒的目标检测（CFAR + 聚类 + 去重）")
    print("  ✅ 多目标独立跟踪")
    print("  ✅ 质量控制（SNR + 置信度）")
    print()
    print("使用方法:")
    print("  from universal_detector import UniversalVitalSignsDetector, DetectorConfig")
    print("  config = DetectorConfig()")
    print("  detector = UniversalVitalSignsDetector(config, params)")
    print("  targets = detector.process_frame(range_fft_data)")
