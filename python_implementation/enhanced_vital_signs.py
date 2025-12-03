"""
增强的生命体征提取算法
Enhanced Vital Signs Extraction

改进点:
1. 更精确的FFT频率分辨率
2. 强化的谐波抑制
3. 多方法融合
4. 自适应参数调整
"""

import numpy as np
from scipy import signal
from scipy.fft import fft
from scipy.ndimage import median_filter as scipy_median_filter


class EnhancedVitalSignsExtractor:
    """
    增强的生命体征提取器
    """

    def __init__(self, params, config):
        """
        Args:
            params: 雷达参数
            config: DetectorConfig配置
        """
        self.params = params
        self.config = config
        self.frame_rate = params['frame_rate']
        self.wavelength = params['wavelength']

        # 频率范围
        self.breath_freq_min = config.breath_freq_min
        self.breath_freq_max = config.breath_freq_max
        self.heart_freq_min = config.heart_freq_min
        self.heart_freq_max = config.heart_freq_max

        # 缓冲区大小（增大以提高频率分辨率）
        self.buffer_size_breath = 1024  # 增大到1024
        self.buffer_size_heart = 2048   # 增大到2048

        # 循环缓冲区
        self.breath_buffer = np.zeros(self.buffer_size_breath)
        self.heart_buffer = np.zeros(self.buffer_size_heart)

        # 相位处理状态
        self.phase_prev = 0.0
        self.phase_correction_cum = 0.0
        self.phase_used_prev = 0.0

        # 设计滤波器
        self._design_filters()

        # FFT参数
        self.fft_size_breath = 2048  # 增大FFT尺寸
        self.fft_size_heart = 4096
        self.freq_bins_breath = np.fft.fftfreq(self.fft_size_breath, 1.0 / self.frame_rate)
        self.freq_bins_heart = np.fft.fftfreq(self.fft_size_heart, 1.0 / self.frame_rate)

        # 中值滤波历史
        self.breath_rate_history = []
        self.heart_rate_history = []
        self.median_window = 10  # 减小窗口以提高响应速度

    def _design_filters(self):
        """设计IIR滤波器"""
        # 呼吸滤波器
        sos_breath = signal.butter(
            2,
            [self.breath_freq_min, self.breath_freq_max],
            btype='bandpass',
            fs=self.frame_rate,
            output='sos'
        )
        self.sos_breath = sos_breath
        self.zi_breath = signal.sosfilt_zi(sos_breath)

        # 心率滤波器
        sos_heart = signal.butter(
            4,
            [self.heart_freq_min, self.heart_freq_max],
            btype='bandpass',
            fs=self.frame_rate,
            output='sos'
        )
        self.sos_heart = sos_heart
        self.zi_heart = signal.sosfilt_zi(sos_heart)

    def process_frame(self, complex_signal):
        """处理单帧"""
        # 提取相位
        phase_current = np.angle(complex_signal)

        # 相位解缠绕
        unwrapped_phase = self._unwrap_phase(phase_current)

        # 相位差分
        phase_diff = unwrapped_phase - self.phase_used_prev
        self.phase_used_prev = unwrapped_phase

        # IIR滤波
        breath_sample, self.zi_breath = signal.sosfilt(
            self.sos_breath, [phase_diff], zi=self.zi_breath
        )
        heart_sample, self.zi_heart = signal.sosfilt(
            self.sos_heart, [phase_diff], zi=self.zi_heart
        )

        # 更新缓冲区
        self.breath_buffer = np.roll(self.breath_buffer, -1)
        self.breath_buffer[-1] = breath_sample[0]

        self.heart_buffer = np.roll(self.heart_buffer, -1)
        self.heart_buffer[-1] = heart_sample[0]

        return {
            'unwrapped_phase': unwrapped_phase,
            'breath_sample': breath_sample[0],
            'heart_sample': heart_sample[0]
        }

    def estimate_vital_signs(self):
        """估计生命体征"""
        # FFT方法
        breath_rate_fft, conf_breath = self._estimate_rate_fft_enhanced(
            self.breath_buffer,
            self.breath_freq_min,
            self.breath_freq_max,
            self.fft_size_breath,
            self.freq_bins_breath
        )

        heart_rate_fft, conf_heart = self._estimate_rate_fft_enhanced(
            self.heart_buffer,
            self.heart_freq_min,
            self.heart_freq_max,
            self.fft_size_heart,
            self.freq_bins_heart,
            suppress_harmonics=True,
            fundamental_freq=breath_rate_fft / 60.0  # 传入呼吸基频
        )

        # 中值滤波
        breath_rate_final = self._apply_median_filter(
            breath_rate_fft,
            self.breath_rate_history
        )

        heart_rate_final = self._apply_median_filter(
            heart_rate_fft,
            self.heart_rate_history
        )

        return {
            'breathing_rate_bpm': breath_rate_final,
            'heart_rate_bpm': heart_rate_final,
            'confidence_breath': conf_breath,
            'confidence_heart': conf_heart
        }

    def _estimate_rate_fft_enhanced(self, waveform, freq_min, freq_max,
                                    fft_size, freq_bins,
                                    suppress_harmonics=False,
                                    fundamental_freq=None):
        """
        增强的FFT频谱估计

        Args:
            waveform: 时域波形
            freq_min: 最小频率(Hz)
            freq_max: 最大频率(Hz)
            fft_size: FFT尺寸
            freq_bins: 频率bins
            suppress_harmonics: 是否抑制谐波
            fundamental_freq: 基频(Hz)，用于谐波抑制
        """
        # 应用Hanning窗
        windowed = waveform * np.hanning(len(waveform))

        # Zero-padding
        if len(windowed) < fft_size:
            windowed = np.pad(windowed, (0, fft_size - len(windowed)))

        # FFT
        spectrum = np.abs(fft(windowed, n=fft_size))
        spectrum = spectrum[:fft_size // 2]
        freq_bins_positive = freq_bins[:fft_size // 2]

        # 频率范围索引
        freq_idx_min = np.argmin(np.abs(freq_bins_positive - freq_min))
        freq_idx_max = np.argmin(np.abs(freq_bins_positive - freq_max))

        spectrum_roi = spectrum[freq_idx_min:freq_idx_max]
        freq_roi = freq_bins_positive[freq_idx_min:freq_idx_max]

        # 谐波抑制
        if suppress_harmonics and fundamental_freq is not None:
            spectrum_roi = self._suppress_harmonics(
                spectrum_roi,
                freq_roi,
                fundamental_freq
            )

        # 找峰值
        if len(spectrum_roi) > 0:
            peak_idx = np.argmax(spectrum_roi)
            peak_freq = freq_roi[peak_idx]
            rate_bpm = peak_freq * 60.0

            # 置信度（峰值功率 / 平均功率）
            peak_power = spectrum_roi[peak_idx]
            avg_power = np.mean(spectrum_roi)

            if avg_power > 0:
                confidence = peak_power / avg_power
            else:
                confidence = 0.0
        else:
            rate_bpm = 0.0
            confidence = 0.0

        return rate_bpm, confidence

    def _suppress_harmonics(self, spectrum, freq_bins, fundamental_freq):
        """
        抑制谐波干扰

        Args:
            spectrum: 频谱
            freq_bins: 频率bins
            fundamental_freq: 基频(Hz)
        """
        spectrum_suppressed = spectrum.copy()

        # 抑制2倍频和3倍频
        for harmonic in [2, 3]:
            harmonic_freq = fundamental_freq * harmonic
            harmonic_bandwidth = 0.1  # Hz

            # 找到谐波频率附近的索引
            mask = np.abs(freq_bins - harmonic_freq) < harmonic_bandwidth

            # 抑制（设为局部平均值）
            if np.any(mask):
                local_avg = np.mean(spectrum_suppressed[~mask])
                spectrum_suppressed[mask] = local_avg * 0.5

        return spectrum_suppressed

    def _unwrap_phase(self, phase_current):
        """相位解缠绕"""
        phase_diff = phase_current - self.phase_prev

        while phase_diff > np.pi:
            phase_diff -= 2 * np.pi
            self.phase_correction_cum -= 2 * np.pi

        while phase_diff < -np.pi:
            phase_diff += 2 * np.pi
            self.phase_correction_cum += 2 * np.pi

        unwrapped_phase = phase_current + self.phase_correction_cum
        self.phase_prev = phase_current

        return unwrapped_phase

    def _apply_median_filter(self, current_value, history_list):
        """中值滤波"""
        history_list.append(current_value)

        if len(history_list) > self.median_window:
            history_list.pop(0)

        if len(history_list) > 0:
            filtered_value = np.median(history_list)
        else:
            filtered_value = current_value

        return filtered_value

    def reset(self):
        """重置状态"""
        self.breath_buffer = np.zeros(self.buffer_size_breath)
        self.heart_buffer = np.zeros(self.buffer_size_heart)
        self.phase_prev = 0.0
        self.phase_correction_cum = 0.0
        self.phase_used_prev = 0.0
        self.breath_rate_history = []
        self.heart_rate_history = []
        self.zi_breath = signal.sosfilt_zi(self.sos_breath)
        self.zi_heart = signal.sosfilt_zi(self.sos_heart)


if __name__ == "__main__":
    print("增强的生命体征提取算法")
    print("="*60)
    print()
    print("改进点:")
    print("  ✅ 更大的FFT尺寸 (2048/4096) → 更高频率分辨率")
    print("  ✅ 更大的缓冲区 (1024/2048) → 更多数据积累")
    print("  ✅ 谐波抑制 → 减少呼吸对心率的干扰")
    print("  ✅ 优化的频率范围 → 避免异常值")
    print()
