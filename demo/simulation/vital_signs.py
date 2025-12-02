from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config_parser import RadarConfig, VitalSignsConfig, MotionDetectionConfig


PI = math.pi
WAVELENGTH_MM = 3.9  # 与固件一致（mm），仅用于可视化
CONVERT_HZ_BPM = 60.0


@dataclass(frozen=True)
class VitalSignsResult:
    unwrap_phase_mm: float
    breath_bpm_fft: float
    heart_bpm_fft: float
    breath_bpm_peakcount: float
    heart_bpm_peakcount: float
    confidence_breath: float
    confidence_heart: float
    breath_wave: float
    heart_wave: float
    # 运动检测标志（对应固件 obj_VS->motionDetected，便于调试/对比）
    motion_detected: bool = False


def _unwrap(phase: float, phase_prev: float, diff_corr_cum: float) -> Tuple[float, float]:
    # 逐帧相位展开（与固件 dss_vitalSignsDemo_utilsFunc.c 对齐）
    diff_phase = phase - phase_prev
    if diff_phase > PI:
        mod_factor = 1.0
    elif diff_phase < -PI:
        mod_factor = -1.0
    else:
        mod_factor = 0.0
    diff_mod = diff_phase - mod_factor * 2.0 * PI
    if (diff_mod == -PI) and (diff_phase > 0):
        diff_mod = PI

    diff_corr = diff_mod - diff_phase
    # Ignore correction when incremental variation is smaller than cutoff
    if (0 < diff_corr < PI) or (-PI < diff_corr < 0):
        diff_corr = 0.0

    diff_corr_cum = diff_corr_cum + diff_corr
    phase_out = phase + diff_corr_cum
    return phase_out, diff_corr_cum


def _autocorr_rate(
    buf: np.ndarray,
    fs_hz: float,
    f_min_hz: float,
    f_max_hz: float,
) -> float:
    """
    近似固件 computeAutoCorrelation + 心率/呼吸率估计：
      - 在 [minLag, maxLag] 内做自相关
      - 找主峰位置，对应的频率 = fs / lag
    """
    n = buf.size
    if n < 4:
        return 0.0
    # 将频率区间映射为 lag 区间：lag = fs / f
    max_lag = int(max(1, fs_hz / max(f_min_hz, 1e-3)))
    min_lag = int(max(1, fs_hz / max(f_max_hz, 1e-3)))
    if max_lag <= min_lag or max_lag >= n:
        return 0.0

    x = buf.astype(np.float32)
    # 简单直接型自相关（长度较小可接受）
    ac = np.zeros(max_lag + 1, dtype=np.float32)
    for lag in range(min_lag, max_lag + 1):
        ac[lag] = float(np.dot(x[: n - lag], x[lag:]))
    # 找最大峰（忽略 0-lag）
    lag_idx = int(np.argmax(ac[min_lag:max_lag + 1])) + min_lag
    if lag_idx <= 0:
        return 0.0
    f = fs_hz / float(lag_idx)
    return CONVERT_HZ_BPM * f


def _apply_agc(
    buf: np.ndarray,
    block_size: int,
    thresh: float,
) -> None:
    """
    近似固件 computeAGC：对滑动窗口能量超过阈值的片段进行缩放。
    这里只做简单遍历，适用于离线仿真场景。
    """
    n = buf.size
    if block_size <= 0 or block_size > n:
        return
    for end in range(block_size, n + 1):
        seg = buf[end - block_size : end]
        energy = float(np.dot(seg, seg))
        if energy > thresh:
            scale = float(np.sqrt(thresh / max(energy, 1e-9)))
            buf[end - block_size : end] = seg * scale


class _IIRBiquadCascade:
    """
    直接使用固件提供的 2nd-order 级联系数（b0,b1,b2,1,a1,a2）与 scale 值。
    每次仅处理一个样本（在线处理）。
    """

    def __init__(self, coefs: List[float], scales: List[float], num_stages: int) -> None:
        self.coefs = np.asarray(coefs, dtype=np.float32).reshape(num_stages, 6)
        self.scales = np.asarray(scales, dtype=np.float32)  # len = num_stages + 1, 但仅使用前 num_stages
        self.delay = np.zeros((num_stages, 3), dtype=np.float32)  # x(n), x(n-1), x(n-2) 的内部状态实现
        self.num_stages = num_stages

    def step(self, x: float) -> float:
        y = float(x)
        for s in range(self.num_stages):
            b0, b1, b2, _, a1, a2 = self.coefs[s]
            scale = float(self.scales[s])
            # 复现 C 代码的结构：
            # pDelay[idx] = scale*input - a1*pDelay[idx+1] - a2*pDelay[idx+2]
            # y =  b0*pDelay[idx] + b1*pDelay[idx+1] + b2*pDelay[idx+2]
            z0, z1, z2 = self.delay[s]
            z0 = scale * y - a1 * z1 - a2 * z2
            y = b0 * z0 + b1 * z1 + b2 * z2
            # 更新
            self.delay[s, 2] = z1
            self.delay[s, 1] = z0
            self.delay[s, 0] = z0  # 保持与索引一致（z0 占位）
        return float(y)


def _remove_impulse_noise(prev2: float, prev1: float, curr: float, thresh: float) -> float:
    back = prev1 - prev2
    fwd = prev1 - curr
    if ((fwd > thresh) and (back > thresh)) or ((fwd < -thresh) and (back < -thresh)):
        # 线性插值
        return prev2 + (curr - prev2) * 0.5
    return curr


def _confidence_metric(spectrum: np.ndarray, peak_idx: int, span: int, start: int, end: int) -> float:
    start = max(start, 0)
    end = min(end, spectrum.size - 1)
    sum_sig = float(np.sum(spectrum[start : end + 1]))
    lo = max(peak_idx - span, start)
    hi = min(peak_idx + span, end)
    sum_peak = float(np.sum(spectrum[lo : hi + 1]))
    if abs(sum_sig - sum_peak) < 1e-6:
        return 0.0
    return float(sum_peak / (sum_sig - sum_peak + 1e-9))


class VitalSignsProcessor:
    """
    复现固件 VS_Processing_Chain 的关键步骤：
      - 相位展开
      - 冲击噪声抑制（可选）
      - IIR(BP) 呼吸/心率波形
      - 峰计数与 FFT 估计
      - 指数平滑
    """

    # 固件 IIR 系数（从 dss_data_path.c 拷贝）
    BREATH_COEFS = [
        1.0000, 0, -1.0000, 1.0000, -1.9196, 0.9252,
        1.0000, 0, -1.0000, 1.0000, -1.6624, 0.7390,
    ]
    BREATH_SCALES = [0.0602, 0.0602, 1.0000]
    HEART4_COEFS = [
        1.0000, 0, -1.0000, 1.0000, -1.4522, 0.6989,
        1.0000, 0, -1.0000, 1.0000,  1.5573, 0.7371,
        1.0000, 0, -1.0000, 1.0000,  1.2189, 0.3932,
        1.0000, 0, -1.0000, 1.0000, -1.0947, 0.3264,
    ]
    HEART4_SCALES = [0.4188, 0.4188, 0.3611, 0.3611, 1.0000]

    def __init__(
        self,
        radar: RadarConfig,
        vital: VitalSignsConfig,
        motion: MotionDetectionConfig,
        remove_impulse: bool = True,
        compute_phase_diff: bool = False,
    ) -> None:
        self.radar = radar
        self.vital = vital
        self.motion = motion
        self.remove_impulse = remove_impulse
        self.compute_phase_diff = compute_phase_diff

        # 采样率（慢时间）= 帧率
        self.slow_fs_hz = 1000.0 / float(radar.frame_periodicity_ms)

        # 环形缓冲区
        self.buf_breath = np.zeros(int(vital.win_len_breath), dtype=np.float32)
        self.buf_heart = np.zeros(int(vital.win_len_heart), dtype=np.float32)

        # 运动检测缓冲区（对应固件 obj_VS->pMotionCircularBuffer）
        self.motion_enabled = bool(motion.enabled)
        self.motion_block_size = int(motion.block_size)
        self.motion_thresh = float(motion.threshold)
        self.motion_gain_control = bool(motion.gain_control)
        if self.motion_block_size < 4:
            self.motion_block_size = 4
        self.buf_motion = np.zeros(self.motion_block_size, dtype=np.float32)
        self._motion_pos = 0
        self.motion_detected = False

        # IIR 滤波器
        self.iir_breath = _IIRBiquadCascade(self.BREATH_COEFS, self.BREATH_SCALES, 2)
        self.iir_heart = _IIRBiquadCascade(self.HEART4_COEFS, self.HEART4_SCALES, 4)

        # 状态
        self._phase_prev = 0.0
        self._diff_corr_cum = 0.0
        self._phase_diff_prev2 = 0.0
        self._phase_diff_prev1 = 0.0

        # 平滑能量
        self._ewma_breath = 0.0
        self._ewma_heart = 0.0

        # 中值滤波缓冲（对应 obj_VS->pBufferHeartRate / pBufferBreathingRate 等）
        self._hist_breath_fft: List[float] = []
        self._hist_heart_fft: List[float] = []
        self._hist_heart_fft_4hz: List[float] = []
        self._median_win = 5  # 远小于固件默认，但足够平滑

        # 峰距阈值（样本）
        # 根据频带近似：最小间距 ~ Fs / f_max，最大间距 ~ Fs / f_min
        def _pk_win(fmin: float, fmax: float) -> Tuple[int, int]:
            dmin = max(1, int(self.slow_fs_hz / max(fmax, 1e-3)))
            dmax = max(dmin + 1, int(self.slow_fs_hz / max(fmin, 1e-3)))
            return dmin, dmax

        self.pk_breath_min, self.pk_breath_max = _pk_win(0.1, 0.5)
        self.pk_heart_min, self.pk_heart_max = _pk_win(0.8, 4.0)

    def _peak_count_rate(self, buf: np.ndarray, dmin: int, dmax: int) -> float:
        # 简单峰计数（不严格逐段验证，近似固件 filterPeaksWfm）
        x = buf.astype(np.float32)
        # 以一阶差分找局部峰
        dx = np.sign(np.diff(x))
        pk = np.where((np.hstack([dx, 0]) <= 0) & (np.hstack([0, dx]) > 0))[0]
        if pk.size == 0:
            return 0.0
        # 基于最小峰距过滤
        selected = [int(pk[0])]
        for i in pk[1:]:
            if (i - selected[-1]) >= dmin:
                selected.append(int(i))
        # 估计频率 = 峰数 / 窗长 * Fs
        rate_hz = (len(selected) * self.slow_fs_hz) / max(1, x.size)
        return CONVERT_HZ_BPM * rate_hz

    def _fft_rate_and_conf(
        self, buf: np.ndarray, band: Tuple[float, float], scale: float
    ) -> Tuple[float, float]:
        # 复刻固件：将实序列填充到复数数组 FFT，计算幅度谱，找峰并给出信心度
        n = buf.size
        x = (buf.astype(np.float32) * float(scale)).astype(np.float32)
        spec = np.fft.rfft(x, n=max(n, 512)).astype(np.complex64)
        mag = np.abs(spec) ** 2
        # 频率分辨率
        freqs = np.fft.rfftfreq(mag.size * 2 - 2, d=1.0 / self.slow_fs_hz)
        f_lo, f_hi = band
        lo = int(np.searchsorted(freqs, f_lo))
        hi = int(np.searchsorted(freqs, f_hi))
        lo = max(1, min(lo, mag.size - 1))
        hi = max(lo + 1, min(hi, mag.size))
        seg = mag[lo:hi]
        if seg.size == 0:
            return 0.0, 0.0
        pk = int(np.argmax(seg)) + lo
        f_est = float(freqs[pk])
        bpm = CONVERT_HZ_BPM * f_est
        conf = _confidence_metric(mag, pk, span=2, start=lo, end=hi - 1)
        return bpm, conf

    def step(self, sample_cplx: complex) -> VitalSignsResult:
        # 由波束形成后的复数样本得到相位
        phase = math.atan2(sample_cplx.imag, sample_cplx.real)
        phase_unwrap, self._diff_corr_cum = _unwrap(phase, self._phase_prev, self._diff_corr_cum)
        self._phase_prev = phase

        phase_used = phase_unwrap
        if self.compute_phase_diff:
            phase_used = phase_unwrap - self._phase_diff_prev1
            self._phase_diff_prev2 = self._phase_diff_prev1
            self._phase_diff_prev1 = phase_unwrap

        if self.remove_impulse:
            phase_used = _remove_impulse_noise(self._phase_diff_prev2, self._phase_diff_prev1, phase_used, 0.5)

        # IIR 滤波
        breath_out = self.iir_breath.step(phase_used)
        heart_out = self.iir_heart.step(phase_used)

        # 呼吸环形缓冲：始终更新
        self.buf_breath = np.roll(self.buf_breath, -1)
        self.buf_breath[-1] = breath_out

        # 心率环形缓冲 + 运动检测（近似 VS_Processing_Chain 的逻辑）
        if self.motion_enabled:
            # 更新运动检测缓冲（使用心率波形）
            self.buf_motion[self._motion_pos] = heart_out
            self._motion_pos = (self._motion_pos + 1) % self.motion_block_size

            # 每填满一个块做一次能量检测
            motion_detected = self.motion_detected
            if self._motion_pos == 0:
                energy = float(np.dot(self.buf_motion, self.buf_motion))
                motion_detected = energy > self.motion_thresh
                self.motion_detected = motion_detected

                # 若当前块“干净”，将该块拼接进心率环形缓冲
                if not motion_detected:
                    # 左移 block_size 个样本，然后将 buf_motion 拷贝到末尾
                    shift = min(self.motion_block_size, self.buf_heart.size)
                    if shift > 0:
                        self.buf_heart = np.roll(self.buf_heart, -shift)
                        self.buf_heart[-shift:] = self.buf_motion[-shift:]
            # 若块尚未填满，则本帧仍按普通帧只更新最后一个样本
            else:
                self.buf_heart = np.roll(self.buf_heart, -1)
                self.buf_heart[-1] = heart_out
        else:
            # 无运动检测：直接更新心率环形缓冲
            self.buf_heart = np.roll(self.buf_heart, -1)
            self.buf_heart[-1] = heart_out

        # 可选 AGC：对应 computeAGC，对心率缓冲能量过大的片段做增益控制
        if self.motion_gain_control:
            _apply_agc(self.buf_heart, self.motion_block_size, self.motion_thresh)

        # 峰计数估计
        breath_bpm_pc = self._peak_count_rate(self.buf_breath, self.pk_breath_min, self.pk_breath_max)
        heart_bpm_pc = self._peak_count_rate(self.buf_heart, self.pk_heart_min, self.pk_heart_max)

        # FFT 估计与置信度（主通道，与固件频带一致）
        breath_bpm_fft, conf_breath = self._fft_rate_and_conf(
            self.buf_breath, (0.1, 0.5), scale=float(self.vital.scale_breath)
        )
        heart_bpm_fft_full, conf_heart_full = self._fft_rate_and_conf(
            self.buf_heart, (0.8, 4.0), scale=float(self.vital.scale_heart)
        )

        # 额外心率 FFT（1.6–4.0 Hz 区间），用于谐波检测（近似 heartRateEst_FFT_4Hz）
        heart_bpm_fft_4hz, _ = self._fft_rate_and_conf(
            self.buf_heart, (1.6, 4.0), scale=float(self.vital.scale_heart)
        )

        # 简化版谐波与呼吸谐波抑制：
        heart_bpm_fft = heart_bpm_fft_full
        conf_heart = conf_heart_full
        if 0.0 < breath_bpm_fft < 200.0 and 0.0 < heart_bpm_fft_4hz < 200.0:
            # 若心率估计与呼吸 2 倍附近过近，则认为被呼吸谐波污染，回退到 0.8–4Hz 主谱估计
            if abs(heart_bpm_fft_4hz - 2.0 * breath_bpm_fft) < 10.0:
                heart_bpm_fft = heart_bpm_fft_full
                conf_heart = conf_heart_full
            else:
                heart_bpm_fft = heart_bpm_fft_4hz

        # 指数平滑能量（仅用于可视化/一致性）
        self._ewma_breath = float(self.vital.alpha_breath) * (breath_out * breath_out) + (1.0 - float(self.vital.alpha_breath)) * self._ewma_breath
        self._ewma_heart = float(self.vital.alpha_heart) * (heart_out * heart_out) + (1.0 - float(self.vital.alpha_heart)) * self._ewma_heart

        # 中值滤波（对应 FLAG_MEDIAN_FILTER）：基于最近 _median_win 帧的 FFT 结果
        def _update_med(history: List[float], value: float) -> float:
            if not np.isfinite(value) or value <= 0.0:
                return 0.0
            history.append(float(value))
            if len(history) > self._median_win:
                del history[0 : len(history) - self._median_win]
            arr = np.asarray(history, dtype=np.float32)
            return float(np.median(arr)) if arr.size > 0 else 0.0

        breath_bpm_fft = _update_med(self._hist_breath_fft, breath_bpm_fft)
        heart_bpm_fft = _update_med(self._hist_heart_fft, heart_bpm_fft)
        _ = _update_med(self._hist_heart_fft_4hz, heart_bpm_fft_4hz)

        return VitalSignsResult(
            unwrap_phase_mm=phase_unwrap * (4.0 * math.pi / WAVELENGTH_MM),  # 与固件相同的比例（仅作标度）
            breath_bpm_fft=float(breath_bpm_fft),
            heart_bpm_fft=float(heart_bpm_fft),
            breath_bpm_peakcount=float(breath_bpm_pc),
            heart_bpm_peakcount=float(heart_bpm_pc),
            confidence_breath=float(conf_breath),
            confidence_heart=float(conf_heart),
            breath_wave=float(breath_out),
            heart_wave=float(heart_out),
            motion_detected=bool(self.motion_detected),
        )





