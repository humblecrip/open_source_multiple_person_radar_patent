from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config_parser import RadarConfig


ODDEMO_MAX_RANGE = 64


@dataclass(frozen=True)
class TargetPeak:
    range_idx: int
    azimuth_idx: int
    power: float
    sample_cplx: complex  # 本帧用于生命体征链的复数输入（波束形成后）


def _steering_matrix(num_rx: int, angle_bins: int, angle_range_deg: float) -> np.ndarray:
    """
    生成均匀线阵（ULA）波束方向向量，阵元间距 = 0.5λ。
    返回形状 [angle_bins, num_rx] 的复数矩阵。
    """
    thetas = np.linspace(-angle_range_deg, angle_range_deg, angle_bins, dtype=np.float32) * np.pi / 180.0
    # 相位差 = 2π(d/λ) sin(theta)；d=0.5λ => 相位步进 = π sin(theta)
    phase_step = np.pi * np.sin(thetas)  # [angle_bins]
    idx = np.arange(num_rx, dtype=np.float32)[None, :]  # [1, num_rx]
    phases = -idx * phase_step[:, None]  # [angle_bins, num_rx]
    return np.exp(1j * phases).astype(np.complex64)


def _capon_heatmap(
    X: np.ndarray,  # [num_rx, num_chirps]
    steering: np.ndarray,  # [angle_bins, num_rx]
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Capon 波束形成热力图（每个角度一个值）。
    返回 (heat_az, invR)，其中 heat_az 形状为 [angle_bins]。
    """
    # clutter removal: 去除每个天线的均值（沿 chirp 方向）
    Xc = X - X.mean(axis=1, keepdims=True)
    # 协方差矩阵
    R = (Xc @ Xc.conj().T) / max(1, Xc.shape[1])
    # 对角加载
    diag_mean = np.real(np.trace(R)) / R.shape[0]
    R = R + (gamma * diag_mean) * np.eye(R.shape[0], dtype=np.complex64)
    # 求逆
    invR = np.linalg.pinv(R).astype(np.complex64)

    # Capon 谱：1 / (a^H invR a)
    # steering: [A, N], invR: [N, N]
    v = (steering @ invR)  # [A, N]
    denom = np.einsum("an,an->a", v.conj(), steering, optimize=True)  # [A]
    denom = np.maximum(np.real(denom), 1e-9)
    heat = 1.0 / denom
    return heat.astype(np.float32), invR


def _mvdr_weights(invR: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    MVDR 权重: w = invR a / (a^H invR a)
    """
    num = invR @ a  # [num_rx]
    denom = np.vdot(a, num)  # 标量
    if np.abs(denom) < 1e-9:
        return np.zeros_like(a)
    return (num / denom).astype(np.complex64)


def _find_peaks_heatmap(
    heatmap: np.ndarray,  # [range, azimuth]
    max_peaks: int = 4,
    min_separation: int = 10,
    abs_threshold: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """
    简化版峰值选择（接近固件 heatmapGetPeaks 的逻辑）。
    - min_separation 在 range/azimuth 的 Chebyshev 距离上进行抑制。
    """
    rng, ang = heatmap.shape
    # 经验阈值：若未提供绝对阈值，用分位数抑制
    if abs_threshold <= 0.0:
        abs_threshold = float(np.quantile(heatmap, 0.98))
    candidates = np.argwhere(heatmap >= abs_threshold)
    peaks = []
    for r, a in candidates:
        val = float(heatmap[r, a])
        if not peaks:
            peaks.append((r, a, val))
            continue
        suppress = False
        for pr, pa, pv in peaks:
            if max(abs(pr - r), abs(pa - a)) < min_separation:
                # 留下更大的
                if pv >= val:
                    suppress = True
                    break
        if not suppress:
            peaks.append((r, a, val))
        # 限制数量
        peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[: max_peaks * 3]
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:max_peaks]
    return peaks


class RangeAngleProcessor:
    """
    复现固件 Occupancy Heatmap + 动态峰值选择 + 生成生命体征链输入的复数样本。
    """

    def __init__(self, radar: RadarConfig) -> None:
        self.radar = radar
        self.angle_bins = int(radar.angle_bins)
        self.angle_range = 60.0  # 固件 ODDEMO_ANGLE_RANGE
        self.max_range_bins = min(ODDEMO_MAX_RANGE, int(radar.num_range_bins))
        self.steering = _steering_matrix(radar.num_rx, self.angle_bins, self.angle_range)
        self.gamma = float(radar.diag_load_factor)
        # 1D FFT 窗
        win = np.blackman(int(radar.num_adc_samples)).astype(np.float32)
        self.window = win / (np.sum(win) + 1e-9)

    def _range_fft(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: [num_chirps_per_frame, num_adc_samples, v_rx]
        返回 range-fft 结果（保留前 max_range_bins）：
        cube: [num_chirps_per_frame, max_range_bins, v_rx]
        """
        n_chirps, n_adc, v_rx = frame.shape
        n_bins = int(self.radar.num_range_bins)
        x = frame * self.window[None, :, None]
        # 复数输入使用复数 FFT（固件中 1D FFT 为复数），保留前 max_range_bins
        spec = np.fft.fft(x.astype(np.complex64, copy=False), n=n_bins, axis=1)  # [chirps, n_bins, v_rx]
        spec = spec[:, : self.max_range_bins, :]
        return spec.astype(np.complex64)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[TargetPeak]]:
        """
        处理一帧：生成 range-azimuth 热力图与峰值，并给出每个峰值对应的本帧复数样本（供生命体征链）。
        返回：
          - heatmap: [max_range_bins, angle_bins] float32
          - peaks: List[TargetPeak]
        """
        n_chirps, _, v_rx = frame.shape
        # 只取 TX0 的 chirps（与固件进行 TDM-MIMO 等价处理），步长为 num_tx
        step = max(1, int(self.radar.num_tx))
        chirp_sel = np.arange(0, n_chirps, step, dtype=np.int32)
        # 对应 TX0 的实 RX 索引（虚拟阵列：每 TX 占据一段）
        rx_sel = np.arange(0, int(self.radar.num_rx), dtype=np.int32)
        # 取 TX0 的一段虚拟阵元
        rx_indices = rx_sel  # 只用 num_rx 元素

        # Range FFT
        cube = self._range_fft(frame)  # [chirps, range, v_rx]

        heatmap = np.zeros((self.max_range_bins, self.angle_bins), dtype=np.float32)
        # 为每个 range 计算 Capon 谱
        for r in range(self.max_range_bins):
            # 选出该 range, TX0 的所有 chirps, 以及 num_rx 阵元
            X = cube[chirp_sel, r, :][:, rx_indices]  # [n_sel_chirps, num_rx]
            X = X.T  # [num_rx, n_sel_chirps]
            heat_az, _ = _capon_heatmap(X, self.steering, self.gamma)
            heatmap[r, :] = heat_az

        # 峰值选择（动态多人）
        peaks_idx = _find_peaks_heatmap(heatmap, max_peaks=4, min_separation=10, abs_threshold=0.0)

        peaks: List[TargetPeak] = []
        for (r, a, p) in peaks_idx:
            # 为峰值计算 MVDR 权重，并生成生命体征输入复数样本（使用该 range 上全部所选 chirps 的均值）
            X = cube[chirp_sel, r, :][:, rx_indices].T  # [num_rx, n_sel_chirps]
            # invR 与权重
            heat_az, invR = _capon_heatmap(X, self.steering[a : a + 1, :], self.gamma)
            a_vec = self.steering[a, :]
            w = _mvdr_weights(invR, a_vec)
            # 将权重应用于每个 chirp 的快拍，取均值以增强稳定度
            y = np.einsum("n,nt->t", w.conj(), X, optimize=True)  # [n_sel_chirps]
            sample = np.mean(y) if y.size > 0 else 0.0 + 0.0j
            peaks.append(TargetPeak(range_idx=int(r), azimuth_idx=int(a), power=float(p), sample_cplx=complex(sample)))

        return heatmap, peaks


