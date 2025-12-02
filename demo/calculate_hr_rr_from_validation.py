"""
从 TI ADS1292R 采集的 ECG/PCG 数据中计算心率（HR）和呼吸率（RR）

数据来源：TI ADS1292R 生物电信号采集芯片
- ECG 列：心电图信号，用于计算心率（HR）和派生呼吸波形（RR）
- PCG 列：心音/呼吸信号，可用于辅助 HR/RR 估计与验证

当前处理算法（重写后的逻辑）：
- 心率检测（HR）：
  1) ECG 去直流
  2) 5–20 Hz 带通滤波（保留 QRS 主要能量）
  3) 一阶差分 + 绝对值 + 0.15 s 滑动平均
  4) 用前 10 s 的最大值估计阈值，结合最小 0.3 s 峰距找到 R 峰
  5) 用相邻 R 峰间隔（RR 间期）计算瞬时 HR 和平均 HR
  6) 将 R 峰构成的脉冲序列做 FFT，获得频域 HR 估计并做谐波校正

- 呼吸率检测（RR）：
  - ECG-derived respiration:
    1) ECG 去直流
    2) 0.1–0.7 Hz 带通滤波（呼吸带）
    3) 对滤波信号归一化后做峰值检测，最小峰距 2 s
    4) 由相邻呼吸峰间隔计算瞬时 RR 和平均 RR
    5) 对呼吸波形做 FFT 得到频域 RR 估计
  - PCG-derived respiration:
    1) PCG 去直流
    2) 20–0.9*Nyquist Hz 带通滤波（突出心音/胸壁振动）
    3) 取绝对值 + 0.1 s 滑动平均得到心音包络
    4) 对包络做 0.1–0.7 Hz 带通滤波，提取呼吸成分
    5) 峰值检测 + FFT 同 ECG-derived RR，用于验证和对比

- RR_Final_BPM：综合 ECG/PCG、峰值与 FFT 的多个 RR 估计，
  做生理范围筛选 + 中位数附近聚类，得到更稳健的最终 RR。
"""

import argparse
import re
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks

# ====================================================================
# matplotlib 中文字体设置（尽量避免 glyph missing 警告）
# ====================================================================
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False


# ====================================================================
# 工具函数：读 CSV、滤波、FFT 估计等
# ====================================================================

def load_validation_csv(
    csv_path: str,
    num_samples: Optional[int] = None,
    segment_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 ADS1292R 采集的 ECG/PCG CSV 文件

    适配当前数据格式：
    - 第 1 行: Column1,Column2
    - 第 2 行: ECG,PCG
    - 第 3 行起: 数据
    """
    df = pd.read_csv(csv_path, skiprows=2, names=["ECG", "PCG"])
    ecg = df["ECG"].astype(float).values
    pcg = df["PCG"].astype(float).values

    # 去除 NaN 和无穷值
    valid_mask = np.isfinite(ecg) & np.isfinite(pcg)
    ecg = ecg[valid_mask]
    pcg = pcg[valid_mask]

    # 分段读取
    if num_samples is not None:
        start_idx = segment_idx * num_samples
        end_idx = min(start_idx + num_samples, len(ecg))

        if start_idx >= len(ecg):
            raise ValueError(
                f"段索引 {segment_idx} 超出数据范围"
                f"（总样本数: {len(ecg)}，每段: {num_samples}）"
            )

        ecg = ecg[start_idx:end_idx]
        pcg = pcg[start_idx:end_idx]

    return ecg, pcg


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """设计 Butterworth 带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_signal(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """对信号进行带通滤波（零相位）"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered = filtfilt(b, a, data)
    return filtered


def estimate_rate_fft(
    signal_data: np.ndarray,
    fs: float,
    freq_range: Tuple[float, float],
    window_size: Optional[int] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    使用 FFT 估计主频（转换为 BPM）

    Args:
        signal_data: 输入信号（可以是原始波形，也可以是 R 峰脉冲序列）
        fs: 采样率 (Hz)
        freq_range: 频率范围 (Hz) - (low, high)
        window_size: FFT 窗口大小（None 则使用 max(N, 512)）

    Returns:
        rate_bpm: 估计的频率（BPM）
        freqs: 频率轴（Hz）
        spectrum: 频谱功率谱
    """
    signal_data = np.asarray(signal_data, dtype=float)
    signal_data = signal_data - np.mean(signal_data)

    n = len(signal_data)
    if n == 0:
        return 0.0, np.array([]), np.array([])

    if window_size is None:
        window_size = max(n, 512)

    fft_result = np.fft.rfft(signal_data, n=window_size)
    spectrum = np.abs(fft_result) ** 2
    freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)

    f_low, f_high = freq_range
    mask = (freqs >= f_low) & (freqs <= f_high)
    if np.sum(mask) == 0:
        return 0.0, freqs, spectrum

    valid_spectrum = spectrum[mask]
    valid_freqs = freqs[mask]
    if len(valid_spectrum) == 0:
        return 0.0, freqs, spectrum

    peak_idx = np.argmax(valid_spectrum)
    peak_freq_hz = valid_freqs[peak_idx]
    rate_bpm = peak_freq_hz * 60.0

    return rate_bpm, freqs, spectrum


def parse_index_expression(expr: Optional[str]) -> Optional[List[int]]:
    """解析形如 "1,3-5" 的位置/编号字符串"""
    if expr is None:
        return None

    expr = expr.strip()
    if not expr or expr.lower() in {"all", "*"}:
        return None

    indices: set[int] = set()
    for chunk in expr.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            if not start_str.strip() or not end_str.strip():
                continue
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            indices.add(int(chunk))

    return sorted(indices) if indices else None


def extract_position_idx(directory: Path) -> Optional[int]:
    """从目录名（如 position_ (3)）解析位置编号"""
    match = re.search(r"(\d+)", directory.name)
    return int(match.group(1)) if match else None


def extract_measurement_idx(filename: str) -> Optional[int]:
    """从 CSV 文件名末尾的括号解析记录编号"""
    match = re.search(r"_\s*\((\d+)\)\.csv$", filename)
    return int(match.group(1)) if match else None


def sanitize_label(label: str) -> str:
    """将标签转换为文件系统友好的字符串"""
    safe = re.sub(r"[^0-9A-Za-z]+", "_", label)
    return safe.strip("_").lower() or "item"


def discover_radar_csvs(
    radar_root: Path,
    freq_band: str,
    target_names: List[str],
    positions: Optional[List[int]] = None,
    measurement_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """
    递归扫描 FMCW Radar 数据集，收集目标 CSV
    
    目录结构:
        radar_root/
        ├── Target1/
        │   ├── position_ (1)/
        │   │   ├── log_Target1_3GHZ_position1_ (1).csv
        │   │   └── ...
        │   └── position_ (6)/
        └── Target2/
            └── ...
    
    文件名格式: log_{Target}_{FreqBand}_position{PosIdx}_ ({MeasurementIdx}).csv
    """
    freq_band = freq_band.upper()
    discovered: list[dict] = []

    for target_name in target_names:
        target_dir = radar_root / target_name
        if not target_dir.exists():
            print(f"⚠️  警告: 目录不存在: {target_dir}")
            continue

        for position_dir in sorted(target_dir.iterdir()):
            if not position_dir.is_dir():
                continue

            pos_idx = extract_position_idx(position_dir)
            if pos_idx is None:
                continue
            if positions is not None and pos_idx not in positions:
                continue

            # 修正 glob pattern: log_Target1_3GHZ_position1_ (*.csv
            # 注意文件名中的空格和括号
            pattern = f"log_{target_name}_{freq_band}_position{pos_idx}_ (*).csv"
            csv_paths = sorted(position_dir.glob(pattern))

            for csv_path in csv_paths:
                measurement_idx = extract_measurement_idx(csv_path.name)
                if measurement_ids is not None and measurement_idx not in measurement_ids:
                    continue

                discovered.append({
                    "target": target_name,
                    "csv_path": csv_path,
                    "position": pos_idx,
                    "measurement": measurement_idx,
                    "freq_band": freq_band,
                })

    discovered.sort(key=lambda item: (
        item.get("target", ""),
        item.get("position", 0),
        item.get("measurement", 0) or 0,
    ))

    if discovered:
        print(
            f"在 {radar_root} 中找到 {len(discovered)} 个 {freq_band} 频段的 CSV "
            f"(targets={target_names}, positions={positions or 'all'}, measurements={measurement_ids or 'all'})"
        )
    else:
        print(
            f"⚠️  未在 {radar_root} 下找到 {freq_band} 频段、目标 {target_names} 的 CSV，"
            f"请检查目录结构或参数"
        )

    return discovered


def detect_and_correct_harmonic(
    fft_bpm: float,
    peak_bpm: float,
    spectrum: np.ndarray,
    freqs: np.ndarray,
    freq_range: Tuple[float, float] = (0.6, 3.0)
) -> Tuple[float, str]:
    """
    检测并校正二次谐波问题（用于 HR）

    当 FFT 和峰值检测结果差异约为 2 倍时，判断是否存在谐波，并返回校正后的心率
    """
    if fft_bpm <= 0 and peak_bpm > 0:
        return peak_bpm, "仅峰值检测有效"
    if peak_bpm <= 0 and fft_bpm > 0:
        return fft_bpm, "仅FFT有效"
    if fft_bpm <= 0 and peak_bpm <= 0:
        return 0.0, "两种方法均失败"

    ratio = fft_bpm / peak_bpm if peak_bpm > 0 else 0
    is_double = 1.8 <= ratio <= 2.2
    is_half = 0.45 <= ratio <= 0.55

    MIN_HR = 40
    MAX_HR = 180

    if is_double:
        base_freq_hz = peak_bpm / 60.0
        f_low, f_high = freq_range
        mask = (freqs >= f_low) & (freqs <= f_high)

        if np.sum(mask) > 0:
            valid_freqs = freqs[mask]
            valid_spectrum = spectrum[mask]

            base_mask = np.abs(valid_freqs - base_freq_hz) < 0.1
            if np.sum(base_mask) > 0:
                base_energy = np.max(valid_spectrum[base_mask])

                harmonic_freq_hz = fft_bpm / 60.0
                harmonic_mask = np.abs(valid_freqs - harmonic_freq_hz) < 0.1
                if np.sum(harmonic_mask) > 0:
                    harmonic_energy = np.max(valid_spectrum[harmonic_mask])

                    if base_energy >= 0.3 * harmonic_energy:
                        if MIN_HR <= peak_bpm <= MAX_HR:
                            return peak_bpm, f"谐波校正: FFT={fft_bpm:.0f}是二次谐波, 实际HR={peak_bpm:.0f}"
        if MIN_HR <= peak_bpm <= MAX_HR:
            return peak_bpm, f"谐波校正: 基于生理范围选择{peak_bpm:.0f}BPM"

    elif is_half:
        if MIN_HR <= fft_bpm <= MAX_HR:
            return fft_bpm, f"峰值漏检校正: 峰值={peak_bpm:.0f}过低, 使用FFT={fft_bpm:.0f}"

    if 0.8 <= ratio <= 1.2:
        avg_bpm = (fft_bpm + peak_bpm) / 2
        if MIN_HR <= avg_bpm <= MAX_HR:
            return avg_bpm, f"两方法一致: 平均={avg_bpm:.0f}BPM"

    fft_in_range = MIN_HR <= fft_bpm <= MAX_HR
    peak_in_range = MIN_HR <= peak_bpm <= MAX_HR

    if fft_in_range and not peak_in_range:
        return fft_bpm, f"选择FFT: {fft_bpm:.0f}在正常范围内"
    elif peak_in_range and not fft_in_range:
        return peak_bpm, f"选择峰值: {peak_bpm:.0f}在正常范围内"
    elif fft_in_range and peak_in_range:
        chosen = min(fft_bpm, peak_bpm)
        return chosen, f"差异较大: 选择较低值{chosen:.0f}BPM"
    else:
        fft_dist = min(abs(fft_bpm - MIN_HR), abs(fft_bpm - MAX_HR))
        peak_dist = min(abs(peak_bpm - MIN_HR), abs(peak_bpm - MAX_HR))
        if fft_dist < peak_dist:
            return fft_bpm, f"异常: FFT={fft_bpm:.0f}更接近正常范围"
        else:
            return peak_bpm, f"异常: 峰值={peak_bpm:.0f}更接近正常范围"


# ====================================================================
# 核心逻辑：HR / RR 计算
# ====================================================================

def simple_hr_from_ecg(ecg: np.ndarray, fs: float) -> dict:
    """
    简化版 HR 检测（基于 ECG 的斜率/能量 + 阈值）
    """
    ecg = ecg.astype(float)
    ecg_d = ecg - np.mean(ecg)

    # 5–20 Hz 带通
    b, a = butter(3, [5 / (fs / 2), 20 / (fs / 2)], btype='band')
    ecg_f = filtfilt(b, a, ecg_d)

    # 一阶差分 + 绝对值 + 0.15 s 平滑
    deriv = np.diff(ecg_f, prepend=ecg_f[0])
    rect = np.abs(deriv)
    win = int(0.15 * fs)
    if win < 1:
        win = 1
    kernel = np.ones(win) / win
    rect_s = np.convolve(rect, kernel, mode='same')

    # 阈值 + 不应期找 R 峰
    init_len = int(min(10 * fs, len(rect_s)))
    thr = 0.5 * np.max(rect_s[:init_len]) if init_len > 0 else 0.5 * np.max(rect_s)
    min_distance = int(0.3 * fs)
    if min_distance < 1:
        min_distance = 1

    peaks, _ = find_peaks(rect_s, height=thr, distance=min_distance)

    if len(peaks) < 2:
        return {
            "bpm": 0.0,
            "inst_bpm": np.array([]),
            "r_peaks": peaks,
            "rr_intervals": np.array([]),
            "signal": ecg_f,
        }

    rr_intervals = np.diff(peaks) / fs
    inst_hr = 60.0 / rr_intervals
    mean_hr = float(np.mean(inst_hr))

    return {
        "bpm": mean_hr,
        "inst_bpm": inst_hr,
        "r_peaks": peaks,
        "rr_intervals": rr_intervals,
        "signal": ecg_f,
    }


def simple_rr_from_ecg(ecg: np.ndarray, fs: float) -> dict:
    """
    从 ECG 派生呼吸（ECG-derived Respiration）
    """
    ecg = ecg.astype(float)
    ecg_d = ecg - np.mean(ecg)

    # 0.1–0.7 Hz 带通
    b, a = butter(2, [0.1 / (fs / 2), 0.7 / (fs / 2)], btype='band')
    resp = filtfilt(b, a, ecg_d)

    resp_n = (resp - np.mean(resp)) / (np.std(resp) + 1e-9)

    min_distance = int(2.0 * fs)
    if min_distance < 1:
        min_distance = 1

    peaks, _ = find_peaks(resp_n, distance=min_distance)

    if len(peaks) < 2:
        return {
            "bpm": 0.0,
            "inst_bpm": np.array([]),
            "resp_signal": resp,
            "peaks": peaks,
        }

    intervals = np.diff(peaks) / fs
    mask = (intervals >= 2.0) & (intervals <= 10.0)
    valid = intervals[mask] if np.any(mask) else intervals

    if len(valid) == 0:
        return {
            "bpm": 0.0,
            "inst_bpm": np.array([]),
            "resp_signal": resp,
            "peaks": peaks,
        }

    inst_rr = 60.0 / valid
    mean_rr = float(np.mean(inst_rr))

    return {
        "bpm": mean_rr,
        "inst_bpm": inst_rr,
        "resp_signal": resp,
        "peaks": peaks,
    }


def simple_rr_from_pcg(pcg: np.ndarray, fs: float) -> dict:
    """
    从 PCG 中估计呼吸率（RR），PCG-derived Respiration (PDR)
    """
    pcg = pcg.astype(float)
    pcg_d = pcg - np.mean(pcg)

    nyq = fs / 2.0

    # ---------- 1) 心音频带带通滤波 ----------
    low_cut = 20.0
    high_cut = min(200.0, 0.9 * nyq)
    if high_cut <= low_cut:
        low_cut = 0.1 * nyq
        high_cut = 0.45 * nyq

    Wn_low = low_cut / nyq
    Wn_high = high_cut / nyq
    b1, a1 = butter(2, [Wn_low, Wn_high], btype="band")
    pcg_hf = filtfilt(b1, a1, pcg_d)

    # ---------- 2) 心音包络：绝对值 + 0.1 s 平滑 ----------
    env = np.abs(pcg_hf)
    win = int(0.1 * fs)
    if win < 1:
        win = 1
    kernel = np.ones(win) / win
    env_s = np.convolve(env, kernel, mode="same")

    # ---------- 3) 包络中提取呼吸带 0.1–0.7 Hz ----------
    low_resp = 0.1
    high_resp = 0.7
    Wn_low_resp = low_resp / nyq
    Wn_high_resp = high_resp / nyq
    if Wn_high_resp >= 1.0:
        Wn_high_resp = 0.9
    if Wn_low_resp <= 0.0:
        Wn_low_resp = 0.01

    b2, a2 = butter(2, [Wn_low_resp, Wn_high_resp], btype="band")
    resp = filtfilt(b2, a2, env_s)

    # ---------- 4) 标准化 + 峰值检测 ----------
    resp_n = (resp - np.mean(resp)) / (np.std(resp) + 1e-9)
    min_distance = int(2.0 * fs)
    if min_distance < 1:
        min_distance = 1
    peaks, _ = find_peaks(resp_n, distance=min_distance)

    if len(peaks) < 2:
        return {
            "bpm": 0.0,
            "inst_bpm": np.array([]),
            "resp_signal": resp,
            "peaks": peaks,
        }

    intervals = np.diff(peaks) / fs
    mask = (intervals >= 2.0) & (intervals <= 10.0)
    valid = intervals[mask] if np.any(mask) else intervals

    if len(valid) == 0:
        return {
            "bpm": 0.0,
            "inst_bpm": np.array([]),
            "resp_signal": resp,
            "peaks": peaks,
        }

    inst_rr = 60.0 / valid
    mean_rr = float(np.mean(inst_rr))

    return {
        "bpm": mean_rr,
        "inst_bpm": inst_rr,
        "resp_signal": resp,
        "peaks": peaks,
    }


def calculate_hr_rr(
    ecg: np.ndarray,
    pcg: np.ndarray,
    fs: float = 125.0,
    use_pan_tompkins: bool = True,
    debug: bool = False
) -> dict:
    """
    计算 HR / RR（ECG + PCG）

    - HR：simple_hr_from_ecg + R 峰脉冲序列 FFT + 谐波校正
    - RR_ECG：simple_rr_from_ecg + FFT
    - RR_PCG：simple_rr_from_pcg + FFT
    """
    results = {}

    # ---------- HR from ECG ----------
    hr_info = simple_hr_from_ecg(ecg, fs)

    # R 峰脉冲序列
    impulse = np.zeros_like(hr_info["signal"], dtype=float)
    r_peaks = hr_info["r_peaks"]
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(impulse))]
    impulse[r_peaks] = 1.0

    hr_fft, hr_freqs, hr_spectrum = estimate_rate_fft(
        impulse, fs, (0.6, 3.0)
    )

    corrected_hr, hr_diagnosis = detect_and_correct_harmonic(
        fft_bpm=hr_fft,
        peak_bpm=hr_info["bpm"],
        spectrum=hr_spectrum,
        freqs=hr_freqs,
        freq_range=(0.6, 3.0)
    )

    results["HR"] = {
        "pan_tompkins_bpm": hr_info["bpm"],
        "fft_bpm": hr_fft,
        "corrected_bpm": corrected_hr,
        "hr_diagnosis": hr_diagnosis,
        "r_peaks": hr_info["r_peaks"],
        "rr_intervals": hr_info["rr_intervals"],
        "freqs": hr_freqs,
        "spectrum": hr_spectrum,
        "signal": hr_info["signal"],
        "ecg_integrated": None,
    }

    # ---------- RR from ECG ----------
    rr_info = simple_rr_from_ecg(ecg, fs)
    rr_fft, rr_freqs, rr_spectrum = estimate_rate_fft(
        rr_info["resp_signal"], fs, (0.1, 0.5)
    )

    results["RR"] = {
        "pcg_detect_bpm": rr_info["bpm"],   # 实际为 ECG-derived RR
        "hrv_detect_bpm": 0.0,
        "fft_bpm": rr_fft,
        "freqs": rr_freqs,
        "spectrum": rr_spectrum,
        "signal": rr_info["resp_signal"],
        "peaks": rr_info["peaks"],
    }

    # ---------- RR from PCG（验证通道） ----------
    rr_pcg_info = simple_rr_from_pcg(pcg, fs)
    rr_pcg_fft, rr_pcg_freqs, rr_pcg_spectrum = estimate_rate_fft(
        rr_pcg_info["resp_signal"], fs, (0.1, 0.5)
    )

    results["RR_PCG"] = {
        "bpm": rr_pcg_info["bpm"],
        "inst_bpm": rr_pcg_info["inst_bpm"],
        "signal": rr_pcg_info["resp_signal"],
        "peaks": rr_pcg_info["peaks"],
        "fft_bpm": rr_pcg_fft,
        "freqs": rr_pcg_freqs,
        "spectrum": rr_pcg_spectrum,
    }

    if debug:
        print(f"  调试: HR ≈ {hr_info['bpm']:.2f} BPM")
        print(f"  调试: ECG-derived RR ≈ {rr_info['bpm']:.2f} 次/分钟")
        print(f"  调试: PCG-derived RR ≈ {rr_pcg_info['bpm']:.2f} 次/分钟")
        print(f"  调试: ECG R 峰 {len(hr_info['r_peaks'])} 个, ECG 呼吸峰 {len(rr_info['peaks'])} 个, PCG 呼吸峰 {len(rr_pcg_info['peaks'])} 个")

    return results


# ====================================================================
# RR 综合选择：RR_Final_BPM
# ====================================================================

def choose_rr_from_results(results: dict) -> float:
    """
    综合 ECG / PCG、峰值与 FFT 的多个 RR 估计，选出一个更可靠的 RR_Final_BPM

    策略：
    1) 收集所有在 6–30 次/分之间的非零 RR：
       - RR_ECG_peak, RR_ECG_FFT, RR_PCG_peak, RR_PCG_FFT
    2) 计算中位数 median
    3) 丢弃与 median 相差超过 40% 的离群点
    4) 对剩余的取平均（或中位数）作为最终 RR
    """
    candidates = []

    ecg_peak = results.get("RR", {}).get("pcg_detect_bpm", 0.0)
    ecg_fft = results.get("RR", {}).get("fft_bpm", 0.0)
    pcg_peak = results.get("RR_PCG", {}).get("bpm", 0.0)
    pcg_fft = results.get("RR_PCG", {}).get("fft_bpm", 0.0)

    for x in [ecg_peak, ecg_fft, pcg_peak, pcg_fft]:
        if 6.0 <= x <= 30.0:
            candidates.append(float(x))

    if not candidates:
        return 0.0

    candidates = np.array(candidates, dtype=float)
    median = float(np.median(candidates))

    if median <= 0:
        keep = candidates
    else:
        # 与中位数差距不超过 40% 的作为一簇
        keep = candidates[np.abs(candidates - median) / median <= 0.4]

    if keep.size == 0:
        keep = candidates

    final_rr = float(np.mean(keep))
    return final_rr


# ====================================================================
# 画图函数
# ====================================================================

def plot_waveforms_60s(
    results: Dict,
    fs: float,
    target_name: str,
    segment_idx: int,
    output_dir: Path,
    file_stub: Optional[str] = None,
):
    """绘制 60 s 的呼吸波形（ECG-derived）和 HR 波形"""
    breath_signal = results["RR"]["signal"]
    heart_signal = results["HR"]["signal"]
    r_peaks = results["HR"].get("r_peaks", np.array([]))

    t_breath = np.arange(len(breath_signal)) / fs
    t_heart = np.arange(len(heart_signal)) / fs

    hr_pt = results["HR"].get("pan_tompkins_bpm", 0.0)
    hr_fft = results["HR"]["fft_bpm"]
    rr_ecg = results["RR"].get("pcg_detect_bpm", 0.0)
    rr_fft = results["RR"]["fft_bpm"]
    rr_final = choose_rr_from_results(results)

    # Respiration waveform (ECG-derived)
    fig_br, ax_br = plt.subplots(figsize=(14, 5))
    ax_br.plot(t_breath, breath_signal, linewidth=1.2, label="Respiration (ECG-derived)")
    ax_br.set_xlabel("Time (s)", fontsize=12)
    ax_br.set_ylabel("Amplitude", fontsize=12)
    ax_br.set_title(
        f"{target_name} Segment {segment_idx} - Respiration Waveform\n"
        f"RR_Final≈{rr_final:.1f} / RR_ECG: {rr_ecg:.1f} (peak) / {rr_fft:.1f} (FFT) breaths/min",
        fontsize=12,
    )
    ax_br.grid(True, alpha=0.3)
    ax_br.legend(loc="upper right")
    ax_br.set_xlim([0, max(t_breath) if len(t_breath) > 0 else 1])

    stub = file_stub or sanitize_label(target_name)
    br_path = output_dir / f"waveform_BR_{stub}_seg{segment_idx}.png"
    fig_br.tight_layout()
    fig_br.savefig(br_path, dpi=150, bbox_inches="tight")
    print(f"保存呼吸波形图: {br_path}")
    plt.close(fig_br)

    # ECG waveform with R-peaks
    fig_hr, ax_hr = plt.subplots(figsize=(14, 5))
    ax_hr.plot(t_heart, heart_signal, linewidth=0.8, label="ECG (5–20 Hz)")
    if len(r_peaks) > 0:
        valid_peaks = r_peaks[r_peaks < len(heart_signal)]
        if len(valid_peaks) > 0:
            ax_hr.scatter(
                valid_peaks / fs,
                heart_signal[valid_peaks],
                c="red",
                s=40,
                marker="v",
                label=f"R-peaks ({len(valid_peaks)})",
                zorder=5,
            )

    ax_hr.set_xlabel("Time (s)", fontsize=12)
    ax_hr.set_ylabel("Amplitude", fontsize=12)
    ax_hr.set_title(
        f"{target_name} Segment {segment_idx} - ECG Waveform with R-peaks\n"
        f"HR: {hr_pt:.1f} (slope-based) / {hr_fft:.1f} (FFT) BPM",
        fontsize=12,
    )
    ax_hr.grid(True, alpha=0.3)
    ax_hr.legend(loc="upper right")
    ax_hr.set_xlim([0, max(t_heart) if len(t_heart) > 0 else 1])

    hr_path = output_dir / f"waveform_HR_{stub}_seg{segment_idx}.png"
    fig_hr.tight_layout()
    fig_hr.savefig(hr_path, dpi=150, bbox_inches="tight")
    print(f"保存心率波形图: {hr_path}")
    plt.close(fig_hr)


def plot_comparison_waveforms(
    results_dict: Dict,
    fs: float,
    segment_idx: int,
    output_dir: Path,
    context_label: str = "",
    file_stub: Optional[str] = None,
):
    """绘制 Target1 和 Target2 的 BR/HR 对比波形图"""
    if len(results_dict) < 2:
        return

    colors = {"Target1": "blue", "Target2": "red"}

    # Respiration comparison (ECG-derived)
    fig_br, ax_br = plt.subplots(figsize=(14, 6))

    for target_name, results in results_dict.items():
        breath_signal = results["RR"]["signal"]
        t = np.arange(len(breath_signal)) / fs
        color = colors.get(target_name, "green")
        rr_final = choose_rr_from_results(results)
        ax_br.plot(
            t,
            breath_signal,
            color=color,
            linewidth=1.2,
            alpha=0.8,
            label=f"{target_name} (RR_Final: {rr_final:.1f}/min)",
        )

    ax_br.set_xlabel("Time (s)", fontsize=12)
    ax_br.set_ylabel("Amplitude", fontsize=12)
    title_suffix = f"Segment {segment_idx}"
    if context_label:
        title_suffix += f" | {context_label}"
    ax_br.set_title(f"Respiration Waveform Comparison (ECG-derived) - {title_suffix}", fontsize=14)
    ax_br.grid(True, alpha=0.3)
    ax_br.legend(loc="upper right", fontsize=11)

    stub = file_stub or (sanitize_label(context_label) if context_label else "")
    suffix = f"_seg{segment_idx}"
    if stub:
        suffix = f"_{stub}{suffix}"
    br_path = output_dir / f"waveform_BR_comparison{suffix}.png"
    fig_br.tight_layout()
    fig_br.savefig(br_path, dpi=150, bbox_inches="tight")
    print(f"保存 BR 对比图: {br_path}")
    plt.close(fig_br)

    # HR comparison
    fig_hr, ax_hr = plt.subplots(figsize=(14, 6))

    for target_name, results in results_dict.items():
        heart_signal = results["HR"]["signal"]
        t = np.arange(len(heart_signal)) / fs
        color = colors.get(target_name, "green")
        hr_bpm = results["HR"].get("pan_tompkins_bpm", results["HR"]["fft_bpm"])
        ax_hr.plot(
            t,
            heart_signal,
            color=color,
            linewidth=0.8,
            alpha=0.8,
            label=f"{target_name} (HR: {hr_bpm:.1f} BPM)",
        )

    ax_hr.set_xlabel("Time (s)", fontsize=12)
    ax_hr.set_ylabel("Amplitude", fontsize=12)
    ax_hr.set_title(f"ECG Waveform Comparison - {title_suffix}", fontsize=14)
    ax_hr.grid(True, alpha=0.3)
    ax_hr.legend(loc="upper right", fontsize=11)

    hr_path = output_dir / f"waveform_HR_comparison{suffix}.png"
    fig_hr.tight_layout()
    fig_hr.savefig(hr_path, dpi=150, bbox_inches="tight")
    print(f"保存 HR 对比图: {hr_path}")
    plt.close(fig_hr)


def plot_results(
    ecg: np.ndarray,
    pcg: np.ndarray,
    results: dict,
    fs: float,
    save_path: Path = None
):
    """绘制综合结果图（3x2 子图）"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    t_ecg = np.arange(len(ecg)) / fs
    t_pcg = np.arange(len(pcg)) / fs

    hr_pt = results["HR"].get("pan_tompkins_bpm", 0.0)
    hr_fft = results["HR"]["fft_bpm"]
    hr_corrected = results["HR"].get("corrected_bpm", hr_pt)
    hr_diagnosis = results["HR"].get("hr_diagnosis", "")
    rr_ecg_peak = results["RR"].get("pcg_detect_bpm", 0.0)
    rr_fft_ecg = results["RR"]["fft_bpm"]
    r_peaks = results["HR"].get("r_peaks", np.array([]))

    rr_pcg_peak = results.get("RR_PCG", {}).get("bpm", 0.0)
    rr_pcg_fft = results.get("RR_PCG", {}).get("fft_bpm", 0.0)
    rr_final = choose_rr_from_results(results)

    # ECG raw
    axes[0, 0].plot(t_ecg, ecg, "b-", alpha=0.7, linewidth=0.5)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title("ECG Signal (ADS1292R Raw)")
    axes[0, 0].grid(True, alpha=0.3)

    # PCG raw
    axes[0, 1].plot(t_pcg, pcg, "r-", alpha=0.7, linewidth=0.5)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].set_title("PCG Signal (ADS1292R Raw)")
    axes[0, 1].grid(True, alpha=0.3)

    # ECG filtered + R-peaks
    ecg_filtered = results["HR"]["signal"]
    t_filtered = np.arange(len(ecg_filtered)) / fs
    axes[1, 0].plot(t_filtered, ecg_filtered, "b-", linewidth=0.8)
    if len(r_peaks) > 0:
        valid_peaks = r_peaks[r_peaks < len(ecg_filtered)]
        if len(valid_peaks) > 0:
            axes[1, 0].scatter(
                valid_peaks / fs,
                ecg_filtered[valid_peaks],
                c="red",
                s=30,
                marker="v",
                zorder=5,
            )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_title(
        f"ECG Filtered + R-peaks (HR: {hr_corrected:.1f} BPM [校正], "
        f"Peak={hr_pt:.1f}, FFT={hr_fft:.1f})"
    )
    axes[1, 0].grid(True, alpha=0.3)

    # Respiration signal (ECG-derived)
    resp_signal = results["RR"]["signal"]
    t_resp = np.arange(len(resp_signal)) / fs
    axes[1, 1].plot(t_resp, resp_signal, "r-", linewidth=1)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].set_title(
        f"Respiration Signal (ECG-derived)\n"
        f"RR_Final: {rr_final:.1f}/min, "
        f"RR_ECG: {rr_ecg_peak:.1f} peak / {rr_fft_ecg:.1f} FFT /min, "
        f"RR_PCG_peak: {rr_pcg_peak:.1f} /min",
    )
    axes[1, 1].grid(True, alpha=0.3)

    # HR spectrum
    hr_freqs = results["HR"]["freqs"]
    hr_spectrum = results["HR"]["spectrum"]
    mask_hr = (hr_freqs >= 0.5) & (hr_freqs <= 3.0)
    if np.sum(mask_hr) > 0:
        axes[2, 0].plot(
            hr_freqs[mask_hr] * 60,
            hr_spectrum[mask_hr],
            "b-",
            linewidth=1.5,
        )
        axes[2, 0].axvline(hr_fft, color="r", linestyle="--", alpha=0.7, label=f"FFT: {hr_fft:.1f} BPM")
        if hr_pt > 0:
            axes[2, 0].axvline(hr_pt, color="g", linestyle=":", alpha=0.7, label=f"Peak: {hr_pt:.1f} BPM")
        axes[2, 0].axvline(hr_corrected, color="purple", linestyle="-", linewidth=2,
                           label=f"校正: {hr_corrected:.1f} BPM")
    axes[2, 0].set_xlabel("Frequency (BPM)")
    axes[2, 0].set_ylabel("Magnitude")
    axes[2, 0].set_title(f"Heart Rate Spectrum ({hr_diagnosis})")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    # RR spectrum (ECG + PCG 标记)
    rr_freqs_ecg = results["RR"]["freqs"]
    rr_spectrum_ecg = results["RR"]["spectrum"]
    mask_rr = (rr_freqs_ecg >= 0.05) & (rr_freqs_ecg <= 1.0)
    if np.sum(mask_rr) > 0:
        axes[2, 1].plot(
            rr_freqs_ecg[mask_rr] * 60,
            rr_spectrum_ecg[mask_rr],
            "r-",
            linewidth=1.5,
            label="ECG-derived spectrum",
        )
        axes[2, 1].axvline(rr_fft_ecg, color="b", linestyle="--",
                           label=f"ECG FFT: {rr_fft_ecg:.1f}/min")
        if rr_ecg_peak > 0:
            axes[2, 1].axvline(rr_ecg_peak, color="g", linestyle=":",
                               label=f"ECG Peak: {rr_ecg_peak:.1f}/min")

    if rr_pcg_fft > 0:
        axes[2, 1].axvline(rr_pcg_fft, color="purple", linestyle="--",
                           label=f"PCG FFT: {rr_pcg_fft:.1f}/min")
    if rr_pcg_peak > 0:
        axes[2, 1].axvline(rr_pcg_peak, color="orange", linestyle=":",
                           label=f"PCG Peak: {rr_pcg_peak:.1f}/min")

    axes[2, 1].set_xlabel("Frequency (breaths/min)")
    axes[2, 1].set_ylabel("Magnitude")
    axes[2, 1].set_title("Respiration Rate Spectrum (ECG & PCG)")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"保存图像: {save_path}")
    else:
        plt.show()

    plt.close()


# ====================================================================
# main
# ====================================================================

def main():
    # 默认参数
    DEFAULT_TARGET = None
    DEFAULT_BOTH = True
    DEFAULT_CSV = None
    DEFAULT_FS = 125.0
    DEFAULT_USE_PAN_TOMPKINS = True
    # 输出目录：基于脚本位置，确保从任意目录运行时路径正确
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_OUTPUT = SCRIPT_DIR / "results"
    DEFAULT_DEBUG = False
    DEFAULT_NUM_SAMPLES = 7500     # 60 s @ 125 Hz
    DEFAULT_SEGMENT = 0
    DEFAULT_ALL_SEGMENTS = False
    DEFAULT_FREQ_BAND = "3GHZ"

    parser = argparse.ArgumentParser(
        description="从验证集 CSV 计算心率和呼吸率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 处理单个文件
  python calculate_hr_rr_from_validation.py --csv path/to/file.csv

  # 2. 处理默认的 Target1 和 Target2（demo/validate 目录）
  python calculate_hr_rr_from_validation.py

  # 3. 扫描数据集，处理 3GHZ 频段的所有 position、所有记录
  python calculate_hr_rr_from_validation.py --scan --freq-band 3GHZ

  # 4. 扫描数据集，只处理 position 1-3 的第 1 次记录
  python calculate_hr_rr_from_validation.py --scan --freq-band 3GHZ --positions 1-3 --measurements 1

  # 5. 处理所有频段（需要分别运行）
  python calculate_hr_rr_from_validation.py --scan --freq-band 2GHZ
  python calculate_hr_rr_from_validation.py --scan --freq-band 2_5GHZ
  python calculate_hr_rr_from_validation.py --scan --freq-band 3GHZ
        """
    )
    parser.add_argument("--csv", type=str, default=None, help="验证集 CSV 文件路径（或使用 --target 自动选择）")
    parser.add_argument("--target", type=int, choices=[1, 2], default=None, help="目标编号（1 或 2）")
    parser.add_argument("--both", action="store_true", default=None, help="同时处理 Target1 和 Target2")
    parser.add_argument("--fs", type=float, default=None, help="采样率 (Hz)")
    parser.add_argument("--no-pan-tompkins", action="store_true", default=None,
                        help="不使用 Pan-Tompkins（保留参数，当前实现已不使用 PT）")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="每段样本数（默认 7500，对应 60 秒 @ 125Hz）")
    parser.add_argument("--segment", type=int, default=None, help="段索引（0 表示第一段）")
    parser.add_argument("--all-segments", action="store_true", default=None, help="处理所有段")
    parser.add_argument("--scan", "--scan-radar-dataset", action="store_true", dest="scan_radar_dataset",
                        help="从 FMCW Radar 数据集按频段/位置批量读取 Target1/2")
    parser.add_argument("--radar-root", type=str, default=None,
                        help="FMCW Radar 日志根目录（默认读取仓库内 1_AsymmetricalPosition/2_Log_data）")
    parser.add_argument("--freq-band", type=str, default=None,
                        help="频段（与文件名一致，例如 3GHZ / 2_5GHZ / 2GHZ）")
    parser.add_argument("--positions", type=str, default=None,
                        help="限定位置，格式如 1,3-5 或 all（默认处理所有 position）")
    parser.add_argument("--measurements", type=str, default=None,
                        help="限定记录编号（括号内数字），格式同 --positions（默认处理所有记录）")

    args = parser.parse_args()

    target = args.target if args.target is not None else DEFAULT_TARGET
    both = args.both if args.both is not None else DEFAULT_BOTH
    if target is not None:
        both = False
    csv_path = args.csv if args.csv is not None else DEFAULT_CSV
    fs = args.fs if args.fs is not None else DEFAULT_FS
    use_pan_tompkins = not args.no_pan_tompkins if args.no_pan_tompkins is not None else DEFAULT_USE_PAN_TOMPKINS
    output = Path(args.output) if args.output is not None else DEFAULT_OUTPUT
    debug = args.debug if args.debug else DEFAULT_DEBUG
    num_samples = args.num_samples if args.num_samples is not None else DEFAULT_NUM_SAMPLES
    segment_idx = args.segment if args.segment is not None else DEFAULT_SEGMENT
    all_segments = args.all_segments if args.all_segments is not None else DEFAULT_ALL_SEGMENTS

    project_root = Path(__file__).resolve().parents[1]
    radar_default_dir = project_root / "FMCW radar-based multi-person vital sign monitoring data" / \
        "1_AsymmetricalPosition" / "2_Log_data"

    radar_root = Path(args.radar_root) if args.radar_root else radar_default_dir
    freq_band = args.freq_band.upper() if args.freq_band else DEFAULT_FREQ_BAND
    positions_filter = parse_index_expression(args.positions)
    measurements_filter = parse_index_expression(args.measurements)
    scan_radar_dataset = args.scan_radar_dataset or any(
        opt is not None for opt in (args.radar_root, args.freq_band, args.positions, args.measurements)
    )

    base_path = Path(__file__).parent
    validate_dir = base_path / "validate"

    if both:
        target_names = ["Target1", "Target2"]
    elif target is not None:
        target_names = [f"Target{target}"]
    else:
        target_names = ["Target1"]

    csv_entries: list[dict] = []

    if csv_path:
        csv_entries = [{
            "target": "Custom",
            "csv_path": Path(csv_path),
            "position": None,
            "measurement": None,
            "freq_band": freq_band,
        }]
    elif scan_radar_dataset:
        csv_entries = discover_radar_csvs(
            radar_root=radar_root,
            freq_band=freq_band,
            target_names=target_names,
            positions=positions_filter,
            measurement_ids=measurements_filter,
        )
    else:
        for tgt in target_names:
            csv_file = validate_dir / f"log_{tgt}_3GHZ_position1_ (1).csv"
            csv_entries.append({
                "target": tgt,
                "csv_path": csv_file,
                "position": None,
                "measurement": None,
                "freq_band": DEFAULT_FREQ_BAND,
            })

    if not csv_entries:
        print("⚠️  未找到可处理的 CSV 文件，请检查参数设置")
        return

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    segment_results_collection: dict[tuple, dict] = {}

    for entry in csv_entries:
        target_name = entry.get("target", "Target")
        csv_path = Path(entry["csv_path"])
        freq_label = entry.get("freq_band")
        position_idx = entry.get("position")
        measurement_idx = entry.get("measurement")

        title_parts = [target_name]
        if freq_label:
            title_parts.append(freq_label)
        if position_idx is not None:
            title_parts.append(f"Pos{position_idx}")
        if measurement_idx is not None:
            title_parts.append(f"Rec{measurement_idx}")

        entry_label = " | ".join(title_parts)
        entry_stub = sanitize_label("_".join(title_parts))

        if not csv_path.exists():
            print(f"⚠️  警告: 文件不存在: {csv_path}")
            continue

        print(f"\n{'='*60}")
        print(f"处理 {entry_label}: {csv_path}")
        print(f"{'='*60}")

        df_full = pd.read_csv(str(csv_path), skiprows=2, names=["ECG", "PCG"])
        total_samples = len(df_full)
        loop_cnt = total_samples // num_samples

        print(f"CSV 总样本数: {total_samples}")
        print(f"每段样本数: {num_samples}")
        print(f"总段数: {loop_cnt}")
        print(f"每段时长: {num_samples / fs:.1f} 秒")
        print(f"采样率: {fs} Hz")

        if loop_cnt == 0:
            loop_cnt = 1

        if all_segments:
            segments_to_process = list(range(loop_cnt))
        else:
            if segment_idx >= loop_cnt:
                print(f"⚠️  警告: 段索引 {segment_idx} 超出范围（最大 {loop_cnt-1}），使用最后一段")
                segments_to_process = [loop_cnt - 1]
            else:
                segments_to_process = [segment_idx]

        for seg_idx in segments_to_process:
            seg_label = f"{entry_stub}_seg{seg_idx}"

            print(f"\n--- 处理第 {seg_idx + 1}/{loop_cnt} 段 "
                  f"(样本 {seg_idx * num_samples} 到 {(seg_idx + 1) * num_samples - 1}) ---")

            ecg, pcg = load_validation_csv(str(csv_path), num_samples=num_samples, segment_idx=seg_idx)
            print(f"加载样本数: {len(ecg)}")
            print(f"信号时长: {len(ecg) / fs:.1f} 秒")

            print("计算心率和呼吸率（ECG HR + ECG/PCG RR）...")
            results = calculate_hr_rr(ecg, pcg, fs=fs, use_pan_tompkins=use_pan_tompkins, debug=debug)

            hr_corrected = results["HR"].get("corrected_bpm", results["HR"]["pan_tompkins_bpm"])
            hr_diagnosis = results["HR"].get("hr_diagnosis", "")
            rr_final = choose_rr_from_results(results)

            print(f"\n{seg_label} 计算结果:")
            print(f"  心率 (HR) - 从 ECG:")
            print(f"    Slope-based: {results['HR']['pan_tompkins_bpm']:.2f} BPM")
            print(f"    检测到 R 峰数量: {len(results['HR']['r_peaks'])}")
            print(f"    FFT 估计(脉冲序列): {results['HR']['fft_bpm']:.2f} BPM")
            print(f"    校正后 HR: {hr_corrected:.2f} BPM, 诊断: {hr_diagnosis}")

            print(f"  呼吸率 (RR) - ECG-derived respiration:")
            print(f"    Peak 估计:  {results['RR']['pcg_detect_bpm']:.2f} 次/分钟")
            print(f"    FFT 估计:   {results['RR']['fft_bpm']:.2f} 次/分钟")

            print(f"  呼吸率 (RR) - PCG-derived respiration:")
            print(f"    Peak 估计:  {results['RR_PCG']['bpm']:.2f} 次/分钟")
            print(f"    FFT 估计:   {results['RR_PCG']['fft_bpm']:.2f} 次/分钟")

            print(f"  综合 RR_Final: {rr_final:.2f} 次/分钟")

            all_results.append({
                "target": target_name,
                "freq_band": freq_label,
                "position": position_idx,
                "measurement": measurement_idx,
                "csv_file": csv_path.name,
                "segment": seg_idx,
                "duration_sec": num_samples / fs,
                # 作为“真值”的最终 HR / RR
                "HR_Ref_BPM": hr_corrected,  # 原 HR_Corrected_BPM
                "RR_Ref_BPM": rr_final,  # 原 RR_Final_BPM
                # 一个简单的质量指标
                "R_peak_count": len(results["HR"]["r_peaks"]),
            })

            if all_segments:
                plot_path = output_dir / f"hr_rr_validation_{entry_stub}_seg{seg_idx}.png"
            else:
                plot_path = output_dir / f"hr_rr_validation_{entry_stub}.png"
            plot_results(ecg, pcg, results, fs, save_path=plot_path)

            plot_waveforms_60s(results, fs, entry_label, seg_idx, output_dir, file_stub=entry_stub)

            key = (freq_label, position_idx, measurement_idx, seg_idx)
            if key not in segment_results_collection:
                segment_results_collection[key] = {}
            segment_results_collection[key][target_name] = results

    if len(segment_results_collection) > 0:
        print(f"\n{'='*60}")
        print("绘制 Target1 与 Target2 对比图")
        print(f"{'='*60}")
        for key, results_dict in segment_results_collection.items():
            freq_label, position_idx, measurement_idx, seg_idx = key
            if len(results_dict) >= 2:
                context_parts = []
                if freq_label:
                    context_parts.append(freq_label)
                if position_idx is not None:
                    context_parts.append(f"Pos{position_idx}")
                if measurement_idx is not None:
                    context_parts.append(f"Rec{measurement_idx}")
                context_label = " | ".join(context_parts)
                context_stub = sanitize_label("_".join(context_parts)) if context_parts else f"seg{seg_idx}"
                print(f"\n段 {seg_idx}: {context_label or '无额外上下文'} - 对比 {list(results_dict.keys())}")
                plot_comparison_waveforms(
                    results_dict,
                    fs,
                    seg_idx,
                    output_dir,
                    context_label=context_label,
                    file_stub=context_stub,
                )

    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # 根据是否扫描数据集，调整输出文件名
        if scan_radar_dataset:
            csv_out = output_dir / f"hr_rr_from_validation_{freq_band}_summary.csv"
        else:
            csv_out = output_dir / "hr_rr_from_validation_summary.csv"
        
        results_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
        print(f"\n{'='*60}")
        print("汇总结果")
        print(f"{'='*60}")
        
        # 如果记录太多，只显示前 20 条
        if len(results_df) > 20:
            print(results_df.head(20).to_string(index=False))
            print(f"... 共 {len(results_df)} 条记录，只显示前 20 条 ...")
        else:
            print(results_df.to_string(index=False))
        
        print(f"\n保存汇总结果到: {csv_out}")
        
        # 如果是批量处理，输出简要统计
        if scan_radar_dataset and len(results_df) > 1:
            print(f"\n{'='*60}")
            print("批量处理统计")
            print(f"{'='*60}")
            print(f"频段: {freq_band}")
            print(f"总记录数: {len(results_df)}")
            
            if 'position' in results_df.columns:
                positions_processed = results_df['position'].dropna().unique()
                print(f"处理的 position: {sorted(positions_processed)}")
            
            if 'target' in results_df.columns:
                for tgt in sorted(results_df['target'].unique()):
                    tgt_df = results_df[results_df['target'] == tgt]
                    hr_mean = tgt_df['HR_Ref_BPM'].mean()
                    hr_std = tgt_df['HR_Ref_BPM'].std()
                    rr_mean = tgt_df['RR_Ref_BPM'].mean()
                    rr_std = tgt_df['RR_Ref_BPM'].std()
                    print(f"\n{tgt} ({len(tgt_df)} 条记录):")
                    print(f"  HR: {hr_mean:.1f} ± {hr_std:.1f} BPM")
                    print(f"  RR: {rr_mean:.1f} ± {rr_std:.1f} /min")


if __name__ == "__main__":
    main()
