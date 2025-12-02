import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

C_LIGHT = 299792458.0


def _count_bits(value: int) -> int:
    return bin(value).count("1")


def _pow2_roundup(x: int) -> int:
    y = 1
    while y < x:
        y <<= 1
    return y


@dataclass(frozen=True)
class VitalSignsConfig:
    start_range_m: float = 0.3
    end_range_m: float = 0.9
    win_len_breath: int = 256
    win_len_heart: int = 512
    rx_antenna_process: float = 4.0
    alpha_breath: float = 0.1
    alpha_heart: float = 0.05
    scale_breath: float = 1.0
    scale_heart: float = 1.0


@dataclass(frozen=True)
class MotionDetectionConfig:
    enabled: bool = False
    block_size: int = 32
    threshold: float = 2.0
    gain_control: bool = False


@dataclass(frozen=True)
class RadarConfig:
    num_rx: int
    num_tx: int
    num_adc_samples: int
    sample_rate_ksps: float
    freq_slope_mhz_per_us: float
    start_freq_ghz: float
    idle_time_us: float
    ramp_end_time_us: float
    frame_periodicity_ms: float
    num_loops: int
    num_chirps_per_frame: int
    num_range_bins: int
    num_doppler_bins: int
    range_resolution_m: float
    range_idx_to_m: float
    doppler_resolution_mps: float
    angle_bins: int = 48
    diag_load_factor: float = 1e-3
    heatmap_window_len: int = 8

    def override(self, **kwargs: float) -> "RadarConfig":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / (self.start_freq_ghz * 1e9)


def _init_defaults() -> Dict[str, Optional[float]]:
    return {
        "numRxAnt": None,
        "numTxAnt": None,
        "numAdcSamples": None,
        "digOutSampleRate": None,
        "freqSlopeConst": None,
        "startFreq": None,
        "idleTime": None,
        "rampEndTime": None,
        "numLoops": None,
        "chirpStartIdx": None,
        "chirpEndIdx": None,
        "framePeriodicity": None,
        "rangeAzimuthHeatMap": 16,
        "numAngleBins": 48,
        "oddemoWindowLen": 8,
        "oddemoDiagLoadFactor": 1e-3,
    }


def parse_cli_config(path: str) -> Tuple[RadarConfig, VitalSignsConfig, MotionDetectionConfig]:
    params = _init_defaults()
    vital = None
    motion = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            tokens = line.split()
            key = tokens[0]

            if key == "channelCfg":
                rx_mask = int(tokens[1])
                tx_mask = int(tokens[2])
                params["numRxAnt"] = _count_bits(rx_mask)
                params["numTxAnt"] = _count_bits(tx_mask)
            elif key == "profileCfg":
                params["startFreq"] = float(tokens[2])
                params["idleTime"] = float(tokens[3])
                params["rampEndTime"] = float(tokens[5])
                params["freqSlopeConst"] = float(tokens[8])
                params["numAdcSamples"] = int(tokens[10])
                params["digOutSampleRate"] = float(tokens[11])
            elif key == "frameCfg":
                params["chirpStartIdx"] = int(tokens[1])
                params["chirpEndIdx"] = int(tokens[2])
                params["numLoops"] = int(tokens[3])
                params["framePeriodicity"] = float(tokens[5])
            elif key == "guiMonitor":
                if len(tokens) > 2:
                    params["rangeAzimuthHeatMap"] = int(tokens[2])
            elif key == "oddemoParms":
                params["oddemoWindowLen"] = int(tokens[1])
                params["oddemoDiagLoadFactor"] = float(tokens[2])
            elif key == "vitalSignsCfg":
                vital = VitalSignsConfig(
                    start_range_m=float(tokens[1]),
                    end_range_m=float(tokens[2]),
                    win_len_breath=int(tokens[3]),
                    win_len_heart=int(tokens[4]),
                    rx_antenna_process=float(tokens[5]),
                    alpha_breath=float(tokens[6]),
                    alpha_heart=float(tokens[7]),
                    scale_breath=float(tokens[8]),
                    scale_heart=float(tokens[9]),
                )
            elif key == "motionDetection":
                motion = MotionDetectionConfig(
                    enabled=bool(int(tokens[1])),
                    block_size=int(tokens[2]),
                    threshold=float(tokens[3]),
                    gain_control=bool(int(tokens[4])),
                )

    missing = [k for k, v in params.items() if v is None and k not in ("numAngleBins",)]
    if missing:
        raise ValueError(f"配置文件缺少字段: {missing}")

    num_chirps_per_frame = ((params["chirpEndIdx"] - params["chirpStartIdx"] + 1) * params["numLoops"])
    num_doppler = num_chirps_per_frame // max(1, params["numTxAnt"] or 1)
    num_range_bins = _pow2_roundup(params["numAdcSamples"])
    dig_rate_hz = params["digOutSampleRate"] * 1e3
    slope_hz_per_s = params["freqSlopeConst"] * 1e12
    range_res = (C_LIGHT * dig_rate_hz) / (2 * slope_hz_per_s * params["numAdcSamples"])
    range_idx_to_m = (C_LIGHT * dig_rate_hz) / (2 * slope_hz_per_s * num_range_bins)
    doppler_res = C_LIGHT / (
        2
        * (params["startFreq"] * 1e9)
        * (params["idleTime"] + params["rampEndTime"])
        * 1e-6
        * num_doppler
        * max(1, params["numTxAnt"])
    )

    radar = RadarConfig(
        num_rx=params["numRxAnt"],
        num_tx=params["numTxAnt"],
        num_adc_samples=params["numAdcSamples"],
        sample_rate_ksps=params["digOutSampleRate"],
        freq_slope_mhz_per_us=params["freqSlopeConst"],
        start_freq_ghz=params["startFreq"],
        idle_time_us=params["idleTime"],
        ramp_end_time_us=params["rampEndTime"],
        frame_periodicity_ms=params["framePeriodicity"],
        num_loops=params["numLoops"],
        num_chirps_per_frame=num_chirps_per_frame,
        num_range_bins=num_range_bins,
        num_doppler_bins=int(num_doppler),
        range_resolution_m=range_res,
        range_idx_to_m=range_idx_to_m,
        doppler_resolution_mps=doppler_res,
        angle_bins=params["numAngleBins"],
        diag_load_factor=params["oddemoDiagLoadFactor"],
        heatmap_window_len=params["oddemoWindowLen"],
    )

    if vital is None:
        vital = VitalSignsConfig()
    if motion is None:
        motion = MotionDetectionConfig()
    return radar, vital, motion


