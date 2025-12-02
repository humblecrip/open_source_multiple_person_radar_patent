import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DCALoadResult:
    # Complex IQ data with shape: [num_frames, num_chirps_per_frame, num_adc_samples, num_virtual_rx]
    frames: np.ndarray
    num_frames: int
    num_chirps_per_frame: int
    num_adc_samples: int
    num_rx: int
    num_tx: int
    num_virtual_rx: int


class DCABinaryReader:
    """
    DCA1000 .bin 读取器（兼容 TI IWR6843ISK TDM-MIMO）。

    - 假设 ADC 为 16bit，复数模式（I/Q）。
    - 默认按 Matlab readDCA1000.m 的复数组装方式解包：
      以 4 个 int16 为一组，组装为两个 complex16：
         c0 = adc[i] + j*adc[i+2]
         c1 = adc[i+1] + j*adc[i+3]
    - 支持 1/2/3 TX 的 TDM-MIMO，虚拟通道数 = num_rx * num_tx。
    """

    def __init__(
        self,
        num_adc_samples: int,
        num_rx: int,
        num_tx: int,
        num_chirps_per_frame: int,
    ) -> None:
        self.num_adc_samples = int(num_adc_samples)
        self.num_rx = int(num_rx)
        self.num_tx = int(num_tx)
        self.num_chirps_per_frame = int(num_chirps_per_frame)

    def _decode_complex_pairs(self, raw: np.ndarray) -> np.ndarray:
        # raw: flat int16 array
        # 4 int16 => 2 complex samples
        n = (raw.size // 4) * 4
        raw = raw[:n].astype(np.int16)
        a = raw.reshape(-1, 4)
        # 依照 Matlab readDCA1000.m 的复数装配
        re0 = a[:, 0].astype(np.float32)
        im0 = a[:, 2].astype(np.float32)
        re1 = a[:, 1].astype(np.float32)
        im1 = a[:, 3].astype(np.float32)
        c0 = re0 + 1j * im0
        c1 = re1 + 1j * im1
        c = np.empty((a.shape[0] * 2,), dtype=np.complex64)
        c[0::2] = c0
        c[1::2] = c1
        return c

    def _reshape_lvds(
        self, cplx_stream: np.ndarray
    ) -> Tuple[np.ndarray, int, int, int, int]:
        """
        将一维复数流 reshape 为 [num_chirps_total, num_adc_samples, num_virtual_rx]
        其中 num_virtual_rx = num_rx * num_tx
        """
        v_rx = self.num_rx * self.num_tx
        # 总 chirp 数（跨所有 TX 的 TDM-MIMO）
        num_chirps_total = cplx_stream.size // (self.num_adc_samples * v_rx)
        if num_chirps_total * self.num_adc_samples * v_rx != cplx_stream.size:
            raise ValueError(
                f"数据长度与配置不匹配: len={cplx_stream.size}, "
                f"adc={self.num_adc_samples}, v_rx={v_rx}"
            )
        data = cplx_stream.reshape(num_chirps_total, self.num_adc_samples, v_rx)
        return data, num_chirps_total, self.num_adc_samples, v_rx, self.num_tx

    def _group_frames(self, data: np.ndarray) -> np.ndarray:
        """
        将 [num_chirps_total, num_adc_samples, v_rx]
        分组为 [num_frames, num_chirps_per_frame, num_adc_samples, v_rx]
        """
        if data.shape[0] % self.num_chirps_per_frame != 0:
            # 截断到整帧
            n_frames = data.shape[0] // self.num_chirps_per_frame
            data = data[: n_frames * self.num_chirps_per_frame, :, :]
        n_frames = data.shape[0] // self.num_chirps_per_frame
        frames = data.reshape(
            n_frames, self.num_chirps_per_frame, self.num_adc_samples, data.shape[2]
        )
        return frames

    def load(self, path: str) -> DCALoadResult:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        raw = np.fromfile(path, dtype=np.int16)
        cplx = self._decode_complex_pairs(raw)
        chirps, n_chirps_total, num_adc, v_rx, _ = self._reshape_lvds(cplx)
        frames = self._group_frames(chirps)
        return DCALoadResult(
            frames=frames,
            num_frames=frames.shape[0],
            num_chirps_per_frame=self.num_chirps_per_frame,
            num_adc_samples=num_adc,
            num_rx=self.num_rx,
            num_tx=self.num_tx,
            num_virtual_rx=v_rx,
        )






