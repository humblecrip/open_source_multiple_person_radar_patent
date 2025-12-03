"""
配置文件 - 不同雷达和场景的预设配置
Configuration Profiles for Different Radars and Scenarios
"""

from universal_detector import DetectorConfig


class ConfigProfiles:
    """
    预设配置文件
    """

    @staticmethod
    def iwr6843isk_default():
        """
        IWR6843ISK 默认配置

        硬件规格:
        - 4 RX天线
        - 3 TX天线
        - 最大虚拟天线: 12 (4x3)
        - 理论最大可检测目标: ~10-15个（取决于场景）
        - 实际推荐: 4-8个目标
        """
        return DetectorConfig(
            # 硬件参数
            num_rx=4,
            num_tx=1,  # 实际使用的TX数量
            num_adc_samples=200,
            num_chirps_per_frame=1,

            # 检测参数
            cfar_guard_len=2,
            cfar_noise_len=8,
            cfar_threshold_scale=3.0,  # 降低以提高检测灵敏度
            min_target_distance_m=0.5,  # 最小目标间距
            max_targets=8,  # IWR6843ISK推荐最大目标数

            # 波束成形参数
            num_azimuth_bins=64,
            angle_range_deg=60,

            # 生命体征参数（优化后）
            breath_freq_min=0.15,  # 9 BPM
            breath_freq_max=0.5,   # 30 BPM
            heart_freq_min=0.8,    # 48 BPM
            heart_freq_max=3.0,    # 180 BPM (降低上限避免谐波)

            # 质量控制
            min_snr_db=10.0,  # 降低SNR要求
            min_confidence=1.2  # 降低置信度要求
        )

    @staticmethod
    def iwr6843isk_high_density():
        """
        IWR6843ISK 高密度场景配置
        适用于多人密集场景
        """
        config = ConfigProfiles.iwr6843isk_default()
        config.max_targets = 12  # 增加最大目标数
        config.min_target_distance_m = 0.3  # 减小最小间距
        config.cfar_threshold_scale = 2.5  # 更敏感的检测
        config.min_snr_db = 8.0  # 更低的SNR要求
        return config

    @staticmethod
    def iwr6843isk_single_chirp():
        """
        IWR6843ISK 单chirp配置
        适用于当前数据集
        """
        config = ConfigProfiles.iwr6843isk_default()
        config.num_chirps_per_frame = 1
        config.max_targets = 4  # 单chirp限制目标数
        return config

    @staticmethod
    def iwr6843isk_multi_chirp():
        """
        IWR6843ISK 多chirp配置
        适用于标准TI固件配置
        """
        config = ConfigProfiles.iwr6843isk_default()
        config.num_chirps_per_frame = 16  # 典型配置
        config.max_targets = 10  # 多chirp可检测更多目标
        config.cfar_threshold_scale = 3.5  # 更严格的检测
        return config

    @staticmethod
    def custom(
        max_targets=8,
        min_target_distance=0.5,
        cfar_threshold=3.0,
        breath_range=(0.15, 0.5),
        heart_range=(0.8, 3.0),
        min_snr=10.0
    ):
        """
        自定义配置

        Args:
            max_targets: 最大目标数
            min_target_distance: 最小目标间距(m)
            cfar_threshold: CFAR阈值倍数
            breath_range: 呼吸频率范围(Hz) tuple
            heart_range: 心率频率范围(Hz) tuple
            min_snr: 最小SNR(dB)
        """
        return DetectorConfig(
            num_rx=4,
            num_tx=1,
            num_adc_samples=200,
            num_chirps_per_frame=1,
            cfar_threshold_scale=cfar_threshold,
            min_target_distance_m=min_target_distance,
            max_targets=max_targets,
            breath_freq_min=breath_range[0],
            breath_freq_max=breath_range[1],
            heart_freq_min=heart_range[0],
            heart_freq_max=heart_range[1],
            min_snr_db=min_snr
        )


# 使用示例
if __name__ == "__main__":
    print("="*60)
    print("IWR6843ISK 配置文件")
    print("="*60)
    print()

    configs = {
        "默认配置": ConfigProfiles.iwr6843isk_default(),
        "高密度场景": ConfigProfiles.iwr6843isk_high_density(),
        "单chirp模式": ConfigProfiles.iwr6843isk_single_chirp(),
        "多chirp模式": ConfigProfiles.iwr6843isk_multi_chirp()
    }

    for name, config in configs.items():
        print(f"📋 {name}:")
        print(f"   最大目标数: {config.max_targets}")
        print(f"   Chirps/Frame: {config.num_chirps_per_frame}")
        print(f"   CFAR阈值: {config.cfar_threshold_scale}")
        print(f"   最小目标间距: {config.min_target_distance_m}m")
        print(f"   呼吸范围: {config.breath_freq_min*60:.0f}-{config.breath_freq_max*60:.0f} BPM")
        print(f"   心率范围: {config.heart_freq_min*60:.0f}-{config.heart_freq_max*60:.0f} BPM")
        print()
