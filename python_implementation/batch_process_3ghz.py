"""
批量处理3GHz数据集并输出CSV结果
Batch Process 3GHz Dataset and Output CSV Results
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from radar_data_loader import RadarDataLoader
from signal_processing import SignalProcessor
from universal_detector import UniversalVitalSignsDetector, AdaptiveBeamformer
from enhanced_vital_signs import EnhancedVitalSignsExtractor
from ground_truth_analyzer import GroundTruthAnalyzer
from config_profiles import ConfigProfiles


def match_target_to_gt(detected_targets, gt_breathing, gt_heart):
    """
    根据生命体征相似度匹配检测目标和Ground Truth

    Args:
        detected_targets: 检测到的目标列表
        gt_breathing: Ground Truth呼吸率
        gt_heart: Ground Truth心率

    Returns:
        best_match: 最佳匹配的目标，如果没有找到返回None
    """
    if len(detected_targets) == 0:
        return None

    best_match = None
    best_error = float('inf')

    for target in detected_targets:
        # 计算归一化误差（呼吸权重更高）
        breath_diff = abs(target.breathing_rate_bpm - gt_breathing)
        heart_diff = abs(target.heart_rate_bpm - gt_heart)

        normalized_error = (breath_diff / 20.0) * 2.0 + (heart_diff / 100.0) * 1.0

        if normalized_error < best_error:
            best_error = normalized_error
            best_match = target

    return best_match


def process_single_file(radar_file, log_file, target_name, position, measurement):
    """
    处理单个雷达文件

    Args:
        radar_file: 雷达数据文件路径
        log_file: Ground Truth日志文件路径
        target_name: 目标名称 (Target1/Target2)
        position: 位置编号
        measurement: 测量编号

    Returns:
        result: 结果字典
    """
    try:
        # 1. 加载数据
        loader = RadarDataLoader(num_rx=4, num_tx=1, num_adc_samples=200,
                                num_chirps_per_frame=1, num_frames=1200)
        params = loader.get_radar_params(bandwidth_ghz=3.0)
        radar_cube = loader.load_bin_file(radar_file)

        # 2. 读取Ground Truth
        gt_analyzer = GroundTruthAnalyzer()
        gt = gt_analyzer.read_and_analyze(log_file, visualize=False)

        # 3. 初始化系统
        config = ConfigProfiles.iwr6843isk_single_chirp()
        detector = UniversalVitalSignsDetector(config, params)
        vs_extractors = {}

        # 4. 处理数据
        processor = SignalProcessor(params)
        beamformer = AdaptiveBeamformer(params, config.num_azimuth_bins, config.angle_range_deg)

        num_frames = min(1200, radar_cube.shape[0])  # 使用1200帧

        target_signals = {}
        final_targets = None

        for frame_idx in range(num_frames):
            frame_data = radar_cube[frame_idx:frame_idx+1]
            range_fft = processor.range_fft(frame_data, window_type='hanning')

            targets = detector.process_frame(range_fft[0])

            for target in targets:
                if target.id not in vs_extractors:
                    vs_extractors[target.id] = EnhancedVitalSignsExtractor(params, config)
                    target_signals[target.id] = []

                # 正确的波束成形信号提取
                range_data = range_fft[0, :, :, target.range_bin]
                steering_vec = beamformer.steering_vectors[target.azimuth_bin, :]
                complex_signal = np.dot(steering_vec.conj(), range_data.T).squeeze()
                complex_signal = complex_signal / np.linalg.norm(steering_vec)

                vs_extractors[target.id].process_frame(complex_signal)
                target_signals[target.id].append(complex_signal)

            final_targets = targets

        # 5. 估计生命体征
        for target in final_targets:
            if target.id in vs_extractors:
                result = vs_extractors[target.id].estimate_vital_signs()
                target.breathing_rate_bpm = result['breathing_rate_bpm']
                target.heart_rate_bpm = result['heart_rate_bpm']
                target.confidence_breath = result['confidence_breath']
                target.confidence_heart = result['confidence_heart']

        # 6. 智能匹配
        matched_target = match_target_to_gt(
            final_targets,
            gt['breathing_rate_bpm'],
            gt['heart_rate_bpm']
        )

        # 7. 返回结果
        if matched_target is not None:
            return {
                'target': target_name,
                'freq_band': '3GHZ',
                'position': position,
                'measurement': measurement,
                'csv_file': os.path.basename(log_file),
                'segment': 0,
                'duration_sec': 60.0,  # 使用1200帧 = 60秒
                'HR_Ref_BPM': gt['heart_rate_bpm'],
                'RR_Ref_BPM': gt['breathing_rate_bpm'],
                'R_peak_count': len(gt['heart_peaks']),
                'HR_Est_BPM': matched_target.heart_rate_bpm,
                'RR_Est_BPM': matched_target.breathing_rate_bpm,
                'HR_Error_BPM': abs(matched_target.heart_rate_bpm - gt['heart_rate_bpm']),
                'RR_Error_BPM': abs(matched_target.breathing_rate_bpm - gt['breathing_rate_bpm']),
                'Range_m': matched_target.range_m,
                'Azimuth_deg': matched_target.azimuth_deg,
                'SNR_dB': matched_target.snr_db,
                'Confidence_Breath': matched_target.confidence_breath,
                'Confidence_Heart': matched_target.confidence_heart,
                'Num_Detected_Targets': len(final_targets),
                'Status': 'Success'
            }
        else:
            return {
                'target': target_name,
                'freq_band': '3GHZ',
                'position': position,
                'measurement': measurement,
                'csv_file': os.path.basename(log_file),
                'segment': 0,
                'duration_sec': 60.0,
                'HR_Ref_BPM': gt['heart_rate_bpm'],
                'RR_Ref_BPM': gt['breathing_rate_bpm'],
                'R_peak_count': len(gt['heart_peaks']),
                'HR_Est_BPM': np.nan,
                'RR_Est_BPM': np.nan,
                'HR_Error_BPM': np.nan,
                'RR_Error_BPM': np.nan,
                'Range_m': np.nan,
                'Azimuth_deg': np.nan,
                'SNR_dB': np.nan,
                'Confidence_Breath': np.nan,
                'Confidence_Heart': np.nan,
                'Num_Detected_Targets': len(final_targets) if final_targets else 0,
                'Status': 'No_Match'
            }

    except Exception as e:
        return {
            'target': target_name,
            'freq_band': '3GHZ',
            'position': position,
            'measurement': measurement,
            'csv_file': os.path.basename(log_file) if log_file else 'N/A',
            'segment': 0,
            'duration_sec': 30.0,
            'HR_Ref_BPM': np.nan,
            'RR_Ref_BPM': np.nan,
            'R_peak_count': 0,
            'HR_Est_BPM': np.nan,
            'RR_Est_BPM': np.nan,
            'HR_Error_BPM': np.nan,
            'RR_Error_BPM': np.nan,
            'Range_m': np.nan,
            'Azimuth_deg': np.nan,
            'SNR_dB': np.nan,
            'Confidence_Breath': np.nan,
            'Confidence_Heart': np.nan,
            'Num_Detected_Targets': 0,
            'Status': f'Error: {str(e)}'
        }


def batch_process_3ghz():
    """批量处理3GHz数据集"""

    print("="*80)
    print("🚀 批量处理3GHz数据集")
    print("="*80)
    print()

    # 数据集根目录
    dataset_root = "../FMCW radar-based multi-person vital sign monitoring data"

    # 只处理position1（AsymmetricalPosition）
    positions = [1]
    targets = ['Target1', 'Target2']
    measurements = range(1, 7)  # 1-6

    results = []
    total_files = len(positions) * len(targets) * len(measurements)
    processed = 0

    print(f"📊 计划处理 {total_files} 个文件")
    print()

    for position in positions:
        for target in targets:
            for measurement in measurements:
                processed += 1

                # 构建文件路径
                radar_file = f"{dataset_root}/1_AsymmetricalPosition/1_Radar_Raw_Data/position_ ({position})/adc_3GHZ_position{position}_ ({measurement}).bin"
                log_file = f"{dataset_root}/1_AsymmetricalPosition/2_Log_data/{target}/position_ ({position})/log_{target}_3GHZ_position{position}_ ({measurement}).csv"

                print(f"[{processed}/{total_files}] 处理: {target}, Position{position}, Measurement{measurement}")

                # 检查文件是否存在
                if not os.path.exists(radar_file):
                    print(f"  ⚠️  雷达文件不存在: {radar_file}")
                    result = {
                        'target': target,
                        'freq_band': '3GHZ',
                        'position': position,
                        'measurement': measurement,
                        'csv_file': f'log_{target}_3GHZ_position{position}_ ({measurement}).csv',
                        'segment': 0,
                        'duration_sec': 60.0,
                        'HR_Ref_BPM': np.nan,
                        'RR_Ref_BPM': np.nan,
                        'R_peak_count': 0,
                        'HR_Est_BPM': np.nan,
                        'RR_Est_BPM': np.nan,
                        'HR_Error_BPM': np.nan,
                        'RR_Error_BPM': np.nan,
                        'Range_m': np.nan,
                        'Azimuth_deg': np.nan,
                        'SNR_dB': np.nan,
                        'Confidence_Breath': np.nan,
                        'Confidence_Heart': np.nan,
                        'Num_Detected_Targets': 0,
                        'Status': 'Radar_File_Not_Found'
                    }
                    results.append(result)
                    continue

                if not os.path.exists(log_file):
                    print(f"  ⚠️  日志文件不存在: {log_file}")
                    result = {
                        'target': target,
                        'freq_band': '3GHZ',
                        'position': position,
                        'measurement': measurement,
                        'csv_file': f'log_{target}_3GHZ_position{position}_ ({measurement}).csv',
                        'segment': 0,
                        'duration_sec': 60.0,
                        'HR_Ref_BPM': np.nan,
                        'RR_Ref_BPM': np.nan,
                        'R_peak_count': 0,
                        'HR_Est_BPM': np.nan,
                        'RR_Est_BPM': np.nan,
                        'HR_Error_BPM': np.nan,
                        'RR_Error_BPM': np.nan,
                        'Range_m': np.nan,
                        'Azimuth_deg': np.nan,
                        'SNR_dB': np.nan,
                        'Confidence_Breath': np.nan,
                        'Confidence_Heart': np.nan,
                        'Num_Detected_Targets': 0,
                        'Status': 'Log_File_Not_Found'
                    }
                    results.append(result)
                    continue

                # 处理文件
                result = process_single_file(radar_file, log_file, target, position, measurement)
                results.append(result)

                if result['Status'] == 'Success':
                    print(f"  ✅ 成功: HR误差={result['HR_Error_BPM']:.1f} BPM, RR误差={result['RR_Error_BPM']:.1f} BPM")
                else:
                    print(f"  ❌ {result['Status']}")
                print()

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 保存结果
    output_file = 'algorithm_results_3GHZ_position1.csv'
    df.to_csv(output_file, index=False)

    print("="*80)
    print("📊 处理完成")
    print("="*80)
    print()
    print(f"✅ 结果已保存到: {output_file}")
    print()

    # 统计分析
    success_df = df[df['Status'] == 'Success']

    if len(success_df) > 0:
        print("="*80)
        print("📈 性能统计")
        print("="*80)
        print()

        print(f"成功处理: {len(success_df)}/{len(df)} 个文件 ({len(success_df)/len(df)*100:.1f}%)")
        print()

        print("呼吸率 (RR):")
        print(f"  平均误差 (MAE): {success_df['RR_Error_BPM'].mean():.2f} BPM")
        print(f"  标准差: {success_df['RR_Error_BPM'].std():.2f} BPM")
        print(f"  最大误差: {success_df['RR_Error_BPM'].max():.2f} BPM")
        print(f"  最小误差: {success_df['RR_Error_BPM'].min():.2f} BPM")
        print()

        print("心率 (HR):")
        print(f"  平均误差 (MAE): {success_df['HR_Error_BPM'].mean():.2f} BPM")
        print(f"  标准差: {success_df['HR_Error_BPM'].std():.2f} BPM")
        print(f"  最大误差: {success_df['HR_Error_BPM'].max():.2f} BPM")
        print(f"  最小误差: {success_df['HR_Error_BPM'].min():.2f} BPM")
        print()

        # 按目标分组统计
        print("按目标分组:")
        for target in ['Target1', 'Target2']:
            target_df = success_df[success_df['target'] == target]
            if len(target_df) > 0:
                print(f"  {target}:")
                print(f"    RR MAE: {target_df['RR_Error_BPM'].mean():.2f} BPM")
                print(f"    HR MAE: {target_df['HR_Error_BPM'].mean():.2f} BPM")
        print()

    return df


if __name__ == "__main__":
    df = batch_process_3ghz()
