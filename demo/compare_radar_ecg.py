"""
雷达 vs ECG/PCG 真值对比脚本

功能：
1. 自动适配两种真值格式：
   - 精简版：HR_Ref_BPM, RR_Ref_BPM
   - 原版：HR_Corrected_BPM, RR_Final_BPM
2. 自动识别雷达 CSV 中的 HR / RR 列名
3. 按 target + segment 对齐
4. 计算误差指标：MAE, RMSE, Bias, 命中率
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 列名映射配置
# ============================================================

# 真值 CSV 可能的列名
REF_HR_COLUMNS = [
    'HR_Ref_BPM',           # 精简版
    'HR_Corrected_BPM',     # 原版（谐波校正后）
    'HR_Final_BPM',         # 备选
    'HR_PanTompkins_BPM',   # Pan-Tompkins
    'HR_Slope_BPM',         # Slope-based
]

REF_RR_COLUMNS = [
    'RR_Ref_BPM',           # 精简版
    'RR_Final_BPM',         # 原版
    'RR_Peak_BPM',          # Peak-based
    'RR_FFT_BPM',           # FFT-based
    'RR_PCG_BPM',           # PCG-based
]

# 雷达 CSV 可能的列名
RADAR_HR_COLUMNS = [
    'HR_radar',
    'HR_Radar',
    'heartRate',
    'HeartRate',
    'HR',
    'hr',
]

RADAR_RR_COLUMNS = [
    'RR_radar',
    'RR_Radar',
    'breathRate',
    'BreathRate',
    'RR',
    'rr',
    'BR',
    'br',
]

# Target 列名
TARGET_COLUMNS = ['target', 'Target', 'target_id', 'Target_ID']
SEGMENT_COLUMNS = ['segment', 'Segment', 'seg', 'Seg', 'seg_idx']


def find_column(df: pd.DataFrame, candidates: list) -> str:
    """从候选列名中找到存在的列"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_target(val) -> str:
    """标准化 target 值（统一格式）"""
    s = str(val).strip().lower()
    # 提取数字部分
    import re
    match = re.search(r'(\d+)', s)
    if match:
        return match.group(1)
    return s


def load_reference_csv(csv_path: str) -> pd.DataFrame:
    """加载真值 CSV 并标准化列名"""
    df = pd.read_csv(csv_path)
    
    # 找到 HR 和 RR 列
    hr_col = find_column(df, REF_HR_COLUMNS)
    rr_col = find_column(df, REF_RR_COLUMNS)
    target_col = find_column(df, TARGET_COLUMNS)
    segment_col = find_column(df, SEGMENT_COLUMNS)
    
    if hr_col is None:
        raise ValueError(f"真值 CSV 中找不到 HR 列，尝试过: {REF_HR_COLUMNS}")
    if rr_col is None:
        raise ValueError(f"真值 CSV 中找不到 RR 列，尝试过: {REF_RR_COLUMNS}")
    
    # 标准化列名
    result = pd.DataFrame()
    result['HR_ref'] = df[hr_col]
    result['RR_ref'] = df[rr_col]
    
    if target_col:
        result['target'] = df[target_col].apply(normalize_target)
    else:
        result['target'] = '1'  # 默认单目标
    
    if segment_col:
        result['segment'] = df[segment_col].astype(int)
    else:
        result['segment'] = 0  # 默认单段
    
    print(f"[真值] 加载 {csv_path}")
    print(f"       HR 列: {hr_col}, RR 列: {rr_col}")
    print(f"       共 {len(result)} 条记录")
    
    return result


def load_radar_csv(csv_path: str) -> pd.DataFrame:
    """加载雷达 CSV 并标准化列名"""
    df = pd.read_csv(csv_path)
    
    # 找到 HR 和 RR 列
    hr_col = find_column(df, RADAR_HR_COLUMNS)
    rr_col = find_column(df, RADAR_RR_COLUMNS)
    target_col = find_column(df, TARGET_COLUMNS)
    segment_col = find_column(df, SEGMENT_COLUMNS)
    
    if hr_col is None:
        raise ValueError(f"雷达 CSV 中找不到 HR 列，尝试过: {RADAR_HR_COLUMNS}")
    if rr_col is None:
        raise ValueError(f"雷达 CSV 中找不到 RR 列，尝试过: {RADAR_RR_COLUMNS}")
    
    # 标准化列名
    result = pd.DataFrame()
    result['HR_radar'] = df[hr_col]
    result['RR_radar'] = df[rr_col]
    
    if target_col:
        result['target'] = df[target_col].apply(normalize_target)
    else:
        result['target'] = '1'
    
    if segment_col:
        result['segment'] = df[segment_col].astype(int)
    else:
        result['segment'] = 0
    
    print(f"[雷达] 加载 {csv_path}")
    print(f"       HR 列: {hr_col}, RR 列: {rr_col}")
    print(f"       共 {len(result)} 条记录")
    
    return result


def calculate_total_error(merged: pd.DataFrame) -> float:
    """计算总误差（HR + RR 的 MAE 之和）"""
    if len(merged) == 0:
        return float('inf')
    hr_mae = np.mean(np.abs(merged['HR_radar'] - merged['HR_ref']))
    rr_mae = np.mean(np.abs(merged['RR_radar'] - merged['RR_ref']))
    return hr_mae + rr_mae


def try_swap_targets(ref_df: pd.DataFrame, radar_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    尝试交换 target 对应关系，如果交换后误差更小则采用交换
    
    Returns:
        radar_df: 可能经过 target 交换的雷达数据
        swapped: 是否进行了交换
    """
    radar_df = radar_df.copy()
    
    # 获取所有 unique targets
    ref_targets = sorted(ref_df['target'].unique())
    radar_targets = sorted(radar_df['target'].unique())
    
    # 只在 target 数量为 2 时进行交换检测
    if len(ref_targets) != 2 or len(radar_targets) != 2:
        return radar_df, False
    
    # 尝试原始匹配
    merged_original = pd.merge(
        ref_df, radar_df,
        on=['target', 'segment'],
        how='inner'
    )
    error_original = calculate_total_error(merged_original)
    
    # 尝试交换 target
    radar_swapped = radar_df.copy()
    target_map = {radar_targets[0]: radar_targets[1], radar_targets[1]: radar_targets[0]}
    radar_swapped['target'] = radar_swapped['target'].map(target_map)
    
    merged_swapped = pd.merge(
        ref_df, radar_swapped,
        on=['target', 'segment'],
        how='inner'
    )
    error_swapped = calculate_total_error(merged_swapped)
    
    # 比较误差
    if error_swapped < error_original * 0.5:  # 交换后误差减少 50% 以上才采用
        print(f"\n[自动修正] 检测到 Target 对应关系可能相反！")
        print(f"           原始匹配误差: {error_original:.2f}")
        print(f"           交换后误差:   {error_swapped:.2f}")
        print(f"           自动交换: 雷达 Target {radar_targets[0]} <-> Target {radar_targets[1]}")
        return radar_swapped, True
    
    return radar_df, False


def merge_data(ref_df: pd.DataFrame, radar_df: pd.DataFrame, auto_swap: bool = True) -> pd.DataFrame:
    """按 target + segment 合并数据"""
    radar_df = radar_df.copy()
    
    # 确保 segment 从 0 或 1 开始都能对齐
    # 雷达可能从 1 开始，真值可能从 0 开始
    radar_min_seg = radar_df['segment'].min()
    ref_min_seg = ref_df['segment'].min()
    
    if radar_min_seg != ref_min_seg:
        print(f"\n[注意] segment 起始值不同: 雷达={radar_min_seg}, 真值={ref_min_seg}")
        print(f"       自动调整雷达 segment -= {radar_min_seg - ref_min_seg}")
        radar_df['segment'] = radar_df['segment'] - (radar_min_seg - ref_min_seg)
    
    # 尝试自动交换 target（如果需要）
    if auto_swap:
        radar_df, swapped = try_swap_targets(ref_df, radar_df)
    
    # 合并
    merged = pd.merge(
        ref_df, radar_df,
        on=['target', 'segment'],
        how='inner',
        suffixes=('_ref', '_radar')
    )
    
    print(f"\n[合并] 成功匹配 {len(merged)} 条记录")
    
    if len(merged) == 0:
        print("\n[警告] 没有匹配的记录！检查 target 和 segment 值：")
        print(f"  真值 targets: {sorted(ref_df['target'].unique())}")
        print(f"  雷达 targets: {sorted(radar_df['target'].unique())}")
        print(f"  真值 segments: {sorted(ref_df['segment'].unique())}")
        print(f"  雷达 segments: {sorted(radar_df['segment'].unique())}")
    
    return merged


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """计算误差指标"""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)
    
    return {
        'name': name,
        'MAE': mae,
        'RMSE': rmse,
        'Bias': bias,
        'errors': errors,
        'abs_errors': abs_errors,
    }


def calculate_hit_rate(abs_errors: np.ndarray, threshold: float) -> float:
    """计算命中率（误差小于阈值的比例）"""
    return np.mean(abs_errors <= threshold) * 100


def print_comparison_table(merged: pd.DataFrame):
    """打印逐条对比表格"""
    print("\n" + "=" * 80)
    print("逐条对比结果")
    print("=" * 80)
    
    # 计算误差
    merged['HR_err'] = merged['HR_radar'] - merged['HR_ref']
    merged['RR_err'] = merged['RR_radar'] - merged['RR_ref']
    
    # 打印表头
    header = f"{'Target':>8} {'Seg':>4} | {'HR_ref':>8} {'HR_radar':>9} {'HR_err':>8} | {'RR_ref':>8} {'RR_radar':>9} {'RR_err':>8}"
    print(header)
    print("-" * 80)
    
    # 打印每条记录
    for _, row in merged.iterrows():
        line = (
            f"{row['target']:>8} {int(row['segment']):>4} | "
            f"{row['HR_ref']:>8.1f} {row['HR_radar']:>9.1f} {row['HR_err']:>+8.1f} | "
            f"{row['RR_ref']:>8.1f} {row['RR_radar']:>9.1f} {row['RR_err']:>+8.1f}"
        )
        print(line)
    
    print("-" * 80)


def print_metrics(merged: pd.DataFrame):
    """打印误差指标汇总"""
    print("\n" + "=" * 80)
    print("误差指标汇总")
    print("=" * 80)
    
    # HR 指标
    hr_metrics = calculate_metrics(
        merged['HR_ref'].values,
        merged['HR_radar'].values,
        'Heart Rate (HR)'
    )
    
    # RR 指标
    rr_metrics = calculate_metrics(
        merged['RR_ref'].values,
        merged['RR_radar'].values,
        'Respiration Rate (RR)'
    )
    
    # 打印表格
    print(f"\n{'指标':<20} {'HR (bpm)':>15} {'RR (次/分钟)':>15}")
    print("-" * 50)
    print(f"{'MAE (平均绝对误差)':<20} {hr_metrics['MAE']:>15.2f} {rr_metrics['MAE']:>15.2f}")
    print(f"{'RMSE (均方根误差)':<20} {hr_metrics['RMSE']:>15.2f} {rr_metrics['RMSE']:>15.2f}")
    print(f"{'Bias (偏差)':<20} {hr_metrics['Bias']:>+15.2f} {rr_metrics['Bias']:>+15.2f}")
    
    # 命中率
    print("\n" + "-" * 50)
    print("命中率 (误差在阈值内的比例)")
    print("-" * 50)
    
    hr_thresholds = [3, 5, 10]  # bpm
    rr_thresholds = [2, 3, 5]   # 次/分钟
    
    for thr in hr_thresholds:
        hit_rate = calculate_hit_rate(hr_metrics['abs_errors'], thr)
        print(f"HR |误差| ≤ {thr} bpm:  {hit_rate:>6.1f}%")
    
    print()
    for thr in rr_thresholds:
        hit_rate = calculate_hit_rate(rr_metrics['abs_errors'], thr)
        print(f"RR |误差| ≤ {thr} 次/分: {hit_rate:>6.1f}%")
    
    print("=" * 80)
    
    return hr_metrics, rr_metrics


def print_per_target_metrics(merged: pd.DataFrame):
    """打印每个目标的误差指标"""
    targets = sorted(merged['target'].unique())
    
    if len(targets) <= 1:
        return
    
    print("\n" + "=" * 80)
    print("按目标分组的误差指标")
    print("=" * 80)
    
    print(f"\n{'Target':<10} {'HR_MAE':>10} {'HR_Bias':>10} {'RR_MAE':>10} {'RR_Bias':>10}")
    print("-" * 50)
    
    for target in targets:
        subset = merged[merged['target'] == target]
        
        hr_err = subset['HR_radar'] - subset['HR_ref']
        rr_err = subset['RR_radar'] - subset['RR_ref']
        
        hr_mae = np.mean(np.abs(hr_err))
        hr_bias = np.mean(hr_err)
        rr_mae = np.mean(np.abs(rr_err))
        rr_bias = np.mean(rr_err)
        
        print(f"Target {target:<4} {hr_mae:>10.2f} {hr_bias:>+10.2f} {rr_mae:>10.2f} {rr_bias:>+10.2f}")
    
    print("=" * 80)


def save_comparison_csv(merged: pd.DataFrame, output_path: str):
    """保存对比结果到 CSV"""
    merged['HR_err'] = merged['HR_radar'] - merged['HR_ref']
    merged['RR_err'] = merged['RR_radar'] - merged['RR_ref']
    merged['HR_abs_err'] = np.abs(merged['HR_err'])
    merged['RR_abs_err'] = np.abs(merged['RR_err'])
    
    merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存到: {output_path}")


def main():
    # 基于脚本位置计算默认路径
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_REF = SCRIPT_DIR / "results" / "hr_rr_from_validation_summary.csv"
    DEFAULT_RADAR = SCRIPT_DIR / "matlab" / "radar_vital_signs_results.csv"
    
    parser = argparse.ArgumentParser(
        description='雷达 vs ECG/PCG 真值对比'
    )
    parser.add_argument(
        '--ref', '-r',
        type=str,
        default=str(DEFAULT_REF),
        help='真值 CSV 文件路径'
    )
    parser.add_argument(
        '--radar', '-d',
        type=str,
        default=str(DEFAULT_RADAR),
        help='雷达 CSV 文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出对比结果 CSV 路径（可选）'
    )
    parser.add_argument(
        '--no-auto-swap',
        action='store_true',
        help='禁用 Target 自动交换修正'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("雷达 vs ECG/PCG 真值对比")
    print("=" * 80)
    
    # 加载数据
    try:
        ref_df = load_reference_csv(args.ref)
    except Exception as e:
        print(f"\n[错误] 加载真值 CSV 失败: {e}")
        return
    
    try:
        radar_df = load_radar_csv(args.radar)
    except Exception as e:
        print(f"\n[错误] 加载雷达 CSV 失败: {e}")
        return
    
    # 合并数据（自动检测并修正 target 对应关系）
    auto_swap = not args.no_auto_swap
    merged = merge_data(ref_df, radar_df, auto_swap=auto_swap)
    
    if len(merged) == 0:
        print("\n[错误] 没有可对比的数据！")
        return
    
    # 打印对比结果
    print_comparison_table(merged)
    print_metrics(merged)
    print_per_target_metrics(merged)
    
    # 保存结果
    if args.output:
        save_comparison_csv(merged, args.output)
    else:
        # 默认保存到 demo/results 目录
        default_output = Path(args.ref).parent / 'radar_ecg_comparison.csv'
        save_comparison_csv(merged, str(default_output))


if __name__ == '__main__':
    main()

