import os
import pandas as pd
import matplotlib.pyplot as plt

def save_overlay_plot(
    csv_path: str, 
    columns_to_compare: list[str], 
    output_dir: str = "./plots",
    time_col: str = "time"
):
    """
    将指定的多个列画在【同一张图】上进行对比 (Overlay)。
    
    参数:
    - csv_path: CSV 文件路径
    - columns_to_compare: 需要叠加对比的列名列表，例如 ["obs_0", "obs_target_0"]
    - output_dir: 保存路径
    - time_col: 时间列名（若不存在则使用索引）
    """
    # 1. 读取数据
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到文件 {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 过滤不存在的列
    valid_cols = [col for col in columns_to_compare if col in df.columns]
    if not valid_cols:
        print(f"❌ 错误: 指定的列 {columns_to_compare} 均不存在于 CSV 中。")
        return
    
    if len(valid_cols) < 1:
        print("❌ 请至少指定一列进行绘制。")
        return

    # 准备目录
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_dir = os.path.join(output_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)

    # 2. 处理 X 轴 (时间 vs 索引)
    if time_col and time_col in df.columns:
        x_data = df[time_col]
        x_label = f"{time_col} (s)"
    else:
        x_data = df.index
        x_label = "Step / Index"
        if time_col and time_col not in df.columns:
            print(f"ℹ️ 提示: 未找到 '{time_col}'，使用索引作为 X 轴。")

    # 3. 绘图 (核心修改：在同一个 ax 上多次 plot)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 自动颜色循环
    colors = plt.cm.tab10.colors  # 使用一组对比度高的颜色
    
    for idx, col in enumerate(valid_cols):
        color = colors[idx % len(colors)]
        # linewidth稍微细一点，alpha设置透明度，方便看重叠部分
        ax.plot(x_data, df[col], label=col, color=color, linewidth=1.5, alpha=0.8)

    # 设置图表细节
    ax.set_title(f"Comparison: {' vs '.join(valid_cols)}", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='best') # 图例自动寻找最佳位置

    # 4. 保存文件名：自动拼接列名，方便识别
    # 如果列太多，文件名太长，就截断一下
    cols_str = "_vs_".join(valid_cols)
    if len(cols_str) > 50: 
        cols_str = "multi_columns_overlay"
        
    save_path = os.path.join(save_dir, f"overlay_{cols_str}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ 对比图已保存: {save_path}")


# --- 使用示例 ---
if __name__ == "__main__":
    csv_file_path = "/home/jf/lab/5_utils/Robot_config_test/logs/calibration_log_20260128_103107.csv"
    
    # 场景：你想看这两个值在同一张图上的重合度
    # 比如：对比 观测值(obs) 和 动作值(act)，或者 目标位置 vs 实际位置
    compare_list = ["left_hip_pitch_obs_pos", "left_hip_pitch_act_raw","left_hip_pitch_act_scaled"] 
    
    save_overlay_plot(csv_file_path, compare_list, time_col="time")