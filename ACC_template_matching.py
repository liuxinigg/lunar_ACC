import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from datetime import datetime, timedelta
import os
import pyscamp
import re
import csv  # 新增导入csv模块
from MyFunction.my_function import read_lunar_data, interpolate_masked_data
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 指定使用GPU 7

# 加载数据段的函数
def load_data_segment(start_time, end_time, component, station):
    try:
        stream = read_lunar_data(start_time.strftime("%Y%m%d"), end_time.strftime("%Y%m%d"), component=component, station=station, mode='all')
        interpolated_stream = interpolate_masked_data(stream)
        traces = interpolated_stream.traces
    except:
        return None, None
    if not traces:
        print(f"No data found for station {station}, component {component} from {start_time} to {end_time}.")
        return None, None
    
    trace_id = traces[0].id.replace('.', '_')
    return traces, trace_id

# 加载模板波形
def load_template_waveform(file_path):
    st = read(file_path)
    return st[0]

# 使用pyscamp计算相似度profile，并根据阈值筛选
def calculate_similarity_with_pyscamp(continuous_trace, template_waveform, threshold=0.95):
    template_length = len(template_waveform.data)
    # print(template_length)
    if len(continuous_trace) < template_length:
        return None, None, None
    matrix_profile, _ = pyscamp.abjoin(continuous_trace, template_waveform.data, m=template_length, pearson=True)
    indices = np.where(np.abs(matrix_profile) >= threshold)[0]
    similarities = matrix_profile[indices]
    return matrix_profile, indices, similarities

# 保存匹配到的片段
def save_matching_segments(trace, template_waveform, indices, similarities, station, component, segment_start_time, trace_id, threshold, template_name):
    sampling_rate = trace.stats.sampling_rate

    if indices.size == 0:
        return  # 如果没有匹配到任何片段，直接返回

    # # 输出目录，不包含 glitch_type
    # output_dir = os.path.join(
    #     f'/data01/liuxin/ACC-paper/template-new_matching/{template_name}',
    #     f'fig{threshold}',
    #     station, 'plots'  # 以台站命名的文件夹
    # )
    # os.makedirs(output_dir, exist_ok=True)

    # 匹配信息文件，与图像保存目录并列
    info_dir = os.path.join(f'/data01/liuxin/ACC-paper/template-new_matching2/{template_name}/fig{threshold}', station)
    os.makedirs(info_dir, exist_ok=True)
    info_file = os.path.join(info_dir, 'matching_segments_info.csv')

    # 检查CSV文件是否已存在
    file_exists = os.path.isfile(info_file)

    # 打开CSV文件，准备写入匹配信息
    with open(info_file, 'a', newline='') as csvfile:
        fieldnames = ['Trace ID', 'Amplitude Range', 'Start time', 'End time', 'Similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()

        # 遍历每个符合条件的索引
        for idx, similarity in zip(indices, similarities):
            start_idx = idx
            end_idx = start_idx + len(template_waveform.data)
            segment = trace.data[start_idx:end_idx]

            amplitude_range = np.max(segment) - np.min(segment)
            start_time_segment = segment_start_time + timedelta(seconds=start_idx / sampling_rate)
            end_time_segment = start_time_segment + timedelta(seconds=len(segment) / sampling_rate)

            # 生成横坐标，以秒为单位
            time_axis = np.arange(0, len(segment)) / sampling_rate

            # # 绘图
            # fig, ax = plt.subplots()
            # ax.plot(time_axis, segment, label=f'Similarity: {similarity:.2f}')
            # ax.plot(time_axis, template_waveform.data, label='Template')
            # ax.set_xlabel("Time (seconds)")
            # ax.set_title(f'Station: {station}, Component: {component}\nTime: {start_time_segment} - {end_time_segment}')
            # ax.legend()

            # # 保存图像
            # output_file = os.path.join(output_dir, f'{station}_{component}_{start_time_segment.strftime("%Y%m%d%H%M%S")}_{similarity:.2f}.png')
            # plt.savefig(output_file)
            # plt.close()

            # 写入匹配信息到CSV文件
            writer.writerow({
                'Trace ID': trace_id,
                'Amplitude Range': f'{amplitude_range:.2f}',
                'Start time': start_time_segment,
                'End time': end_time_segment,
                'Similarity': f'{similarity:.2f}'
            })

# 主函数
def main():
    # 设置参数
    start_time_str = '19690101'
    end_time_str = '19780101'
    components = ['MHZ', 'MH1', 'MH2']
    stations = ['16','15','12','14']  # 根据需要调整
    threshold = 0.95  # 相似度阈值
    segment_length_days = 1  # 每段数据长度（天）

    # 解析开始和结束时间
    start_time = datetime.strptime(start_time_str, "%Y%m%d")
    end_time = datetime.strptime(end_time_str, "%Y%m%d")

    # 获取模板文件列表
    template_folder = '/data01/liuxin/ACC-paper/template-new/'
    template_files = [os.path.join(template_folder, f) for f in os.listdir(template_folder) if f.endswith('.sac')]

    # 遍历每个模板文件
    for template_file_path in template_files:
        # 加载模板波形
        template_waveform = load_template_waveform(template_file_path)
        # 提取模板名字
        template_name = os.path.splitext(os.path.basename(template_file_path))[0]
        print(template_name)

        # 遍历台站和分量
        for station in stations:
            for component in components:
                current_start_time = start_time
                while current_start_time < end_time:
                    current_end_time = min(current_start_time + timedelta(days=segment_length_days), end_time)

                    # 加载当前段的数据
                    traces, trace_id = load_data_segment(current_start_time, current_end_time, component, station)

                    if traces is None:
                        current_start_time = current_end_time
                        continue

                    for trace in traces:
                        sampling_rate = trace.stats.sampling_rate

                        # 进行模板匹配
                        matrix_profile, indices, similarities = calculate_similarity_with_pyscamp(trace.data, template_waveform, threshold)
                        if matrix_profile is None:
                            continue
                        # 保存匹配片段（不区分正负相关）
                        save_matching_segments(trace, template_waveform, indices, similarities, station, component, current_start_time, trace_id, threshold, template_name)

                    current_start_time = current_end_time

if __name__ == "__main__":
    main()
