import whisper
import os
import argparse
import time
import torch
import subprocess
import glob
import sys
import re
import wave
from tqdm import tqdm

# 将当前脚本所在目录临时添加到系统的 PATH 环境变量中
# 将当前脚本所在目录及其子目录 ffmpeg/bin 临时添加到系统的 PATH 环境变量中
# 这样哪怕 ffmpeg 没有加到系统环境变量，只要放在本脚本同目录下，Whisper 也能正常调用它
script_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_bin_dir = os.path.join(script_dir, "ffmpeg", "bin")
os.environ["PATH"] = script_dir + os.pathsep + ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")

def get_wav_duration(file_path):
    """使用内置的 wave 模块获取 wav 文件的时长（秒）"""
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate)

class WhisperProgressLogger:
    """拦截 sys.stdout，通过解析 Whisper 打印的时间戳来更新全局进度条"""
    def __init__(self, pbar):
        self.pbar = pbar
        self.max_sec_seen = 0.0

    def write(self, message):
        # 匹配 Whisper 的输出格式，如: [00:00.000 --> 00:05.000] 或 [01:02:03.000 --> 01:02:08.000]
        # 我们只提取箭号后面的“结束时间”
        match = re.search(r'-->\s*(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})', message)
        if match:
            h, m, s, ms = match.groups()
            h = int(h) if h else 0
            current_sec = h * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
            
            # 如果时间有推进，则计算差值并更新 tqdm
            if current_sec > self.max_sec_seen:
                update_val = current_sec - self.max_sec_seen
                self.pbar.update(update_val)
                self.max_sec_seen = current_sec

    def flush(self):
        pass

def transcribe_audio(input_path, model_name="small"):
    """
    使用 OpenAI Whisper 将音频或视频转换为文字。
    采用分段处理方式，防止长音频导致系统内存(RAM)溢出。
    """
    device_name = "GPU (CUDA 加速)" if torch.cuda.is_available() else "纯 CPU (速度较慢)"
    print(f"🚀 当前实际运行模式: {device_name}")
    print(f"正在加载 Whisper 模型 '{model_name}' (首次运行会自动下载模型，请耐心等待)...")
    try:
        # 加载模型
        model = whisper.load_model(model_name)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("提示：如果遇到网络问题，请检查网络连接或代理设置。")
        return

    print(f"正在预处理音频：正在自动安全分段...")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_转写结果.txt"
    
    # 先清理可能残留的历史临时文件
    for f in glob.glob("temp_chunk_*.wav"):
        try:
            os.remove(f)
        except:
            pass

    # 使用 ffmpeg 切割文件，每 600 秒 (10分钟) 一段
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-f", "segment", "-segment_time", "600",
        "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "temp_chunk_%03d.wav"
    ]
    
    # 隐藏 ffmpeg 的大量输出
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    chunk_files = sorted(glob.glob("temp_chunk_*.wav"))
    if not chunk_files:
        print("❌ 音频分割失败，请检查输入文件是否有效或 ffmpeg 是否正常工作。")
        return

    # 精确计算切片后的总音频时长
    total_audio_duration = sum(get_wav_duration(f) for f in chunk_files)
    print(f"✅ 音频预处理完毕！总时长约 {total_audio_duration/60:.1f} 分钟。开始智能识别...\n")
    
    start_time = time.time()
    use_fp16 = torch.cuda.is_available()
    
    # 清空并创建最终的输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")

    try:
        # 自定义好看的中文进度条格式
        pbar_format = "{l_bar}{bar}| 进度: {n:.1f}/{total:.1f} 秒音频 [已耗时: {elapsed}, 预计剩余: {remaining}]"
        
        with tqdm(total=total_audio_duration, unit="秒", bar_format=pbar_format) as pbar:
            for idx, chunk_file in enumerate(chunk_files):
                chunk_duration = get_wav_duration(chunk_file)
                
                # 狸猫换太子：暂时把系统的打印输出重定向到我们的解析器，以拦截底层时间戳
                logger = WhisperProgressLogger(pbar)
                old_stdout = sys.stdout
                sys.stdout = logger
                
                try:
                    # verbose=True 会输出时间戳，刚好被上面的 logger 捕获并转换成进度条
                    result = model.transcribe(chunk_file, language="zh", fp16=use_fp16, verbose=True)
                finally:
                    # 本段结束，把系统的打印输出还回去
                    sys.stdout = old_stdout
                
                # 补齐误差（比如最后半秒钟 Whisper 没有打印出来）
                unprocessed = chunk_duration - logger.max_sec_seen
                if unprocessed > 0:
                    pbar.update(unprocessed)
                
                # 识别完一段，立刻将文字追加保存到文件中
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(result["text"] + "\n")
                    
                # 处理完一个片段就删掉一个临时文件
                try:
                    os.remove(chunk_file)
                except:
                    pass
            
    except Exception as e:
        # 万一报错了，确保控制台输出被恢复
        sys.stdout = sys.__stdout__
        print(f"\n❌ 识别过程出错: {e}")
        return

    end_time = time.time()
    print(f"\n🎉 识别大功告成！总处理耗时: {end_time - start_time:.2f} 秒")
    print(f"📄 所有的文字结果已合并并保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="免费高质量语音/视频转文字工具 (基于 OpenAI Whisper)")
    parser.add_argument("input_file", help="要转换的音频或视频文件路径")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large", "turbo"], 
                        help="使用的模型大小，默认为 'small'。想要更高质量可以选择 'medium' 或 'large'")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ 错误: 找不到文件 '{args.input_file}'")
    else:
        transcribe_audio(args.input_file, args.model)
