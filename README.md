# Whisper 语音/视频转文字工具 (Speech-to-Text)

这是一个基于 [OpenAI Whisper](https://github.com/openai/whisper) 开发的自动化音视频转文字脚本。
旨在提供**完全免费**、**高质量中文识别**、并且**防内存溢出(OOM)**的本地化转写方案。

## 🌟 核心特性

- **智能切片防爆内存**：自动使用 `ffmpeg` 将超长音视频无损切分为 10 分钟小段进行处理。哪怕是长达几小时的视频，在 4GB 甚至更低内存/显存的电脑（如 vGPU 实例）上也能稳定运行，拒绝崩溃。
- **实时保存防丢失**：处理完每一个小片段后会自动追加写入到 txt 文件中。即使中途断电关机，已识别的文字也会被安全保存。
- **GPU (CUDA) 自动加速**：代码会自动检测 NVIDIA 显卡。如果检测到支持 CUDA 的环境，将自动开启 `fp16` 半精度推理，速度飙升 10 倍以上。
- **实时进度条**：控制台实时滚动打印识别到的内容和时间戳，让你对识别进度一目了然。

## 🛠️ 安装说明

### 1. 安装 FFmpeg (核心依赖)
Whisper 的底层音频解码需要依赖 FFmpeg。
请下载 Windows 版本的 FFmpeg，并将其提取出来的 `ffmpeg.exe` 放置在与脚本同目录下的 `ffmpeg/bin/` 文件夹中（或直接放置在同目录下）。脚本启动时会自动寻找它，**无需手动配置系统环境变量**。

### 2. 配置 Python 环境

建议使用 Python 虚拟环境以隔离依赖：

```powershell
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境 (Windows)
.\venv\Scripts\activate

# 3. 安装依赖包
pip install -r requirements.txt
```

### 3. (强烈推荐) 配置 GPU 专属 PyTorch
如果你拥有 NVIDIA 独立显卡（如 Tesla P4 等），请务必卸载默认的纯 CPU 版本 PyTorch，并安装 CUDA 专属版本以获得数十倍的速度提升：

```powershell
# 先卸载默认版本
pip uninstall torch -y

# 例如安装带 CUDA 11.8 支持的 GPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 使用方法

将需要转换的音频（如 `.wav`, `.mp3`）或视频（如 `.mp4`, `.mkv`）放入文件夹，打开终端并激活虚拟环境，然后运行：

```powershell
python speech_to_text.py "你的音视频文件路径"
```

**进阶模型选择：**
脚本默认使用 `small` 模型（占用显存约 2GB，兼顾速度与质量）。如果你希望获得极致的中文识别质量，可以通过 `--model` 参数切换为 `medium`（推荐，约 5GB 显存）或 `large` 模型：

```powershell
python speech_to_text.py "你的音视频文件路径" --model medium
```

*注：首次使用某个特定尺寸的模型时，脚本会在后台自动从官网下载模型文件，请耐心等待。*

## 📄 结果输出
转换完成后，脚本会自动在原音视频同目录下生成一个以 `_转写结果.txt` 结尾的文本文件，里面包含了所有提取出的纯文本内容。
