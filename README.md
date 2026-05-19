# Speech-to-Text

这是一个本地音视频转文字脚本，当前支持：

- 本地音频 / 视频文件转写
- 直接传入 B 站视频链接转写
- B 站媒体缓存复用
- NVIDIA `CUDA`
- AMD 官方 `ROCm + PyTorch`
- Intel `OpenVINO`
- Hugging Face 镜像源与超时控制

## 安装

### 1. 准备 FFmpeg

把 `ffmpeg.exe` 放到项目目录下的 `ffmpeg/bin/` 里，或者确保系统 `PATH` 里已经有 `ffmpeg`。

### 2. 安装 Python 依赖

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 基本用法

### 本地文件

```powershell
python speech_to_text.py "你的音视频文件路径"
```

### B 站链接

```powershell
python speech_to_text.py "https://www.bilibili.com/video/BVxxxxxxxxxx"
```

如果视频需要登录态：

```powershell
python speech_to_text.py "https://www.bilibili.com/video/BVxxxxxxxxxx" --cookies ".\cookies.txt"
```

## 解码参数

脚本现在支持统一的 `--beam-size` 参数，默认值是 `5`，同时作用于：

- `Whisper` 后端的 `beam_size`
- `OpenVINO` 后端的 `num_beams`

示例：

```powershell
python speech_to_text.py "你的音视频文件路径" --beam-size 5
```

如果你想关闭 beam search，可以传 `1`：

```powershell
python speech_to_text.py "你的音视频文件路径" --beam-size 1
```

如果你在 Intel `OpenVINO + GPU` 路径下使用 `--beam-size 2` 或更大，命中当前 OpenVINO 运行时的 beam search 兼容性问题时，脚本会直接报错，并提示你改用 `--openvino-device CPU` 或 `--beam-size 1`。

## 后端说明

### NVIDIA

```powershell
python speech_to_text.py "你的音视频文件路径" --backend cuda
```

### AMD

脚本不再走 `DirectML`。AMD 机器优先建议使用官方 `ROCm + PyTorch`：

```powershell
python speech_to_text.py "你的音视频文件路径" --backend rocm
```

如果当前 Windows / Linux 机器没有可用的官方 ROCm 环境，建议直接回退 CPU：

```powershell
python speech_to_text.py "你的音视频文件路径" --backend cpu
```

### Intel

脚本不再走 `DirectML`。Intel 机器优先建议使用 `OpenVINO`：

```powershell
python speech_to_text.py "你的音视频文件路径" --backend openvino --openvino-device GPU
```

支持的 OpenVINO 设备参数：

- `AUTO`
- `CPU`
- `GPU`
- `NPU`

如果你准备在 Intel 机器上启用 beam search，建议把这几个包保持为同一版本线，并优先使用 `2025.2+` 或更新版本：

- `openvino`
- `openvino-genai`
- `openvino-tokenizers`

当前 OpenVINO 路径支持的模型：

- `tiny`
- `base`
- `small`
- `medium`
- `large`

`turbo` 不走 OpenVINO 官方预转换模型；如果你要用 `turbo`，请改用 `--backend cpu / cuda / rocm`。

## 大陆网络下的 Hugging Face 下载

如果你在首次下载 OpenVINO Whisper 模型时遇到 `ConnectTimeout`、`WinError 10060` 或 Hugging Face 连接超时，现在脚本支持：

- `--hf-endpoint`：指定 Hugging Face Hub 镜像
- `--hf-timeout`：拉长 metadata / download 超时
- 自动回退：如果没显式指定 endpoint，会先试官方源，再试 `https://hf-mirror.com`

推荐命令：

```powershell
python speech_to_text.py "你的音视频文件路径" --backend openvino --openvino-device GPU --hf-endpoint "https://hf-mirror.com" --hf-timeout 60
```

也可以长期设置环境变量：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_ETAG_TIMEOUT = "60"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "60"
python speech_to_text.py "你的音视频文件路径" --backend openvino --openvino-device GPU
```

## 缓存目录

### B 站媒体缓存

首次下载成功后，脚本会把媒体缓存到：

```text
cache/bilibili/
```

同一个 B 站链接再次转写时，会优先复用这里的本地媒体文件，不会重复下载。

### OpenVINO 模型缓存

OpenVINO Whisper 模型会缓存到：

```text
cache/openvino/
```

如果目录下已经有完整模型文件，脚本会直接复用缓存，不会重新下载。

## `cookies.txt` 获取方式

脚本的 `--cookies` 参数需要的是 Netscape / Mozilla 格式的 `cookies.txt` 文件。

推荐直接用 `yt-dlp` 从浏览器导出：

```powershell
yt-dlp --cookies-from-browser chrome --cookies cookies.txt "https://www.bilibili.com/video/BVxxxxxxxxxx"
```

也可以用：

- Chrome / Edge：`Get cookies.txt LOCALLY`
- Firefox：`cookies.txt`

## `yt-dlp` 能不能获取高清版视频

可以，但当前这个脚本的目标是转文字，不是保存高清视频文件。

- `yt-dlp` 本身可以下载最佳可用画质
- 当前脚本代码里明确使用的是 `bestaudio/best`
- 也就是说，脚本只会下载最佳音频流用于转写，不会额外保存高清视频

如果你只是想查看某个 B 站视频有哪些格式可下：

```powershell
yt-dlp -F "https://www.bilibili.com/video/BVxxxxxxxxxx"
```

如果你想手工下载最佳视频 + 音频并合并：

```powershell
yt-dlp -f "bv+ba/b" "https://www.bilibili.com/video/BVxxxxxxxxxx"
```

## 输出位置

- 本地文件输入：输出到原文件所在目录
- B 站链接输入：输出到你运行命令时所在的当前目录

输出文件名格式：

```text
<原文件名或视频标题>_转写结果.txt
```
