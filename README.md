# Speech-to-Text

这是一个本地音视频转文字脚本，当前支持：

- 本地音频 / 视频文件转写
- 直接传入 B 站视频链接转写
- B 站媒体缓存复用
- NVIDIA `CUDA`
- AMD 官方 `ROCm + PyTorch`
- AMD iGPU `whisper.cpp + Vulkan`
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

### 3. 可选：为 AMD iGPU 准备 `whisper/` 运行环境

如果你想在不支持 ROCm 的 AMD iGPU 上运行转写，可以准备一个带 Vulkan 支持的 `whisper-cli`：

```powershell
git clone https://github.com/ggml-org/whisper.cpp.git whisper
cd whisper
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release
```

脚本会优先从项目根目录下的 `whisper/` 目录查找运行环境。构建完成后，常见可执行文件路径包括：

- `whisper\build\bin\Release\whisper-cli.exe`
- `whisper\build\bin\whisper-cli.exe`
- `whisper\whisper-cli.exe`

如果你保持这个目录结构，通常不需要额外传参；脚本会自动发现。只有在你放到了别的位置时，才需要通过 `--whispercpp-binary` 显式指定。

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
- `whisper.cpp` 后端的 `-bs`

示例：

```powershell
python speech_to_text.py "你的音视频文件路径" --beam-size 5
```

如果你想关闭 beam search，可以传 `1`：

```powershell
python speech_to_text.py "你的音视频文件路径" --beam-size 1
```

如果你在 Intel `OpenVINO + GPU` 路径下使用 `--beam-size 2` 或更大，命中当前 OpenVINO 运行时的 beam search 兼容性问题时，脚本会直接报错，并提示你改用 `--openvino-device CPU` 或 `--beam-size 1`。

## 音频分段与时间戳

默认情况下，脚本会先把输入音视频转换成一个完整的 16kHz 单声道 WAV，然后整体转写。这样可以避免按固定长度切片后，时间戳偏移和原视频逐渐对不上的问题。

如果整体转写失败，脚本会自动启用分段备用方案。分段备用现在会按每个实际片段时长累计时间偏移，不再使用固定的 `600` 秒偏移。

如果你想强制使用分段转写，可以传：

```powershell
python speech_to_text.py "你的音视频文件路径" --segment-audio
```

## 后端说明

### NVIDIA

```powershell
python speech_to_text.py "你的音视频文件路径" --backend cuda
```

### AMD

脚本不再走 `DirectML`。AMD 机器会优先尝试官方 `ROCm + PyTorch`：

```powershell
python speech_to_text.py "你的音视频文件路径" --backend rocm
```

如果当前机器没有可用的官方 ROCm 环境，但你已经准备好了带 Vulkan 支持的 `whisper-cli`，可以改用：

```powershell
python speech_to_text.py "你的音视频文件路径" --backend whispercpp-vulkan --whispercpp-binary ".\whisper\build\bin\Release\whisper-cli.exe"
```

在 `auto` 模式下，AMD 机器会按这个顺序尝试：

- `ROCm`
- `whisper.cpp + Vulkan`
- `CPU`

如果这台 AMD 机器既没有官方 ROCm，也没有可用的 Vulkan 版 `whisper-cli`，脚本才会回退 CPU：

```powershell
python speech_to_text.py "你的音视频文件路径" --backend cpu
```

`whisper.cpp + Vulkan` 这条路径不会替代官方 ROCm。对于 AMD 官方支持的 dGPU / APU，仍然优先建议使用 `ROCm`；它主要用于补齐像 `Ryzen 7 8745HS + Radeon 780M` 这类 Windows AMD iGPU 场景。

`--backend whispercpp-vulkan` 会强制要求实际启用 Vulkan GPU。脚本会检查 `whisper-cli` 输出中是否出现 `ggml_vulkan:`，如果只检测到 `no GPU found` 或 `device 0: CPU`，会直接报错退出，不会继续静默使用 CPU。

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

同样地，`whisper.cpp` 模型下载也复用这套镜像和超时参数，例如：

```powershell
python speech_to_text.py "你的音视频文件路径" --backend whispercpp-vulkan --whispercpp-binary ".\whisper\build\bin\Release\whisper-cli.exe" --hf-endpoint "https://hf-mirror.com" --hf-timeout 60
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

### whisper.cpp 模型缓存

whisper.cpp 的 ggml 模型会缓存到：

```text
cache/whispercpp/models/
```

如果目录下已经有对应的 `ggml-*.bin` 文件，脚本会直接复用缓存，不会重新下载。

## B 站 cookies 获取方式

如果 B 站视频需要登录态，建议显式传入 `cookies.txt`：

```powershell
python speech_to_text.py "https://www.bilibili.com/video/BVxxxxxxxxxx" --cookies ".\cookies.txt"
```

`cookies.txt` 需要是 Netscape / Mozilla 格式。推荐直接用 `yt-dlp` 从浏览器导出：

### 方法 1：用 yt-dlp 导出 `cookies.txt`

```powershell
yt-dlp --cookies-from-browser chrome --cookies cookies.txt "https://www.bilibili.com/video/BVxxxxxxxxxx"
```

导出后再传给脚本：

```powershell
python speech_to_text.py "https://www.bilibili.com/video/BVxxxxxxxxxx" --cookies ".\cookies.txt"
```

### 方法 2：用浏览器扩展导出 `cookies.txt`

`Get cookies.txt LOCALLY` 不是在网页控制台里执行的代码，它是一个浏览器扩展。用法是：

- 在 Chrome / Edge 扩展商店安装 `Get cookies.txt LOCALLY`
- 打开并登录 `bilibili.com`
- 点击扩展图标，导出当前站点的 cookies
- 把导出的文件保存为 `cookies.txt`
- 运行脚本时传 `--cookies ".\cookies.txt"`

Firefox 也有类似的 `cookies.txt` 扩展，流程基本相同。

## `yt-dlp` 视频下载说明

可以。当前脚本传入 B 站链接时，会优先下载视频文件，再由 `ffmpeg` 从视频中提取 16kHz 单声道 WAV 给后续转写使用。

- 当前脚本使用的格式选择是 `bv*+ba/b`
- `yt-dlp` 会优先下载最佳视频流并合并最佳音频流
- 如果站点只提供单文件格式，则会回退到最佳单文件格式
- 合并后的视频会缓存到 `cache/bilibili/` 下的 `source_video.*`
- 旧版脚本留下的 `source.m4a` 音频缓存不会再直接复用，脚本会重新下载视频缓存

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
