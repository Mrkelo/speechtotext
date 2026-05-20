import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from urllib.parse import parse_qs, urlparse

import torch
import whisper
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_BIN_DIR = os.path.join(SCRIPT_DIR, "ffmpeg", "bin")
os.environ["PATH"] = SCRIPT_DIR + os.pathsep + FFMPEG_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
BILIBILI_CACHE_ROOT = os.path.join(SCRIPT_DIR, "cache", "bilibili")
OPENVINO_CACHE_ROOT = os.path.join(SCRIPT_DIR, "cache", "openvino")
WHISPERCPP_CACHE_ROOT = os.path.join(SCRIPT_DIR, "cache", "whispercpp")
WHISPERCPP_MODEL_REPO_ID = "ggerganov/whisper.cpp"
WHISPER_RUNTIME_ROOT = os.path.join(SCRIPT_DIR, "whisper")
BILIBILI_DOWNLOAD_KIND = "video"
BILIBILI_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".flv", ".mov"}

OPENVINO_WHISPER_MODELS = {
    "tiny": "OpenVINO/whisper-tiny-fp16-ov",
    "base": "OpenVINO/whisper-base-fp16-ov",
    "small": "OpenVINO/whisper-small-fp16-ov",
    "medium": "OpenVINO/whisper-medium-fp16-ov",
    "large": "OpenVINO/whisper-large-v3-int4-ov",
}

OPENVINO_REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "openvino_encoder_model.bin",
    "openvino_encoder_model.xml",
    "openvino_decoder_model.bin",
    "openvino_decoder_model.xml",
    "openvino_tokenizer.bin",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.bin",
    "openvino_detokenizer.xml",
    "preprocessor_config.json",
]

WHISPERCPP_MODEL_FILENAMES = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large": "ggml-large-v3.bin",
    "turbo": "ggml-large-v3-turbo.bin",
}

OPENVINO_LANGUAGE_TOKEN = "<|zh|>"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_ETAG_TIMEOUT = 30
DEFAULT_HF_DOWNLOAD_TIMEOUT = 120
DEFAULT_BEAM_SIZE = 5
SEGMENT_SECONDS = 600
TEMP_FULL_AUDIO_FILE = os.path.join(SCRIPT_DIR, "temp_full_audio.wav")
TEMP_CHUNK_PATTERN = os.path.join(SCRIPT_DIR, "temp_chunk_%03d.wav")
TEMP_CHUNK_GLOB = os.path.join(SCRIPT_DIR, "temp_chunk_*.wav")


def get_wav_duration(file_path):
    with wave.open(file_path, "r") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate)


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class WhisperProgressLogger:
    def __init__(self, pbar):
        self.pbar = pbar
        self.max_sec_seen = 0.0

    def write(self, message):
        match = re.search(r'-->\s*(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})', message)
        if not match:
            return

        hours, minutes, secs, millis = match.groups()
        hours = int(hours) if hours else 0
        current_sec = hours * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000.0

        if current_sec > self.max_sec_seen:
            self.pbar.update(current_sec - self.max_sec_seen)
            self.max_sec_seen = current_sec

    def flush(self):
        pass


def sanitize_filename(name):
    sanitized = INVALID_FILENAME_CHARS.sub("_", name).strip().rstrip(".")
    return sanitized or "bilibili_video"


def is_url(value):
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def is_bilibili_url(value):
    hostname = urlparse(value).netloc.lower().split("@")[-1].split(":")[0]
    return hostname == "b23.tv" or hostname == "bilibili.com" or hostname.endswith(".bilibili.com")


def detect_host_vendor():
    hints = " ".join(
        filter(
            None,
            [
                os.environ.get("PROCESSOR_IDENTIFIER"),
                platform.processor(),
                platform.machine(),
            ],
        )
    ).lower()

    if "intel" in hints:
        return "intel"
    if "amd" in hints or "ryzen" in hints:
        return "amd"
    return "unknown"


def is_rocm_available():
    return torch.cuda.is_available() and bool(getattr(torch.version, "hip", None))


def is_cuda_available():
    return torch.cuda.is_available() and not is_rocm_available()


def can_use_openvino():
    try:
        import huggingface_hub  # noqa: F401
        import openvino_genai  # noqa: F401
        return True
    except ImportError:
        return False


def get_whispercpp_binary_candidates(explicit_path=None):
    executable_name = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    candidates = [
        explicit_path,
        os.environ.get("WHISPERCPP_BINARY"),
        os.path.join(WHISPER_RUNTIME_ROOT, executable_name),
        os.path.join(WHISPER_RUNTIME_ROOT, "build", "bin", executable_name),
        os.path.join(WHISPER_RUNTIME_ROOT, "build", "bin", "Release", executable_name),
        os.path.join(WHISPER_RUNTIME_ROOT, "build", "bin", "RelWithDebInfo", executable_name),
        os.path.join(WHISPER_RUNTIME_ROOT, "bin", executable_name),
        os.path.join(WHISPER_RUNTIME_ROOT, "bin", "Release", executable_name),
        os.path.join(SCRIPT_DIR, executable_name),
        shutil.which(executable_name),
        shutil.which("whisper-cli"),
    ]
    normalized = []
    for candidate in candidates:
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def resolve_whispercpp_binary(explicit_path=None):
    for candidate in get_whispercpp_binary_candidates(explicit_path):
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def can_use_whispercpp_vulkan(explicit_path=None):
    return bool(resolve_whispercpp_binary(explicit_path))


def normalize_hf_endpoint(endpoint):
    if not endpoint:
        return None
    return endpoint.strip().rstrip("/")


def get_effective_hf_endpoint(cli_endpoint=None):
    return normalize_hf_endpoint(cli_endpoint or os.environ.get("HF_ENDPOINT"))


def parse_positive_int(raw_value):
    if raw_value in (None, ""):
        return None

    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError):
        return None

    return parsed_value if parsed_value > 0 else None


def parse_beam_size_arg(raw_value):
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("beam size 必须是正整数。") from exc

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("beam size 必须大于 0。")

    return parsed_value


def get_effective_hf_timeouts(cli_timeout=None):
    cli_value = parse_positive_int(cli_timeout)
    if cli_value is not None:
        return cli_value, cli_value

    env_etag_timeout = parse_positive_int(os.environ.get("HF_HUB_ETAG_TIMEOUT"))
    env_download_timeout = parse_positive_int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"))

    return (
        env_etag_timeout or DEFAULT_HF_ETAG_TIMEOUT,
        env_download_timeout or DEFAULT_HF_DOWNLOAD_TIMEOUT,
    )


def build_hf_endpoint_candidates(preferred_endpoint=None):
    candidates = []
    for endpoint in [preferred_endpoint, DEFAULT_HF_ENDPOINT, DEFAULT_HF_MIRROR_ENDPOINT]:
        normalized = normalize_hf_endpoint(endpoint)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def format_hf_endpoint_label(endpoint):
    if endpoint == DEFAULT_HF_ENDPOINT:
        return "Hugging Face 官方源"
    if endpoint == DEFAULT_HF_MIRROR_ENDPOINT:
        return "中国大陆镜像 hf-mirror.com"
    return endpoint


def is_likely_network_error(exc):
    error_text = str(exc).lower()
    network_markers = [
        "connecttimeout",
        "readtimeout",
        "timed out",
        "winerror 10060",
        "max retries exceeded",
        "temporary failure in name resolution",
        "name or service not known",
        "failed to establish a new connection",
        "connection aborted",
        "connection reset",
        "proxyerror",
    ]
    return any(marker in error_text for marker in network_markers)


def get_installed_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def format_openvino_runtime_versions():
    package_names = ["openvino", "openvino-genai", "openvino-tokenizers"]
    version_items = []

    for package_name in package_names:
        version_text = get_installed_package_version(package_name)
        if version_text:
            version_items.append(f"{package_name}={version_text}")

    return ", ".join(version_items) if version_items else "unknown"


def format_openvino_download_failure(model_repo, model_dir, endpoint_errors, etag_timeout, download_timeout):
    attempted = " -> ".join(format_hf_endpoint_label(endpoint) for endpoint, _ in endpoint_errors)
    lines = [
        f"OpenVINO Whisper 模型下载失败: {model_repo}",
        f"缓存目录: {model_dir}",
        f"已尝试地址: {attempted}",
        f"当前超时设置: metadata={etag_timeout}s, download={download_timeout}s",
    ]

    for endpoint, exc in endpoint_errors:
        lines.append(f"- {format_hf_endpoint_label(endpoint)}: {exc}")

    lines.extend(
        [
            "可尝试：",
            "1. 在中国大陆网络下加 `--hf-endpoint https://hf-mirror.com`。",
            "2. 慢网环境下加 `--hf-timeout 60` 或更大。",
            "3. 如果之前已下载过完整模型，确认 `cache/openvino/<model>/` 下的 .bin / .xml 文件是否齐全。",
        ]
    )

    return "\n".join(lines)


def get_whispercpp_model_cache_dir():
    return os.path.join(WHISPERCPP_CACHE_ROOT, "models")


def get_whispercpp_model_filename(model_name):
    if model_name not in WHISPERCPP_MODEL_FILENAMES:
        supported = ", ".join(WHISPERCPP_MODEL_FILENAMES.keys())
        raise RuntimeError(f"whisper.cpp Vulkan 后端当前仅支持 {supported}，收到的是 '{model_name}'。")
    return WHISPERCPP_MODEL_FILENAMES[model_name]


def get_whispercpp_model_path(model_name):
    return os.path.join(get_whispercpp_model_cache_dir(), get_whispercpp_model_filename(model_name))


def format_whispercpp_download_failure(model_filename, model_dir, endpoint_errors, etag_timeout, download_timeout):
    attempted = " -> ".join(format_hf_endpoint_label(endpoint) for endpoint, _ in endpoint_errors)
    lines = [
        f"whisper.cpp 模型下载失败: {model_filename}",
        f"缓存目录: {model_dir}",
        f"已尝试地址: {attempted}",
        f"当前超时设置: metadata={etag_timeout}s, download={download_timeout}s",
    ]

    for endpoint, exc in endpoint_errors:
        lines.append(f"- {format_hf_endpoint_label(endpoint)}: {exc}")

    lines.extend(
        [
            "可尝试：",
            "1. 在中国大陆网络下加 `--hf-endpoint https://hf-mirror.com`。",
            "2. 慢网环境下加 `--hf-timeout 60` 或更大。",
            "3. 也可以手工下载 ggml 模型后放到 `cache/whispercpp/models/`。",
        ]
    )
    return "\n".join(lines)


def ensure_whispercpp_model(model_name, hf_endpoint=None, hf_timeout=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少依赖 huggingface_hub，请先执行 `pip install -r requirements.txt`。") from exc

    model_dir = get_whispercpp_model_cache_dir()
    model_filename = get_whispercpp_model_filename(model_name)
    model_path = os.path.join(model_dir, model_filename)
    if os.path.exists(model_path):
        return model_path

    os.makedirs(model_dir, exist_ok=True)
    endpoint_candidates = build_hf_endpoint_candidates(hf_endpoint)
    etag_timeout, download_timeout = get_effective_hf_timeouts(hf_timeout)

    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(etag_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout)

    print(f"正在下载 whisper.cpp 模型: {model_filename}")
    print(f"metadata 超时: {etag_timeout}s, download 超时: {download_timeout}s")

    endpoint_errors = []
    for endpoint in endpoint_candidates:
        print(f"正在尝试模型下载源: {format_hf_endpoint_label(endpoint)}")
        try:
            snapshot_download(
                repo_id=WHISPERCPP_MODEL_REPO_ID,
                local_dir=model_dir,
                allow_patterns=[model_filename],
                endpoint=endpoint,
                etag_timeout=etag_timeout,
                max_workers=4,
            )
            break
        except Exception as exc:
            endpoint_errors.append((endpoint, exc))
            if os.path.exists(model_path):
                print("检测到本地 whisper.cpp 模型缓存，继续使用本地缓存。")
                return model_path

            if is_likely_network_error(exc):
                print(f"当前下载源连接超时或失败，准备尝试下一个地址: {exc}")
            else:
                print(f"当前下载源失败，准备尝试下一个地址: {exc}")
    else:
        raise RuntimeError(
            format_whispercpp_download_failure(
                model_filename,
                model_dir,
                endpoint_errors,
                etag_timeout,
                download_timeout,
            )
        )

    if not os.path.exists(model_path):
        raise RuntimeError("whisper.cpp 模型下载完成，但缓存目录中仍未找到目标 ggml 模型文件。")

    print(f"✅ whisper.cpp 模型已缓存到: {model_path}")
    return model_path


def get_execution_backend(preferred_backend, model_name, whispercpp_binary=None):
    if preferred_backend == "cuda":
        if not is_cuda_available():
            raise RuntimeError("当前环境没有可用的 NVIDIA CUDA。")
        return "cuda", "NVIDIA GPU (CUDA 加速)"

    if preferred_backend == "rocm":
        if not is_rocm_available():
            raise RuntimeError("当前环境没有检测到 AMD ROCm PyTorch。")
        return "rocm", "AMD GPU (ROCm 加速)"

    if preferred_backend == "openvino":
        if not can_use_openvino():
            raise RuntimeError("当前环境缺少 OpenVINO 依赖，请先执行 `pip install -r requirements.txt`。")
        if model_name not in OPENVINO_WHISPER_MODELS:
            supported = ", ".join(OPENVINO_WHISPER_MODELS.keys())
            raise RuntimeError(f"OpenVINO 后端当前仅支持 {supported}，收到的是 '{model_name}'。")
        return "openvino", "Intel OpenVINO 后端"

    if preferred_backend == "whispercpp-vulkan":
        if not can_use_whispercpp_vulkan(whispercpp_binary):
            raise RuntimeError(
                "当前环境没有找到 whisper.cpp 的 `whisper-cli` 可执行文件。"
                " 请先把运行环境放到项目根目录的 `whisper/` 下，或通过 `--whispercpp-binary` 指定路径。"
            )
        if model_name not in WHISPERCPP_MODEL_FILENAMES:
            supported = ", ".join(WHISPERCPP_MODEL_FILENAMES.keys())
            raise RuntimeError(f"whisper.cpp Vulkan 后端当前仅支持 {supported}，收到的是 '{model_name}'。")
        return "whispercpp-vulkan", "whisper.cpp 路径 (期望 Vulkan GPU)"

    if preferred_backend == "cpu":
        return "cpu", "纯 CPU"

    if is_rocm_available():
        return "rocm", "AMD GPU (ROCm 加速)"
    if is_cuda_available():
        return "cuda", "NVIDIA GPU (CUDA 加速)"
    if detect_host_vendor() == "amd" and can_use_whispercpp_vulkan(whispercpp_binary):
        return "whispercpp-vulkan", "AMD 路径 (whisper.cpp，期望 Vulkan GPU)"
    if detect_host_vendor() == "intel" and can_use_openvino() and model_name in OPENVINO_WHISPER_MODELS:
        return "openvino", "Intel OpenVINO 后端"
    return "cpu", "纯 CPU"


def get_openvino_model_cache_dir(model_name):
    return os.path.join(OPENVINO_CACHE_ROOT, model_name)


def is_openvino_model_ready(model_dir):
    return all(os.path.exists(os.path.join(model_dir, file_name)) for file_name in OPENVINO_REQUIRED_FILES)


def ensure_openvino_model(model_name, hf_endpoint=None, hf_timeout=None):
    if model_name not in OPENVINO_WHISPER_MODELS:
        supported = ", ".join(OPENVINO_WHISPER_MODELS.keys())
        raise RuntimeError(f"OpenVINO 后端当前仅支持 {supported}，收到的是 '{model_name}'。")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少依赖 huggingface_hub，请先执行 `pip install -r requirements.txt`。") from exc

    model_dir = get_openvino_model_cache_dir(model_name)
    if is_openvino_model_ready(model_dir):
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    model_repo = OPENVINO_WHISPER_MODELS[model_name]
    endpoint_candidates = build_hf_endpoint_candidates(hf_endpoint)
    etag_timeout, download_timeout = get_effective_hf_timeouts(hf_timeout)

    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(etag_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout)

    print(f"正在下载 OpenVINO Whisper 模型: {model_repo}")
    print(f"metadata 超时: {etag_timeout}s, download 超时: {download_timeout}s")

    endpoint_errors = []
    for endpoint in endpoint_candidates:
        print(f"正在尝试模型下载源: {format_hf_endpoint_label(endpoint)}")
        try:
            snapshot_download(
                repo_id=model_repo,
                local_dir=model_dir,
                endpoint=endpoint,
                etag_timeout=etag_timeout,
                max_workers=4,
            )
            break
        except Exception as exc:
            endpoint_errors.append((endpoint, exc))
            if is_openvino_model_ready(model_dir):
                print("检测到完整本地模型缓存，继续使用本地缓存。")
                return model_dir

            if is_likely_network_error(exc):
                print(f"当前下载源连接超时或失败，准备尝试下一个地址: {exc}")
            else:
                print(f"当前下载源失败，准备尝试下一个地址: {exc}")
    else:
        raise RuntimeError(
            format_openvino_download_failure(
                model_repo,
                model_dir,
                endpoint_errors,
                etag_timeout,
                download_timeout,
            )
        )

    if not is_openvino_model_ready(model_dir):
        raise RuntimeError("OpenVINO Whisper 模型下载完成，但缓存目录中的关键文件仍不完整。")

    print(f"✅ OpenVINO Whisper 模型已缓存到: {model_dir}")
    return model_dir


def get_openvino_device_candidates(preferred_device):
    preferred_device = preferred_device.upper()
    if preferred_device != "AUTO":
        return [preferred_device]

    if detect_host_vendor() == "intel":
        return ["GPU", "CPU"]
    return ["CPU"]


def load_openvino_pipeline(model_name, preferred_device, hf_endpoint=None, hf_timeout=None):
    try:
        import openvino_genai as ov_genai
    except ImportError as exc:
        raise RuntimeError("缺少依赖 openvino-genai，请先执行 `pip install -r requirements.txt`。") from exc

    model_dir = ensure_openvino_model(model_name, hf_endpoint=hf_endpoint, hf_timeout=hf_timeout)
    last_error = None

    for device_name in get_openvino_device_candidates(preferred_device):
        try:
            pipeline = ov_genai.WhisperPipeline(model_dir, device_name)
            return pipeline, device_name
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"OpenVINO 后端初始化失败: {last_error}")


def build_openvino_generation_config(ov_pipeline, beam_size):
    generation_config = ov_pipeline.get_generation_config()
    generation_config.return_timestamps = True
    generation_config.task = "transcribe"
    generation_config.language = OPENVINO_LANGUAGE_TOKEN
    generation_config.num_beams = beam_size
    return generation_config


def run_openvino_transcription(ov_pipeline, audio_samples, beam_size):
    generation_config = build_openvino_generation_config(ov_pipeline, beam_size)
    return ov_pipeline.generate(audio_samples, generation_config)


def is_openvino_beam_search_known_issue(exc, beam_size):
    if beam_size <= 1:
        return False

    error_text = str(exc).lower()
    markers = [
        "beam idx batch",
        "b == b_state",
        "scaleddotproductattentionwithkvcache",
        "not implemented",
        "iremote_tensor",
        "remote_tensor",
    ]
    return any(marker in error_text for marker in markers)


def format_openvino_beam_search_failure(exc, device_name, beam_size):
    versions = format_openvino_runtime_versions()
    lines = [
        f"OpenVINO beam search 运行失败: device={device_name}, beam_size={beam_size}",
        f"当前 OpenVINO 版本: {versions}",
        f"原始错误: {exc}",
        "可尝试：",
        "1. 升级 `openvino` / `openvino-genai` / `openvino-tokenizers` 到同一 2025.2+ 或更新版本。",
        "2. 保持 beam search，但改用 `--openvino-device CPU`。",
        "3. 如果优先追求 Intel GPU 速度，改用 `--beam-size 1` 关闭 beam search。",
    ]
    return "\n".join(lines)


def load_whispercpp_runtime(model_name, whispercpp_binary=None, hf_endpoint=None, hf_timeout=None):
    binary_path = resolve_whispercpp_binary(whispercpp_binary)
    if not binary_path:
        raise RuntimeError(
            "没有找到 whisper.cpp 的 `whisper-cli` 可执行文件。"
            " 请先把运行环境放到项目根目录的 `whisper/` 下，或通过 `--whispercpp-binary` 指定路径。"
        )

    model_path = ensure_whispercpp_model(model_name, hf_endpoint=hf_endpoint, hf_timeout=hf_timeout)
    return binary_path, model_path


def parse_srt_timestamp(raw_value):
    hours_text, minutes_text, seconds_text = raw_value.split(":")
    seconds_text, millis_text = seconds_text.split(",")
    return (
        int(hours_text) * 3600
        + int(minutes_text) * 60
        + int(seconds_text)
        + int(millis_text) / 1000.0
    )


def parse_srt_segments(srt_path):
    with open(srt_path, "r", encoding="utf-8") as srt_file:
        blocks = re.split(r"\r?\n\r?\n", srt_file.read().strip())

    segments = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        timing_line = lines[1]
        match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})",
            timing_line,
        )
        if not match:
            continue

        text = " ".join(lines[2:]).strip()
        if not text:
            continue

        segments.append(
            {
                "start": parse_srt_timestamp(match.group(1)),
                "end": parse_srt_timestamp(match.group(2)),
                "text": text,
            }
        )

    return segments


def format_whispercpp_failure(command, returncode, stderr_text, stdout_text):
    lines = [
        "whisper.cpp 转写失败。",
        f"退出码: {returncode}",
        f"命令: {' '.join(command)}",
    ]

    if stderr_text.strip():
        lines.append(f"stderr: {stderr_text.strip()}")
    if stdout_text.strip():
        lines.append(f"stdout: {stdout_text.strip()}")

    if returncode == 3221225781:
        lines.append(
            "诊断：Windows 退出码 3221225781 (0xC0000135) 通常表示程序依赖的 DLL 或运行时组件缺失。"
        )
        lines.append(
            "优先检查：不要只单独复制 `whisper-cli.exe`，应当保留同一构建/发布目录中的配套 DLL，并优先使用项目根目录 `whisper/` 下的整套运行环境。"
        )

    lines.extend(
        [
            "可尝试：",
            "1. 确认 `whisper-cli` 是用 `-DGGML_VULKAN=1` 构建的 Vulkan 版本。",
            "2. 确认系统 Vulkan 运行时和 AMD 驱动正常。",
            "3. 如果只是要先跑通流程，可改用 `--backend cpu` 或已支持的 `--backend rocm`。",
        ]
    )
    return "\n".join(lines)


def detect_whispercpp_runtime_mode(stdout_text, stderr_text):
    combined_text = f"{stdout_text}\n{stderr_text}".lower()

    if "ggml_vulkan:" in combined_text:
        return "vulkan"

    if "no gpu found" in combined_text or "whisper_backend_init_gpu: device 0: cpu" in combined_text:
        return "cpu-fallback"

    if "use gpu" in combined_text and "ggml_vulkan" not in combined_text:
        return "unknown-gpu"

    return "unknown"


def format_whispercpp_gpu_required_failure(binary_path, runtime_mode, stdout_text, stderr_text):
    lines = [
        "whisper.cpp Vulkan 后端没有成功启用 GPU，已停止转写。",
        f"whisper-cli: {binary_path}",
        f"检测结果: {runtime_mode}",
        "原因：当前选择的是 `--backend whispercpp-vulkan`，脚本要求日志中出现 `ggml_vulkan:` 才继续运行。",
    ]

    if stderr_text.strip():
        lines.append(f"stderr: {stderr_text.strip()}")
    if stdout_text.strip():
        lines.append(f"stdout: {stdout_text.strip()}")

    lines.extend(
        [
            "可尝试：",
            "1. 重新构建 whisper.cpp：`cmake -B build -DGGML_VULKAN=1`，并使用 build 输出目录中的 `whisper-cli`。",
            "2. 确认 AMD 驱动和 Vulkan Runtime 正常，手工运行时应能看到 `ggml_vulkan: Found ...`。",
            "3. 如果只想先跑通转写，请改用 `--backend cpu`。",
        ]
    )
    return "\n".join(lines)


def build_whispercpp_subprocess_env(binary_path):
    env = os.environ.copy()
    path_entries = [
        os.path.dirname(binary_path),
        WHISPER_RUNTIME_ROOT,
        os.path.join(WHISPER_RUNTIME_ROOT, "bin"),
        os.path.join(WHISPER_RUNTIME_ROOT, "build", "bin"),
        os.path.join(WHISPER_RUNTIME_ROOT, "build", "bin", "Release"),
        os.path.join(WHISPER_RUNTIME_ROOT, "build", "bin", "RelWithDebInfo"),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)
    return env


def run_whispercpp_transcription(binary_path, model_path, chunk_file, beam_size, temp_dir):
    chunk_file = os.path.abspath(chunk_file)
    model_path = os.path.abspath(model_path)
    output_base = os.path.abspath(os.path.join(temp_dir, os.path.splitext(os.path.basename(chunk_file))[0]))
    srt_path = f"{output_base}.srt"

    command = [
        binary_path,
        "-m",
        model_path,
        "-f",
        chunk_file,
        "-l",
        "zh",
        "-bs",
        str(beam_size),
        "-osrt",
        "-of",
        output_base,
        "-np",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=WHISPER_RUNTIME_ROOT if os.path.isdir(WHISPER_RUNTIME_ROOT) else os.path.dirname(binary_path),
        env=build_whispercpp_subprocess_env(binary_path),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            format_whispercpp_failure(command, completed.returncode, completed.stderr, completed.stdout)
        )

    if not os.path.exists(srt_path):
        raise RuntimeError("whisper.cpp 运行完成，但没有生成预期的 SRT 输出文件。")

    return {
        "segments": parse_srt_segments(srt_path),
        "runtime_mode": detect_whispercpp_runtime_mode(completed.stdout, completed.stderr),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def load_whisper_model(model_name, backend_kind):
    device_name = "cuda" if backend_kind in {"cuda", "rocm"} else "cpu"
    return whisper.load_model(model_name, device=device_name)


def load_wav_samples_for_openvino(file_path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("缺少依赖 numpy，请先执行 `pip install -r requirements.txt`。") from exc

    with wave.open(file_path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise RuntimeError("OpenVINO 输入 wav 必须是 16kHz / 16-bit / 单声道 PCM。")

    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def normalize_bilibili_page(page_value):
    if not page_value:
        return None

    try:
        page_number = int(page_value)
        return None if page_number <= 1 else str(page_number)
    except (TypeError, ValueError):
        return str(page_value)


def get_bilibili_cache_key(url):
    parsed = urlparse(url)
    hostname = parsed.netloc.lower().split("@")[-1].split(":")[0]
    path = parsed.path.rstrip("/")
    query = parse_qs(parsed.query)

    video_match = re.search(r"/video/((?:BV|av)[A-Za-z0-9]+)", path, re.IGNORECASE)
    if video_match:
        video_id = video_match.group(1)
        page = normalize_bilibili_page(query.get("p", [None])[0])
        return sanitize_filename(f"{video_id}_p{page}" if page else video_id)

    if hostname == "b23.tv":
        short_code = path.strip("/") or "index"
        return sanitize_filename(f"b23_{short_code}")

    normalized_url = f"{hostname}{path}"
    page = normalize_bilibili_page(query.get("p", [None])[0])
    if page:
        normalized_url = f"{normalized_url}?p={page}"

    digest = hashlib.sha1(normalized_url.encode("utf-8")).hexdigest()[:16]
    return f"bilibili_{digest}"


def get_bilibili_cache_paths(url):
    cache_dir = os.path.join(BILIBILI_CACHE_ROOT, get_bilibili_cache_key(url))
    metadata_path = os.path.join(cache_dir, "metadata.json")
    return cache_dir, metadata_path


def find_cached_media_file(cache_dir, require_video=False):
    patterns = [
        os.path.join(cache_dir, "source_video.*"),
        os.path.join(cache_dir, "source.*"),
    ]
    candidates = []
    seen = set()

    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if candidate in seen or not os.path.isfile(candidate):
                continue
            seen.add(candidate)

            extension = os.path.splitext(candidate)[1].lower()
            if extension in {".json", ".part", ".temp", ".ytdl"}:
                continue
            if require_video and extension not in BILIBILI_VIDEO_EXTENSIONS:
                continue

            candidates.append(candidate)

    return candidates[0] if candidates else None

    return None


def load_cached_bilibili_media(url):
    cache_dir, metadata_path = get_bilibili_cache_paths(url)
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError):
            metadata = {}

    media_path = find_cached_media_file(cache_dir, require_video=True)
    if not media_path:
        legacy_media_path = find_cached_media_file(cache_dir)
        if legacy_media_path:
            print(f"检测到旧版音频缓存，将重新下载视频缓存: {legacy_media_path}")
        return None, None

    if metadata.get("download_kind") != BILIBILI_DOWNLOAD_KIND:
        print(f"检测到旧版 B 站缓存，将重新下载视频缓存: {media_path}")
        return None, None

    output_stem = sanitize_filename(
        metadata.get("output_stem")
        or metadata.get("title")
        or os.path.splitext(os.path.basename(media_path))[0]
    )
    return media_path, output_stem


def write_bilibili_cache_metadata(metadata_path, source_url, info, output_stem):
    metadata = {
        "source_url": source_url,
        "resolved_url": info.get("webpage_url") or source_url,
        "video_id": info.get("id"),
        "title": info.get("title"),
        "output_stem": output_stem,
        "download_kind": BILIBILI_DOWNLOAD_KIND,
        "format_id": info.get("format_id"),
        "ext": info.get("ext"),
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)


def build_bilibili_ydl_opts(cache_dir, cookies_path=None):
    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(cache_dir, "source_video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path

    return ydl_opts


def download_bilibili_media(url, cookies_path=None):
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("缺少依赖 yt-dlp，请先执行 `pip install -r requirements.txt`。") from exc

    if cookies_path and not os.path.exists(cookies_path):
        raise FileNotFoundError(f"找不到 cookies 文件: '{cookies_path}'")

    cached_media_path, cached_output_stem = load_cached_bilibili_media(url)
    if cached_media_path:
        print(f"检测到本地缓存，跳过下载: {cached_media_path}")
        return cached_media_path, cached_output_stem

    cache_dir, metadata_path = get_bilibili_cache_paths(url)
    os.makedirs(cache_dir, exist_ok=True)

    print("检测到 B 站链接，正在下载源视频...")
    if cookies_path:
        print(f"正在使用 cookies 文件: {cookies_path}")

    ydl_opts = build_bilibili_ydl_opts(cache_dir, cookies_path)
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise RuntimeError(f"B 站视频下载失败: {exc}") from exc

    downloaded_media_path = find_cached_media_file(cache_dir, require_video=True)
    if not downloaded_media_path:
        raise RuntimeError("B 站视频下载完成，但没有找到合并后的视频文件。")

    title = info.get("title") or info.get("id") or "bilibili_video"
    output_stem = sanitize_filename(title)
    write_bilibili_cache_metadata(metadata_path, url, info, output_stem)

    print(f"✅ B 站视频下载完成，已缓存到: {cache_dir}")
    return downloaded_media_path, output_stem


def resolve_input_source(input_value, cookies_path=None):
    if is_url(input_value):
        if not is_bilibili_url(input_value):
            raise ValueError("当前仅支持直接传入 B 站视频链接（bilibili.com 或 b23.tv）。")

        media_path, output_stem = download_bilibili_media(input_value, cookies_path)
        return media_path, output_stem, os.getcwd()

    if not os.path.exists(input_value):
        raise FileNotFoundError(f"找不到文件: '{input_value}'")

    output_stem = os.path.splitext(os.path.basename(input_value))[0]
    output_dir = os.path.dirname(os.path.abspath(input_value))
    return input_value, output_stem, output_dir


def cleanup_temp_audio_files():
    for chunk_file in glob.glob(TEMP_CHUNK_GLOB):
        try:
            os.remove(chunk_file)
        except OSError:
            pass

    try:
        os.remove(TEMP_FULL_AUDIO_FILE)
    except OSError:
        pass


def preprocess_audio_to_wav(input_path):
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        TEMP_FULL_AUDIO_FILE,
    ]

    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if not os.path.exists(TEMP_FULL_AUDIO_FILE):
        raise RuntimeError("音频预处理失败，请检查输入文件是否有效或 ffmpeg 是否正常工作。")

    return TEMP_FULL_AUDIO_FILE


def split_wav_to_chunks(wav_path):
    for chunk_file in glob.glob(TEMP_CHUNK_GLOB):
        try:
            os.remove(chunk_file)
        except OSError:
            pass

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        wav_path,
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_SECONDS),
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        TEMP_CHUNK_PATTERN,
    ]

    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    chunk_files = sorted(glob.glob(TEMP_CHUNK_GLOB))
    if not chunk_files:
        raise RuntimeError("音频分割失败，请检查输入文件是否有效或 ffmpeg 是否正常工作。")

    return chunk_files


def build_chunk_offsets(wav_files):
    offsets = []
    current_offset = 0.0

    for wav_file in wav_files:
        offsets.append(current_offset)
        current_offset += get_wav_duration(wav_file)

    return offsets


def write_segment(output_file, start_time, end_time, text):
    text = str(text).strip()
    if text:
        output_file.write(f"[{format_time(start_time)} - {format_time(end_time)}] {text}\n")


def transcribe_wav_file(
    wav_file,
    chunk_offset,
    output_file,
    pbar,
    backend_kind,
    ov_pipeline,
    ov_device_name,
    whispercpp_runtime,
    whispercpp_temp_dir,
    whispercpp_runtime_state,
    model,
    use_fp16,
    whisper_beam_size,
    beam_size,
):
    chunk_duration = get_wav_duration(wav_file)

    if backend_kind == "openvino":
        audio_samples = load_wav_samples_for_openvino(wav_file)
        try:
            result = run_openvino_transcription(ov_pipeline, audio_samples, beam_size)
        except Exception as exc:
            if is_openvino_beam_search_known_issue(exc, beam_size):
                raise RuntimeError(
                    format_openvino_beam_search_failure(exc, ov_device_name, beam_size)
                ) from exc
            raise

        chunks = getattr(result, "chunks", [])
        if chunks:
            for segment in chunks:
                write_segment(
                    output_file,
                    chunk_offset + float(segment.start_ts),
                    chunk_offset + float(segment.end_ts),
                    segment.text,
                )
        else:
            write_segment(
                output_file,
                chunk_offset,
                chunk_offset + chunk_duration,
                getattr(result, "text", result),
            )

        pbar.update(chunk_duration)
        return

    if backend_kind == "whispercpp-vulkan":
        binary_path, model_path = whispercpp_runtime
        whispercpp_result = run_whispercpp_transcription(
            binary_path,
            model_path,
            wav_file,
            beam_size,
            whispercpp_temp_dir,
        )

        if not whispercpp_runtime_state["checked"]:
            runtime_mode = whispercpp_result["runtime_mode"]
            if runtime_mode == "vulkan":
                print("✅ whisper.cpp 已检测到 Vulkan GPU 后端。")
            else:
                raise RuntimeError(
                    format_whispercpp_gpu_required_failure(
                        binary_path,
                        runtime_mode,
                        whispercpp_result["stdout"],
                        whispercpp_result["stderr"],
                    )
                )
            whispercpp_runtime_state["checked"] = True

        for segment in whispercpp_result["segments"]:
            write_segment(
                output_file,
                chunk_offset + segment["start"],
                chunk_offset + segment["end"],
                segment["text"],
            )

        pbar.update(chunk_duration)
        return

    logger = WhisperProgressLogger(pbar)
    old_stdout = sys.stdout
    sys.stdout = logger

    try:
        result = model.transcribe(
            wav_file,
            language="zh",
            beam_size=whisper_beam_size,
            fp16=use_fp16,
            verbose=True,
        )
    finally:
        sys.stdout = old_stdout

    unprocessed = chunk_duration - logger.max_sec_seen
    if unprocessed > 0:
        pbar.update(unprocessed)

    for segment in result.get("segments", []):
        write_segment(
            output_file,
            chunk_offset + segment["start"],
            chunk_offset + segment["end"],
            segment["text"],
        )


def transcribe_wav_files(
    wav_files,
    output_path,
    backend_kind,
    ov_pipeline,
    ov_device_name,
    whispercpp_runtime,
    whispercpp_temp_dir,
    model,
    use_fp16,
    whisper_beam_size,
    beam_size,
):
    total_audio_duration = sum(get_wav_duration(wav_file) for wav_file in wav_files)
    chunk_offsets = build_chunk_offsets(wav_files) if len(wav_files) > 1 else [0.0]
    whispercpp_runtime_state = {"checked": False}

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("")

    pbar_format = "{l_bar}{bar}| 进度: {n:.1f}/{total:.1f} 秒音频 [已耗时: {elapsed}, 预计剩余: {remaining}]"

    with tqdm(total=total_audio_duration, unit="秒", bar_format=pbar_format) as pbar:
        for wav_file, chunk_offset in zip(wav_files, chunk_offsets):
            with open(output_path, "a", encoding="utf-8") as output_file:
                transcribe_wav_file(
                    wav_file,
                    chunk_offset,
                    output_file,
                    pbar,
                    backend_kind,
                    ov_pipeline,
                    ov_device_name,
                    whispercpp_runtime,
                    whispercpp_temp_dir,
                    whispercpp_runtime_state,
                    model,
                    use_fp16,
                    whisper_beam_size,
                    beam_size,
                )

    return total_audio_duration


def should_use_segment_fallback(exc):
    error_text = str(exc).lower()
    non_segment_errors = [
        "whisper.cpp vulkan 后端没有成功启用 gpu",
        "ggml_vulkan",
        "openvino beam search 运行失败",
        "no gpu found",
        "device 0: cpu",
    ]
    return not any(marker in error_text for marker in non_segment_errors)


def transcribe_audio(
    input_path,
    model_name="small",
    output_stem=None,
    output_dir=None,
    backend="auto",
    openvino_device="AUTO",
    whispercpp_binary=None,
    force_segmented=False,
    beam_size=DEFAULT_BEAM_SIZE,
    hf_endpoint=None,
    hf_timeout=None,
):
    backend_kind, backend_name = get_execution_backend(
        backend,
        model_name,
        whispercpp_binary=whispercpp_binary,
    )
    use_fp16 = backend_kind in {"cuda", "rocm"}
    whisper_beam_size = None if beam_size == 1 else beam_size

    print(f"🚀 当前实际运行模式: {backend_name}")
    print(f"🧠 当前解码 beam size: {beam_size}")
    if backend_kind == "whispercpp-vulkan":
        print(f"正在准备 whisper.cpp Vulkan 模型 '{model_name}' (首次运行会自动下载模型，请耐心等待)...")
    else:
        print(f"正在加载 Whisper 模型 '{model_name}' (首次运行会自动下载模型，请耐心等待)...")

    try:
        model = None
        ov_pipeline = None
        ov_device_name = None
        whispercpp_runtime = None

        if backend_kind == "openvino":
            ov_pipeline, ov_device_name = load_openvino_pipeline(
                model_name,
                openvino_device,
                hf_endpoint=hf_endpoint,
                hf_timeout=hf_timeout,
            )
            print(f"✅ OpenVINO 设备初始化成功: {ov_device_name}")
        elif backend_kind == "whispercpp-vulkan":
            whispercpp_runtime = load_whispercpp_runtime(
                model_name,
                whispercpp_binary=whispercpp_binary,
                hf_endpoint=hf_endpoint,
                hf_timeout=hf_timeout,
            )
            print(f"✅ whisper.cpp 可执行文件: {whispercpp_runtime[0]}")
            print(f"✅ whisper.cpp 模型文件: {whispercpp_runtime[1]}")
        else:
            model = load_whisper_model(model_name, backend_kind)
    except Exception as exc:
        print(f"❌ 模型加载失败: {exc}")
        if backend_kind == "openvino":
            print("提示：请确认已安装 openvino / openvino-genai / huggingface_hub。")
            print("提示：如果是大陆网络，建议加 `--hf-endpoint https://hf-mirror.com`。")
            print("提示：如果是慢网，建议加 `--hf-timeout 60`。")
        elif backend_kind == "whispercpp-vulkan":
            print("提示：请确认已把支持 Vulkan 的 whisper.cpp 运行环境放到项目根目录的 `whisper/` 下，或通过 `--whispercpp-binary` 指向 `whisper-cli`。")
            print("提示：模型会缓存到 `cache/whispercpp/models/`，也支持配合 `--hf-endpoint https://hf-mirror.com`。")
        elif backend_kind == "rocm":
            print("提示：请确认当前安装的是 AMD 官方 ROCm PyTorch，并且硬件在 ROCm 支持矩阵内。")
        else:
            print("提示：如果遇到网络问题，请检查网络连接或代理设置。")
        return

    print("正在预处理音频：转换为完整 WAV...")
    base_name = sanitize_filename(output_stem or os.path.splitext(os.path.basename(input_path))[0])
    output_directory = output_dir or os.getcwd()
    output_path = os.path.join(output_directory, f"{base_name}_转写结果.txt")

    process_start_time = time.time()
    whispercpp_temp_dir = None
    if backend_kind == "whispercpp-vulkan":
        whispercpp_temp_dir = tempfile.mkdtemp(prefix="whispercpp_", dir=SCRIPT_DIR)

    try:
        cleanup_temp_audio_files()
        full_audio_path = preprocess_audio_to_wav(input_path)
        total_audio_duration = get_wav_duration(full_audio_path)
        print(f"✅ 音频预处理完毕，总时长约 {total_audio_duration / 60:.1f} 分钟。开始智能识别...\n")

        if force_segmented:
            print("当前已启用强制分段模式，将使用分段备用路径。")
            chunk_files = split_wav_to_chunks(full_audio_path)
            transcribe_wav_files(
                chunk_files,
                output_path,
                backend_kind,
                ov_pipeline,
                ov_device_name,
                whispercpp_runtime,
                whispercpp_temp_dir,
                model,
                use_fp16,
                whisper_beam_size,
                beam_size,
            )
        else:
            try:
                transcribe_wav_files(
                    [full_audio_path],
                    output_path,
                    backend_kind,
                    ov_pipeline,
                    ov_device_name,
                    whispercpp_runtime,
                    whispercpp_temp_dir,
                    model,
                    use_fp16,
                    whisper_beam_size,
                    beam_size,
                )
            except Exception as exc:
                if not should_use_segment_fallback(exc):
                    raise

                print(f"\n⚠️ 整体转写失败，准备启用分段备用方案: {exc}")
                print("分段时间戳将按实际片段时长累计，避免固定 600 秒偏移造成漂移。\n")
                chunk_files = split_wav_to_chunks(full_audio_path)
                transcribe_wav_files(
                    chunk_files,
                    output_path,
                    backend_kind,
                    ov_pipeline,
                    ov_device_name,
                    whispercpp_runtime,
                    whispercpp_temp_dir,
                    model,
                    use_fp16,
                    whisper_beam_size,
                    beam_size,
                )

    except Exception as exc:
        sys.stdout = sys.__stdout__
        print(f"\n❌ 识别过程出错: {exc}")
        return
    finally:
        cleanup_temp_audio_files()
        if whispercpp_temp_dir:
            shutil.rmtree(whispercpp_temp_dir, ignore_errors=True)

    elapsed = time.time() - process_start_time
    print(f"\n🎉 识别完成，总处理耗时: {elapsed:.2f} 秒")
    print(f"📄 转写结果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="本地音视频转文字工具（支持 Whisper / whisper.cpp Vulkan / OpenVINO / B 站链接）"
    )
    parser.add_argument("input_file", help="本地音视频文件路径，或 B 站视频链接")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper 模型大小，默认 small",
    )
    parser.add_argument("--cookies", help="可选：B 站 cookies.txt 文件路径")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "cuda", "rocm", "openvino", "whispercpp-vulkan"],
        help="推理后端。auto 会优先尝试 AMD ROCm，其次 NVIDIA CUDA，再其次 AMD whisper.cpp Vulkan，再其次 Intel OpenVINO，最后回退 CPU。",
    )
    parser.add_argument(
        "--openvino-device",
        default="AUTO",
        choices=["AUTO", "CPU", "GPU", "NPU"],
        help="当 backend=openvino 时使用的设备。AUTO 在 Intel 机器上会优先尝试 GPU，再回退 CPU。",
    )
    parser.add_argument(
        "--whispercpp-binary",
        help="可选：指定 whisper.cpp 的 `whisper-cli` 可执行文件路径。若不传，脚本会优先在项目根目录的 `whisper/` 及其 build/bin 下自动查找。",
    )
    parser.add_argument(
        "--beam-size",
        type=parse_beam_size_arg,
        default=DEFAULT_BEAM_SIZE,
        help="解码 beam size，默认 5。传 1 可关闭 beam search。",
    )
    parser.add_argument(
        "--segment-audio",
        action="store_true",
        help="强制使用旧的分段转写路径。默认会先整体转写，整体失败后才自动分段备用。",
    )
    parser.add_argument(
        "--hf-endpoint",
        help="可选：指定 Hugging Face Hub 地址或镜像源，例如 https://hf-mirror.com 。",
    )
    parser.add_argument(
        "--hf-timeout",
        type=int,
        help="可选：指定 Hugging Face metadata / download 超时秒数，例如 60。",
    )

    args = parser.parse_args()

    try:
        resolved_input_path, output_stem, output_dir = resolve_input_source(args.input_file, args.cookies)
        transcribe_audio(
            resolved_input_path,
            model_name=args.model,
            output_stem=output_stem,
            output_dir=output_dir,
            backend=args.backend,
            openvino_device=args.openvino_device,
            whispercpp_binary=args.whispercpp_binary,
            force_segmented=args.segment_audio,
            beam_size=args.beam_size,
            hf_endpoint=get_effective_hf_endpoint(args.hf_endpoint),
            hf_timeout=args.hf_timeout,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"❌ 错误: {exc}")
