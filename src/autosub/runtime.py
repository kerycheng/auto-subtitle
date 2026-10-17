"""Check the requested compute device; never silently change transcription settings."""
from __future__ import annotations


def check_runtime(device):
    import ctranslate2
    if device not in {"cpu", "cuda"}:
        raise ValueError("運算裝置必須是 cpu 或 cuda")
    compute = "int8" if device == "cpu" else "float16"
    runtime_dirs = []
    try:
        if device == "cuda" and ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("沒有偵測到可用的 NVIDIA CUDA 裝置")
        if device == "cuda":
            from .cuda_runtime import prepare_cuda_runtime
            runtime_dirs = prepare_cuda_runtime()
        supported = ctranslate2.get_supported_compute_types(device)
        if compute not in supported:
            raise RuntimeError(f"裝置不支援 {compute}，支援：{', '.join(sorted(supported))}")
    except Exception as exc:
        raise RuntimeError(runtime_error(device, exc)) from exc
    return {"device": device, "compute_type": compute, "supported": sorted(supported),
            "ctranslate2": ctranslate2.__version__, "cuda_runtime_directories": runtime_dirs}


def runtime_error(device, exc):
    detail = str(exc)
    if detail.startswith(("GPU 辨識無法執行。", "CPU 辨識無法執行；")):
        return detail
    if device == "cuda":
        return ("GPU 辨識無法執行。可從「專案與工具 → GPU 失敗後改用 CPU 繼續」接續已保存區段；"
                "或依目前 CTranslate2 版本安裝相容的 NVIDIA 驅動、CUDA、cuBLAS／cuDNN。"
                "偵測到顯示卡不代表推論所需 DLL 齊全。\n原始錯誤：" + detail)
    return "CPU 辨識無法執行；請檢查模型與執行環境。\n原始錯誤：" + detail
