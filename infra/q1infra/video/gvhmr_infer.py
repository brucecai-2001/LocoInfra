"""在 GVHMR 环境里运行（工作目录 = GVHMR 根目录）：视频 → SMPL-X 参数。

    python gvhmr_infer.py <video> <out_dir> [--static-cam] [--f-mm 24] [--device auto|cuda|mps|cpu]

GVHMR 的 demo 把输入逐帧拷贝并当作 30 fps，所以先用 ffmpeg 重采样到 30 fps。
只跑检测跟踪、ViTPose、ViT 特征、（动态相机时）视觉里程计和 GVHMR 预测，不渲染。
out_dir 写 0_input_video.mp4（30 fps）、preprocess/、bbx.pt、vitpose.pt、hmr4d_results.pt、meta.json。

GVHMR 代码里写死了 CUDA（.cuda()、YOLO device="cuda"）。设备不是 CUDA 时（macOS 的 MPS / CPU），
use_device() 在本进程内把这些调用改到所选设备，GVHMR 仓库不用改。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 不支持的算子退回 CPU

import torch  # noqa: E402


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


_dev = torch.device("cuda")  # GVHMR 里 .cuda() 实际去的设备


def use_device(device: str) -> None:
    global _dev
    _dev = torch.device(device)
    if device == "cuda":
        return

    def tensor_to(t, *args, **kwargs):
        if _dev.type == "mps" and t.dtype == torch.float64:  # MPS 没有 float64
            t = t.float()
        return t.to(_dev)

    torch.Tensor.cuda = tensor_to
    torch.nn.Module.cuda = lambda m, *args, **kwargs: m.to(_dev)
    from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin

    _DeviceDtypeModuleMixin.cuda = lambda m, *args, **kwargs: m.to(_dev)  # Lightning 模块自带的 cuda() 不走 nn.Module
    load = torch.load
    torch.load = lambda f, map_location=None, **kw: load(f, map_location=map_location or "cpu", **kw)

    from ultralytics.engine.model import Model

    track = Model.track
    Model.track = lambda self, *args, **kw: track(self, *args, **{**kw, "device": device})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--static-cam", action="store_true")
    p.add_argument("--f-mm", type=int, default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    a = p.parse_args()
    device = pick_device(a.device)
    use_device(device)
    print(f"[gvhmr] device = {device}", flush=True)

    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    v30 = out / "0_input_video.mp4"  # GVHMR 的 cfg.video_path（output_dir = out）
    if not v30.exists():
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(a.video), "-vf", "fps=30", "-an", "-c:v", "libx264", "-crf", "17", str(v30)], check=True)

    # 根目录也显式加上：macOS 会给 venv 里的 .pth 打 hidden 标记，GVHMR 的可编辑安装随之失效
    sys.path[:0] = [str(Path("tools/demo").resolve()), str(Path.cwd())]
    from hydra import compose, initialize_config_module

    import demo
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.video_io_utils import get_video_lwh

    length, width, height = get_video_lwh(v30)
    q = lambda s: "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"  # noqa: E731  路径可能含空格
    overrides = [f"video_name={q(out.name)}", f"static_cam={a.static_cam}", "verbose=False", "use_dpvo=False", f"output_root={q(out.parent)}"]
    if a.f_mm is not None:
        overrides.append(f"f_mm={a.f_mm}")
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)
    assert Path(cfg.video_path) == v30, (cfg.video_path, v30)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    demo.run_preprocess(cfg)
    data = demo.load_data_dict(cfg)
    global _dev
    if _dev.type == "mps":
        _dev = torch.device("cpu")  # GVHMR 主网络在 MPS（torch 2.3）上会卡死在 GPU 同步里；它很小，CPU 够快
    print(f"[gvhmr] predict on {_dev}", flush=True)
    model = __import__("hydra").utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    with torch.no_grad():
        pred = detach_to_cpu(model.predict(data, static_cam=cfg.static_cam))
    torch.save(pred, out / "hmr4d_results.pt")
    torch.save(torch.load(cfg.paths.bbx), out / "bbx.pt")
    torch.save(torch.load(cfg.paths.vitpose), out / "vitpose.pt")
    meta = {"frames": length, "fps": 30, "width": width, "height": height, "static_cam": a.static_cam, "f_mm": a.f_mm, "source_video": a.video.name,
            "device": device}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("GVHMR_DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
