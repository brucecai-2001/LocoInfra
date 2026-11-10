#!/usr/bin/env bash
# 在 GPU 机器上部署 HY-Motion 1.0 生成服务的环境与权重（需要约 26 GB 显存、约 22 GB 磁盘）。
#
#   git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0 && cd HY-Motion-1.0 && git lfs pull
#   HY_ROOT=$PWD bash <LocoInfra>/infra/q1infra/text2motion/setup_gpu.sh
#
# 【需要使用者处理】
#   * HY-Motion 1.0 以腾讯混元社区许可发布（不适用于欧盟、英国、韩国），使用前请自行阅读并遵守其许可；
#   * HY_ROOT：HY-Motion 仓库根目录（必填）；HY_VENV：虚拟环境（默认 $HY_ROOT/.venv）；
#     HY_CKPT：权重目录（默认 $HY_ROOT/ckpts，磁盘不够时指到大盘，脚本会软链回 $HY_ROOT/ckpts/）；
#   * 权重默认从 ModelScope 下载（国内较快），HF_SOURCE=1 改从 Hugging Face 下载（仓库名以 HY-Motion 官方 README 为准）；
#   * 官方的提示词改写模型 Text2MotionPrompter（Qwen3-30B，61 GB）不下载：服务复用 Qwen3-8B 文本编码器改写中文。
set -euo pipefail
: "${HY_ROOT:?请设置 HY_ROOT=<HY-Motion-1.0 仓库根目录>}"
HY_VENV=${HY_VENV:-$HY_ROOT/.venv}
HY_CKPT=${HY_CKPT:-$HY_ROOT/ckpts}

command -v uv >/dev/null || { echo "需要 uv：https://docs.astral.sh/uv/" >&2; exit 1; }
[ -x "$HY_VENV/bin/python" ] || uv venv --python 3.10 "$HY_VENV"
# fbxsdkpy 只用于导出 FBX（另一个索引），服务不需要
grep -v -e '^--extra-index-url' -e '^fbxsdkpy' "$HY_ROOT/requirements.txt" > /tmp/hymotion_req.txt
uv pip install --python "$HY_VENV/bin/python" -r /tmp/hymotion_req.txt modelscope "huggingface_hub[cli]"
"$HY_VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p "$HY_CKPT"
if [ "${HF_SOURCE:-0}" = 1 ]; then
  HF="$HY_VENV/bin/huggingface-cli"
  "$HF" download tencent/HY-Motion-1.0 --include "HY-Motion-1.0/*" --local-dir "$HY_CKPT/tencent"
  "$HF" download openai/clip-vit-large-patch14 --exclude "*.msgpack" "*.h5" "pytorch_model.bin" --local-dir "$HY_CKPT/clip-vit-large-patch14"
  "$HF" download Qwen/Qwen3-8B --local-dir "$HY_CKPT/Qwen3-8B"
else
  MS="$HY_VENV/bin/modelscope"
  "$MS" download --model Tencent-Hunyuan/HY-Motion-1.0 --include "HY-Motion-1.0/*" --local_dir "$HY_CKPT/tencent"
  "$MS" download --model AI-ModelScope/clip-vit-large-patch14 --exclude "*.msgpack" "*.h5" "pytorch_model.bin" --local_dir "$HY_CKPT/clip-vit-large-patch14"
  "$MS" download --model Qwen/Qwen3-8B --local_dir "$HY_CKPT/Qwen3-8B"
fi

if [ "$(cd "$HY_CKPT" && pwd)" != "$(mkdir -p "$HY_ROOT/ckpts" && cd "$HY_ROOT/ckpts" && pwd)" ]; then
  for d in tencent clip-vit-large-patch14 Qwen3-8B; do
    ln -sfn "$HY_CKPT/$d" "$HY_ROOT/ckpts/$d"
  done
fi
echo "完成。启动服务：cd \$HY_ROOT && \$HY_VENV/bin/python <LocoInfra>/infra/q1infra/text2motion/hymotion_server.py"
