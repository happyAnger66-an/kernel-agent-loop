# Thor 部署（Phase 2.5）

目标机示例：`ssh nvidia@tz -p 2240`（hostname `thor-003`）。

## 布局

```text
/home/nvidia/kal/
  src/kernel-agent-loop/   # 可编辑检出
  trusted/                 # evald 可信树（rsync 自 src）
  jobs/                    # overlay 工作目录
  inbox/                   # SSH 上传的候选包
  venv/                    # Python 环境（含 cuda-python）
  evald.config.json
  hidden_shapes.json       # 只在 Thor 上维护，不进仓库
  gpu.lock
```

`hidden_shapes.json` 不提交到仓库（Agent 能读到仓库就能针对它调参）。bootstrap 首次会从
`hidden_shapes.example.json` 生成一份，之后请直接在 Thor 上改成私有的 shape，再次 bootstrap 不会覆盖。

## 从开发机同步并引导

```bash
# 在开发机仓库根目录
HOST=nvidia@tz
PORT=2240
rsync -az --delete -e "ssh -p $PORT" \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '.kal' \
  ./ ${HOST}:/home/nvidia/kal/src/kernel-agent-loop/

ssh -p $PORT $HOST 'bash /home/nvidia/kal/src/kernel-agent-loop/deploy/thor/bootstrap.sh'
```

## 开发机提交评测

```python
from kernel_agent_loop.evald.client import EvalClient

client = EvalClient(
    mode="ssh",
    host="tz",
    user="nvidia",
    port=2240,
    remote_config="/home/nvidia/kal/evald.config.json",
    evald_cmd="/home/nvidia/kal/venv/bin/python -m kernel_agent_loop.evald.serve",
    remote_workdir="/home/nvidia/kal/inbox",
)
result = client.submit_sync(bundle)
```

完整示例见 `client.tz.example.py`。

## 真机冒烟

```bash
ssh -p 2240 nvidia@tz 'export PATH=/usr/local/cuda/bin:$PATH
  source /home/nvidia/kal/venv/bin/activate
  cd /home/nvidia/kal/src/kernel-agent-loop
  pytest -m thor --override-ini=addopts= -q'
```

## 环境说明

- Jetson 上 `nvidia-smi` 的 pstate/clocks 常为 `[N/A]`；`read_gpu_env` 会改走 `nvpmodel` + `/sys/class/devfreq/gpu-gpc-0`。
- `jetson_clocks` / `nvpmodel` 需要 `sudo -n`；bootstrap 会尝试锁频。
- ncu 采样需要 `nvidia` 用户能跑 `docker`（nvidia runtime），并且本机已有 `profile_image` 指定的镜像（默认 `adas-img.nioint.com/r-lab/blackwell-aarch64:latest`，自带 ncu、Python 3.12、numpy、cuda-python）。不想采样时把配置里的 `profile_image` 删掉。
