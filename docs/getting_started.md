# 安装与首次运行

## 配置环境

从仓库根目录运行以下命令。项目使用 Python 3.10 或以上版本，推荐使用 Conda 环境。

```bash
git clone https://github.com/Donaure/Snooker.git
cd Snooker
conda create -n pool python=3.10
conda activate pool
python -m pip install -r requirements.txt
```

对局物理在 CPU 上运行；GPU 用于 EGL 渲染或学生的学习模型。headless 渲染设置 `MUJOCO_GL=egl`。


## 运行最小选手示例

```bash
python scripts/tools/play_snooker_match.py \
  --player-0 'python scripts/tools/snooker_example_player.py' \
  --player-1 'python scripts/tools/snooker_example_player.py' \
  --max-shots 2
```

该示例只展示 JSON-lines 协议，朝最近的合法目标球击球。
`--max-shots 2` 用于两杆调试。选手程序准备好后，
将两个命令替换为自己的程序；移除 `--max-shots` 可运行至终局，默认每局最多 500 杆，达到上限按比分结算。
真实击球使用较小的物理时间步，执行过程可能较慢。

每位选手每局使用一个进程：初始化后先输出 `{"op":"ready"}` 并 flush，
再循环从标准输入读取单行 JSON 状态，将动作 `[angle, speed, u, v]` 写到标准输出。
初始化默认限时 300 秒，不占用每回合 120 秒决策预算；日志写到标准错误。
规则请求、超时与进程生命周期见[完整协议说明](snooker.md)。

## 渲染场景

```bash
MUJOCO_GL=egl python scripts/render/render_snooker_diagnostics.py --images-only
```

诊断图片保存到 `outputs/snooker_realism/`。去掉 `--images-only` 可生成物理诊断视频；
对局录像可在 `play_snooker_match.py` 命令中添加
`--record outputs/my_match --fps 20`，headless 环境设置 `MUJOCO_GL=egl`。

## 可选学习依赖

```bash
python -m pip install -r requirements-rl.txt
```

可选依赖不会增加训练接口。学生需要自行实现策略、Gymnasium 封装（若使用）、
采样与训练流程。课程任务与评估要求以教师发布的安排为准。
