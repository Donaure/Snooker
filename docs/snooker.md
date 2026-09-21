# 完整接口与规则说明

`SnookerMatchEnv` 在 `player_0`（开球方）与 `player_1` 之间运行一局比赛。
它采用本项目规定的规则：最终同分判平、不重置黑球决胜，单次决策超过 120 秒则判负。
不会仅因落后一方需要通过做斯诺克追分就提前结束比赛。每次合法得分后继续击球，
并开始新的 120 秒决策预算。

## 启动对局

```bash
conda activate pool
python scripts/tools/play_snooker_match.py \
  --player-0 'python scripts/tools/snooker_example_player.py' \
  --player-1 'python scripts/tools/snooker_example_player.py'
```

每个决策回合都会启动新的 player 进程：
1. 裁判将当前状态以一行 JSON 写入选手的标准输入。
2. 选手读取状态，输出击球动作，例如 [angle, speed]。
3. 环境执行这一杆，更新状态，再调用接下来应击球的 player。

`--max-shots 2` 可停止诊断运行。
运行器在每个完成的回合后输出完整 JSON 状态。

## Python 接口

从仓库中的独立程序导入时，设置 `PYTHONPATH=src`。

```python
from snooker_env.snooker_env import SnookerMatchEnv

with SnookerMatchEnv() as env:
    state = env.get_match_state("player_0")
    player, token = state["current_player"], state["turn_id"]
    # 可选：将初始白球移到 D 区内的合法位置。
    env.place_cue_ball(player, 0.18, -1.16, turn_id=token)
    # 可选：归一化直角坐标击球点，向右为 +u，世界向上为 +v。
    state = env.submit_shot(player, 1.60, 1.2, u=0.2, v=-0.3, turn_id=token)
    # 省略 u/v 时击打水平视角下的正中点（u=v=0）。
```

角度从世界 +X 向 +Y 逆时针增加，单位为弧度。速度必须为正，表示球杆速度，
不是击球后白球的速度。可选的 `u, v` 是以球半径为单位的直角坐标击球点，
满足 `u*u + v*v < 1`，默认均为零。沿水平击球方向看向白球时，向右为正 `u`，
世界向上为正 `v`。非有限值、布尔值以及单位圆上或圆外的坐标会在物理执行前被拒绝。

击球点固定在水平击球方向坐标系中，不随球杆抬升而改变。设球心为 `c`、半径为 `R`，
`h=(cos(angle),sin(angle),0)`、`right=(sin(angle),-cos(angle),0)`，则球面接触点为：

```text
p = c + R * (u * right + v * (0,0,1) - sqrt(1-u*u-v*v) * h)
```

规划器在定位球杆时考虑球形皮头帽的半径。它优先使用水平杆，内部求解无碰撞的最低仰角，
最多抬杆 75 度，并在需要时缩短后摆。抬杆改变接近方向，也可能改变旋转响应，但不会移动
请求的球面接触点。零偏移抬杆仍击打面向球杆的赤道点，不会强行让抬起的杆轴穿过球心。
原有的双标量输入仍有效，其物理含义遵循这一约定。

开单位圆只是几何输入范围，不保证每个点都可到达或能产生稳定旋转。
不可行的杆路会抛出 `StrokeInfeasibleError`，不会替换击球点，也不会将其判为已完成的一杆。
大偏移可能使皮头滑动；结果由接触动力学决定，而非预设角速度或经验滑杆阈值。
MuJoCo 实际积分球杆接触、球间碰撞、库边接触和入袋过程。
即使白球已经落袋，其他球也会继续运动至停止。

状态包含比分、当前与最高单杆得分、当前选手、回合令牌、剩余时间、全部 22 个球的位置和
有效标志、当前合法目标球（Ball On）、彩球与自由球指定、犯规选择、手中球状态、球台尺寸、
点位、袋口，以及上一杆的物理事件和判罚。快照与环境内部数据独立，可 JSON 序列化；
`get_game_state` 是别名。`action` 元数据描述偏移坐标系、约束和以米为单位的半径。
`last_shot` 记录实际执行的 `u`、`v`，包括省略时的零值。

可选规则操作不会给击球动作增加字段：

| 方法 | 用途 |
| --- | --- |
| `nominate_colour(player, ball, turn_id=...)` | 红球得分后指定彩球 |
| `nominate_free_ball(player, ball, turn_id=...)` | 获得自由球资格时指定一颗仍在台面上的非目标球；`None` 表示放弃 |
| `place_cue_ball(player, x, y, turn_id=...)` | 手中球时在 D 区摆放或重新摆放白球 |
| `choose_foul(player, choice, turn_id=...)` | 选择 `accept`、`play_again`，或在判定 miss 后选择 `restore` |
| `forfeit(player, turn_id=...)` | 当前选手认输 |

开球方从手中球开始，环境提供合法的默认 D 区位置。白球落袋或被击出台面后，也会再次提供
空闲的 D 区位置；选手可以在击球前修改。彩球阶段中，明确、无遮挡的直接瞄准可以隐式指定
彩球；借库解球或目标不明确时需显式指定。自由球始终需要显式指定。
开始摆球、指定球或击球即表示接受犯规后的局面。`accept` 保留剩余决策时间；
要求犯规方重打则给该选手开启一个新的计时回合。

恢复局面时保留双方比分和全部罚分，只恢复击球前的球位、手中球状态和 Ball On 阶段。
若恢复到红球后的彩球阶段，犯规方可以重新选择彩球。要求犯规方重打时不保留自由球资格。

`reset` 或前一杆结束后开放新回合时，决策计时开始。读取状态、指定彩球、摆白球和接受犯规
均不重置计时。即使选手始终不返回，监视计时器也会判定超时。内部杆路规划、物理仿真和
自动裁判的时间不计入选手预算。无效请求会抛出异常，计时继续；仿真失败会恢复击球前局面
和未使用的决策时间。正式比赛没有暂停接口。调用 `close()` 或使用上下文管理器释放计时器。
测试中可以缩短 `turn_seconds`，正式比赛默认值为 120。

## 独立选手程序

`SnookerProgramPlayer` 为每个决策回合启动一个进程，将 JSON 状态作为单行写入 stdin，
然后读取 stdout。最简单的选手返回：

```json
[1.5707963267948966, 1.2]
```

如需选择击球点，则返回四个标量：

```json
[1.5707963267948966, 1.2, 0.2, -0.3]
```

此例在固定水平视角下向右偏移 `0.2R`、向下偏移 `0.3R`。
程序协议只接受两个或四个标量。Python 调用者可按位置或关键字传入 `u`、`v`，二者默认均为零。

如需击球前的规则操作，发送一行 JSON 请求、flush stdout，并在下一次请求或最终击球前
读取一行更新后的状态：

```json
{"op": "nominate_colour", "ball": 21}
```

支持的操作包括 `state`、`nominate_colour`（参数 `ball`）、
`nominate_free_ball`（参数 `ball`，可为 `null`）、`place_cue_ball`（参数 `x`、`y`）和
`choose_foul`（参数 `choice`）。要求对手重打会结束非犯规方的进程回合，随后由运行器调用
犯规方程序。诊断信息写入 stderr。命令按参数数组执行，不经过 shell。
每行 stdout 协议数据上限为 64 KiB。协议或动作格式错误、程序崩溃会判负；
仿真错误会向宿主传播以便诊断。

进程启动、不完整输出和所有击球前请求共用同一截止时间。超时或完成后，运行器终止整个
进程组，包括后代进程。进程内直接调用也受比赛级监视计时器约束，不过环境无法中断其他
线程中的任意代码。

## 球、球台与物理资源

球编号 `0` 为白球，`1..15` 为红球，`16..21` 依次为黄、绿、棕、蓝、粉、黑。
红球计 1 分，各彩球分别计 2 至 7 分。

尺寸参考 [WPBSA 2024–25 规则手册的设备章节](https://wpbsa.com/wp-content/uploads/2198_WPBSA-Rulebook-2024-25.pdf)。
库边内侧的有效台面为 3.569 × 1.778 m，球直径为 52.5 mm，开球线距底库 737 mm，
D 区半径为 292 mm，黑球点距顶库 324 mm。蓝球位于中心，粉球位于蓝球与顶库之间的中点。
所有球质量均为 140.6 g，与校准所用的公开碰撞实验一致。橡胶最高点距展示地板 864 mm。
台呢保持在世界坐标 z=1.05 m，地板位于 z=0.229 m。

`models/snooker_scene.xml` 与 `models/snooker_balls.xml` 是由
`scripts/assets/build_snooker_assets.py` 生成的原创程序化资源；共享形状定义位于
`src/snooker_env/snooker_table_assets.py`。球台包含成型橡胶库边、弯曲的角袋和中袋袋角、
石板袋沿与实际开口、缝制皮革袋口、金属袋架、编织网袋、木制围板、装饰线条与八根车制桌腿。
台呢具有细微的程序化织物纹理，不下载新的第三方美术资源。场景复用已有许可的球杆外观和
物理皮头、杆身。本课程副本包含所有引用资源，无需原仓库、相邻球台仓库或自定义 SDF 插件。

台呢和袋角碰撞使用 MuJoCo 原生凸几何块。袋口由几何块之间的开口构成，凹形外观网格的
凸包不会封闭袋口。袋角外观与碰撞分段遵循相同轮廓。球必须进入石板开口并下降到台呢以下
100 mm 才算入袋；只触碰袋角或袋沿不算入袋，球回到台呢高度时会清除待入袋标记。
网袋和袋口五金仅为外观细节，不施加额外捕获力。离台球被停放到场外，关闭接触和重力。

Snooker 接触参数位于 `src/snooker_env/snooker_parameters.py`。
滑动、滚动和自旋摩擦分别拟合公开实验，另有球间恢复系数及随速度变化的库边回弹损失。
`SnookerPhysics` 在求解冲量前，通过 MuJoCo 原生分步与约束 API 应用橡胶阻尼曲线。
直接调用 `mujoco.mj_step` 只使用 XML 中的静态参考阻尼；需要完整材料模型的诊断仿真应使用
`SnookerPhysics._step()`。共享实现中的 Pool 场景保留原有接触设置与步进方式。

参见[物理参数、复现命令与局限](snooker_calibration.md)和
[球台资源尺寸与来源](https://github.com/Donaure/Snooker/blob/master/assets/table/snooker/README.md)。
校准针对公开实验条件，并非对某张实际安装球台的物理测量。通用袋口轮廓不是经过认证的
WPBSA 袋口模板。

## 规则与裁判范围

引擎处理单杆多颗红球、单颗指定彩球、最后一颗红球后的额外彩球、按顺序清彩、
每杆取最高罚分的 4–7 分犯规处罚、白球落袋和球出台、彩球重置、连续得分、自由球、
重打选择，以及最后黑球进袋或犯规后的终局。任何犯规都会取消击球方该杆得分。
犯规中离台的红球不回台，除非恢复 miss 前的局面；犯规中离台的彩球会重置，清彩阶段也如此。
自由球按 Ball On 的分值计分并重置，不消耗实际红球。清彩阶段仅打入自由球不会清除原彩球；
同时打入自由球和原彩球时，只计算一次该彩球分值。

重置先尝试本球点位，再尝试最高分的可用彩球点，之后沿本球点位的纵向线向顶库方向寻找
最近的不接触位置。该方向无可用位置时再向底库搜索。多颗彩球需要重置时按分值从高到低处理。

仿真与裁判使用以下明确的近似：

- 袋口轮廓为通用程序化模型，并非认证的 WPBSA 模板。台呢、球和正碰库边响应拟合公开测量；
  将橡胶模型用于斜碰和袋角是一项假设。没有拟合台呢绒向、湿度、局部磨损、库边老化和皮头差异。
  该模型不能确定真实球台的进袋概率。
- 固定白球位置的斯诺克判断检查目标球两侧的极限切线路径，不将其他 Ball On 当作斯诺克遮挡。
  手中球资格检查以 5 mm 网格搜索合法 D 区位置，可能漏掉更小的无阻挡窗口。
- 自动 miss 判定搜索直接路径和最多四次理想库边反弹，每个镜像目标尝试 17 个碰撞偏移。
  若找到可行路径而首碰失败，则判 miss；狭窄或更长的解球路径可能被漏掉。
  可传入 `route_oracle(positions, targets) -> bool` 替代这一判断。
  当前采用项目给定的可行路径规则，不包含国际规则中的主观尽力或分差例外，也不会自动增加三次 miss 判负。
- 球杆避障以 1 度搜索、0.1 度细化，并采样检查球台扫掠。不可行出杆、未停稳或不稳定的仿真会抛出异常，
  不会虚构裁判结果。接触先后顺序在积分步精度上判定。选手接口不提供跳球控制。

## 验证

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/assets/build_snooker_assets.py
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_tests/run_snooker_rules_smoke.py
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_tests/run_snooker_players_smoke.py
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_tests/run_snooker_cue_offset_smoke.py
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/smoke_tests/run_snooker_physics_smoke.py --render
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_tests/run_snooker_table_smoke.py
CUDA_VISIBLE_DEVICES=0 python scripts/tools/inspect_snooker_dynamics.py \
  --pockets --output outputs/snooker_realism/calibrated_dynamics.json
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/render/render_snooker_diagnostics.py
```

规则检查覆盖完整 147 和 155 分单杆、犯规计分、最后红球和黑球、平局、恢复与重打、
被占用的重置点、几何、输入验证、快照、截止时间和回滚。选手程序检查覆盖击球前请求、
错误响应、不完整写入、程序卡住和后代进程清理。物理检查覆盖编译后尺寸、22 球稳定性、
固定接触点的抬杆、实际首碰、白球与目标球入袋、场外停放、回滚和完整球堆开球。
初始球阵渲染保存到 `outputs/snooker/initial_rack.png`，不纳入 Git。
球台检查还覆盖六个袋口的落球和弯曲袋角的阻挡。

诊断渲染器将球台全景、袋口近景及 `snooker_diagnostics.mp4` 保存到
`outputs/snooker_realism/`。标注的诊断轨迹通过预设球的初始滚动速度，分别研究台呢、球、
库边和袋口响应；它们不是对局录像，也不是球杆速度校准。

击球点检查验证有无遮挡导致抬杆时的胶囊体与球接触几何、默认与显式零偏移的一致性、
左右塞及高低杆的旋转符号、非法输入拒绝，以及未完成偏心击球的回滚。

## 球杆尺寸与近似

皮头直径为 **9.5 mm**，采用短胶囊的球冠近似，突出杆身前端约 **5 mm**；后半部嵌入前端。
杆身用五段胶囊近似锥度，直径从杆尾的 29 mm 逐段减小至前端的 9.5 mm，所有分段均参与避障检查。

球杆总质量保持 **530 g**，其中皮头为 **0.5 g**；按分段质量计算，重心距杆尾约 **496.5 mm**。
惯量由 MuJoCo 根据各段几何和质量计算。外观网格不参与碰撞或质量计算。

这些是教学仿真的几何和质量近似，未模拟皮革形变、真实皮头曲率、巧粉或滑杆极限；
执行器仍逐步指定球杆运动。自动抬杆不改变指定的母球接触点。
可运行 `scripts/smoke_tests/run_snooker_cue_offset_smoke.py` 检查尺寸、质量、避障、
偏心接触和左右塞/高低杆旋转方向。

验证样例：沿 +Y 方向、杆速 0.6 m/s 时，`u=±0.3` 得到约 `ωz=±9.80 rad/s` 的左右旋转；
`v=+0.3/-0.3` 分别得到约 `ωx=-9.12/+9.59 rad/s`。旋转在皮头最后一次接触时采样，
首次接触误差均小于 0.006 mm；默认输入与显式零偏移的最终状态一致。
这些结果验证方向与实现一致性，不代表已实测校准真实球杆的旋转幅值。
