# 接口速查

实现策略前，请阅读[选手程序需要处理的全部决策情况](snooker.md#选手程序需要处理的全部决策情况)，
覆盖普通回合、犯规选择、自由球、阶段切换、终局和协议处理顺序。

比赛决策期间，禁止将当前局面在 MuJoCo 中复制或反复恢复，试打多个候选动作后
根据仿真结果择优出杆；串行或并行执行均不允许。
允许调用课程环境进行出杆可行性检查：该检查只检测球杆是否会碰到其他球，
不模拟球的运动，不属于上述禁止的仿真试打。详见[选手程序限制](snooker.md#选手程序限制)。

## Python 入口

从仓库根目录运行自己的 Python 程序时设置 `PYTHONPATH=src`，然后导入：

```python
from snooker_env.snooker_env import SnookerMatchEnv

with SnookerMatchEnv() as env:
    state = env.get_match_state()
    print(state["current_player"], state["legal_targets"])
```

环境创建时已经开始一局比赛和决策计时；使用 `with` 在退出时释放计时器。
以上示例只读取状态，具体决策逻辑由学生实现。

| 接口 | 功能 |
| --- | --- |
| `SnookerMatchEnv(*, physics=None, turn_seconds=120., shot_limit=500, route_oracle=...)` | 创建对局；正式比赛使用默认 120 秒决策预算 |
| `reset()` | 重置比赛并开始新的计时 |
| `get_match_state(player=None)` | 返回独立、可 JSON 序列化的公开状态快照 |
| `submit_shot(player, angle, speed, u=0., v=0., *, turn_id)` | 执行一杆并返回更新状态 |
| `nominate_colour(player, ball, *, turn_id)` | 在任选彩球阶段指定目标彩球；击球前必须显式指定 |
| `nominate_free_ball(player, ball, *, turn_id)` | 指定自由球，传入 `None` 放弃指定 |
| `place_cue_ball(player, x, y, *, turn_id)` | 手中球时在 D 区合法位置摆球 |
| `choose_foul(player, choice, *, turn_id)` | 从当前 `foul_options` 选择处理方式 |
| `forfeit(player, *, turn_id, reason="forfeit")` | 当前选手认输 |
| `close()` | 释放环境计时器 |

修改状态的调用需要当前 `current_player` 与 `turn_id`；每次操作后以返回状态为准。
读取状态、指定球或摆球不会重置决策预算。环境没有供正式比赛使用的暂停接口。

## 状态字段

| 字段 | 含义 |
| --- | --- |
| `balls` | 固定 22 个球的编号、名称、分值、有效标志与三维位置 |
| `table` | 球台尺寸、袋口、球半径、彩球点位、D 区参数 |
| `stage` | `red`、`colour` 或 `clearance` |
| `phase` | `shot`、`foul_choice`、`initializing` 或 `finished`；摆球与指定还需检查对应字段 |
| `legal_targets` | 当前可以作为目标的球编号 |
| `ball_in_hand` | 是否允许在 D 区摆白球 |
| `free_ball_available` / `nominated_free_ball` | 自由球资格与当前指定 |
| `nominated_colour` / `clearance_ball` | 指定彩球与清彩阶段目标 |
| `foul_options` | 当前可选的 `accept`、`play_again`、`restore` |
| `scores` / `current_break` / `highest_breaks` | 双方比分与单杆得分 |
| `current_player` / `turn_id` | 当前选手与回合令牌 |
| `shot_count` / `shot_limit` | 已完成总杆数与每局上限（默认 500）；达到上限后按比分结算 |
| `seconds_remaining` | 当前决策剩余秒数 |
| `last_shot` | 上一杆动作、物理事件和裁判结果；开局为 `None` |
| `terminated` / `winner` / `draw` / `reason` | 终局与结果信息 |

`0` 为白球，`1–15` 为红球，`16–21` 依次为黄、绿、棕、蓝、粉、黑。
离台球的 `active=False` 且 `position=None`，计算特征前应先过滤。
长轴为 `+Y`，短轴为 `+X`，台呢表面高度为 `z=1.05 m`。

## 击球动作

```text
[angle, speed, u, v]
```

`angle` 是从世界 `+X` 向 `+Y` 逆时针增加的弧度；`speed` 是正的球杆速度（m/s）。
`u, v` 是用球半径归一化的击球点坐标，满足 `u*u + v*v < 1`，
沿水平击球方向看向白球时向右和向上为正。自动抬杆不会移动所请求的球面接触点。
两元素动作 `[angle, speed]` 等价于 `u=v=0`。

几何合法的偏移仍可能无法出杆。不可行的杆路会抛出 `StrokeInfeasibleError`；
仿真异常也不会凭空生成比赛结果。完整的计时、回滚与异常处理见[接口说明](snooker.md)。

## 独立选手程序

`SnookerProgramPlayer(command)` 接受程序及参数列表，`play_turn(env)` 完成当前回合并返回状态。
每位选手每局使用一个常驻进程。初始化后先输出 `{"op":"ready"}` 并 flush，
再循环读取状态、提交动作。标准输出只能写协议内容。规则请求后读取更新状态；
要求对手重打时，下一行输入是再次轮到自己时的状态。

| JSON 请求 | 作用 |
| --- | --- |
| `{"op": "state"}` | 获取更新状态 |
| `{"op": "nominate_colour", "ball": 21}` | 指定黑球 |
| `{"op": "nominate_free_ball", "ball": null}` | 放弃自由球指定 |
| `{"op": "place_cue_ball", "x": 0.18, "y": -1.16}` | 请求在指定位置摆白球 |
| `{"op": "choose_foul", "choice": "accept"}` | 接受犯规后的局面 |

每个请求都必须符合当前规则状态。外部程序不必在请求中传递 `player` 与 `turn_id`，
适配器会自动附加。要求对手重打后保留当前进程，裁判向对手的常驻进程发送状态。
首次初始化默认限时 300 秒（`startup_seconds`），不计入 120 秒决策预算。
就绪后，当前回合所有请求共用决策期限，单行输出上限为 64 KiB。
使用 `with` 或 `close()` 清理进程；完整生命周期见[接口说明](snooker.md)。

可阅读仓库中的
[最小选手示例](https://github.com/Donaure/Snooker/blob/master/scripts/tools/snooker_example_player.py)
和[协议适配器](https://github.com/Donaure/Snooker/blob/master/src/snooker_env/snooker_players.py)。
