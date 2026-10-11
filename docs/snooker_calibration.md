# 物理参数与复现

本页记录仓库内的参数定义、诊断入口与模型局限。参数来源是
`src/snooker_env/snooker_parameters.py` 中的 `SNOOKER_PHYSICS`；
场景生成和校准脚本共享这份配置。以下数值是配置值，实际测量结果以本地生成的报告为准。

## 坐标与基本参数

| 参数 | 当前配置 |
| --- | --- |
| 有效台面长 × 宽 | 3.569 × 1.778 m |
| 台呢表面高度 | 1.05 m |
| 球直径 | 0.0525 m |
| 球质量 | 0.1406 kg |
| 物理时间步 | 0.00005 s |
| 滚动减速度参考值 | 0.125 m/s² |
| 滑动减速度参考值 | 0.212 × 9.81 m/s² |
| 自旋减速度参考值 | 22 rad/s² |
| 球间恢复系数参考值 | 0.89 |

MuJoCo 的扭转和滚动摩擦系数以长度为单位，不能当作普通的无量纲摩擦系数使用。
库边采用随入射法向速度变化的阻尼插值表；当前表覆盖 0.28–3.5 m/s，范围外取端点值。
上述参考值与接触求解器参数用途不同，球间碰撞由实际接触积分执行。

## 校准和诊断

在 `pool` 环境中检查 `nvidia-smi` 后选择空闲 GPU。以下命令以 GPU 1 为例：

```bash
conda activate pool
CUDA_VISIBLE_DEVICES=1 python scripts/tools/calibrate_snooker_physics.py \
  --check --plot --output outputs/snooker_calibration
CUDA_VISIBLE_DEVICES=1 python scripts/tools/inspect_snooker_dynamics.py \
  --pockets --output outputs/snooker_realism/calibrated_dynamics.json
```

校准脚本生成 `calibration.json`；`--plot` 增加 `calibration.png`，
`--check` 检查脚本定义的数值回归阈值。可选的 `--fit` 会拟合参数并输出 `fit.json`，
不会自动修改仓库中的参数文件。正式比赛应固定环境版本与参数。

脚本中的 `OBLIQUE_EXPERIMENTS` 标注了 Mathavan 等（2014）的球间碰撞数据；
`cushion_reference()` 标注了 2009 年滚动正碰库边回弹的实验回归。
查阅或复现实验时应同时检查脚本中的条件、测量时刻和比较指标。
本课程副本没有随仓库提交校准报告，本页不将参数值当作本次重新测得的误差或性能结果。

诊断库边动态阻尼时使用 `SnookerPhysics` 的步进路径。直接调用 `mujoco.mj_step`
只采用 XML 中的静态参考阻尼，不能据此验证完整的速度相关材料模型。

## 模型边界

球台是程序化几何，袋口轮廓是教学模型，并非经过认证的 WPBSA 袋口模板。
球必须进入开口并下降到台呢以下 100 mm 才算入袋，接触袋角或袋沿本身不算入袋。
网袋及部分球台外观不参与碰撞。

校准使用公开实验条件，没有测量某张实际安装球台的台呢、湿度、磨损或库边老化。
正碰库边模型用于斜碰及袋角是近似；模型不能直接证明真实球台的进袋概率。
皮头形变、巧粉和真实滑杆极限未被完整建模，偏心击球的方向验证也不等同于旋转幅值的实测校准。

球台资源尺寸与来源见
[球台资源说明](https://github.com/Donaure/Snooker/blob/master/assets/table/snooker/README.md)，
球杆视觉资源许可见
[资源许可](https://github.com/Donaure/Snooker/blob/master/assets/cue/sketchfab_pool_table_traditional/license.txt)。
再分发时保留这些说明与许可；裁判和几何近似详见[完整接口与规则说明](snooker.md)。
