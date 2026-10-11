# Snooker MuJoCo 项目

在统一的 MuJoCo 球台和自动裁判环境中，从零实现自己的 Snooker 对局策略。
环境负责执行击球、模拟碰撞和入袋、判定犯规、更新比分；策略负责读取状态并作出决策。

仓库提供最小协议示例，完整的选球、进攻、防守及学习策略由学生实现。

## 从这里开始

1. 按[安装与首次运行](getting_started.md)配置环境，验证规则与选手协议。
2. 对照[接口速查](api.md)与[完整接口及规则说明](snooker.md)，接入自己的选手程序。
3. 在固定物理参数下完成实验，参考[物理参数与复现](snooker_calibration.md)解释结果。

```{toctree}
:maxdepth: 2
:caption: 使用指南

getting_started
api
snooker
snooker_calibration
```

源码与问题反馈：[Donaure/Snooker](https://github.com/Donaure/Snooker)。
比赛规则以本项目裁判实现为准。
