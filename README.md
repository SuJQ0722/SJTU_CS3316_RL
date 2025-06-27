# SJTU_CS3316_RL Final Project

This project implements and evaluates several model-free reinforcement learning algorithms for Atari and MuJoCo environments as part of the course final project. An innovative model-based algorithm, MuZero, is also implemented as a bonus.

## Project Structure
```txt
Reinforcement_Learning_Final_Project/
├── agents/                      # 存放所有RL算法的实现
│   ├── d3qn/
│   │   ├── __init__.py
│   │   ├── agent.py             # D3QN Agent类，包含act, learn, save, load等方法
│   │   └── model.py             # Dueling DQN网络结构定义
│   ├── ppo/
│   │   ├── __init__.py
│   │   ├── agent.py             # PPO Agent类
│   │   └── model.py             # Actor-Critic网络结构定义
│   └── muzero/                  # (Bonus)
│       ├── __init__.py
│       ├── agent.py             # MuZero Agent类
│       ├── model.py             # 表征、动态、预测网络
│       └── mcts.py              # 蒙特卡洛树搜索的实现
│
├── configs/                     # 存放所有超参数配置文件
│   ├── d3qn_breakout.yaml
│   └── ppo_hopper.yaml
│
├── envs/                        # 环境封装 (如果需要预处理)
│   ├── __init__.py
│   └── atari_wrappers.py        # Atari环境的常用封装，如灰度化、帧堆叠等
│
├── utils/                       # 通用工具模块
│   ├── __init__.py
│   ├── replay_buffer.py         # 经验回放池的实现
│   ├── logger.py                # 用于记录训练日志和性能指标 (支持TensorBoard)
│   └── scheduler.py             # 学习率、探索率等动态调整的调度器
│
├── run.py                      # 主程序入口，用于训练和评估
├── requirements.txt             # 项目依赖库
└── README.md                    # 项目说明文档，必须详细！
```

## Setup
1. Clone the repository:
   `git clone ...`
2. Create a conda or venv environment (Python 3.8+ recommended).
3. Install dependencies:
   `pip install -r requirements.txt`

## How to Run

### Training
To train an agent, use `main.py` with the desired algorithm and environment.

**Train D3QN on Breakout:**
```bash
python main.py --algo d3qn --env_name BreakoutNoFrameskip-v4 --config configs/d3qn_breakout.yaml
```

## **Final Report**

*   **1. 引言**: 项目目标，选择的算法和环境概述。
*   **2. 算法详述**:
    *   **D3QN**: 详细公式推导，网络结构图，与DQN的对比。
    *   **PPO**: 详细公式推导（包括GAE和Clipped Objective），Actor-Critic结构图。
    *   **MuZero (Bonus)**: 详细介绍其三大网络和MCTS的协同工作原理，画出其规划和学习流程图。
*   **3. 实验设置**:
    *   环境描述 (Atari/MuJoCo)。
    *   超参数表（从`configs/`文件整理）。
    *   实验平台（硬件、软件版本）。
*   **4. 结果与分析**:
    *   **性能曲线**: 绘制每个算法在对应环境中的**平均奖励曲线**（带标准差阴影更佳）。
    *   **性能对比**: 如果你在一个环境上测试了多个算法（比如DQN vs D3QN），直接对比它们的性能曲线。
    *   **超参数敏感性分析 (加分项)**: 选择1-2个关键超参数（如PPO的clip_epsilon，D3QN的学习率），展示不同取值对性能的影响。
    *   **行为分析 (加分项)**: 录制一段agent玩游戏的视频（gif），分析其学到的策略是否符合直觉。
*   **5. 结论**: 总结算法表现，讨论遇到的挑战和未来可改进的方向。
*   **6. 参考文献**: 引用所有相关的论文 (DQN, Dueling, Double DQN, PPO, MuZero)。