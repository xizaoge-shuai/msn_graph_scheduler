# MSN Graph-Aware LLM Scheduler

这是根据《面向移动边缘网络的图感知大模型推理调度》方案实现的第一版可运行实验代码。目标不是一次性复现完整生产系统，而是先把论文中的算法闭环、动机测量和基线比较跑通，再逐步替换成真实轨迹与本地 GPU profiling。

## 已实现

- 边缘—云基础设施时变图与固定模型块副本布局；
- 同一服务锚点队列中的节点条件滚动动态规划合批；
- 云端安全容量检查与后续完整路径兜底；
- 连续计算块组动作 `g`；
- 边属性感知 GAT（纯 PyTorch，无需 PyG）；
- 可变动作空间的共享 Dueling Double DQN；
- 预填充逐步奖励与映射完成后的预期解码成本；
- 云端、锚点优先、时延贪心基线；
- 两个最先需要做的动机实验；
- 通用电信轨迹/Azure 请求轨迹适配器；
- 本地 Hugging Face 模型 profiling 脚本。

## 尚未伪装成“已完成”的部分

- 上海电信和 Azure 数据集的真实字段格式需要拿到数据后配置映射；
- 当前默认性能曲线是可运行的解析模型，需要用你的 GPU 测量 CSV 替换；
- 第三组“MLP 与 GAT 跨拓扑泛化”实验需要先训练两类策略，建议在主训练稳定后补；
- 没有实现在线模型块迁移、KV Cache 迁移或跨基站合批，和会议版边界一致；
- 当前模拟器以一个预填充批次的一条映射作为一个 RL episode，后续可扩展为长期离散事件仿真。

## 安装

```bash
cd msn_graph_scheduler
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

运行测试：

```bash
pytest -q
```

## 先跑动机实验

### 1. 节点依赖的合批收益

```bash
python scripts/motivation_batch_node.py
```

输出位于 `outputs/motivation_batch_node/`，包含不同节点/批大小下的时延、吞吐和显存结果。

### 2. 连续计算块组长度权衡

```bash
python scripts/motivation_group_length.py
```

输出位于 `outputs/motivation_group_length/`。

### 3. 未知拓扑泛化的代理实验

```bash
python scripts/motivation_topology.py \
  --train-samples 300 \
  --test-samples 100 \
  --epochs 40
```

这个脚本先用“选择最低预计时延节点”的监督代理任务比较边属性 GAT 和固定长度 MLP。它不是最终 RL 主结果，但能在主训练前快速检查图表示是否对未见节点规模更稳健。

## 训练调度策略

先做短 smoke run：

```bash
python scripts/train.py --episodes 50 --output outputs/train_smoke
```

正式训练：

```bash
python scripts/train.py --episodes 1200 --output outputs/train_main
```

## 评价与基线

```bash
python scripts/evaluate.py \
  --checkpoint outputs/train_main/agent_final.pt \
  --episodes 50 \
  --output outputs/eval_main.csv
```

不传 `--checkpoint` 时只跑规则调度基线。

合批基线单独比较：

```bash
python scripts/evaluate_batchers.py \
  --episodes 30 \
  --fixed-batch-size 4 \
  --output outputs/eval_batchers.csv
```

该脚本比较不合批、固定批大小、逐请求贪心和节点条件滚动 DP，并固定使用同一时延贪心调度器，避免把调度器差异混入合批比较。

## 本地大模型 profiling

额外安装：

```bash
pip install -e '.[profile]'
```

以 Qwen2.5-1.5B 为例：

```bash
python scripts/profile_llm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --batches 1 2 4 8 \
  --tokens 128 256 512 1024 \
  --output profiles/qwen_full_model.csv
```

该脚本先测整模型 prefill/decode。第一阶段可按 `g/L` 近似得到块组时间；论文正式实验前，应进一步加入分块实测或在不同 GPU 上分别测量。

## 真实轨迹接入原则

- 电信轨迹只决定移动、基站关联和时变活跃用户数；
- 请求到达强度可由 `lambda_b(t)=lambda0+kappa*N_active_b(t)` 生成；
- Azure 轨迹在主实验中用于输入/输出 token 分布，不与电信时间戳对齐；
- 阿里生产集群轨迹只作为可选资源波动压力测试；
- 所有算法必须共享同一组请求、拓扑、块副本和资源状态。

通用适配入口：

- `src/msn_scheduler/data/telecom.py`
- `src/msn_scheduler/data/azure.py`

## 推荐实验顺序

1. 先运行两个动机脚本，确认趋势；
2. 在实际 GPU 上运行 `profile_llm.py`；
3. 用实测曲线替换解析 profile；
4. 跑三个规则基线，检查仿真器是否符合直觉；
5. 训练 GAT-Dueling-DDQN；
6. 做固定组长度、无移动成本、全局贪心合批等消融；
7. 最后接入真实移动轨迹和跨拓扑测试。
