# 实现状态与论文方案对应关系

## 已经落到代码中的核心机制

| 论文设计 | 代码位置 |
|---|---|
| 边缘—云基础设施图 | `src/msn_scheduler/graph.py` |
| 固定模型块副本与云端完整模型 | `src/msn_scheduler/deployment.py` |
| 同锚点请求队列 | `src/msn_scheduler/synthetic.py`、数据适配器 |
| 节点条件滚动 DP 合批 | `src/msn_scheduler/batching.py` |
| 不合批、固定批、逐请求贪心基线 | `src/msn_scheduler/batching.py` |
| 硬过滤、Top-N、锚点优先与云端回退 | `src/msn_scheduler/candidates.py` |
| 连续计算块组调度环境 | `src/msn_scheduler/env.py` |
| 边属性感知 GAT | `src/msn_scheduler/models.py` |
| 共享 Dueling Double DQN | `src/msn_scheduler/models.py`、`agent.py` |
| 预填充单步成本与终止解码成本 | `src/msn_scheduler/env.py` |
| 电信/Azure 通用适配器 | `src/msn_scheduler/data/` |
| 本地 Qwen profiling | `scripts/profile_llm.py` |

## 当前实验代码的定位

当前版本是论文实验的 **v0.1 可运行骨架**：可以进行合成环境 smoke test、动机分析、基线比较和 DDQN 训练，但默认解析性能曲线不能作为论文最终数值。正式结果必须替换为本地 GPU 测量，并接入确定版本的真实轨迹。

## 下一批必须补的内容

1. 根据实际上海电信数据字段完成轨迹预处理脚本；
2. 根据实际 Azure trace 字段完成 token 分布拟合与跨工作负载重放；
3. 在所有目标 GPU 上建立 `node_type` 对应的 profile CSV；
4. 加入长期离散事件仿真：多个锚点队列、请求持续到达、资源占用释放；
5. 实现固定边缘—云切分、FC-DDQN、GAT-greedy 等完整调度基线；
6. 加入多随机种子、置信区间和统一绘图脚本；
7. 对 DP 在线复杂度、候选动作数量和决策时延做专门记录。

## 不能直接宣称的事项

- 代码目前没有复现两篇参考论文的原始实验；
- 默认合成 profile 只用于检查趋势和程序正确性；
- 当前一个 episode 对应一个预填充批次映射，不是完整生产服务系统；
- 第三组 GAT/MLP 动机脚本是监督代理任务，不是最终 RL 主结果。
