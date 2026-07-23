from __future__ import annotations

from pathlib import Path
import re
import shutil
import textwrap

import yaml


ROOT = Path.cwd()


def backup(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing expected file: {path}")

    dst = path.with_suffix(
        path.suffix + ".bak_calibration"
    )

    if not dst.exists():
        shutil.copy2(path, dst)
        print(f"Backup: {dst}")


for relative in [
    "src/msn_scheduler/profiles.py",
    "src/msn_scheduler/synthetic.py",
    "src/msn_scheduler/batching.py",
    "src/msn_scheduler/agent.py",
    "scripts/evaluate_batchers.py",
    "configs/default.yaml",
]:
    backup(ROOT / relative)


# ============================================================
# 1. profiles.py
# 所有逻辑节点都基于 edge_fast 实测曲线，
# 再通过 compute_scale 构造逻辑异构性能。
# ============================================================

path = ROOT / "src/msn_scheduler/profiles.py"
text = path.read_text(encoding="utf-8")

methods = textwrap.indent(
    textwrap.dedent(
        '''
        def prefill_ms(
            self,
            node: ComputeNode,
            batch_size: int,
            max_input_tokens: int,
            group_size: int,
        ) -> float:
            measured = self._nearest_measured(
                node.node_type,
                "prefill",
                batch_size,
                max_input_tokens,
                group_size,
            )

            if measured is None:
                measured = self._nearest_measured(
                    "edge_fast",
                    "prefill",
                    batch_size,
                    max_input_tokens,
                    group_size,
                )

                if measured is not None:
                    measured = (
                        measured
                        / max(node.compute_scale, 0.1)
                    )

            if measured is not None:
                return (
                    measured
                    * (
                        1.0
                        + 0.85
                        * node.background_load
                    )
                    + node.queue_delay_ms
                )

            token_factor = (
                max_input_tokens / 128.0
            )

            batch_factor = (
                0.55
                + 0.45
                * (batch_size ** 0.72)
            )

            layer_factor = (
                group_size
                / max(
                    self.model.num_blocks,
                    1,
                )
            )

            base_full_model_ms = (
                32.0
                / max(
                    node.compute_scale,
                    0.1,
                )
            )

            return (
                base_full_model_ms
                * layer_factor
                * batch_factor
                * (
                    0.60 * token_factor
                    + 0.40
                    * token_factor**1.55
                )
                * (
                    1.0
                    + 1.1
                    * node.background_load
                )
                + node.queue_delay_ms
            )

        def decode_per_token_ms(
            self,
            node: ComputeNode,
            batch_size: int,
            context_tokens: int,
            group_size: int,
        ) -> float:
            measured = self._nearest_measured(
                node.node_type,
                "decode",
                batch_size,
                context_tokens,
                group_size,
            )

            if measured is None:
                measured = self._nearest_measured(
                    "edge_fast",
                    "decode",
                    batch_size,
                    context_tokens,
                    group_size,
                )

                if measured is not None:
                    measured = (
                        measured
                        / max(
                            node.compute_scale,
                            0.1,
                        )
                    )

            if measured is not None:
                return (
                    measured
                    * (
                        1.0
                        + 0.70
                        * node.background_load
                    )
                    + 0.02
                    * node.queue_delay_ms
                )

            layer_factor = (
                group_size
                / max(
                    self.model.num_blocks,
                    1,
                )
            )

            context_factor = (
                1.0
                + 0.18
                * np.log2(
                    max(
                        context_tokens,
                        1,
                    )
                    / 128.0
                    + 1.0
                )
            )

            batch_factor = (
                0.72
                + 0.28
                * (batch_size ** 0.55)
            )

            return (
                3.4
                * layer_factor
                * context_factor
                * batch_factor
                / max(
                    node.compute_scale,
                    0.1,
                )
                * (
                    1.0
                    + 0.8
                    * node.background_load
                )
            )
        '''
    ).strip("\n"),
    "    ",
)

pattern = re.compile(
    r"    def prefill_ms\(.*?\n"
    r"(?=    def activation_gb\()",
    flags=re.S,
)

text, count = pattern.subn(
    methods + "\n\n",
    text,
    count=1,
)

if count != 1:
    raise SystemExit(
        "Could not patch ProfileTable "
        "prefill/decode methods"
    )

path.write_text(
    text,
    encoding="utf-8",
)

print(f"Patched: {path}")


# ============================================================
# 2. synthetic.py
# 使用输入、输出长度相关的 SLO。
# ============================================================

path = ROOT / "src/msn_scheduler/synthetic.py"
text = path.read_text(encoding="utf-8")

old = (
    '        deadline = float('
    'rng.uniform('
    'q_cfg["deadline_ms_min"], '
    'q_cfg["deadline_ms_max"]'
    '))'
)

new = textwrap.indent(
    textwrap.dedent(
        '''
        # Length-conditioned SLO calibrated
        # from the measured Qwen profile.
        reference_service_ms = (
            float(
                q_cfg.get(
                    "deadline_base_ms",
                    250.0,
                )
            )
            + float(
                q_cfg.get(
                    "deadline_input_ms_per_token",
                    0.10,
                )
            )
            * input_tokens
            + float(
                q_cfg.get(
                    "deadline_decode_ms_per_token",
                    24.0,
                )
            )
            * out
        )

        slack_factor = float(
            rng.uniform(
                q_cfg.get(
                    "deadline_slack_factor_min",
                    1.05,
                ),
                q_cfg.get(
                    "deadline_slack_factor_max",
                    1.80,
                ),
            )
        )

        deadline = float(
            np.clip(
                reference_service_ms
                * slack_factor,
                q_cfg.get(
                    "deadline_ms_min",
                    1200,
                ),
                q_cfg.get(
                    "deadline_ms_max",
                    12000,
                ),
            )
        )
        '''
    ).strip("\n"),
    "        ",
)

if old not in text:
    raise SystemExit(
        "Could not find old synthetic "
        "deadline line"
    )

text = text.replace(
    old,
    new,
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)

print(f"Patched: {path}")


# ============================================================
# 3. batching.py
# 在批次可行性判断中加入：
# anchor 传输、局部 prefill/decode、
# 云端剩余层计算、云端传输。
# ============================================================

path = ROOT / "src/msn_scheduler/batching.py"
text = path.read_text(encoding="utf-8")

old_import = (
    "from .profiles import "
    "ProfileTable, "
    "shortest_path_latency_ms"
)

new_import = (
    "from .profiles import "
    "ProfileTable, "
    "shortest_path_latency_ms, "
    "transfer_time_ms"
)

if old_import not in text:
    raise SystemExit(
        "Could not find batching.py "
        "profile import"
    )

text = text.replace(
    old_import,
    new_import,
    1,
)

completion_method = textwrap.indent(
    textwrap.dedent(
        '''
        def _estimate_completion_reference_ms(
            self,
            infra: Infrastructure,
            node_id: str,
            batch_size: int,
            max_len: int,
            group_size: int,
            expected_out: int,
        ) -> float:
            """Run the first group locally and
            remaining blocks in the cloud."""
            node = infra.nodes[node_id]

            cloud = infra.nodes[
                infra.cloud_node
            ]

            activation_mb = (
                self.profile.intermediate_mb(
                    batch_size,
                    max_len,
                    decode=False,
                )
            )

            token_mb = (
                self.profile.intermediate_mb(
                    batch_size,
                    1,
                    decode=True,
                )
            )

            anchor_transfer = (
                transfer_time_ms(
                    infra,
                    infra.anchor_node,
                    node_id,
                    activation_mb,
                )
            )

            local_prefill = (
                self.profile.prefill_ms(
                    node,
                    batch_size,
                    max_len,
                    group_size,
                )
            )

            local_decode = (
                self.profile.decode_per_token_ms(
                    node,
                    batch_size,
                    (
                        max_len
                        + expected_out // 2
                    ),
                    group_size,
                )
                * expected_out
            )

            remaining_blocks = (
                self.profile.model.num_blocks
                - group_size
            )

            if remaining_blocks <= 0:
                return float(
                    anchor_transfer
                    + local_prefill
                    + local_decode
                )

            cloud_prefill_transfer = (
                transfer_time_ms(
                    infra,
                    node_id,
                    infra.cloud_node,
                    activation_mb,
                )
            )

            cloud_decode_transfer = (
                transfer_time_ms(
                    infra,
                    node_id,
                    infra.cloud_node,
                    token_mb,
                )
                * expected_out
            )

            cloud_prefill = (
                self.profile.prefill_ms(
                    cloud,
                    batch_size,
                    max_len,
                    remaining_blocks,
                )
            )

            cloud_decode = (
                self.profile.decode_per_token_ms(
                    cloud,
                    batch_size,
                    (
                        max_len
                        + expected_out // 2
                    ),
                    remaining_blocks,
                )
                * expected_out
            )

            return float(
                anchor_transfer
                + local_prefill
                + local_decode
                + cloud_prefill_transfer
                + cloud_decode_transfer
                + cloud_prefill
                + cloud_decode
            )
        '''
    ).strip("\n"),
    "    ",
)

pattern = re.compile(
    r"    def _estimate_cloud_fallback_ms"
    r"\(.*?\n"
    r"(?=    def _cache_key\()",
    flags=re.S,
)

text, count = pattern.subn(
    completion_method + "\n\n",
    text,
    count=1,
)

if count != 1:
    raise SystemExit(
        "Could not replace "
        "_estimate_cloud_fallback_ms"
    )

old = '''                fallback = self._estimate_cloud_fallback_ms(
                    infra, target_b, max_len, group_size, expected_out_pool
                )
'''

new = '''                completion_reference = self._estimate_completion_reference_ms(
                    infra=infra,
                    node_id=node_id,
                    batch_size=target_b,
                    max_len=max_len,
                    group_size=group_size,
                    expected_out=expected_out_pool,
                )
'''

if old not in text:
    raise SystemExit(
        "Could not find DP "
        "fallback estimate"
    )

text = text.replace(
    old,
    new,
    1,
)

old = (
    "                    if "
    "batch_prefill + fallback "
    "> min_deadline "
    "+ self.slack_ms:\n"
)

new = (
    "                    if "
    "completion_reference "
    "> min_deadline "
    "+ self.slack_ms:\n"
)

if old not in text:
    raise SystemExit(
        "Could not find DP deadline check"
    )

text = text.replace(
    old,
    new,
    1,
)

old = '''        fallback = self._estimate_cloud_fallback_ms(
            infra, batch_size, max_len, group_size, expected_out
        )
'''

new = '''        completion_reference = self._estimate_completion_reference_ms(
            infra=infra,
            node_id=node_id,
            batch_size=batch_size,
            max_len=max_len,
            group_size=group_size,
            expected_out=expected_out,
        )
'''

if old not in text:
    raise SystemExit(
        "Could not find heuristic "
        "fallback estimate"
    )

text = text.replace(
    old,
    new,
    1,
)

old = (
    "        if prefill + fallback "
    "> min_deadline "
    "+ self.slack_ms:\n"
)

new = (
    "        if completion_reference "
    "> min_deadline "
    "+ self.slack_ms:\n"
)

if old not in text:
    raise SystemExit(
        "Could not find heuristic "
        "deadline check"
    )

text = text.replace(
    old,
    new,
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)

print(f"Patched: {path}")


# ============================================================
# 4. default.yaml
# ============================================================

path = ROOT / "configs/default.yaml"

cfg = yaml.safe_load(
    path.read_text(
        encoding="utf-8",
    )
)

cfg["requests"].update(
    {
        "queue_size": 24,
        "window_size": 16,
        "input_token_min": 64,
        "input_token_max": 2048,
        "output_token_min": 8,
        "output_token_max": 768,
        "deadline_base_ms": 250.0,
        "deadline_input_ms_per_token": 0.10,
        "deadline_decode_ms_per_token": 24.0,
        "deadline_slack_factor_min": 1.05,
        "deadline_slack_factor_max": 1.80,
        "deadline_ms_min": 1200,
        "deadline_ms_max": 12000,
        "handover_prob_max": 0.35,
    }
)

cfg["batching"].update(
    {
        "max_batch_size": 8,
        "token_capacity": 8192,
        "token_quantum": 32,
        "length_buckets": [
            128,
            256,
            512,
            1024,
            1536,
            2048,
        ],
    }
)

path.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
    ),
    encoding="utf-8",
)

print(f"Patched: {path}")


# ============================================================
# 5. agent.py
# 显式指定 weights_only=False，消除 warning。
# ============================================================

path = ROOT / "src/msn_scheduler/agent.py"
text = path.read_text(encoding="utf-8")

old = (
    "        ckpt = torch.load("
    "path, "
    "map_location=self.device"
    ")"
)

new = '''        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )'''

if old in text:
    text = text.replace(
        old,
        new,
        1,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    print(f"Patched: {path}")
else:
    print(
        f"Skipped: {path} "
        "already changed or load line differs"
    )


# ============================================================
# 6. evaluate_batchers.py
# 不再静默跳过失败 episode，并增加进度输出。
# ============================================================

path = ROOT / "scripts/evaluate_batchers.py"

path.write_text(
    '''from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.baselines import (
    choose_min_lower_bound,
)
from msn_scheduler.batching import (
    FixedSizeBatcher,
    NodeConditionedDPBatcher,
    SequentialGreedyBatcher,
    SingleRequestBatcher,
)
from msn_scheduler.config import load_config
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import (
    ModelProfile,
    ProfileTable,
)
from msn_scheduler.synthetic import (
    make_request_queue,
    make_synthetic_infrastructure,
)


def run_episode(
    env,
    infra,
    queue,
    max_steps: int,
):
    obs = env.reset(
        infra,
        queue,
    )

    result = None

    for _ in range(max_steps):
        action = choose_min_lower_bound(obs)

        next_obs, _, done, result = (
            env.step(action)
        )

        if done:
            if result is None:
                raise RuntimeError(
                    "Episode finished "
                    "without a result"
                )

            return result

        if next_obs is None:
            raise RuntimeError(
                "Missing next observation "
                "before termination"
            )

        obs = next_obs

    raise RuntimeError(
        f"Episode exceeded "
        f"{max_steps} mapping steps"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs/default.yaml"
        ),
    )

    parser.add_argument(
        "--profile-csv",
        default=None,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "outputs/eval_batchers.csv"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    model = ModelProfile(
        num_blocks=int(
            cfg["model"]["num_blocks"]
        ),
        hidden_size=int(
            cfg["model"]["hidden_size"]
        ),
        bytes_per_element=int(
            cfg["model"][
                "bytes_per_element"
            ]
        ),
        block_parameter_gb=float(
            cfg["model"][
                "block_parameter_gb"
            ]
        ),
        kv_bytes_per_token_per_block=float(
            cfg["model"][
                "kv_bytes_per_token_per_block"
            ]
        ),
    )

    profile = (
        ProfileTable.from_csv(
            model,
            args.profile_csv,
        )
        if args.profile_csv
        else ProfileTable(model)
    )

    if args.profile_csv:
        print(
            f"Using measured profile: "
            f"{args.profile_csv}",
            flush=True,
        )
    else:
        print(
            "Warning: using analytic "
            "profile fallback",
            flush=True,
        )

    batchers = {
        "no_batch": SingleRequestBatcher(
            cfg,
            profile,
        ),
        f"fixed_{args.fixed_batch_size}":
            FixedSizeBatcher(
                cfg,
                profile,
                args.fixed_batch_size,
            ),
        "sequential_greedy":
            SequentialGreedyBatcher(
                cfg,
                profile,
            ),
        "node_conditioned_dp":
            NodeConditionedDPBatcher(
                cfg,
                profile,
            ),
    }

    rows: list[dict] = []

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_runs = (
        args.episodes
        * len(batchers)
    )

    completed = 0
    start = time.perf_counter()

    print(
        "Starting batching evaluation: "
        f"episodes={args.episodes}, "
        f"methods={list(batchers)}, "
        f"total_runs={total_runs}",
        flush=True,
    )

    for episode in range(
        args.episodes
    ):
        seed = (
            int(cfg["seed"])
            + 20000
            + episode
        )

        for name, batcher in (
            batchers.items()
        ):
            rng = np.random.default_rng(
                seed
            )

            infra = (
                make_synthetic_infrastructure(
                    cfg,
                    rng,
                )
            )

            queue = make_request_queue(
                cfg,
                rng,
            )

            env = SchedulingEnv(
                cfg,
                profile,
                batcher,
            )

            try:
                result = run_episode(
                    env,
                    infra,
                    queue,
                    args.max_steps,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Batcher failed: "
                    f"batcher={name}, "
                    f"episode={episode}, "
                    f"seed={seed}, "
                    f"error={exc}"
                ) from exc

            rows.append(
                {
                    "episode": episode,
                    "batcher": name,
                    "e2e_est_ms": (
                        result.total_prefill_ms
                        + result.expected_decode_ms
                        + result.handover_ms
                    ),
                    "prefill_ms":
                        result.total_prefill_ms,
                    "decode_ms":
                        result.expected_decode_ms,
                    "transfer_mb":
                        result.transfer_mb,
                    "handover_ms":
                        result.handover_ms,
                    "slo_violations":
                        result.slo_violations,
                    "batch_size":
                        result.batch_size,
                    "reward":
                        result.total_reward,
                }
            )

            completed += 1

        if (
            episode == 0
            or (
                episode + 1
            )
            % max(
                args.log_every,
                1,
            )
            == 0
            or episode + 1
            == args.episodes
        ):
            pd.DataFrame(rows).to_csv(
                out,
                index=False,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            rate = (
                completed / elapsed
                if elapsed > 0
                else 0.0
            )

            eta = (
                (
                    total_runs
                    - completed
                )
                / rate
                if rate > 0
                else float("inf")
            )

            print(
                f"episode="
                f"{episode + 1:4d}/"
                f"{args.episodes} "
                f"runs="
                f"{completed:4d}/"
                f"{total_runs} "
                f"elapsed="
                f"{elapsed:7.1f}s "
                f"eta="
                f"{eta:7.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        out,
        index=False,
    )

    print(
        df.groupby("batcher")
        .mean(numeric_only=True)
        .round(3),
        flush=True,
    )

    print(
        f"Saved {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
''',
    encoding="utf-8",
)

print(f"Patched: {path}")

print(
    "\nCalibration patch "
    "applied successfully."
)
