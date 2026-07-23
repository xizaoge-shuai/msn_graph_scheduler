from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AzureColumns:
    input_tokens: str = "input_tokens"
    output_tokens: str = "output_tokens"
    timestamp: str = "timestamp"
    request_type: str = "request_type"


class EmpiricalRequestSampler:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, rng: np.random.Generator):
        if len(inputs) == 0 or len(outputs) == 0:
            raise ValueError("Empty Azure request sample")
        self.inputs = np.asarray(inputs, dtype=int)
        self.outputs = np.asarray(outputs, dtype=int)
        self.rng = rng

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, len(self.inputs), size=n)
        return self.inputs[idx], self.outputs[idx]


def load_generic_azure_trace(path: str, columns: AzureColumns = AzureColumns()) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [columns.input_tokens, columns.output_tokens]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Azure trace missing columns: {missing}")
    out = df.rename(
        columns={
            columns.input_tokens: "input_tokens",
            columns.output_tokens: "output_tokens",
            columns.timestamp: "timestamp",
            columns.request_type: "request_type",
        }
    ).copy()
    out = out[(out["input_tokens"] > 0) & (out["output_tokens"] > 0)]
    return out
