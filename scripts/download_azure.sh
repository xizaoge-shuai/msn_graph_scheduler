#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_ROOT="$ROOT/data/raw"

AZURE_2023="$RAW_ROOT/azure_2023"
AZURE_2024="$RAW_ROOT/azure_2024"

mkdir -p "$AZURE_2023"
mkdir -p "$AZURE_2024"

download_file() {
  local url="$1"
  local output="$2"

  echo
  echo "Downloading:"
  echo "  URL:    $url"
  echo "  Output: $output"

  curl \
    --location \
    --fail \
    --show-error \
    --retry 30 \
    --retry-delay 5 \
    --connect-timeout 30 \
    --max-time 0 \
    --continue-at - \
    --output "$output" \
    "$url"

  if [[ ! -s "$output" ]]; then
    echo "Downloaded file is empty: $output" >&2
    exit 1
  fi

  echo "Finished: $(du -h "$output" | cut -f1)"
}

# Azure LLM inference trace 2023.
download_file \
  "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv" \
  "$AZURE_2023/code.csv"

download_file \
  "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv" \
  "$AZURE_2023/conversation.csv"

# Azure LLM inference trace 2024.
download_file \
  "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv" \
  "$AZURE_2024/code.csv"

download_file \
  "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv" \
  "$AZURE_2024/conversation.csv"

echo
echo "Azure traces downloaded successfully."
echo "Raw data directory: $RAW_ROOT"

find "$RAW_ROOT" \
  -maxdepth 2 \
  -type f \
  -printf "%p\t%k KB\n" \
  | sort
