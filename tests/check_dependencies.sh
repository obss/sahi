#!/bin/bash

##################################################################################
# Checks, if the imports of the supported models are working as expected.
# Not all frameworks are available on all python versions.
#
# It checks it for all python versions
# Also, it checks, if the command line tool works on different examples
##################################################################################

# This script should abort on error
set -e

# Python versions to check
PYTHON_VERSIONS=("3.8 " "3.9 " "3.10" "3.11" "3.12")

# Commands to check (okay means: return code 0)
COMMANDS=(
  "uv run python -c 'from mmdet.apis.det_inferencer import DetInferencer'"
  "uv run python -c 'import torch'"
  "uv run python -c 'import ultralytics'"
  "tests/check_commandline.sh"
  "uv run pytest -x"
)

# Corresponding to the commands, the expected behaviour
CONTEXTS=(
  "mmdet/mmcv with Python < 3.11"
  "torch, should work for all python versions"
  "ultralytics, should work for all python versions"
  "command line"
  "pytest"
)

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Initialize an array to store results
declare -A RESULTS

# Loop over each Python version
for version in "${PYTHON_VERSIONS[@]}"; do
  echo "Checking Python $version..."
  uv python pin "$version"
  uv sync -U
  # uv sync

  # Loop over each command
  for cmd in "${COMMANDS[@]}"; do
    echo -n "Checking $cmd..."
    # Check if the command runs without errors
    if eval "$cmd"; then
      RESULTS["$version $cmd"]="${GREEN}✅ okay${NC}"
    else
      RESULTS["$version $cmd"]="${RED}❌ not working${NC}"
    fi
    echo -e "${RESULTS["$version $cmd"]}"
  done
done

# Display the results
for index in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[$index]}"
  context="${CONTEXTS[$index]}"

  echo -e "\n$context:"
  for version in "${PYTHON_VERSIONS[@]}"; do
    echo -e "$version : ${RESULTS["$version $cmd"]}"
  done
done
