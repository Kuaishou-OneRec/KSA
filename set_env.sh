#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
    echo "This script must be run with bash, e.g. 'bash set_env.sh'." >&2
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: ${ENV_FILE} not found"
    exit 1
fi

set -a
source "${ENV_FILE}"
set +a

echo "Loaded environment variables from ${ENV_FILE}:"
cat "${ENV_FILE}"

nohup rm -rf hs_err_pid*.log &

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install sortedcontainers easydict"
