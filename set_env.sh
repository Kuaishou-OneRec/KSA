#!/usr/bin/env bash
# Optional local environment only. No installs, remote commands or cleanup.
KSA_ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
if [[ -f "$KSA_ENV_FILE" ]]; then
    case $- in
        *a*) source "$KSA_ENV_FILE" ;;
        *) set -a; source "$KSA_ENV_FILE"; set +a ;;
    esac
fi
unset KSA_ENV_FILE
