#!/usr/bin/env bash

set -o pipefail

INSTALL_DIR="${PWD}"

if [[ ! -d "${INSTALL_DIR}/venv" ]]; then
  echo "Creating Python Virtual Environment"
  python3 -m venv venv
  source "venv/bin/activate"
  pip install pip --upgrade > /dev/null
  pip -q install -r requirements.txt
  echo "$HOSTNAME"
fi
