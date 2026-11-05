#!/bin/bash
# Micrograph Maker（Mac）: ダブルクリックで起動。初回だけ必要なパッケージを入れます。
cd "$(dirname "$0")" || exit 1
PY="$(command -v python3)"
if [ -z "$PY" ]; then
  echo "Python 3 が見つかりません。https://www.python.org から 3.10 以降を入れてください。"
  read -r -p "Enter で閉じます"; exit 1
fi
if [ ! -x ".venv/bin/python" ]; then
  echo "初回セットアップ中です（数分かかります）…"
  "$PY" -m venv .venv || { echo "仮想環境を作れませんでした"; read -r -p "Enter で閉じます"; exit 1; }
  if [ -d wheels ]; then
    .venv/bin/python -m pip install --no-index --find-links wheels -r requirements.txt
  else
    .venv/bin/python -m pip install -r requirements.txt
  fi
  if [ $? -ne 0 ]; then
    echo "パッケージを入れられませんでした（オフラインなら wheels フォルダが必要です。README参照）"
    rm -rf .venv; read -r -p "Enter で閉じます"; exit 1
  fi
fi
echo "ブラウザで http://localhost:8501 が開きます。終了するときはこのウィンドウを閉じてください。"
exec .venv/bin/python -m streamlit run app.py --browser.gatherUsageStats false
