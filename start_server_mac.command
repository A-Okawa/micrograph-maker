#!/bin/bash
# Micrograph Maker を研究室の共有アプリとして起動（Mac）：同じ学内ネットワークの PC からブラウザで使えます。
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 serve.py --check >/dev/null 2>&1; then
  PY="$(command -v python3)"  # Anaconda など、必要なパッケージが入っている Python
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 が見つかりません。https://www.python.org から 3.10 以降を入れてください。"
    read -r -p "Enter で閉じます"; exit 1
  fi
  echo "初回セットアップ中です（数分かかります）…"
  python3 -m venv .venv || { echo "仮想環境を作れませんでした"; read -r -p "Enter で閉じます"; exit 1; }
  if [ -d wheels ]; then
    .venv/bin/python -m pip install --no-index --find-links wheels -r requirements.txt
  else
    .venv/bin/python -m pip install -r requirements.txt
  fi
  if [ $? -ne 0 ]; then
    echo "パッケージを入れられませんでした（オフラインなら wheels フォルダが必要です。README参照）"
    rm -rf .venv; read -r -p "Enter で閉じます"; exit 1
  fi
  PY=".venv/bin/python"
fi
"$PY" serve.py
read -r -p "終了しました。Enter で閉じます"
