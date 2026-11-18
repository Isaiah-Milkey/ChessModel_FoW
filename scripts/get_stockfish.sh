#!/usr/bin/env bash
# Download the official Stockfish 19 Linux build (x86-64) into ./stockfish/ and print the
# line to add to your shell setup. On other systems, install Stockfish yourself and set STOCKFISH.
# Pinned to Stockfish 19, the version that labelled this project's data: the "latest" release no
# longer has the old stockfish-ubuntu-x86-64-avx2.tar file, and a newer engine would label differently.
set -euo pipefail
NAME=stockfish-linux-x86-64-universal
URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_19/$NAME.tar.gz"
mkdir -p stockfish
curl -fsSL "$URL" -o stockfish/stockfish.tar.gz
tar -xzf stockfish/stockfish.tar.gz -C stockfish
BIN=$(find "$PWD/stockfish" -type f -name "$NAME" | head -n 1)
[ -n "$BIN" ] || { echo "no $NAME found in the download"; exit 1; }
chmod +x "$BIN"
echo "Stockfish is at: $BIN"
echo "Add this to your shell setup:  export STOCKFISH=$BIN"
