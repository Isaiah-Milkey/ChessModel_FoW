#!/usr/bin/env bash
# Download the official Stockfish 19 Linux build (x86-64) into ./stockfish/ and print the
# line to add to your shell setup. On other systems, install Stockfish yourself and set STOCKFISH.
# Pinned to Stockfish 19, the version that labelled this project's data: the "latest" release no
# longer has the old stockfish-ubuntu-x86-64-avx2.tar file, and a newer engine would label differently.
#
# The official binary needs glibc 2.35. On older systems (Sol has 2.28) it does not start, so the
# script builds the same Stockfish 19 from the source in the download, with conda-forge's compilers
# against an old glibc (2.17) and the C++ runtime linked in. That needs conda or mamba
# (on Sol: module load mamba/latest) and about 5 minutes; Stockfish's build downloads its two
# network files from tests.stockfishchess.org.
set -euo pipefail
NAME=stockfish-linux-x86-64-universal
URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_19/$NAME.tar.gz"
DIR=$PWD/stockfish
BUILT=$DIR/sf19-built/stockfish
starts() { [ -x "$1" ] && echo uci | timeout 60 "$1" 2>/dev/null | grep -q uciok; }
done_at() {
    echo "Stockfish is at: $1"
    echo "Add this to your shell setup:  export STOCKFISH=$1"
    exit 0
}

starts "$BUILT" && done_at "$BUILT"                  # built on an earlier run
mkdir -p "$DIR"
if [ ! -f "$DIR/stockfish.tar.gz" ]; then
    curl -fsSL "$URL" -o "$DIR/stockfish.tar.gz"
    tar -xzf "$DIR/stockfish.tar.gz" -C "$DIR"
fi
BIN=$(find "$DIR" -type f -name "$NAME" | head -n 1)
[ -n "$BIN" ] || { echo "no $NAME found in the download"; exit 1; }
chmod +x "$BIN"
starts "$BIN" && done_at "$BIN"

echo "The official binary does not start here:"
(echo uci | "$BIN" 2>&1 || true) | head -n 2
SRC=$(dirname "$(find "$DIR" -type f -path '*/src/Makefile' | head -n 1)")
[ -f "$SRC/Makefile" ] || { echo "no Stockfish source found in the download"; exit 1; }
CONDA=$(command -v mamba || command -v conda || true)
[ -n "$CONDA" ] || { echo "building it needs conda or mamba (on Sol: module load mamba/latest)"; exit 1; }
ENV=$DIR/build-env
echo "== building Stockfish 19 from its source (about 5 minutes)"
[ -x "$ENV/bin/x86_64-conda-linux-gnu-g++" ] || \
    "$CONDA" create -y -q -p "$ENV" -c conda-forge gxx_linux-64 "sysroot_linux-64=2.17" make
# Stockfish's own compiler flags only (conda's activation flags are not wanted), and the C++ runtime
# linked statically so the binary runs without this environment.
(cd "$SRC" && env -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u CFLAGS PATH="$ENV/bin:$PATH" \
    make -j 8 build ARCH=x86-64-avx2 COMP=gcc COMPCXX="$ENV/bin/x86_64-conda-linux-gnu-g++" \
    EXTRALDFLAGS="-static-libstdc++ -static-libgcc")
mkdir -p "$(dirname "$BUILT")"
cp "$SRC/stockfish" "$BUILT"
starts "$BUILT" || { echo "the built Stockfish does not start either:"; (echo uci | "$BUILT" 2>&1 || true) | head -n 3; exit 1; }
echo "$(echo uci | "$BUILT" | grep -m 1 '^id name') built from source"
done_at "$BUILT"
