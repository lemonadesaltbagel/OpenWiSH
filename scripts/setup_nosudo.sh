#!/usr/bin/env bash
# Set up OpenWiSH on Linux or WSL without sudo. Each step checks first and skips when already done, so a second run
# changes nothing. See docs/fork.md.
#
#   scripts/setup_nosudo.sh                     # bubblewrap, Rust, uv, Python 3.12
#   scripts/setup_nosudo.sh --venvs sandbox     # also assets/venvs/sandbox (CPU torch)
#   scripts/setup_nosudo.sh --venvs sandbox,qwen   # also assets/venvs/qwen (vLLM, for the exp8 GPU machine)
#
# OPENWISH_PREFIX (default ~/.local) receives bwrap and uv. OPENWISH_ROOT (default: the checkout holding this script)
# receives the venvs. bubblewrap needs unprivileged user namespaces: WSL2 and most desktops have them, RunPod and
# Docker containers do not (no user namespaces, no CAP_SYS_ADMIN), so run OpenWiSH itself outside such containers.
set -euo pipefail

PREFIX="${OPENWISH_PREFIX:-$HOME/.local}"
ROOT="${OPENWISH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENVS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --venvs) VENVS="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
export PATH="$PREFIX/bin:$HOME/.cargo/bin:$PATH"
export HF_HUB_DISABLE_XET=1   # Hugging Face downloads over Xet stalled for us, the plain HTTP path did not

step() { printf '\n== %s\n' "$*"; }
skip() { printf '   done already: %s\n' "$*"; }

step "Linux with unprivileged user namespaces"
[ "$(uname -s)" = Linux ] || { echo "OpenWiSH needs Linux (or WSL2)" >&2; exit 1; }
if command -v unshare >/dev/null && ! unshare -Ur true 2>/dev/null; then
  echo "no unprivileged user namespaces here (a RunPod or Docker container?). bubblewrap cannot run." >&2
  exit 1
fi
command -v prlimit >/dev/null || { echo "prlimit missing (util-linux)" >&2; exit 1; }
echo "   ok"

step "bubblewrap (bwrap) into $PREFIX"
if command -v bwrap >/dev/null; then
  skip "$(command -v bwrap) ($(bwrap --version))"
else
  pkgs="$PREFIX/pkgs"
  mkdir -p "$pkgs" "$PREFIX/bin"
  (cd "$pkgs" && ls bubblewrap_*.deb >/dev/null 2>&1 || apt download bubblewrap)
  dpkg-deb -x "$(ls "$pkgs"/bubblewrap_*.deb | tail -1)" "$pkgs/bwrap-root"
  ln -sfn "$pkgs/bwrap-root/usr/bin/bwrap" "$PREFIX/bin/bwrap"
  bwrap --version
fi
bwrap --ro-bind / / --unshare-all --dev /dev true && echo "   bwrap starts a sandbox"

step "Rust (rustup, stable, with rustfmt)"
if command -v cargo >/dev/null; then
  skip "$(cargo --version)"
else
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -c rustfmt --no-modify-path
fi

step "uv into $PREFIX/bin"
if command -v uv >/dev/null; then
  skip "$(uv --version)"
else
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PREFIX/bin" UV_NO_MODIFY_PATH=1 sh
fi

step "Python 3.12 (uv python install --default)"
if uv python find 3.12 >/dev/null 2>&1; then
  skip "$(uv python find 3.12)"
else
  uv python install --default 3.12 || uv python install --preview --default 3.12
fi

venv() {  # venv NAME PACKAGES...
  local d="$ROOT/assets/venvs/$1"
  shift
  if [ -x "$d/bin/python" ]; then
    skip "$d"
    return 1
  fi
  uv venv -q -p 3.12 "$d" || exit 1
  VIRTUAL_ENV="$d" uv pip install -q "$@" || exit 1
}

for v in ${VENVS//,/ }; do
  case "$v" in
    sandbox)
      step "venv assets/venvs/sandbox"
      if venv sandbox numpy scipy scikit-learn pandas sympy networkx safetensors minatar tokenizers; then
        VIRTUAL_ENV="$ROOT/assets/venvs/sandbox" uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu || exit 1
      fi
      ;;
    qwen)
      step "venv assets/venvs/qwen (the exp8 GPU worker)"
      if venv qwen "vllm==0.33.0" "transformers==5.20.2" safetensors math-verify; then
        # FlashInfer kernels without nvcc: vLLM downloads prebuilt ones
        PATH="$ROOT/assets/venvs/qwen/bin:$PATH" vllm download-kernels || exit 1
      fi
      ;;
    *) echo "unknown venv $v (sandbox, qwen)" >&2; exit 2 ;;
  esac
done

step "next"
echo "   put $PREFIX/bin and ~/.cargo/bin on PATH, then: nice -n 10 cargo build --workspace -j 4"
