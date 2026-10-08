#!/usr/bin/env bash
# One-command boot for the Laya Call Router demo.
#
#   ./scripts/dev.sh              # install/build what is missing, then serve
#   ./scripts/dev.sh --rebuild    # force a frontend rebuild first
#   ./scripts/dev.sh --no-serve   # set up and check, but do not start the server
#   ./scripts/dev.sh --open       # open the browser once the server answers
#   ./scripts/dev.sh --port 9000  # use a different port
#   PORT=9000 ./scripts/dev.sh    # serve on another port
#
# Re-running skips completed downloads and frontend work. The setup check only
# reports whether optional Kaggle auth appears configured; it doesn't display
# credential values or change authentication settings.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8765}"
REBUILD=0
SERVE=1
OPEN=0

usage() {
    cat <<'EOF'
One-command boot for the Laya Call Router demo.

  ./scripts/dev.sh              # install/build what is missing, then serve
  ./scripts/dev.sh --rebuild    # force a frontend rebuild first
  ./scripts/dev.sh --no-serve   # set up and check, but do not start the server
  ./scripts/dev.sh --open       # open the browser once the server answers
  ./scripts/dev.sh --port 9000  # use a different port
  PORT=9000 ./scripts/dev.sh    # serve on another port

Re-running skips completed downloads and frontend work. The setup check only
reports whether optional Kaggle auth appears configured; it doesn't display
credential values or change authentication settings.
EOF
}

die() {
    printf '\033[1;31mxx\033[0m %s\n' "$*" >&2
    exit 1
}

log() {
    printf '\033[1;36m==>\033[0m %s\n' "$*"
}

warn() {
    printf '\033[1;33m!!\033[0m %s\n' "$*" >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --rebuild) REBUILD=1 ;;
        --no-serve) SERVE=0 ;;
        --open) OPEN=1 ;;
        --port)
            shift
            [ $# -ge 1 ] || die "--port needs a value"
            PORT="$1"
            ;;
        --port=*) PORT="${1#*=}" ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
    shift
done

# ------------------------------------------------------------------ prerequisites
for cmd in uv git node npm; do
    command -v "$cmd" >/dev/null 2>&1 || die "$cmd is required but is not on PATH"
done

# ------------------------------------------------------------------ model weights
# Fresh clones carry a small Git LFS pointer until `git lfs pull` replaces it.
WEIGHTS="models/active/model.safetensors"
is_lfs_pointer() {
    [ -f "$1" ] && head -c 200 "$1" 2>/dev/null | grep -q "git-lfs.github.com/spec"
}

if [ ! -f "$WEIGHTS" ] || [ ! -s "$WEIGHTS" ]; then
    log "Active checkpoint is missing; fetching model weights with Git LFS"
    command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1 ||
        die "Git LFS is not installed; see https://git-lfs.com/"
    git lfs pull
elif is_lfs_pointer "$WEIGHTS"; then
    log "Active checkpoint is a Git LFS pointer; fetching model weights"
    command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1 ||
        die "Git LFS is not installed; see https://git-lfs.com/"
    git lfs pull
else
    log "Model weights present ($WEIGHTS)"
fi

# ------------------------------------------------------------------ python deps
log "Syncing Python dependencies (uv sync)"
uv sync

# ------------------------------------------------------------------ frontend
if [ ! -d web/node_modules ]; then
    log "Installing frontend dependencies (npm ci)"
    (cd web && npm ci)
else
    log "Frontend dependencies present (web/node_modules)"
fi

need_build=$REBUILD
if [ ! -f web/dist/index.html ]; then
    need_build=1
elif [ "$need_build" = 0 ] &&
    find web -path web/node_modules -prune -o -path web/dist -prune -o \
        -type f -newer web/dist/index.html -print -quit | grep -q .; then
    need_build=1
fi

if [ "$need_build" = 1 ]; then
    log "Building frontend (npm run build)"
    (cd web && npm run build)
else
    log "Frontend build is up to date (web/dist)"
fi

# ------------------------------------------------------------------ git hooks
if [ "$(git config --local --get core.hooksPath || true)" != ".githooks" ]; then
    log "Enabling repository Git hooks (.githooks)"
    git config core.hooksPath .githooks
else
    log "Repository Git hooks already enabled"
fi

# ------------------------------------------------------------------ setup check
log "Checking setup (scripts/doctor.py)"
uv run python scripts/doctor.py || die "setup check failed (see above)"

# ------------------------------------------------------------------ serve
if [ "$SERVE" = 0 ]; then
    log "Setup complete. Start the server with:"
    printf '    uv run uvicorn jev_classifier.api:app --host 127.0.0.1 --port %s\n' "$PORT"
    exit 0
fi

if command -v lsof >/dev/null 2>&1 &&
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "port $PORT is already in use; stop the other process or set PORT"
fi

URL="http://127.0.0.1:$PORT"

if [ "$OPEN" = 1 ]; then
    (
        for _ in $(seq 1 60); do
            if curl -fsS "$URL/api/health" >/dev/null 2>&1; then
                command -v open >/dev/null 2>&1 && open "$URL"
                exit 0
            fi
            sleep 0.5
        done
    ) &
fi

log "Starting Laya Call Router at $URL (Ctrl-C to stop)"
exec uv run uvicorn jev_classifier.api:app --host 127.0.0.1 --port "$PORT"
