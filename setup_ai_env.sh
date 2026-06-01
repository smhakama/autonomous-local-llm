#!/usr/bin/env bash
# ==============================================================================
# setup_ai_env.sh — Rocky Linux 10 (WSL2) 自律型 AI 開発環境セットアップ
#   target: Ollama (qwen2.5-coder:7b + deepseek-r1:14b) + Aider + Qdrant
#   constraint: VRAM 8GB / RAM 32GB
# ==============================================================================
set -euo pipefail

LOG_PREFIX="[setup_ai_env]"
log()  { echo "${LOG_PREFIX} $*"; }
warn() { echo "${LOG_PREFIX} WARN: $*" >&2; }
die()  { echo "${LOG_PREFIX} ERROR: $*" >&2; exit 1; }

trap 'die "line $LINENO で異常終了 (exit=$?)"' ERR

# WSL 検出 (systemd 系コマンドの挙動分岐用)
IS_WSL=0
if grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=1
  log "WSL2 環境を検出"
fi

# ------------------------------------------------------------------------------
# 1. システムアップデート + 必須開発ツール
# ------------------------------------------------------------------------------
log "Step 1/5: dnf update + dev tools"
sudo dnf -y makecache
sudo dnf -y upgrade --refresh

# Rocky 10 は python3.12 が標準。3.11 fallback も試す。
if ! sudo dnf -y install python3 python3-pip python3-devel \
                        git curl wget gcc gcc-c++ make tar which jq zstd; then
  warn "python3 メタパッケージ失敗 — 3.12/3.11 を明示指定で再試行"
  sudo dnf -y install python3.12 python3.12-pip python3.12-devel \
                      git curl wget gcc gcc-c++ make tar which jq zstd \
    || sudo dnf -y install python3.11 python3.11-pip python3.11-devel \
                            git curl wget gcc gcc-c++ make tar which jq zstd zstd
fi

# 最良の Python を選定 (3.11+ 必須)
PYTHON_BIN=""
for cand in python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    major=${ver%.*}; minor=${ver#*.}
    if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
      PYTHON_BIN=$(command -v "$cand"); break
    fi
  fi
done
[ -n "$PYTHON_BIN" ] || die "Python 3.11+ が見つかりません"
log "Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# ------------------------------------------------------------------------------
# 2. Docker + Qdrant
# ------------------------------------------------------------------------------
log "Step 2/5: Docker + Qdrant"
if ! command -v docker >/dev/null 2>&1; then
  log "Docker 未導入 — 公式インストーラ実行"
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
  if [ "$IS_WSL" -eq 0 ]; then
    sudo systemctl enable --now docker
  else
    warn "WSL2 では Docker Desktop の WSL Integration を有効にしてください"
  fi
  sudo usermod -aG docker "$USER" || true
else
  log "Docker 既存: $(docker --version)"
fi

# docker daemon 疎通確認
if ! docker info >/dev/null 2>&1; then
  if [ "$IS_WSL" -eq 1 ]; then
    die "Docker daemon に到達できません。Docker Desktop の Settings → Resources → WSL Integration で Rocky 10 を有効化し、本スクリプトを再実行してください"
  else
    die "Docker daemon が起動していません"
  fi
fi

QDRANT_DIR="${HOME}/qdrant_storage"
mkdir -p "$QDRANT_DIR"

# 既存 qdrant コンテナがあれば差し替え (冪等性確保)
if docker ps -a --format '{{.Names}}' | grep -qx qdrant; then
  log "既存 qdrant コンテナを停止・削除"
  docker rm -f qdrant >/dev/null 2>&1 || true
fi

log "Qdrant 起動 (ports 6333, 6334 / volume ${QDRANT_DIR})"
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v "${QDRANT_DIR}:/qdrant/storage" \
  qdrant/qdrant:latest

# 起動待ち
for i in $(seq 1 30); do
  if curl -fsS http://localhost:6333/readyz >/dev/null 2>&1 \
     || curl -fsS http://localhost:6333/ >/dev/null 2>&1; then
    log "Qdrant ready (${i}s)"; break
  fi
  sleep 1
  [ "$i" -eq 30 ] && warn "Qdrant 起動確認タイムアウト — docker logs qdrant を確認"
done

# ------------------------------------------------------------------------------
# 3. Ollama インストール + モデル pull
# ------------------------------------------------------------------------------
log "Step 3/5: Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  log "Ollama 未導入 — 公式 install.sh 実行"
  curl -fsSL https://ollama.com/install.sh | sh
else
  log "Ollama 既存: $(ollama --version 2>/dev/null || echo unknown)"
fi

# ollama serve 常駐確認 (systemd 不可な WSL も考慮)
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  if [ "$IS_WSL" -eq 1 ]; then
    log "ollama serve をバックグラウンド起動 (nohup)"
    nohup ollama serve >"${HOME}/ollama.log" 2>&1 &
    disown || true
  else
    sudo systemctl enable --now ollama || nohup ollama serve >"${HOME}/ollama.log" 2>&1 &
  fi
  for i in $(seq 1 30); do
    curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi

log "モデル pull: qwen2.5-coder:7b (chat/edit 主力)"
ollama pull qwen2.5-coder:7b

log "モデル pull: deepseek-r1:14b (思考/監査 / 部分 offload 想定)"
ollama pull deepseek-r1:14b

log "モデル pull: bge-m3 (CPU embedding / RAM 側で Qdrant に投入)"
ollama pull bge-m3

# ------------------------------------------------------------------------------
# 4. Python venv + エージェント (aider, browser-use, langchain-ollama, playwright)
# ------------------------------------------------------------------------------
log "Step 4/5: venv at ~/ai_agents_env"
VENV_DIR="${HOME}/ai_agents_env"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "venv 既存 — 再利用"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools
pip install --upgrade \
  aider-chat \
  browser-use \
  langchain-ollama \
  playwright \
  pytest

log "playwright chromium インストール"
playwright install chromium
# 依存ライブラリ (Rocky 10 系は手動補完が必要なケースあり)
playwright install-deps chromium 2>/dev/null || \
  warn "playwright install-deps 失敗 (root 不要パスを試行) — ブラウザ起動時に libnss3 等が不足する場合は手動で dnf install"

deactivate

# ------------------------------------------------------------------------------
# 5. 疎通テスト + 起動ワンライナー
# ------------------------------------------------------------------------------
log "Step 5/5: 疎通テスト"

echo "--- Ollama /api/tags ---"
curl -fsS http://localhost:11434/api/tags | jq '.models[] | {name, size}' 2>/dev/null \
  || curl -fsS http://localhost:11434/api/tags

echo "--- Qdrant root ---"
curl -fsS http://localhost:6333/ | jq . 2>/dev/null \
  || curl -fsS http://localhost:6333/

echo "--- docker ps (qdrant) ---"
docker ps --filter name=qdrant --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

cat <<'EOF'

==============================================================================
[setup_ai_env]  環境構築完了
==============================================================================

▼ Aider をローカル LLM (qwen2.5-coder:7b) で即起動するワンライナー:

    OLLAMA_API_BASE=http://127.0.0.1:11434 aider --model ollama_chat/qwen2.5-coder:7b

▼ venv 有効化:

    source ~/ai_agents_env/bin/activate

▼ 思考/監査用に deepseek-r1:14b へ切替える場合:

    OLLAMA_API_BASE=http://127.0.0.1:11434 aider --model ollama_chat/deepseek-r1:14b

▼ browser-use を 8GB VRAM で安定動かす設定 (Vision 無効化):

    from browser_use import Agent
    # use_vision=False で screenshot を LLM に送らず、HTML/AX-tree のみで解析
    agent = Agent(task="...", llm=..., use_vision=False)

▼ Qdrant に embedding 投入する Python サンプル (bge-m3, CPU):

    from langchain_ollama import OllamaEmbeddings
    emb = OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")
    vec = emb.embed_query("test text")   # 1024 次元

▼ サービスエンドポイント:

    Ollama API   : http://localhost:11434
    Qdrant REST  : http://localhost:6333
    Qdrant gRPC  : localhost:6334
    Qdrant UI    : http://localhost:6333/dashboard

▼ 永続化パス:

    Qdrant data  : ~/qdrant_storage
    venv         : ~/ai_agents_env
    ollama log   : ~/ollama.log  (WSL で nohup 起動時)

==============================================================================
EOF
