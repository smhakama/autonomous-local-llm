# cron/ — 定期実行スクリプト集

このディレクトリには 2 系統の dispatcher があり、互いに独立して動く:

| 系統 | スクリプト | 起動方式 | Phase |
|---|---|---|---|
| Web research | `web_research.sh` | crontab (crond) | Phase 2.1b |
| Nightly experiment matrix | `llm_nightly_experiment.sh` | systemd user timer | Phase 3.8a |

それぞれ前提と起動手順が違うので以下別 section に分けて記載。

---

## Phase 2.1b — 定期実行 (cron) 手順

`themes.txt` の各テーマを `web_research.py --themes-file` でバッチ処理し、
Qdrant `web_brain` collection に蓄積する日次運用の雛形。

## 前提

- `setup_ai_env.sh` 実行済 (Ollama / Qdrant docker / venv `~/ai_agents_env`)
- Ollama と Qdrant が WSL 起動時に立ち上がる状態 (起動方法は INSTALL.md 参照)
- repo root に `themes.txt` を編集 (sample 同梱)

## crontab 設定

毎日 03:00 (ローカル時刻) に走らせる例:

```cron
0 3 * * * /home/makoto/projects/autonomous-local-llm/cron/web_research.sh
```

`crontab -e` で追記。WSL2 では cron daemon が auto-start しないため初回:

```bash
sudo systemctl enable --now crond   # Rocky 10 / RHEL
# または
sudo service cron start             # Ubuntu / Debian
```

## ログ確認

```bash
tail -f ~/cron-web_research.log
```

環境変数 `WEB_RESEARCH_LOG` でログパス上書き可:

```cron
0 3 * * * WEB_RESEARCH_LOG=/var/log/web_research.log /home/makoto/projects/autonomous-local-llm/cron/web_research.sh
```

## 多重起動防止

`/tmp/web_research.lock` を `fcntl.LOCK_EX | LOCK_NB` で排他取得。
先行プロセスが走っていれば即終了 (rc=1)、cron が連発しても破壊しない。
reboot で `/tmp/` 配下は消えるため明示削除不要。

`--lock-file` で別パス指定可 (複数 themes 構成を並行運用する場合等)。

## 失敗ハンドリング

default は **部分失敗を許容して継続**: 1 テーマで DDG / fetch / embed が失敗しても
次のテーマへ進み、最後に `[SUMMARY] N/M themes failed: [...]` を stderr に出力 (rc=0)。

すべて成功させたい運用では `--strict` を追加:

```bash
exec python web_research.py --themes-file themes.txt --strict ...
```

`--strict` 時は最初の失敗で即 stop (rc=1) し、後続テーマは処理しない。

## モデル warm-up (任意)

cron 実行前に bge-m3 を暖めておくと初回 fetch が速い:

```cron
55 2 * * * curl -s -X POST http://127.0.0.1:11434/api/embeddings -d '{"model":"bge-m3","prompt":"warmup"}' > /dev/null
0  3 * * * /home/makoto/projects/autonomous-local-llm/cron/web_research.sh
```

`OLLAMA_KEEP_ALIVE=2h` を ollama 起動時に export しておけば eviction 抑制可。

## rollback

```bash
crontab -e                                # 該当行を削除
rm -f /tmp/web_research.lock              # (任意) 残骸があれば
git revert <Phase 2.1b の commit hash>    # コードを Phase 2.1a 状態に戻す
```

## themes.txt フォーマット (再掲)

- 1 行 1 テーマ
- 空行スキップ
- 行頭 `#` はコメントスキップ
- UTF-8 BOM 自動除去 (`utf-8-sig`)
- 行末空白 trim
- 大文字小文字無視で重複検出 → 警告 + 1 回だけ採用

---

## Phase 3.8a — 夜間実験マトリクス (systemd user timer)

`bench/parallel_capacity_check.py` を `experiments/matrices/*.yaml` 駆動で
連続実行し、metrics JSONL に追記する nightly job。本体ロジックは
`experiments/runner.py` 側にあり、ここでは起動 / スケジュール設定のみ扱う。

詳細な matrix schema / 追加方法は `experiments/README.md` を参照。

### entrypoint

```bash
# 既定 matrix を手動実行 (動作確認)
cron/llm_nightly_experiment.sh

# 別 matrix を指定
MATRIX_YAML=experiments/matrices/other.yaml cron/llm_nightly_experiment.sh

# ログ参照
tail -f ~/cron-llm-nightly.log
```

`LLM_NIGHTLY_LOG` で出力先上書き、`LLM_RUNNER_PYTHON` で venv 切替可。
ollama service が active でないと早期 exit する safety guard あり。

### systemd user timer 設定 (推奨)

WSL2 + Rocky 10 では systemd user manager が動くので crontab より systemd
timer のほうが状態確認が容易。template 同梱:

```bash
mkdir -p ~/.config/systemd/user
sed "s|__REPO__|$HOME/projects/autonomous-local-llm|g" \
  cron/llm-nightly.service.template > ~/.config/systemd/user/llm-nightly.service
sed "s|__REPO__|$HOME/projects/autonomous-local-llm|g" \
  cron/llm-nightly.timer.template > ~/.config/systemd/user/llm-nightly.timer

systemctl --user daemon-reload
systemctl --user enable --now llm-nightly.timer

# 確認
systemctl --user list-timers llm-nightly.timer
systemctl --user status llm-nightly.service
```

`OnCalendar=*-*-* 02:00:00` + `Persistent=true` で WSL が落ちていても
次回起動時に catch up する。`loginctl enable-linger $USER` 適用済なら
ログアウト中も発火する。

### rollback

```bash
systemctl --user disable --now llm-nightly.timer
rm -f ~/.config/systemd/user/llm-nightly.{service,timer}
systemctl --user daemon-reload
# 完全削除する場合は experiments/runs/ の蓄積ログも掃除
rm -rf experiments/runs/
```

`metrics/parallel_capacity_checks.jsonl` の既存 record (E0/E1/.../NT*)
には触らない。Phase 3.7e-2 verdict は git で記録済なので、experiments
を消しても結論側のドキュメントは残る。
