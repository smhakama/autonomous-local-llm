# Phase 2.1b — 定期実行 (cron) 手順

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
