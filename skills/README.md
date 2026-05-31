# skills/ — Auto-Distilled Helpers (Karpathy Layer 4)

`corpus2skill.py` が `web_brain_clean` collection (Karpathy Layer 2 Wiki Pages)
からテーマ別 chunks を取り出し、`deepseek-r1:14b` で蒸留して生成する Python
関数群。

## 構成

```
skills/
├── __init__.py
├── README.md  (このファイル)
└── <slug>.py  ← corpus2skill.py が theme 別に自動生成
```

各 `.py` モジュールはヘッダー docstring に以下を記録:

- Theme (元 chunks の payload.theme)
- Source chunks 数 + URL 一覧
- 生成日時 (ISO 8601)
- 使用モデル

## 再生成

```bash
python corpus2skill.py --theme "FastAPI dependency injection patterns"
```

オプション:

- `--collection web_brain_clean` (デフォルト)
- `--model deepseek-r1:14b` (デフォルト)
- `--timeout 600` 秒
- `--max-retries 3`
- `--no-verify` (subprocess import 検証を skip)

## 利用例 (将来の Phase 4 想定)

```python
# Aider script 内 / browser-use action / hotfix_loop など
from skills.fastapi_dependency_injection_patterns import build_dependency_chain
chain = build_dependency_chain(...)
```

これにより 7B モデル (qwen2.5-coder:7b) は「巨大ドキュメント」を context に
持たず、「1 関数を呼ぶだけ」で目的達成できる (Phase 3 decision drawer 参照)。

## 手動編集の扱い

corpus2skill.py の次回実行で **上書きされる** 想定。改修したい場合は:

1. `corpus2skill.py` 内の `DISTILL_PROMPT_TEMPLATE` を改善して再生成
2. または、生成 skill を `skills/<slug>_manual.py` 等にコピーして別名管理

## 品質ゲート

`corpus2skill.py` が以下 3 段を自動検査:

| Layer | 検査 |
|---|---|
| L1 | `ast.parse()` で syntax 確認 |
| L2 | subprocess で `import skills.<slug>` 成功確認 |
| L3 | top-level callable が ≥1 存在 |

L1 で fail なら retry (最大 `--max-retries` 回)、L2/L3 で fail は exit code 2/3。
追加で `ask_gemini` wrapper 経由で Gemini Pro レビューに投げる運用も可
(Phase 3 decision drawer 8e24705d 参照)。

## 由来

- 上位設計: Phase 3 = Corpus2Skill 採択 (palace decisions drawer 参照)
- 理論基盤: Karpathy LLM Wiki 3 層 + Layer 4 Skills
- 前提: `web_brain_clean` に該当 theme の Wiki Pages が存在すること (Phase 2.5b3 完了)
