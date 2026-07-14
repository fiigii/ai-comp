# 2026-07-14 — cocoindex-code (`ccc`) installed and indexed

## What happened

Installed [cocoindex-io/cocoindex-code](https://github.com/cocoindex-io/cocoindex-code) — an AST-based semantic code search CLI — for use with Claude Code in this repo.

### Install method
Chose the "Claude Code skill" path:
```
npx skills add cocoindex-io/cocoindex-code
```
Note: on this machine, plain `npx` gets mangled by the `rtk` shell hook (silently rewrites it into a broken `npm` call). Had to bypass with:
```
rtk proxy npx -y skills add cocoindex-io/cocoindex-code
```

This created (untracked, not yet committed):
- `.agents/skills/ccc/` — the actual skill (universal format)
- `.claude/skills/ccc/` — symlink for Claude Code
- `cocoindex_command.md` — CLI reference
- `skills-lock.json` — lock file

### CLI install
The skill only registers the prompt/docs — the `ccc` binary itself needed separate install. `pipx` wasn't available, so used `uv` instead:
```
rtk proxy uv tool install --upgrade 'cocoindex-code[full]'
```
This pulled in torch + sentence-transformers (~8 min build/install). Binary lands in `C:\Users\minht\.local\bin\ccc`, which is **not on PATH** by default — needs `export PATH="$HOME/.local/bin:$PATH"` per session, or run `uv tool update-shell`.

### Config
`ccc init` defaults to a cloud LiteLLM embedding model requiring `OPENAI_API_KEY`. Since we installed `[full]` specifically for local embeddings, manually edited `~/.cocoindex_code/global_settings.yml` to:
```yaml
embedding:
  provider: sentence-transformers
  model: Snowflake/snowflake-arctic-embed-xs
```
No API key needed this way (384-dim local model).

### Windows gotcha
`ccc doctor` / other commands crash with `UnicodeEncodeError` on Windows due to box-drawing Unicode chars hitting the cp1252 console codec. Fix: set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` before invoking `ccc`.

### Result
Indexed the repo successfully:
- 2117 chunks, 105 files (python: 1914, markdown: 172, html: 25, json: 6)
- Index db: `D:\THANG\GIT\ai-comp\.cocoindex_code\target_sqlite.db` (auto-gitignored)

Ran `ccc search optimization` — top hits were `compiler/pass_config.json` (the actual pass pipeline order), `compiler/__init__.py` (pass exports), and design docs `docs/load_store_optimizations.md` / `docs/slp_vectorization_design.md`.

## To reproduce / use again in a fresh shell

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
ccc search "<query>"
```

Or invoke via the `ccc` Claude Code skill (auto-triggers on phrases like "search the codebase", "ccc", "cocoindex-code").

## Open items
- New files (`.agents/`, `.claude/skills/ccc/`, `cocoindex_command.md`, `skills-lock.json`) are untracked in git — not committed yet, no decision made on whether to commit or gitignore.
- `ccc` binary still not on PATH outside tool-mediated Bash sessions — consider running `uv tool update-shell` for a permanent fix.
