---
name: ai-agentic-update
description: Refresh the current Git repository from the latest centrally managed agentic-coding rules and skills. Use proactively at the start of work in a managed repository and when the user requests ai-agentic-update, an agentic compliance refresh, or an OTA rules update.
---

# AI Agentic Update

Refresh only the current repository from the authoritative `agentic-coding`
repository.

## Required action

Run this skill's deterministic updater before substantive repository work:

```bash
.agents/skills/ai-agentic-update/scripts/update
```

If the current directory is below the repository root, resolve the repository
root first and execute the script from there. After a successful update, reread
the root `AGENTS.md` and any applicable updated skills before continuing.

The helper fetches the central repository, hydrates the current repository from
the fetched default-branch snapshot, and verifies the result. It preserves
instructions outside the managed markers and differently named local skills.

Do not replace this command with hand-written copying. Do not commit, push,
deploy, reset, rebase, or discard worktree changes as part of the refresh.

If the updater fails or cannot verify the central source, stop before making
other repository mutations and report the exact failure. Do not silently work
against an unverified or stale standard.

An explicit invocation authorizes the updater's fetch and its narrowly scoped
changes to the central checkout/cache and current repository. It does not
authorize any other external or destructive action.
