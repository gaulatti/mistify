---
name: ai-agentic-update
description: Refresh the current Git repository from the latest centrally managed bastille rules and skills. Use proactively at the start of work in a managed repository and when the user requests ai-agentic-update, an agentic compliance refresh, or an OTA rules update.
---

# AI Agentic Update

Refresh only the current repository from the authoritative `bastille`
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
the fetched default-branch snapshot, creates a narrowly scoped Conventional
Commit when managed files changed, and verifies the result. Wiki management
applies only to public repositories, determined from the GitHub visibility of
the repository's `origin`: it verifies the repository's separate `./wiki`
checkout, clones the wiki when its derived remote exists, and validates an
existing checkout's origin. Private repositories are skipped unless a `./wiki`
checkout already exists, in which case it is preserved and kept current. It
preserves instructions outside the managed markers, differently named local
skills, uncommitted wiki work, and unrelated product changes. Its commit owns
only `AGENTS.md`, the managed-skills manifest, and centrally owned skill paths;
it must not absorb staged or unstaged product files.

Do not replace this command with hand-written copying. After it completes,
inspect the relevant wiki pages in public repositories before planning code
changes. Do not separately commit the refresh or carry its files into the
product change. If its scoped commit fails, stop rather than continuing with
dirty managed metadata. The updater does not push, deploy, reset, rebase, or
discard product or wiki work.

If the fetched revision changes the updater implementation, the current process
hands off once to a temporary copy of the fetched updater. A guard prevents
handoff loops and makes a failure visible. The central sync also recognizes the
exact no-commit call made by pre-commit legacy updaters and creates the required
narrow bootstrap commit during that same first invocation; this compatibility
does not make ordinary manual `sync --write --repository` calls commit.

Access to private bastille must not block agent execution. If central fetch or
clone fails because access or connectivity is unavailable, the updater reports
`bastille-unavailable` and uses the complete checked-in managed snapshot without
changing it. Continue with that snapshot; no private-repository credential is
required of contributors. Report the unavailable refresh in the handoff.

Malformed or incomplete installed metadata, conflicting ownership, unsafe central
content, and failed managed commits still stop work and must be reported. Do not
invent a new snapshot or ignore these errors. Initial installation without a
committed snapshot needs an authorized maintainer.

The unavailable-access check rejects tracked edits and all untracked additions
inside centrally owned skill directories, including ignored files. Remove or
commit intended managed changes before relying on a checked-in snapshot. Files
in differently named local skills and unrelated product work remain untouched.

For a task worktree already stopped by the legacy dirty-metadata failure, do not
blindly invoke the updater again. Preserve unrelated work and restart the ticket
from a fresh current-default-branch worktree after the compatibility release is
published.

An explicit invocation authorizes the updater's fetch and its narrowly scoped
changes to the central checkout/cache and current repository. It does not
authorize any other external or destructive action.
