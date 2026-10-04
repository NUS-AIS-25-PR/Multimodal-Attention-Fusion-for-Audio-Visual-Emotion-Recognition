# AGENTS.md

## Purpose

This repository is being revised for the IEEE SPMB 2026 mandatory-revision round. The immediate goal is **not** to redesign the whole research system. The immediate goal is to make the experimental evidence rigorous enough to address the reviewers' two most important concerns within one week:

1. Replace the paper's random sample split with a strict **speaker/actor-independent evaluation protocol**.
2. Add a credible **external published fusion baseline** that can be evaluated fairly under the same protocol.

Read these files before changing code:

1. `docs/VISION.md`
2. `docs/DECISIONS.md`
3. `docs/SPEC.md`
4. `docs/ACCEPTANCE.md`
5. `docs/EXAMPLES.md`
6. `docs/CONTEXT.md`

## Decision priority

When instructions conflict, use this priority:

1. User's newest explicit decision
2. `docs/DECISIONS.md`
3. `docs/VISION.md`
4. `docs/SPEC.md`
5. Implementation convenience

Do not silently preserve conflicting old behavior when a newer project decision supersedes it.

## Implementation principles

- Work on a new branch. Do **not** commit directly to `main`.
- Keep the scope focused on experiment infrastructure and the adapted external baseline.
- Do **not** start the final six-fold production training in this first implementation PR.
- Reuse the current data/model/training stack instead of creating a parallel research codebase.
- Preserve the paper's current core encoders unless a change is explicitly required:
  - audio: `microsoft/wavlm-base`
  - video: ImageNet-pretrained ResNet18
- Do not introduce a new architecture merely to improve scores.
- Do not reuse checkpoints trained with the old random split in any new actor-independent fold.
- Treat speaker leakage as a correctness bug.
- Every fold must be reproducible and auditable after training.
- The final test metrics for a run must come from the **best validation checkpoint**, not merely the last epoch.
- Prefer small, explicit, tested changes over broad refactors.
- Keep backend/frontend/deployment code out of scope unless a direct dependency forces a minimal change.

## Suggested branch

`codex/ieee-spmb-revision-eval`

## First PR scope

Implement and test:

- fixed six-fold actor-independent split definitions
- dataset preflight audit
- leakage/integrity guards
- fold-aware paths and metadata
- best-checkpoint reload before test evaluation
- standalone evaluation consistency fixes, especially WavLM handling
- aggregate per-fold metrics with mean/std
- adapted Chumachenko et al. Intermediate Attention baseline
- tests for all of the above
- concise documentation for running one smoke fold and the full six-fold suite

Do not run the final expensive experiment matrix as part of the PR unless needed for a very small smoke test.

## Required PR evidence

The PR description should include:

- files changed
- exact commands used for tests
- test results
- one cheap smoke-run result showing the fold machinery executes end-to-end
- explicit statement that no old random-split unimodal checkpoint is reused
- explicit statement that train/val/test actors are disjoint
- any unresolved environment or runtime risks

If an important requirement is ambiguous, prefer preserving scientific validity over implementation convenience and document the open question rather than guessing.
