# P29 kickoff prompt (paste into a fresh Claude Code session on `main`)

> Use this **after** the `training` branch (P21–P28) is merged to `main`. Start P29 on a
> new branch off the updated `main`.

---

You're picking up work on the rhdram-env repo. Your task is to implement **Phase 29**
(release readiness) of `spec/IMPLEMENTATION_PLAN_V3.md`. P21–P28 are merged. P29 is the
standalone pre-release de-phasing restructure plus SPEC §12 acceptance hardening. Do the
background reading before writing any code.

## Read first, in this order
1. `spec/P29_RELEASE_READINESS.md` — **your execution spec.** It reconciles the older
   restructure doc with the current tree and lists every task (A–E), the reconciled rename
   tables, the test-migration map, the determinism/no-mock/docs/acceptance work, and the
   definition of done. Follow it; where it and the PR-doc agree, the PR-doc is normative.
2. `spec/SPEC.md` — §2 (no mocks/fallbacks; fail closed with a stable error code), §9
   (reward only from trusted state), and **§12 (acceptance)** — P29 is measured against §12.
3. `spec/PRE_RELEASE_RESTRUCTURE.md` — the "PR" restructure P29 finally executes (target
   component layout §2, renames §3, gate→test migration §4, archive §5, done-when §6).
4. `spec/IMPLEMENTATION_PLAN_V3.md` §0–§1 and the **P29** section (the plan-level intent),
   and `docs/adr-0004-secret-address-mapping.md` (the P24 secret mapper you must keep intact).

Then open the actual files `P29_RELEASE_READINESS.md` cites (the env chain
`rowhammer_env/phase{1,2,4,5}_env.py`, the already-existing `server/`/`tools/`/`tasks/`/
`rewards/`/`observability/`/`llm/` dirs, `scripts/verify_release.py`) and **re-verify the
counts and file lists** — the spec was accurate on 2026-07-12 but the tree moves.

## Before you touch anything
- **Capture the baseline.** Run `python3 -m unittest discover -s tests` and record the exact
  pass/skip/fail set. Expected on the dev host: ~263 tests, only `test_phase19` erroring
  (missing `fastapi`/`uvicorn`/`websockets` — a host gap, not a defect). The post-P29 suite
  must match this *meaning* (same assertions, re-homed into `tests/<component>/`).
- Confirm the worker builds (`scripts/build_phase2.py` → `build/phase2/ramulator_worker`);
  many acceptance tests are worker-gated. If you can't build the C++ worker on this host,
  say so rather than working around it.
- Confirm you're on a fresh solo branch off `main`.

## Hard constraints
- **This is behavior-preserving.** No feature logic changes. Every test that passed before
  passes after, unchanged in meaning. If a rename seems to force a logic change, STOP and
  flag it — it belongs in a feature phase, not here.
- **Atomic, forward-only, solo.** Land the rename + all fixups as one coherent change set so
  imports never dangle. This repo has an **auto-commit hook** and a live co-editing caveat:
  do **not** run `git stash`/`git stash pop` (it has surfaced unrelated old stashes before).
- **No mocks / no fake fallbacks** (SPEC §2). Unavailable features stay absent from
  capability discovery and fail closed with a stable error code. Extend the existing
  `verify_release.py` no-mock scan; don't weaken it.
- **No new `verify_phaseN.py` gate scripts.** Acceptance is `python3 -m unittest discover`
  plus the single `scripts/verify_release.py`. Delete the per-phase gates as you migrate
  their assertions into `tests/<component>/`.
- **Keep `torch`/`trl` out of importable library modules** — they stay local to the training
  entrypoints (`scripts/train_grpo.py`), so the host suite imports without them.
- Match the surrounding code's style and idiom; read neighboring files first.

## Suggested sequence (re-run `discover -s tests` green after each)
1. Collapse the env chain → `server/environment.py` + `server/types.py` (compose, don't
   inherit); rename the public symbols; fix every importer. Preserve the load-bearing env
   internals the tests reach into (§2.2 of the spec: `_issue`, `_issue_acts_ceiling`,
   `expand_commands`, `_TimingDigest`, `_compiled`, `_decode`, …).
2. Move the remaining product modules into component dirs (incl. `dram/`, `mitigations/`,
   `sandbox/`, `profiles/`, `disturbance/`); delete `phase1_env.py`.
3. Migrate every test into `tests/<component>/` (fix cross-imports like
   `test_multiturn_rollout`'s `from test_reference_policy import _FakeDiscoveryEnv`).
4. Consolidate scripts (`build.py`, `fetch_sources.py`), delete `verify_phase*.py`, rename
   configs (`configs/dram/ddr4.py`), purge the cpp/manifest "phase" strings, **rebuild the
   worker**, re-run worker-gated tests.
5. Add the new release tests: determinism incl. mapper selection (§12.7, Task B),
   no-mock + capability-discovery assertions (§12.8, Task C), and the §12 acceptance index
   (Task E). Update docs (Task D). Archive the planning docs to `spec/history/`.

## Definition of done
All six gates in `spec/P29_RELEASE_READINESS.md` §7 hold: zero "phase" hits over the release
surface (the two greps return nothing), the public-API smoke resolves the new names and none
of the old ones, `unittest discover` matches the baseline meaning, `verify_release.py` is
green, `build.py` produces a working worker at the de-phased path, and determinism +
no-mock + capability-discovery are green. Report the before/after test count, the two
zero-hit greps, the API smoke, and anything in the tree that contradicted the spec.
