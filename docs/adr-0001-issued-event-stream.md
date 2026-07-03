# ADR 0001 — Real issued-event stream via a Ramulator plugin (P11)

Status: Accepted (Phase 11)
Resolves: defect A (disturbance driven by frontend completions, not issued
commands) and, by the design decision below, defect D (disturbance engine in
Python rather than the `cpp/disturbance/` tree the SPEC §11 skeleton implies).

## Context

SPEC §4/§5 require read-disturbance to be driven by the **actual post-schedule
DRAM commands** (ACT/PRE/RD/WR/REF/RFM) with decoded coordinates and row-buffer
state — not by frontend read/write completions, which ignore row-buffer
locality (N reads to one open row are a single ACT, not N hammers). Before P11:

- `cpp/simulator_service/ramulator_worker.cpp` emitted one synthetic frontend
  `RD`/`WR` event per request, with no ACT/PRE/REF and no decoded coordinates.
- `rowhammer_env/disturbance.py` accumulated exposure on every `RD`, so it could
  not tell a row hit from a row activation and used a fictitious `row_bytes=8192`
  (the row-buffer *size*) as if it were the logical stride between rows.

Ramulator 2.1 already exposes every issued command through the controller-plugin
`on_issue(const Request&)` hook (fired immediately after
`m_device.issue_command(...)` in `generic_ddr_controller.cpp`), carrying the
command id and the fully decoded `addr_vec`. `CmdTraceRecorder`,
`BinTraceRecorder`, and `LiveTraceStreamer` are worked examples of consuming it.

## Decision

Keep the fitted statistical flip model in **Python** (where the empirical profile
lives) and feed it a **real C++ issued-event stream**, rather than porting the
disturbance engine into `cpp/disturbance/`.

- A new controller plugin, `cpp/ramulator_extensions/issued_event_recorder.*`,
  captures each issued command (`clk`, command name, decoded `addr_vec`,
  `type_id`, and a row-buffer `row_hit` flag) into a process-global in-memory
  sink. The worker process is one episode, so the global sink is per-episode
  isolated; the worker drains it after each request. The plugin publishes DRAM
  geometry (level names/sizes, `tx_bytes`, prefetch) once, so coordinates can be
  named and the RoBaRaCoCh row stride derived from the real `DRAMSpec`.
- The plugin is compiled into the worker binary (not `libramulator.so`) and
  injected into the worker YAML as a `controller_plugins` entry. Its
  registration still lands in the shared factory registry because
  `Factory::register_implementation` is defined in the library and mutates the
  library's registry.
- `ramulator_worker.cpp` now emits, per response, a `request` block (the frontend
  op/addr/size) and an `events[]` array of the real issued commands, plus `acts`
  and `refreshes` public counters and an `INFO` command for geometry.
- `DisturbanceEngine.consume()` accumulates exposure from **ACT** events keyed by
  decoded `(channel, rank, bankgroup, bank, row)`; WRITE restores the cells it
  overwrote; REF/RFM restoration and decay are stubbed for P14. Flips remain
  keyed by linear (logical) address — since RoBaRaCoCh makes Row the most
  significant field, a victim's linear address is the aggressor's request
  address offset by one true row stride, which decodes back to the adjacent
  physical row in the same bank.

## Consequences

- Row-buffer locality is now respected: reads to an open row issue no ACT and
  cause no disturbance; only row conflicts/misses hammer. `verify_phase11.py`
  proves that the *same* read budget flips a victim when interleaved (many ACTs)
  and does not when blocked (~1 ACT/row).
- Rejected/illegal commands never tick Ramulator, so they emit no events and
  cannot change exposure (TEST_PLAN D3).
- The `cpp/disturbance/` C++ port remains a valid future alternative but is out
  of scope; this ADR records why Python + a real event stream was chosen instead.
- A full bidirectional physical/opaque address projection (using the geometry the
  plugin now exposes) is deferred to P12; P11 adds only the narrow row-stride
  derivation it needs.
