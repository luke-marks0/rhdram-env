# RowHammer OpenEnv Policy API

Policies connect over the OpenEnv websocket client and receive
`Phase2Observation` objects. Reward is sparse and trusted: a policy cannot earn
reward by declaring success; `episode.finish` returns `0` until the simulator and
disturbance engine have produced the task condition.

## Tool Surface

- `dram.info`: disclosed address forms, commands, mitigations, task metadata,
  and remaining public budget.
- `dram.read`: read bytes from a disclosed logical, physical, or handle address.
- `dram.write`: write base64 bytes to a disclosed address.
- `dram.issue`: issue `RD`, `WR`, or `WAIT` commands through the Ramulator worker.
- `script.run`: run `python-rh-sdk` code in the P18 OS sandbox; `rh` calls are
  brokered back to the same tools.
- `episode.finish`: terminate the episode and receive reward from trusted state.

`rowhammer_env.llm.TOOL_SCHEMAS` exposes this surface as function/tool schemas
for tool-calling chat backends. `OpenAICompatibleToolPolicy` can call any
OpenAI-compatible chat-completions endpoint configured with
`RHD_LLM_CHAT_COMPLETIONS_URL`, `RHD_LLM_MODEL`, and optionally
`RHD_LLM_API_KEY`.

## Discovery families and the secret address mapping

Discovery families (`bounded_sweep`, and the Tier 2b numeric-address family) hide
which candidate rows are same-bank physical neighbours of the victim. For these,
the DRAM **address→bank mapping is a per-episode secret**: the worker runs the
authored `RoBaRaCoChRowXOR` mapper (RoBaRaCoCh decode plus a Row→Bank XOR with a
seedable offset), so `victim ± row_stride` lands in a *different* bank and
adjacency cannot be computed from a numeric address — it must be reverse-engineered
through the bank-conflict timing channel (`public_counters.acts` /
`last_action.cycle_delta` / the `timing_digest`, disclosed under `full_trace`). The
mapper identity/offset is never disclosed in any observation, error, handle name,
or digest. The disclosed geometry (`dram.info.geometry`) is unchanged — the row
stride is identical to the public mapper. Success is scored on the trusted decoded
victim flip, never on a policy read or claim. See
`docs/adr-0004-secret-address-mapping.md`.

The server-internal `DECODE` request (true coordinates under the active mapper) is
used only by the task compiler to build candidate sets; it is **not** in the policy
tool surface and cannot be reached through `step`.

## CI Fixture Policy

`CIHammerFixturePolicy` is a deterministic test fixture for CI. It is not
advertised as an LLM policy; it exists to prove the HTTP adapter, rollouts, and
metrics work without a network model dependency.
