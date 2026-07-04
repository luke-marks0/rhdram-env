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

## CI Fixture Policy

`CIHammerFixturePolicy` is a deterministic test fixture for CI. It is not
advertised as an LLM policy; it exists to prove the HTTP adapter, rollouts, and
metrics work without a network model dependency.
