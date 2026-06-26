# Repository skeleton

```text
rowhammer-openenv/
  README.md
  pyproject.toml
  CMakeLists.txt
  SOURCE_MANIFEST.yaml
  third_party/
    ramulator2/                 # pinned submodule or vendored source reference
  cpp/
    protocol/                   # worker RPC IDL and generated bindings
    simulator_service/          # per-episode Ramulator worker process
    ramulator_extensions/       # issued-event hooks, refresh hooks, mitigation adapters
    memory/                     # sparse functional memory overlay
    disturbance/                # exposure accounting and flip engine
    mitigations/                # real admitted mitigation ports only
  rowhammer_env/
    server/                     # OpenEnv reset/step/state implementation
    tools/                      # dram.info/read/write/issue, script.run, episode.finish
    tasks/                      # task compiler and disclosure projection
    rewards/                    # trusted success predicates
    profiles/                   # profile loader/verifier
    sandbox/                    # policy script runtime and broker
    security/                   # denylist, attestation, information-flow checks
    observability/              # metrics and trace summaries
  sdk/
    rh_sdk/                     # policy-facing Python SDK for scripts
  profile_builder/
    ingest/                     # source-specific parsers
    canonicalize/               # canonical tables and hash generation
    fit/                        # fitted empirical models
    validate/                   # held-out statistical tests
    model_cards/                # generated profile documentation
  configs/
    dram/                       # Ramulator configs
    tasks/                      # task YAMLs
    mitigations/                # mitigation configs
  schemas/
    action.schema.json
    observation.schema.json
    task.schema.json
  examples/
    action.issue.json
    observation.reset.json
    task.known_target.yaml
  tests/
    contract/
    ramulator/
    memory/
    disturbance/
    profiles/
    tasks/
    sandbox/
    security/
    mitigations/
    performance/
    release/
  docs/
    api.md
    threat_model.md
    profile_modeling.md
    operations.md
```
