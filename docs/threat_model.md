# Threat Model

The environment is a simulation-only DRAM research target exposed as a
separate OpenEnv server, for example a hosted Space or a Docker service.
Policies may be untrusted clients. They must never interact with host physical
memory or host devices, whether the server runs locally or remotely.

## Trusted Components

- OpenEnv server ownership of reset, step, state, budgets, reward, and
  termination;
- simulator worker ownership of simulated memory state, issued events, and
  disturbance state;
- profile verifier ownership of admitted empirical profile packages;
- sandbox broker ownership of policy script tool calls.

## Untrusted Components

- policy-authored actions;
- policy-authored scripts;
- script stdout, logs, tags, and claimed success conditions.

## Required Controls

- all policy-visible actions go through `step()` or a brokered equivalent;
- clients receive only the public OpenEnv API and never receive direct handles
  to simulator internals, host files, sockets, devices, or process controls;
- reward and success are computed from trusted simulator state only;
- hidden target coordinates and profile internals are not exposed through
  errors, handles, traces, logs, or timing summaries;
- host physical-address APIs, host memory devices, KVM, huge-page discovery,
  cache-control attack utilities, RDMA, PCIe/GPU memory handles, network, host
  mounts, and general device access are denied;
- missing or unvalidated features fail closed and are not advertised.

## Denied Interfaces

- `/proc/pagemap`
- `/dev/mem`
- `/dev/kmem`
- `/dev/kvm`
- huge-page discovery
- cache-control attack utilities
- RDMA
- PCIe memory handles
- GPU memory handles
- host device access
- host mounts
- network from policy sandbox
- unsandboxed script execution

## Out Of Scope

Phase 0 does not implement the sandbox, Ramulator integration, disturbance
engine, profiles, tasks, rewards, or mitigations. Those features remain
unavailable until admitted by later phase gates.
