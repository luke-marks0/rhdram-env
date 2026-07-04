# ADR 0003: OS-isolated script sandbox for `script.run`

## Status

Accepted for P18.

## Context

The previous `script.run` path evaluated a tiny AST subset in the environment
process. That was useful for early broker semantics, but it did not satisfy the
SPEC sandbox requirements: policy-authored code must not share the environment
process, host filesystem, devices, or network.

The local approved runtime options are:

- `bwrap`: present and supports user, mount, PID, IPC, and UTS isolation.
- `unshare`: present and can create a network namespace for the sandbox user.
- `nsjail`, `firejail`, and Python seccomp bindings: not installed.

`bwrap --unshare-net` cannot configure loopback in this environment because the
required netlink operation is denied. `unshare --net --user --map-root-user`
does work, so the selected runtime composes both tools.

## Decision

`script.run` uses a subprocess runtime:

1. `unshare --net --user --map-root-user` creates an isolated user and network
   namespace.
2. `bwrap` creates a read-only Python runtime view, a private `/tmp`, a fresh
   `/proc`, a minimal `/dev`, and PID/IPC/UTS namespaces.
3. The child Python process runs with `-I -S`, a cleared environment, rlimits
   for CPU, address space, files, file size, process count, and a parent-side
   wall-time timeout.
4. The child installs only an in-memory `rh_sdk` module. Calls to `rh` are sent
   to the parent over JSON-line stdin/stdout IPC; the parent alone owns the
   environment and dispatches the normal tools.
5. A small AST policy in the child is retained as defense-in-depth to reject
   imports, dunder escapes, dynamic calls, and non-broker APIs before execution.

There is no permissive fallback. If the runtime is missing or fails attestation,
`script.run` fails closed with `UNAVAILABLE_CAPABILITY`.

## Consequences

- Policy code no longer executes in the environment process.
- The sandbox cannot read repository files, `/proc/pagemap`, `/dev/mem`, or
  `/dev/kvm`, and it has no usable host network path.
- Trace semantics are unchanged because every `rh_sdk` call becomes the same
  `Phase2Action` tool call used by direct policies.
- VM-level isolation such as Kata or Firecracker remains unnecessary for this
  phase, but can replace the runtime later behind the same IPC broker.
