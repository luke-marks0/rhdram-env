# Second-standard worker config (P15): a real Ramulator HBM2 memory system driven
# by the External frontend. HBM2 has a genuinely different addressing structure —
# a PseudoChannel level, no Rank — so it proves the disturbance engine's geometry
# and standard-adapter machinery carries no DDR4 assumptions (defect F).
#
# No HBM2 empirical read-disturbance profile is admitted (the source is
# deferred_pending_license_and_hash in SOURCE_MANIFEST.yaml), so this config is used
# only for geometry/adapter/differential-decode checks, never for reward.
import ramulator


frontend = ramulator.frontend.External(clock_ratio=1)
dram = ramulator.dram.HBM2(
    org_preset="HBM2_8Gb",
    timing_preset="HBM2_2400Mbps",
)
ctrl = ramulator.controller.GenericDDR(
    dram=dram,
    scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
)
mem = ramulator.memory_system.GenericDRAM(
    clock_ratio=3,
    controllers=[ctrl],
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)

ramulator.Simulation(frontend, mem)
