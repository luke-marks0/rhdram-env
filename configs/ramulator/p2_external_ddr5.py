# Second-standard worker config (P15): a real Ramulator DDR5 memory system driven
# by the External frontend, used to prove the disturbance engine is standard-generic
# (its geometry comes from the real DDR5 DRAMSpec, not DDR4 constants). No DDR5
# empirical profile is admitted, so this config is used only for geometry/adapter
# checks, not for reward-bearing disturbance.
import ramulator


frontend = ramulator.frontend.External(clock_ratio=1)
dram = ramulator.dram.DDR5(
    org_preset="DDR5_16Gb_x8",
    timing_preset="DDR5_4800AN",
    rank=1,
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
