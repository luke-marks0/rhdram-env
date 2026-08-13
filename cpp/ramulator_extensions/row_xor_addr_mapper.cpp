// RoBaRaCoChRowXOR — a real row->bank address-scrambling mapper (P24).
//
// Ramulator v2.1.0's stock mappers do NOT scatter linearly-adjacent rows across
// banks: RoBaRaCoCh keeps the bank a pure low-bit slice, and MOP4CLXOR XORs
// *column* bits into the bank index (cache-line interleaving for bank
// parallelism), so `addr + row_stride` stays in the SAME bank under both. That
// makes address->bank membership computable from the linear address, which
// defeats the DRAMA-style timing-discovery threat model this environment targets
// (spec/IMPLEMENTATION_PLAN_V3 P24; see the ADR in docs/adr/).
//
// This mapper implements the real, documented behaviour the plan actually needs:
// the undocumented controller bank-select XOR that Pessl et al. reverse-engineer
// in the DRAMA paper (USENIX Security 2016) and that AMD/Intel controllers apply.
// It decodes exactly like RoBaRaCoCh (Column at the LSB, then Rank..Row, Row the
// most-significant field) — so the linear row stride is byte-identical to
// RoBaRaCoCh and the publicly disclosed geometry (P21) stays honest — then XORs a
// slice of the Row bits into the Bank (and, with a seedable offset, BankGroup)
// index. The map stays a bijection (XOR by a function of Row is invertible given
// Row), so no two linear addresses collide. Because bit 0 of Row always folds
// into the Bank index, `addr` and `addr + row_stride` land in DIFFERENT banks, so
// physical adjacency is not computable from the numeric address alone and must be
// reverse-engineered via the bank-conflict timing channel. This is real address
// mapping, not fabricated disturbance physics (SPEC §2).
#include "ramulator/base/param.h"
#include "ramulator/controller/addr_mapper/addr_mapper_base.h"
#include "ramulator/controller/addr_mapper/i_addr_mapper.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/dram/dram_spec.h"

#include <stdexcept>

namespace Ramulator {

class RoBaRaCoChRowXOR : public IAddrMapper, public AddrMapperBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IAddrMapper, RoBaRaCoChRowXOR, AddrMapperBase, "RoBaRaCoChRowXOR")

  int m_xor_offset = 0;      // seedable per-episode secret: which Row bits feed the BankGroup XOR
  int m_bank_idx = -1;       // index into m_addr_bits (channel excluded), like m_row_idx
  int m_bankgroup_idx = -1;

  void init() override {
    AddrMapperBase::init();
    RAMULATOR_PARSE_PARAM(m_xor_offset, int, "xor_offset").default_val(0);
    if (m_xor_offset < 0) {
      throw std::runtime_error("RoBaRaCoChRowXOR: xor_offset must be non-negative");
    }
    const auto& spec = *m_ctrl->m_device.m_spec;
    m_bank_idx = spec.has_level("Bank") ? spec.get_level_id("Bank") - 1 : -1;
    m_bankgroup_idx = spec.has_level("BankGroup") ? spec.get_level_id("BankGroup") - 1 : -1;
    // The all-scattering property this mapper exists for (`addr` and
    // `addr + row_stride` land in different banks) comes from folding Row bit 0 into
    // Bank. On a geometry with no mapped Bank level that fold silently disappears and
    // the mapper degrades into stock RoBaRaCoCh, which would hand a discovery episode
    // a bank function the policy *can* compute from the address. Fail closed instead.
    if (m_bank_idx < 0 || m_addr_bits[m_bank_idx] <= 0) {
      throw std::runtime_error(
          "RoBaRaCoChRowXOR: requires a mapped Bank level with a non-zero width");
    }
    // `apply` shifts the Row value right by m_xor_offset. C++ leaves a shift at or
    // beyond the operand width undefined, and an offset past the Row field would make
    // the BankGroup scramble a silent no-op, so bound it by the real Row width.
    if (m_row_idx < 0 || m_addr_bits[m_row_idx] <= 0) {
      throw std::runtime_error("RoBaRaCoChRowXOR: requires a mapped Row level");
    }
    if (m_xor_offset >= m_addr_bits[m_row_idx]) {
      throw std::runtime_error(
          "RoBaRaCoChRowXOR: xor_offset must be smaller than the Row field width");
    }
  }

  void apply(Request& req) override {
    req.addr_vec.resize(m_num_mapped_levels + 1, -1);
    Addr_t addr = req.intra_channel_addr >> m_tx_offset;
    // RoBaRaCoCh base decode: Column at the LSB, then Rank..Row (Row is the MSB).
    req.addr_vec[m_col_idx + 1] = slice_lower_bits(addr, m_addr_bits[m_col_idx]);
    for (int i = 0; i <= m_row_idx; i++) {
      req.addr_vec[i + 1] = slice_lower_bits(addr, m_addr_bits[i]);
    }
    // Row -> Bank / BankGroup XOR. Bit 0 of Row always folds into Bank, so
    // `addr` and `addr + row_stride` (Row differs by 1) land in different banks.
    // Unsigned for the shift/mask arithmetic: `init` has already bounded m_xor_offset
    // by the Row width, and an unsigned operand keeps the shift well-defined rather
    // than relying on the sign of a sliced field.
    const unsigned row = static_cast<unsigned>(req.addr_vec[m_row_idx + 1]);
    {
      const unsigned mask = (1u << m_addr_bits[m_bank_idx]) - 1u;
      req.addr_vec[m_bank_idx + 1] ^= static_cast<int>(row & mask);
    }
    if (m_bankgroup_idx >= 0 && m_addr_bits[m_bankgroup_idx] > 0) {
      const unsigned mask = (1u << m_addr_bits[m_bankgroup_idx]) - 1u;
      req.addr_vec[m_bankgroup_idx + 1] ^= static_cast<int>((row >> m_xor_offset) & mask);
    }
  }
};

}  // namespace Ramulator
