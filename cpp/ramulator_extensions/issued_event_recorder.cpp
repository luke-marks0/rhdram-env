// IssuedEventRecorder — a Ramulator controller plugin that captures every
// post-schedule DRAM command into the process-global `IssuedEventSink`.
//
// Modeled on `controller/plugin/impl/cmd_trace_recorder.cpp`, but instead of
// writing a per-channel trace file it feeds an in-memory sink that the worker
// drains after each request (keeps per-episode isolation; see the header).
//
// Registered under the name "IssuedEventRecorder"; add it to a controller's
// `controller_plugins` list in the worker YAML to enable capture. The plugin is
// linked into the worker binary (not `libramulator.so`); its registration still
// lands in the shared factory registry because `Factory::register_implementation`
// is defined in the library and mutates the library's registry.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ramulator/base/base.h"
#include "ramulator/base/request.h"
#include "ramulator/base/type.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/plugin/i_controller_plugin.h"
#include "ramulator/dram/dram_spec.h"

#include "issued_event_recorder.h"

namespace Ramulator {

// Inverts the controller's *active* address mapper: decoded coordinates -> the
// linear address that decodes back to them.
//
// `IAddrMapper` declares only the forward `apply()`, and the mappers in scope
// (`RoBaRaCoCh`, `MOP4CLXOR`, the authored `RoBaRaCoChRowXOR`) all build
// `addr_vec` from bit slices and XORs of the address bits — i.e. each output bit
// is an affine function over GF(2) of the input bits. So the forward map is
// recovered exactly by probing it on 0 and on each single address bit, and
// inverted by solving the resulting linear system.
//
// The affinity assumption is never trusted: `encode()` decodes its own answer
// through the same mapper and reports `NotInvertible` unless every coordinate
// matches. A mapper this construction cannot invert therefore fails closed
// rather than returning a plausible-looking wrong address.
class AddrInverter {
 public:
  AddrInverter(std::vector<int> level_bits, int tx_offset,
               std::function<std::vector<int>(uint64_t)> decode)
      : m_level_bits(std::move(level_bits)), m_tx_offset(tx_offset), m_decode(std::move(decode)) {
    for (int bits : m_level_bits) m_addr_bits += bits;
    // 64 address bits is far beyond any DRAMSpec organization (DDR4 uses 27);
    // the solver packs a coordinate vector into one word, so refuse rather than
    // silently truncate.
    if (m_addr_bits <= 0 || m_addr_bits >= 64) return;

    m_constant = pack(m_decode(0));
    for (int j = 0; j < m_addr_bits; j++) {
      const uint64_t probe = static_cast<uint64_t>(1) << j;
      insert(pack(m_decode(probe << m_tx_offset)) ^ m_constant, probe);
    }
    m_ready = true;
  }

  rhdram::EncodeStatus encode(const std::vector<int>& coords, uint64_t& linear) const {
    if (!m_ready) return rhdram::EncodeStatus::NotInvertible;
    if (coords.size() != m_level_bits.size()) return rhdram::EncodeStatus::OutOfRange;
    for (size_t i = 0; i < coords.size(); i++) {
      const int64_t limit = static_cast<int64_t>(1) << m_level_bits[i];
      if (coords[i] < 0 || coords[i] >= limit) return rhdram::EncodeStatus::OutOfRange;
    }

    uint64_t address_bits = 0;
    if (!solve(pack(coords) ^ m_constant, address_bits)) return rhdram::EncodeStatus::NotInvertible;
    const uint64_t candidate = address_bits << m_tx_offset;

    const std::vector<int> round_trip = m_decode(candidate);
    if (round_trip.size() != coords.size()) return rhdram::EncodeStatus::NotInvertible;
    for (size_t i = 0; i < coords.size(); i++) {
      if (round_trip[i] != coords[i]) return rhdram::EncodeStatus::NotInvertible;
    }
    linear = candidate;
    return rhdram::EncodeStatus::Ok;
  }

 private:
  // One row of the GF(2) basis: a reduced coordinate vector and the combination
  // of address bits that produces it.
  struct BasisRow {
    uint64_t vector = 0;
    uint64_t address_bits = 0;
    bool used = false;
  };

  std::vector<int> m_level_bits;  // field width per DRAMSpec level (channel included, width 0)
  int m_tx_offset = 0;
  std::function<std::vector<int>(uint64_t)> m_decode;
  int m_addr_bits = 0;
  uint64_t m_constant = 0;  // the decode of address 0
  BasisRow m_basis[64];
  bool m_ready = false;

  // Concatenate the per-level coordinates into one bit vector, each masked to its
  // own field width. Widths sum to the mapped address width, so the packing is
  // lossless for any in-range coordinate vector.
  uint64_t pack(const std::vector<int>& coords) const {
    uint64_t packed = 0;
    int shift = 0;
    for (size_t i = 0; i < m_level_bits.size(); i++) {
      const int bits = m_level_bits[i];
      if (bits > 0 && i < coords.size()) {
        const uint64_t mask = (static_cast<uint64_t>(1) << bits) - 1;
        packed |= (static_cast<uint64_t>(coords[i]) & mask) << shift;
      }
      shift += bits;
    }
    return packed;
  }

  void insert(uint64_t vector, uint64_t address_bits) {
    for (int p = 63; p >= 0; p--) {
      if (!((vector >> p) & 1)) continue;
      if (m_basis[p].used) {
        vector ^= m_basis[p].vector;
        address_bits ^= m_basis[p].address_bits;
        continue;
      }
      m_basis[p] = {vector, address_bits, true};
      return;
    }
  }

  bool solve(uint64_t target, uint64_t& address_bits) const {
    uint64_t remainder = target;
    address_bits = 0;
    for (int p = 63; p >= 0; p--) {
      if (!((remainder >> p) & 1)) continue;
      if (!m_basis[p].used) return false;
      remainder ^= m_basis[p].vector;
      address_bits ^= m_basis[p].address_bits;
    }
    return remainder == 0;
  }
};

class IssuedEventRecorder : public IControllerPlugin, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IControllerPlugin, IssuedEventRecorder, "IssuedEventRecorder")

 public:
  void init() override {}

  void setup(IFrontEnd* /*frontend*/, IMemorySystem* /*memory_system*/) override {
    m_ctrl = cast_parent<ControllerBase>();
    const auto& spec = *m_ctrl->m_device.m_spec;

    m_level_count = spec.level_count;
    m_bank_idx = spec.has_level("Bank") ? spec.get_level_id("Bank") : -1;
    m_bankgroup_idx = spec.has_level("BankGroup") ? spec.get_level_id("BankGroup") : -1;
    m_row_idx = spec.has_level("Row") ? spec.get_level_id("Row") : -1;
    m_num_banks = (m_bank_idx >= 0) ? spec.organization.level_sizes[m_bank_idx] : 1;
    m_command_meta = spec.command_meta;
    m_bank_targets = spec.bank_targets;
    m_command_names = spec.command_names;

    // Publish geometry once so the worker can name decoded coordinates and
    // Python can reconstruct the RoBaRaCoCh address mapping.
    rhdram::Geometry g;
    g.valid = true;
    g.standard = spec.standard_name;
    g.tx_bytes = spec.get_tx_bytes();
    g.prefetch = spec.internal_prefetch_size;
    g.channel_width = spec.channel_width;
    g.level_names = spec.level_names;
    g.level_sizes = spec.organization.level_sizes;
    g.command_names = spec.command_names;
    rhdram::IssuedEventSink::instance().set_geometry(g);

    // Publish a side-effect-free decoder over the controller's *active* mapper so
    // the worker's DECODE op (P24) reports true coordinates for any mapper without
    // re-deriving the bit function in Python. The admitted config is single-channel
    // (CacheLineInterleave collapses to channel 0), so the intra-channel address is
    // the linear address itself; the mapper fills addr_vec[1..N] from it.
    ControllerBase* ctrl = m_ctrl;
    const int level_count = m_level_count;
    std::function<std::vector<int>(uint64_t)> decode = [ctrl, level_count](uint64_t linear) {
      Request req;
      req.addr = static_cast<Addr_t>(linear);
      req.intra_channel_addr = static_cast<Addr_t>(linear);
      ctrl->m_addr_mapper->apply(req);
      std::vector<int> vec(req.addr_vec.begin(), req.addr_vec.end());
      vec.resize(level_count, 0);
      if (!vec.empty()) vec[0] = 0;  // single-channel: channel is always 0
      return vec;
    };
    rhdram::IssuedEventSink::instance().set_decoder(decode);

    // Publish the decoder's inverse for the worker's ENCODE op. Field widths come
    // from the same DRAMSpec organization the mapper itself slices (Column loses
    // the prefetch bits, which live below the transaction offset), and Channel is
    // width 0 because the admitted config is single-channel.
    std::vector<int> level_bits(level_count, 0);
    for (int i = 1; i < level_count; i++) {
      level_bits[i] = calc_log2(spec.organization.level_sizes[i]);
    }
    if (level_count > 1) level_bits[level_count - 1] -= calc_log2(spec.internal_prefetch_size);
    auto inverter = std::make_shared<AddrInverter>(std::move(level_bits), calc_log2(spec.get_tx_bytes()),
                                                   std::move(decode));
    rhdram::IssuedEventSink::instance().set_encoder(
        [inverter](const std::vector<int>& coords, uint64_t& linear) {
          return inverter->encode(coords, linear);
        });
  }

  void on_issue(const Request& req) override {
    const int cmd = req.command;
    const bool is_opening = m_command_meta[cmd].is_opening;
    const bool is_closing = m_command_meta[cmd].is_closing;
    const bool is_accessing = m_command_meta[cmd].is_accessing;
    const bool is_refreshing = m_command_meta[cmd].is_refreshing;
    const bool all_banks = (cmd < static_cast<int>(m_bank_targets.size())) &&
                           m_bank_targets[cmd] == BankTarget::All;

    const int flat_bank = flat_bank_id(req.addr_vec);
    const int row = (m_row_idx >= 0 && m_row_idx < static_cast<int>(req.addr_vec.size()))
                        ? req.addr_vec[m_row_idx]
                        : -1;

    // Row-buffer state: an accessing command is a row hit iff it targets the
    // row currently open in its bank.
    bool row_hit = false;
    if (is_accessing) {
      auto it = m_open_row.find(flat_bank);
      row_hit = (it != m_open_row.end() && row >= 0 && it->second == row);
    }

    // Track open rows so `row_hit` is meaningful and so refresh/precharge close
    // rows the way the device does.
    if (is_opening) {
      m_open_row[flat_bank] = row;
    } else if (is_closing || is_refreshing) {
      if (all_banks) {
        m_open_row.clear();
      } else {
        m_open_row.erase(flat_bank);
      }
    }
    if (is_accessing && is_auto_precharge(cmd)) {
      m_open_row.erase(flat_bank);
    }

    rhdram::IssuedEvent ev;
    ev.clk = static_cast<int64_t>(m_ctrl->m_clk);
    ev.command = m_command_names[cmd];
    ev.addr_vec.assign(req.addr_vec.begin(),
                       req.addr_vec.begin() + std::min<int>(m_level_count, req.addr_vec.size()));
    ev.type_id = req.type_id;
    ev.row_hit = row_hit;
    rhdram::IssuedEventSink::instance().push(std::move(ev));
  }

 private:
  ControllerBase* m_ctrl = nullptr;
  int m_level_count = 0;
  int m_bank_idx = -1;
  int m_bankgroup_idx = -1;
  int m_row_idx = -1;
  int m_num_banks = 1;
  std::vector<DRAMCommandMeta> m_command_meta;
  std::vector<BankTarget> m_bank_targets;
  std::vector<std::string> m_command_names;
  std::unordered_map<int, int> m_open_row;  // flat bank id -> open row (or absent)

  int flat_bank_id(const AddrVec_t& av) const {
    int bank = (m_bank_idx >= 0 && m_bank_idx < static_cast<int>(av.size())) ? av[m_bank_idx] : 0;
    int bg = (m_bankgroup_idx >= 0 && m_bankgroup_idx < static_cast<int>(av.size())) ? av[m_bankgroup_idx] : 0;
    if (bank < 0) bank = 0;
    if (bg < 0) bg = 0;
    return bg * m_num_banks + bank;
  }

  bool is_auto_precharge(int cmd) const {
    const std::string& name = m_command_names[cmd];
    return name == "RDA" || name == "WRA";
  }
};

}  // namespace Ramulator
