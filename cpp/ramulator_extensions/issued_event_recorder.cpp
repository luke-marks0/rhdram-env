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
    rhdram::IssuedEventSink::instance().set_decoder([ctrl](uint64_t linear) {
      Request req;
      req.addr = static_cast<Addr_t>(linear);
      req.intra_channel_addr = static_cast<Addr_t>(linear);
      ctrl->m_addr_mapper->apply(req);
      std::vector<int> vec(req.addr_vec.begin(), req.addr_vec.end());
      if (!vec.empty()) vec[0] = 0;  // single-channel: channel is always 0
      return vec;
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
