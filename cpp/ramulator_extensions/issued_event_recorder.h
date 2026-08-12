// Issued-event capture for the RowHammer simulator worker (SPEC §5.2).
//
// The disturbance model (SPEC §4/§5) must be driven by *actual post-schedule
// DRAM commands* — ACT/PRE/RD/WR/REF/RFM — not by frontend read/write
// completions that ignore row-buffer locality. Ramulator surfaces every issued
// command through the controller-plugin `on_issue(const Request&)` hook, which
// fires immediately after `m_device.issue_command(...)` (see
// `generic_ddr_controller.cpp`). The `IssuedEventRecorder` plugin captures those
// commands here; the worker drains this sink after each request and emits the
// records in its JSON `events[]`.
//
// This header is compiled into the worker binary together with the plugin
// (`issued_event_recorder.cpp`). The worker process owns exactly one Ramulator
// instance per episode, so a single process-global sink gives per-episode
// isolation for free. Both translation units live in the same executable, so the
// function-local `static` below is a single shared instance (no cross-DSO
// registry sharing is required for the sink itself).
#ifndef RHDRAM_ISSUED_EVENT_RECORDER_H
#define RHDRAM_ISSUED_EVENT_RECORDER_H

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace rhdram {

// Outcome of an ENCODE (decoded coordinates -> linear address).
enum class EncodeStatus {
  Ok,
  OutOfRange,      // a coordinate does not fit its level's field width
  NotInvertible,   // the active mapper has no linear preimage for these coordinates
};

// One post-schedule DRAM command with its decoded coordinates.
struct IssuedEvent {
  int64_t clk = 0;              // controller clock when the command was issued
  std::string command;          // command name, e.g. "ACT", "RD", "REFab"
  std::vector<int> addr_vec;    // decoded coordinates, in DRAMSpec level order
  int type_id = -1;             // request type id (-1 for internal/maintenance)
  bool row_hit = false;         // accessing command that hit an already-open row
};

// DRAM geometry published once by the plugin at setup time, so the worker can
// name the decoded coordinates and Python can derive the address mapping.
struct Geometry {
  bool valid = false;
  std::string standard;
  int tx_bytes = 0;
  int prefetch = 0;
  int channel_width = 0;
  std::vector<std::string> level_names;
  std::vector<int> level_sizes;
  std::vector<std::string> command_names;
};

// Process-global, per-episode sink. Single-threaded (Ramulator ticks on one
// thread), so no locking is required.
class IssuedEventSink {
 public:
  static IssuedEventSink& instance() {
    static IssuedEventSink sink;
    return sink;
  }

  // First controller/channel to publish wins; identical geometry from other
  // channels is redundant.
  void set_geometry(const Geometry& g) {
    if (!m_geometry.valid) m_geometry = g;
  }
  const Geometry& geometry() const { return m_geometry; }

  // A side-effect-free decode of a linear (intra-channel) address into decoded
  // coordinates under the *active* mapper. Published by the recorder plugin at
  // setup (it holds the controller, hence the constructed addr_mapper), so the
  // worker's DECODE op reuses the real mapper instead of duplicating its logic in
  // Python — which would drift for any mapper other than RoBaRaCoCh (P24). First
  // controller to publish wins, mirroring the geometry rule.
  void set_decoder(std::function<std::vector<int>(uint64_t)> decoder) {
    if (!m_decoder) m_decoder = std::move(decoder);
  }
  bool has_decoder() const { return static_cast<bool>(m_decoder); }
  std::vector<int> decode(uint64_t linear) const {
    return m_decoder ? m_decoder(linear) : std::vector<int>{};
  }

  // The exact inverse of the decoder: decoded coordinates (in DRAMSpec level
  // order, one per level) -> the linear address that decodes back to them.
  // Published by the recorder plugin at setup alongside the decoder, and used by
  // the worker's ENCODE op so the disturbance model can resolve a victim row's
  // own column-0 address under *any* mapper instead of assuming the RoBaRaCoCh
  // arithmetic (`aggressor ± d * row_stride`), which lands in the wrong bank
  // under a row->bank scrambling mapper. First controller to publish wins,
  // mirroring the geometry rule.
  void set_encoder(std::function<EncodeStatus(const std::vector<int>&, uint64_t&)> encoder) {
    if (!m_encoder) m_encoder = std::move(encoder);
  }
  bool has_encoder() const { return static_cast<bool>(m_encoder); }
  EncodeStatus encode(const std::vector<int>& coords, uint64_t& linear) const {
    return m_encoder ? m_encoder(coords, linear) : EncodeStatus::NotInvertible;
  }

  void push(IssuedEvent&& ev) { m_events.push_back(std::move(ev)); }

  // Return everything captured since the last drain and reset the buffer.
  std::vector<IssuedEvent> drain() {
    std::vector<IssuedEvent> out;
    out.swap(m_events);
    return out;
  }

  void clear() { m_events.clear(); }

 private:
  IssuedEventSink() = default;
  Geometry m_geometry;
  std::vector<IssuedEvent> m_events;
  std::function<std::vector<int>(uint64_t)> m_decoder;
  std::function<EncodeStatus(const std::vector<int>&, uint64_t&)> m_encoder;
};

}  // namespace rhdram

#endif  // RHDRAM_ISSUED_EVENT_RECORDER_H
