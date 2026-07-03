#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

#include "issued_event_recorder.h"

namespace {

using ByteMap = std::unordered_map<Ramulator::Addr_t, unsigned char>;

std::string escape(const std::string& value) {
  std::string out;
  for (char ch : value) {
    if (ch == '"' || ch == '\\') out.push_back('\\');
    out.push_back(ch);
  }
  return out;
}

std::string lower(const std::string& value) {
  std::string out = value;
  for (char& ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return out;
}

std::string error_json(const std::string& id, const std::string& code, const std::string& msg) {
  return "{\"ok\":false,\"id\":\"" + escape(id) + "\",\"error\":{\"code\":\"" + code +
         "\",\"message\":\"" + escape(msg) + "\"}}";
}

std::string str_array(const std::vector<std::string>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out += ',';
    out += '"';
    out += escape(values[i]);
    out += '"';
  }
  return out + ']';
}

std::string int_array(const std::vector<int>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out += ',';
    out += std::to_string(values[i]);
  }
  return out + ']';
}

std::vector<std::string> split(const std::string& line) {
  std::istringstream in(line);
  std::vector<std::string> out;
  for (std::string token; in >> token;) out.push_back(token);
  return out;
}

bool parse_u64(const std::string& text, Ramulator::Addr_t& value) {
  char* end = nullptr;
  value = std::strtoull(text.c_str(), &end, 0);
  return end && *end == '\0';
}

int hex_digit(char ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'a' && ch <= 'f') return 10 + ch - 'a';
  if (ch >= 'A' && ch <= 'F') return 10 + ch - 'A';
  return -1;
}

bool parse_hex(const std::string& hex, std::vector<unsigned char>& bytes) {
  if (hex.size() % 2 != 0) return false;
  bytes.clear();
  for (size_t i = 0; i < hex.size(); i += 2) {
    int hi = hex_digit(hex[i]);
    int lo = hex_digit(hex[i + 1]);
    if (hi < 0 || lo < 0) return false;
    bytes.push_back(static_cast<unsigned char>((hi << 4) | lo));
  }
  return true;
}

std::string to_hex(const std::vector<unsigned char>& bytes) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (auto b : bytes) out << std::setw(2) << static_cast<int>(b);
  return out.str();
}

class Worker {
 public:
  explicit Worker(const std::string& config_path) {
    auto config = Ramulator::Config::parse_config_file(config_path);
    frontend_.reset(Ramulator::Factory::create_frontend(config));
    memory_.reset(Ramulator::Factory::create_memory_system(config));
    frontend_->connect_memory_system(memory_.get());
    memory_->connect_frontend(frontend_.get());
    tx_bytes_ = memory_->get_tx_bytes();
    // The IssuedEventRecorder plugin publishes geometry during setup() (called
    // from the connect_* calls above). Cache it and start every episode with an
    // empty issued-event buffer.
    geometry_ = rhdram::IssuedEventSink::instance().geometry();
    rhdram::IssuedEventSink::instance().clear();
  }

  std::string handle(const std::string& line) {
    auto t = split(line);
    if (t.empty()) return error_json("", "BAD_SCHEMA", "empty request");
    if (t[0] == "QUIT") return "{\"ok\":true,\"quit\":true}";
    if (t.size() < 2) return error_json("", "BAD_SCHEMA", "missing request id");
    const std::string& id = t[1];

    if (t[0] == "INFO") return info(id);
    if (t[0] == "READ") return read(id, t);
    if (t[0] == "WRITE") return write(id, t, false);
    if (t[0] == "ISSUE") return issue(id, t);
    return error_json(id, "BAD_SCHEMA", "unknown request type");
  }

 private:
  std::unique_ptr<Ramulator::IFrontEnd> frontend_;
  std::unique_ptr<Ramulator::IMemorySystem> memory_;
  ByteMap bytes_;
  rhdram::Geometry geometry_;
  long cycle_ = 0;
  int tx_bytes_ = 0;
  long reads_ = 0;
  long writes_ = 0;
  long acts_ = 0;
  long refreshes_ = 0;

  std::string info(const std::string& id) {
    if (!geometry_.valid) {
      return error_json(id, "UNAVAILABLE_CAPABILITY", "issued-event geometry not published");
    }
    std::ostringstream out;
    out << "{\"ok\":true,\"id\":\"" << escape(id) << "\",\"geometry\":{"
        << "\"standard\":\"" << escape(geometry_.standard) << "\""
        << ",\"tx_bytes\":" << geometry_.tx_bytes
        << ",\"prefetch\":" << geometry_.prefetch
        << ",\"channel_width\":" << geometry_.channel_width
        << ",\"level_names\":" << str_array(geometry_.level_names)
        << ",\"level_sizes\":" << int_array(geometry_.level_sizes)
        << "}}";
    return out.str();
  }

  std::string read(const std::string& id, const std::vector<std::string>& t) {
    if (t.size() != 4) return error_json(id, "BAD_SCHEMA", "READ id addr length");
    Ramulator::Addr_t addr = 0;
    Ramulator::Addr_t len = 0;
    if (!parse_u64(t[2], addr) || !parse_u64(t[3], len) || len == 0 || len > static_cast<unsigned>(tx_bytes_)) {
      return error_json(id, "BAD_SCHEMA", "invalid read address or length");
    }
    long before = cycle_;
    auto result = complete_request(Ramulator::Request::Type::Read, addr, static_cast<int>(len));
    if (!result.ok) return error_json(id, result.code, result.message);
    reads_++;

    std::vector<unsigned char> out;
    for (Ramulator::Addr_t i = 0; i < len; ++i) out.push_back(bytes_[addr + i]);
    return ok_json(id, "RD", addr, len, before, to_hex(out));
  }

  std::string write(const std::string& id, const std::vector<std::string>& t, bool issued) {
    if (t.size() != 4) return error_json(id, "BAD_SCHEMA", issued ? "ISSUE id WR addr hex" : "WRITE id addr hex");
    Ramulator::Addr_t addr = 0;
    std::vector<unsigned char> data;
    if (!parse_u64(t[2], addr) || !parse_hex(t[3], data) || data.empty() || data.size() > static_cast<size_t>(tx_bytes_)) {
      return error_json(id, "BAD_SCHEMA", "invalid write address or hex data");
    }
    long before = cycle_;
    auto result = complete_request(Ramulator::Request::Type::Write, addr, static_cast<int>(data.size()));
    if (!result.ok) return error_json(id, result.code, result.message);
    for (size_t i = 0; i < data.size(); ++i) bytes_[addr + i] = data[i];
    writes_++;
    return ok_json(id, "WR", addr, data.size(), before, "");
  }

  std::string issue(const std::string& id, const std::vector<std::string>& t) {
    if (t.size() < 3) return error_json(id, "BAD_SCHEMA", "ISSUE id op ...");
    const std::string& op = t[2];
    if (op == "WAIT") {
      if (t.size() != 4) return error_json(id, "BAD_SCHEMA", "ISSUE id WAIT cycles");
      Ramulator::Addr_t cycles = 0;
      if (!parse_u64(t[3], cycles)) return error_json(id, "BAD_SCHEMA", "invalid wait cycles");
      long before = cycle_;
      for (Ramulator::Addr_t i = 0; i < cycles; ++i) tick();
      return ok_json(id, "WAIT", 0, cycles, before, "");
    }
    if (op == "RD") {
      std::vector<std::string> read_req{"READ", id, t.size() > 3 ? t[3] : "", std::to_string(tx_bytes_)};
      return read(id, read_req);
    }
    if (op == "WR") {
      std::vector<std::string> write_req{"WRITE", id, t.size() > 3 ? t[3] : "", t.size() > 4 ? t[4] : ""};
      return write(id, write_req, true);
    }
    // Commands beyond the admitted set are rejected before ticking Ramulator, so
    // they produce no issued events and cannot change disturbance state (D3).
    return error_json(id, "ILLEGAL_COMMAND", op + " is not admitted in Phase 2");
  }

  struct RequestResult {
    bool ok = false;
    std::string code;
    std::string message;
    Ramulator::Clk_t depart = -1;
  };

  RequestResult complete_request(int type, Ramulator::Addr_t addr, int size) {
    bool done = false;
    Ramulator::Clk_t depart = -1;
    bool accepted = frontend_->receive_external_requests(type, addr, 0, [&](Ramulator::Request& req) {
      done = true;
      depart = req.depart;
    }, size);
    if (!accepted) return {false, "QUEUE_FULL", "request queue is full", -1};

    const long deadline = cycle_ + 200000;
    while (!done && cycle_ < deadline) tick();
    if (!done) return {false, "INTERNAL_SIMULATOR_ERROR", "request did not complete", -1};
    return {true, "", "", depart};
  }

  void tick() {
    memory_->tick();
    cycle_++;
  }

  // Serialize the issued commands captured since the last drain, naming decoded
  // coordinates by the DRAMSpec level order, and accumulate ACT/refresh counters.
  std::string events_json(const std::vector<rhdram::IssuedEvent>& events) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < events.size(); ++i) {
      const auto& e = events[i];
      if (i) out << ',';
      out << "{\"op\":\"" << escape(e.command) << "\",\"clk\":" << e.clk;
      for (size_t k = 0; k < e.addr_vec.size() && k < geometry_.level_names.size(); ++k) {
        out << ",\"" << lower(geometry_.level_names[k]) << "\":" << e.addr_vec[k];
      }
      out << ",\"type_id\":" << e.type_id << ",\"row_hit\":" << (e.row_hit ? "true" : "false") << '}';

      if (e.command == "ACT") {
        acts_++;
      } else if (e.command.rfind("REF", 0) == 0 || e.command.rfind("RFM", 0) == 0) {
        refreshes_++;
      }
    }
    out << ']';
    return out.str();
  }

  std::string ok_json(const std::string& id, const std::string& op, Ramulator::Addr_t addr, Ramulator::Addr_t size,
                      long before, const std::string& data_hex) {
    auto events = rhdram::IssuedEventSink::instance().drain();
    std::string events_serialized = events_json(events);  // updates acts_/refreshes_
    std::ostringstream out;
    out << "{\"ok\":true,\"id\":\"" << escape(id) << "\",\"cycle\":" << cycle_
        << ",\"last_action\":{\"accepted\":1,\"rejected\":0,\"cycle_delta\":" << (cycle_ - before) << "}"
        << ",\"public_counters\":{\"reads\":" << reads_ << ",\"writes\":" << writes_
        << ",\"acts\":" << acts_ << ",\"refreshes\":" << refreshes_ << "}"
        << ",\"request\":{\"op\":\"" << op << "\",\"addr\":" << addr << ",\"size\":" << size << "}"
        << ",\"events\":" << events_serialized;
    if (!data_hex.empty()) out << ",\"data_hex\":\"" << data_hex << "\"";
    out << "}";
    return out.str();
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << error_json("", "BAD_SCHEMA", "usage: ramulator_worker <config.yaml>") << "\n";
    return 1;
  }
  try {
    Worker worker(argv[1]);
    std::string line;
    while (std::getline(std::cin, line)) {
      std::string response = worker.handle(line);
      std::cout << response << "\n" << std::flush;
      if (response.find("\"quit\":true") != std::string::npos) break;
    }
  } catch (const std::exception& err) {
    std::cout << error_json("", "INTERNAL_SIMULATOR_ERROR", err.what()) << "\n";
    return 1;
  }
  return 0;
}
