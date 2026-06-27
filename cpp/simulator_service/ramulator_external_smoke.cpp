#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace {

std::string json_escape(const std::string& value) {
  std::string out;
  for (char ch : value) {
    if (ch == '"' || ch == '\\') out.push_back('\\');
    out.push_back(ch);
  }
  return out;
}

int fail(const std::string& code, const std::string& message) {
  std::cout << "{\"ok\":false,\"error\":{\"code\":\"" << code
            << "\",\"message\":\"" << json_escape(message) << "\"}}\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    return fail("BAD_SCHEMA", "usage: ramulator_external_smoke <config.yaml> [max_ticks]");
  }

  const std::string config_path = argv[1];
  const long max_ticks = argc == 3 ? std::strtol(argv[2], nullptr, 10) : 200000;
  if (max_ticks <= 0) return fail("BAD_SCHEMA", "max_ticks must be positive");

  try {
    auto config = Ramulator::Config::parse_config_file(config_path);
    std::unique_ptr<Ramulator::IFrontEnd> frontend(Ramulator::Factory::create_frontend(config));
    std::unique_ptr<Ramulator::IMemorySystem> memory(Ramulator::Factory::create_memory_system(config));

    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());

    const int tx_bytes = memory->get_tx_bytes();
    bool write_done = false;
    bool read_done = false;
    Ramulator::Clk_t write_depart = -1;
    Ramulator::Clk_t read_depart = -1;

    const Ramulator::Addr_t addr = 0x1000;
    const bool write_accepted = frontend->receive_external_requests(
        Ramulator::Request::Type::Write, addr, 0,
        [&](Ramulator::Request& req) {
          write_done = true;
          write_depart = req.depart;
        },
        tx_bytes);
    if (!write_accepted) return fail("QUEUE_FULL", "write request was not accepted");

    long ticks = 0;
    while (!write_done && ticks++ < max_ticks) memory->tick();
    if (!write_done) return fail("INTERNAL_SIMULATOR_ERROR", "write request did not complete");

    const bool read_accepted = frontend->receive_external_requests(
        Ramulator::Request::Type::Read, addr, 0,
        [&](Ramulator::Request& req) {
          read_done = true;
          read_depart = req.depart;
        },
        tx_bytes);
    if (!read_accepted) return fail("QUEUE_FULL", "read request was not accepted");

    while (!read_done && ticks++ < max_ticks) memory->tick();
    if (!read_done) return fail("INTERNAL_SIMULATOR_ERROR", "read request did not complete");

    frontend->finalize();
    memory->finalize();

    std::cout << "{\"ok\":true"
              << ",\"frontend\":\"External\""
              << ",\"request_path\":\"Ramulator 2.1 External\""
              << ",\"addr\":" << addr
              << ",\"tx_bytes\":" << tx_bytes
              << ",\"ticks\":" << ticks
              << ",\"write\":{\"accepted\":true,\"completed\":true,\"depart\":" << write_depart << "}"
              << ",\"read\":{\"accepted\":true,\"completed\":true,\"depart\":" << read_depart << "}"
              << "}\n";
    return 0;
  } catch (const std::exception& err) {
    return fail("INTERNAL_SIMULATOR_ERROR", err.what());
  }
}
