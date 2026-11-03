// t3sim: `simulate --config <scenario.json> --out <dir>/trace.parquet`
//        `simulate-batch --batch-dir <dir> --out-dir <dir>`
#include <sys/resource.h>
#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/prctl.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif
#include <dirent.h>

#include <atomic>
#include <functional>
#include <chrono>
#include <memory>
#include <thread>
#include <variant>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "engine.h"
#include "parquet.h"
#include "sha256.h"

using clk = std::chrono::steady_clock;

// dev switch T3_PHASES: print the process phase times to stderr at exit
static clk::time_point g_ph[16];
static int g_nph = 0;
static bool g_phase_on = true;  // single-scenario runs only (batch writes outputs on several threads)
static void phase() { if (g_phase_on && g_nph < 16) g_ph[g_nph++] = clk::now(); }

#if T3_ZB_READ
// plain read(2): the first use of an ifstream costs ~50 us of locale set-up
static std::string read_file(const std::string& p) {
  const int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error(T3_MSG("cannot read ", "E1 ") + p);
  std::string s;
  char buf[1 << 15];
  for (ssize_t r; (r = read(fd, buf, sizeof buf)) > 0;) s.append(buf, size_t(r));
  close(fd);
  return s;
}
#else
static std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error(T3_MSG("cannot read ", "E1 ") + p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
#endif
static void write_file(const std::string& p, const void* data, size_t n) {
  FILE* f = fopen(p.c_str(), "wb");
  if (!f) throw std::runtime_error(T3_MSG("cannot write ", "E2 ") + p);
  if (n && fwrite(data, 1, n, f) != n) throw std::runtime_error(T3_MSG("short write ", "E3 ") + p);
  fclose(f);
}
static void mkdirs(const std::string& p) {
  std::string cur;
  for (size_t i = 0; i < p.size(); i++) {
    cur += p[i];
    if (p[i] == '/' && cur.size() > 1) mkdir(cur.c_str(), 0755);
  }
  mkdir(p.c_str(), 0755);
}
static std::string dirname_of(const std::string& p) {
  size_t k = p.find_last_of('/');
  if (k == std::string::npos) return ".";
  if (k == 0) return "/";
  return p.substr(0, k);
}
static int64_t peak_rss_bytes() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  return (int64_t)ru.ru_maxrss;
#else
  return (int64_t)ru.ru_maxrss * 1024;
#endif
}
static std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += c;
  }
  return o + "\"";
}
static std::string fmt_double(double x) {
  char b[64];
  snprintf(b, sizeof b, "%.17g", x);
  std::string s = b;
  if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
  return s;
}

struct Result {
  std::string scenario_id;
  int64_t seed = 0, n_events = 0, n_messages = 0;
  double wall = 0;
  std::string trace_sha;
};

static const std::vector<std::string> TRACE_DICT = {t3::TRACE_NAMES[0], t3::TRACE_NAMES[1], t3::TRACE_NAMES[2],
                                                    t3::TRACE_NAMES[3], t3::TRACE_NAMES[4], t3::TRACE_NAMES[5]};
static const std::vector<std::string> SIDE_DICT = {"BID", "ASK"};
static std::vector<std::string> msg_dict() {
  std::vector<std::string> d;
  for (int i = 0; i < t3::M_NTYPES; i++) d.push_back(t3::MSG_NAMES[i]);
  return d;
}

// Run one scenario; outputs go to out_dir (trace.parquet, message_trace.parquet, events.json).
// Timed region of one scenario: the whole simulation (kernel initialization and event loop, including the
// latency draws and output-page faulting done on the helper thread, which starts working only here) and
// the trace extraction (grouping, quote collapse, ordering, fill labelling). Parsing/seeding before and
// output formatting/writing after are not timed, exactly as in the reference adapter.
template <class S>
static double simulate_timed(S& sim) {
#ifdef T3_PROF_TIMED
  CACHEGRIND_START_INSTRUMENTATION;
#endif
  auto t0 = clk::now();
#ifdef T3_DIAG_PHASES
  sim.diag_t0 = t0;
#endif
  sim.begin();
  sim.run();
  sim.finalize();
  auto t1 = clk::now();
#ifdef T3_PROF_TIMED
  CACHEGRIND_STOP_INSTRUMENTATION;
#endif
#ifdef T3_DIAG_PHASES
  for (size_t k = 0; k < sim.diag_k; k++)
    fprintf(stderr, "deliveries %zu at %.1f us\n", (size_t)1 << k, std::chrono::duration<double>(sim.diag_t[k] - t0).count() * 1e6);
  fprintf(stderr, "end %.1f us\n", std::chrono::duration<double>(t1 - t0).count() * 1e6);
#endif
  sim.end_threads();
  return std::chrono::duration<double>(t1 - t0).count();
}

// threads used to encode parquet column chunks (outputs are identical for any thread count)
static unsigned g_pq_threads = 1;
// a parquet file whose uncompressed column chunks exceed this is GZIP-compressed instead, so that a unit's whole output tree
// (two files per scenario) stays under the platform's 256 MiB limit; a batch lowers it to its share per sub-scenario file
#ifndef T3_GZIP_ABOVE
#define T3_GZIP_ABOVE (int64_t(120) << 20)
#endif
static int64_t g_gzip_above = T3_GZIP_ABOVE;

// The profiling build exercises the multi-CPU paths even on a 2-CPU build host. A volatile value rather than an #ifdef, so that
// cpu_quota() has the same control flow in both builds (the profile must match it).
#ifdef T3_PGO_GEN
static volatile size_t g_min_cpus = 4;
#else
static volatile size_t g_min_cpus = 1;
#endif

// CPUs granted to the container (cgroup v2 cpu.max quota), else the CPUs this process may run on.
static size_t cpu_quota() {
  if (const char* e = T3_DEVENV("T3_CPUS")) return (size_t)std::max(1, atoi(e));  // (test aid: the CPUs a quota would grant)
  size_t hw = std::max(1u, std::thread::hardware_concurrency());
#if defined(__linux__)
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof set, &set) == 0 && CPU_COUNT(&set) > 0) hw = std::min(hw, (size_t)CPU_COUNT(&set));
#endif
  FILE* f = fopen("/sys/fs/cgroup/cpu.max", "r");
  if (f) {
    char q[64] = {0};
    long long per = 0;
    if (fscanf(f, "%63s %lld", q, &per) == 2 && strcmp(q, "max") != 0 && per > 0) {
      long long quota = atoll(q);
      size_t c = (size_t)((quota + per - 1) / per);
      if (c >= 1 && c < hw) hw = c;
    }
    fclose(f);
  }
  return std::max<size_t>(hw, (size_t)g_min_cpus);
}

// The engine is instantiated twice (see SimT): SimDet for the deterministic latency model, Sim for the others.
// The instantiation is chosen when the simulation is created, from the parsed config.
using SimPtr = std::variant<std::unique_ptr<t3::Sim>, std::unique_ptr<t3::SimDet>, std::unique_ptr<t3::SimNL>, std::unique_ptr<t3::SimW>,
                            std::unique_ptr<t3::SimDetW>>;

// message_trace.parquet is written unless a single-scenario run's family makes it optional (a batch needs it for every sub)
static bool want_msgs(const t3::Config& cfg, bool batch) {
  if (const char* e = T3_DEVENV("T3_MSGS")) return atoi(e) != 0;
  return batch || !cfg.ledger_optional;
}

// A spare CPU encodes output pages while the simulation runs (single-scenario runs with enough messages).
template <class S>
static bool use_output_ahead(const S& sim) {
  // est_msgs (5 per wakeup) undercounts market-maker-heavy runs up to ~7x, hence the low bar
  return sim.helper_allowed && sim.cpus_available >= 3 && sim.est_msgs > 4096 && !T3_DEVENV("T3_BENCH") && want_msgs(sim.cfg, false);
}

// A spare CPU encodes finished trace pages while a no-ledger simulation runs (see TraceAhead)
template <class S>
static bool use_trace_ahead(const S& sim) {
  if constexpr (!S::TA || !T3_ZB_CHAIN || !T3_AA_AHEAD) {
    return false;
  } else {
    if (const char* e = T3_DEVENV("T3_AHEAD")) return atoi(e) != 0 && sim.label_live;
    if constexpr (S::NOLEDGER) return sim.helper_allowed && sim.cpus_available >= 3 && sim.est_msgs > 250000 && sim.label_live && !T3_DEVENV("T3_BENCH");
    else return false;  // a run with a ledger starts it from the ledger encoder once the trace is long enough (use_trace_ahead_late)
  }
}

// A run with a message ledger has four busy threads when the trace is also encoded ahead: main, latency helper, ledger encoder, trace
// encoder. The ledger encoder starts the trace encoder when the trace has this many rows (the estimated message count is no guide
// for the length of the trace: market-maker ladders make it up to 7x too low).
template <class S>
static bool use_trace_ahead_late(const S& sim) {
  if constexpr (!S::TA || S::NOLEDGER || !T3_ZB_CHAIN || !T3_AA_AHEAD || !T3_CC_LTA) {
    return false;
  } else {
    if (const char* e = T3_DEVENV("T3_LTA")) return atoi(e) != 0 && sim.helper_allowed && sim.label_live;
    return sim.helper_allowed && sim.cpus_available >= 4 && sim.label_live && !T3_DEVENV("T3_BENCH") && !T3_DEVENV("T3_AHEAD");
  }
}

template <class S>
static std::unique_ptr<S> prepare_as(const t3::Config& cfg, bool helper) {
  std::unique_ptr<S> sim(new S());
  sim->helper_allowed = helper;
  sim->cpus_available = cpu_quota();
  phase();
  sim->setup(cfg);
  phase();
  sim->prepare_threads(use_output_ahead(*sim));  // that thread also takes over the helper's pre-faulting
  return sim;
}

// The unit card, when it is mounted next to the scenario, decides whether message_trace.parquet is required, as the scorer reads it
// (limits.requires_message_ledger): a boolean [scoring.params].requires_message_ledger, else the card's [task].scenario_family
// default (not required only for throughput-scale); without a card the scenario's family default holds.
static void card_ledger_rule(t3::Config& cfg, const std::string& config_path) {
  const size_t sl = config_path.find_last_of('/');
  const std::string card = (sl == std::string::npos ? std::string() : config_path.substr(0, sl + 1)) + "card.toml";
  if (access(card.c_str(), R_OK) != 0) return;
  const std::string t = read_file(card);
  std::string line, table, family;
  int declared = -1;
  bool has_family = false;
  auto trim = [](std::string x) {
    const size_t b = x.find_first_not_of(" \t\r"), e = x.find_last_not_of(" \t\r");
    return b == std::string::npos ? std::string() : x.substr(b, e - b + 1);
  };
  for (size_t pos = 0; pos < t.size();) {
    const size_t nl = t.find('\n', pos);
    line = t.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    pos = nl == std::string::npos ? t.size() : nl + 1;
    bool q = false;
    for (size_t k = 0; k < line.size(); k++) {  // drop a comment (a '#' outside a basic string)
      if (line[k] == '"' && (k == 0 || line[k - 1] != '\\')) q = !q;
      else if (line[k] == '#' && !q) { line.resize(k); break; }
    }
    line = trim(line);
    if (line.empty()) continue;
    if (line[0] == '[') {
      table = trim(line.substr(1, line.find(']') == std::string::npos ? std::string::npos : line.find(']') - 1));
      continue;
    }
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = trim(line.substr(0, eq));
    const std::string val = trim(line.substr(eq + 1));
    if (!table.empty()) key = table + "." + key;
    if (key == "scoring.params.requires_message_ledger") declared = val == "true" ? 1 : val == "false" ? 0 : -1;
    else if (key == "task.scenario_family" && val.size() >= 2 && val.front() == '"' && val.back() == '"') { family = val.substr(1, val.size() - 2); has_family = true; }
  }
  if (declared >= 0) cfg.ledger_optional = declared == 0;
  else cfg.ledger_optional = has_family && family == "throughput-scale";
}

static SimPtr prepare(const std::string& config_path, bool helper, bool single = false) {
  js::V sc = js::parse(read_file(config_path));
  t3::Config cfg = t3::parse_config(sc);
  card_ledger_rule(cfg, config_path);
  if (cfg.agents.size() >= 32767) {  // 32768 agents or more with the exchange: the wide instances (see SimT)
    if (cfg.lat == t3::L_DETERMINISTIC) return prepare_as<t3::SimDetW>(cfg, helper);
    return prepare_as<t3::SimW>(cfg, helper);
  }
  if (cfg.lat == t3::L_DETERMINISTIC) return prepare_as<t3::SimDet>(cfg, helper);
  // a single run without a message file: the engine does not record the ledger (the park-first loop keeps the general instance)
  if (T3_AA_NL && single && !want_msgs(cfg, false) && !t3::park_policy(cfg)) return prepare_as<t3::SimNL>(cfg, helper);
  return prepare_as<t3::Sim>(cfg, helper);
}

// An output file opened while the simulation runs. Helper threads fill its page cache with zeros up to `known` bytes, a lower bound
// of its final size; the end of the run then overwrites cached pages, which costs far less than appending new ones (page allocation,
// block reservation and dirty accounting are paid in the idle time of the run). The filling is best effort: it stops for good if a
// write fails or blocks, and never goes beyond g_fill_cap (a file of the gzip kind ends up shorter than its filler and is cut).
static std::atomic<bool> g_fill_off{false};
static size_t g_fill_cap = size_t(24) << 20;
static constexpr size_t g_fill_min = size_t(1) << 20;  // a smaller file is not worth the zero writes (its final write is short anyway)
struct EarlyFile {
  std::atomic<int> fd{-1};         // set by the thread that opens it (while the simulation runs)
  std::atomic<size_t> known{4};    // "PAR1" is certain; the owner raises it as the file's contents become known
  std::atomic<size_t> claimed{0};  // bytes claimed by the fillers (a filler writes its claim right away)
  void open_at(const std::string& path) { fd.store(open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644), std::memory_order_release); }
  void raise(size_t k) {
    size_t cur = known.load(std::memory_order_relaxed);
    while (k > cur && !known.compare_exchange_weak(cur, k, std::memory_order_release, std::memory_order_relaxed)) {}
  }
  // writes up to `step` more bytes of filler; false when there is nothing to do
  bool fill(size_t step) {
    static char zeros[size_t(1) << 18];
    const int f = fd.load(std::memory_order_acquire);
    if (f < 0 || g_fill_off.load(std::memory_order_relaxed)) return false;
    size_t cur = claimed.load(std::memory_order_relaxed), n;
    do {
      const size_t k = std::min(known.load(std::memory_order_acquire), g_fill_cap);
      if (k < g_fill_min || cur >= k) return false;
      n = std::min({k - cur, step, sizeof zeros});
    } while (!claimed.compare_exchange_weak(cur, cur + n, std::memory_order_relaxed));
    const auto t0 = clk::now();
    const bool ok = pwrite(f, zeros, n, (off_t)cur) == (ssize_t)n;
    if (!ok || clk::now() - t0 > std::chrono::milliseconds(2)) g_fill_off.store(true, std::memory_order_relaxed);
    return true;
  }
};

// Output column chunks of one scenario: pages of pq::kPageRows rows, each encoded from one pass over its rows.
struct OutChunks {
  std::vector<pq::Chunk> tc, mc;  // trace.parquet, message_trace.parquet
  EarlyFile mf;                   // message_trace.parquet when opened early (fd >= 0)
  OutChunks() : tc(7), mc(10) {
    const char* tn[] = {"t_ns", "agent_id", "msg_type", "side", "price", "size", "order_id"};
    const int tt[] = {pq::TYPE_INT64, pq::TYPE_INT32, pq::TYPE_STRING, pq::TYPE_STRING, pq::TYPE_INT64, pq::TYPE_INT64, pq::TYPE_INT64};
    for (int c = 0; c < 7; c++) tc[c] = {tn[c], tt[c], tt[c] == pq::TYPE_STRING, {}};
    const char* mn[] = {"seq", "t_recv_ns", "t_send_ns", "latency_ns", "src_id", "dst_id", "message_id", "msg_type", "order_id", "causal_parent"};
    const int mt[] = {pq::TYPE_INT64, pq::TYPE_INT64, pq::TYPE_INT64, pq::TYPE_INT64, pq::TYPE_INT32, pq::TYPE_INT32, pq::TYPE_INT64, pq::TYPE_STRING, pq::TYPE_INT64, pq::TYPE_INT64};
    for (int c = 0; c < 10; c++) mc[c] = {mn[c], mt[c], mt[c] == pq::TYPE_STRING, {}};
  }
};

// Page pg (rows a .. a+n-1) of trace.parquet (each column encoded in its own pass over the page's rows). W: rows of a wide instance.
template <bool W, class Rows>
static void encode_trace_page(const Rows& tr, size_t a, size_t n, std::vector<pq::Chunk>& tc, size_t pg) {
  const t3::TraceRow* x = &tr[a];
  { pq::IntPage c(n); c.encode([&](size_t i) { return x[i].t; }); c.finish(tc[0].pages[pg]); }
  if constexpr (W) { pq::IntPage c(n); c.encode([&](size_t i) { return int64_t(uint16_t(x[i].agent)) | int64_t(x[i].side >> 1) << 16; }); c.finish(tc[1].pages[pg]); }
  else { pq::IntPage c(n); c.encode([&](size_t i) { return int64_t(x[i].agent); }); c.finish(tc[1].pages[pg]); }
  // trace.parquet has one row group per page, so every page of a string column carries its dictionary page
  { pq::DictPage c(n, TRACE_DICT, true); c.encode([&](size_t i) { return x[i].code; }); c.finish(tc[2].pages[pg]); }
  if constexpr (W) { pq::DictPage c(n, SIDE_DICT, true); c.encode([&](size_t i) { return uint8_t(x[i].side & 1); }); c.finish(tc[3].pages[pg]); }
  else { pq::DictPage c(n, SIDE_DICT, true); c.encode([&](size_t i) { return x[i].side; }); c.finish(tc[3].pages[pg]); }
  { pq::IntPage c(n); c.encode([&](size_t i) { return x[i].price; }); c.finish(tc[4].pages[pg]); }
  { pq::IntPage c(n); c.encode([&](size_t i) { return x[i].size; }); c.finish(tc[5].pages[pg]); }
  { pq::IntPage c(n); c.encode([&](size_t i) { return int64_t(x[i].oid); }); c.finish(tc[6].pages[pg]); }
}

// Presence bits of a ledger page (bit i & 7 of byte i >> 3): sent = latency >= 0 (t_send/causal_parent present),
// has_oid = sent and type >= M_LIMIT. Bits past row n are clear; the arrays need (n + 7) / 8 + 8 bytes.
__attribute__((target("sse4.1,popcnt"))) static void ledger_presence(const t3::LedgerRow* x, size_t n, uint8_t* sent, uint8_t* has_oid, size_t& msent, size_t& moid) {
  static_assert(sizeof(t3::LedgerRow) == 40 && offsetof(t3::LedgerRow, type) == 32 && t3::M_NTYPES < 128, "ledger row layout");
  size_t cs = 0, co = 0;
  const __m128i lim = _mm_set1_epi8(char(t3::M_LIMIT - 1));
  auto lat2 = [&](size_t i) {
    __m128d v = _mm_load_sd(reinterpret_cast<const double*>(&x[i].latency));
    v = _mm_loadh_pd(v, reinterpret_cast<const double*>(&x[i + 1].latency));
    return unsigned(_mm_movemask_pd(v));  // 1 for a wakeup
  };
  const size_t gfull = n >> 3;
  for (size_t g = 0; g < gfull; g++) {
    const size_t i = g * 8;
    const unsigned wake = lat2(i) | lat2(i + 2) << 2 | lat2(i + 4) << 4 | lat2(i + 6) << 6;
    __m128i t = _mm_cvtsi32_si128(x[i].type);
    t = _mm_insert_epi8(t, x[i + 1].type, 1);
    t = _mm_insert_epi8(t, x[i + 2].type, 2);
    t = _mm_insert_epi8(t, x[i + 3].type, 3);
    t = _mm_insert_epi8(t, x[i + 4].type, 4);
    t = _mm_insert_epi8(t, x[i + 5].type, 5);
    t = _mm_insert_epi8(t, x[i + 6].type, 6);
    t = _mm_insert_epi8(t, x[i + 7].type, 7);
    const unsigned s = ~wake & 0xFF, o = s & unsigned(_mm_movemask_epi8(_mm_cmpgt_epi8(t, lim)));
    sent[g] = uint8_t(s);
    has_oid[g] = uint8_t(o);
    cs += size_t(__builtin_popcount(s));
    co += size_t(__builtin_popcount(o));
  }
  if (n & 7) {
    unsigned s = 0, o = 0;
    for (size_t j = 0, i = gfull * 8; j < (n & 7); j++) {
      const bool p = x[i + j].latency >= 0;
      s |= unsigned(p) << j;
      o |= unsigned(p && x[i + j].type >= t3::M_LIMIT) << j;
    }
    sent[gfull] = uint8_t(s);
    has_oid[gfull] = uint8_t(o);
    cs += size_t(__builtin_popcount(s));
    co += size_t(__builtin_popcount(o));
  }
  msent = cs;
  moid = co;
}

// Page pg (rows a .. a+n-1) of message_trace.parquet. A wakeup (latency < 0) has null t_send/causal_parent/order_id
// and latency 0; order_id is also null for message types below M_LIMIT (the engine leaves oid/causal stale there). W: a wide instance's rows.
template <bool W, class Rows>
static void encode_msg_page(const Rows& led, size_t a, size_t n, std::vector<pq::Chunk>& mc, size_t pg) {
  static const std::vector<std::string> MD = msg_dict();
  const t3::LedgerRow* x = &led[a];
  std::vector<uint8_t> sent((n + 7) / 8 + 8), has_oid((n + 7) / 8 + 8);
  size_t msent, moid;
  ledger_presence(x, n, sent.data(), has_oid.data(), msent, moid);
  { pq::IntPage c(n); c.push_sequence(int64_t(a), n); c.finish(mc[0].pages[pg]); }
  { pq::IntPage c(n); c.encode([&](size_t i) { return x[i].t_recv; }); c.finish(mc[1].pages[pg]); }
  { pq::IntPage c(n); c.encode_present(sent.data(), msent, [&](size_t i) { return x[i].t_recv - x[i].latency; }); c.finish(mc[2].pages[pg]); }
  { pq::IntPage c(n); c.encode([&](size_t i) { return x[i].latency < 0 ? int64_t(0) : x[i].latency; }); c.finish(mc[3].pages[pg]); }
  if constexpr (W) {
    { pq::IntPage c(n); c.encode([&](size_t i) { return int64_t(x[i].src) | int64_t(x[i].srch) << 16; }); c.finish(mc[4].pages[pg]); }
    { pq::IntPage c(n); c.encode([&](size_t i) { return int64_t(x[i].dst) | int64_t(x[i].dsth) << 16; }); c.finish(mc[5].pages[pg]); }
  } else {
    { pq::IntPage c(n); c.encode([&](size_t i) { return int64_t(x[i].src); }); c.finish(mc[4].pages[pg]); }
    { pq::IntPage c(n); c.encode([&](size_t i) { return int64_t(x[i].dst); }); c.finish(mc[5].pages[pg]); }
  }
  { pq::IntPage c(n); c.encode([&](size_t i) { return int64_t(x[i].id); }); c.finish(mc[6].pages[pg]); }
  { pq::DictPage c(n, MD, pg == 0); c.encode([&](size_t i) { return x[i].type; }); c.finish(mc[7].pages[pg]); }
  { pq::IntPage c(n); c.encode_present(has_oid.data(), moid, [&](size_t i) { return int64_t(x[i].oid); }); c.finish(mc[8].pages[pg]); }
  { pq::IntPage c(n); c.encode_present(sent.data(), msent, [&](size_t i) { return int64_t(x[i].causal); }); c.finish(mc[9].pages[pg]); }
}

// Encodes full message-ledger pages on a spare CPU while the simulation runs (ledger rows are final once written;
// the newest published row may still be being written, so it is never read).
// Give back the memory of rows [a, b) of a stream (whole system pages inside the range only): encoded rows are
// never read again, and freeing them here (in parallel encoding threads) is cheaper than at process exit.
template <class Rows>
static void release_rows(const Rows& rows, size_t a, size_t b) {
  using T = typename std::remove_reference<decltype(rows[0])>::type;
  const uintptr_t base = (uintptr_t)&rows[0];
  const uintptr_t lo = (base + a * sizeof(T) + 4095) & ~uintptr_t(4095), hi = (base + b * sizeof(T)) & ~uintptr_t(4095);
  if (hi > lo) madvise((void*)lo, hi - lo, MADV_DONTNEED);
}

template <class S>
struct TraceAhead;

template <class S>
struct LedgerAhead {
  std::thread th;
  std::atomic<bool> stop{false};
  TraceAhead<S>* late_ta = nullptr;  // started from here once the trace has late_rows rows
  size_t late_rows = size_t(T3_CC_LTA_PAGES) * pq::kPageRows;
  size_t done = 0;  // pages [0, done) of oc.mc are encoded
  // message_trace.parquet will hold every piece encoded so far, so that many bytes of it are certain
  void note_msg_page(OutChunks& oc) {
    size_t add = 0;
    for (auto& c : oc.mc) add += c.pages[done].len;
    oc.mf.raise(oc.mf.known.load(std::memory_order_relaxed) + add);  // only this thread raises it
  }
  void start(S& sim, OutChunks& oc, const std::string& out_dir) {
    th = std::thread([this, &sim, &oc, out_dir] {
#if T3_ZB_PREFILL
      // the output file is created here, not on the calling thread: creating it costs ~50 us on the critical path of the start
      struct rlimit rl;
      if (getrlimit(RLIMIT_FSIZE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) g_fill_cap = std::min<size_t>(g_fill_cap, size_t(rl.rlim_cur) / 2);
      mkdirs(out_dir);
      oc.mf.open_at(out_dir + "/message_trace.parquet");
#else
      (void)out_dir;
#endif
      while (!stop.load(std::memory_order_acquire)) {
        const size_t pub = sim.led.pub.load(std::memory_order_acquire);
        if ((done + 1) * pq::kPageRows < pub) {
          for (auto& c : oc.mc) c.pages.emplace_back();
          encode_msg_page<S::WIDE>(sim.led, done * pq::kPageRows, pq::kPageRows, oc.mc, done);
          note_msg_page(oc);
          done++;
          release_rows(sim.led, (done - 1) * pq::kPageRows, done * pq::kPageRows);
        } else if (late_ta && sim.trace.pub.load(std::memory_order_relaxed) >= late_rows) {
          if constexpr (S::TA) late_ta->start(sim, oc.tc, out_dir);
          late_ta = nullptr;
        } else if (!sim.prefault_step() && !oc.mf.fill(size_t(1) << 18)) {
          struct timespec ts = {0, 50000};
          nanosleep(&ts, nullptr);
        }
      }
    });
  }
  size_t finish() {
    stop.store(true, std::memory_order_release);
    th.join();
    return done;
  }
};

// Column msg_type (the code) of page pg again: the only column of a finished row that can change (see SimT::nl_checkpoint).
template <class Rows>
static void encode_trace_code(const Rows& tr, size_t a, size_t n, std::vector<pq::Chunk>& tc, size_t pg) {
  const t3::TraceRow* x = &tr[a];
  pq::DictPage c(n, TRACE_DICT, true);
  c.encode([&](size_t i) { return x[i].code; });
  c.finish(tc[2].pages[pg]);
}

// Encodes the trace pages of a no-ledger run (SimNL) on a spare CPU while the simulation runs, and writes and digests the oldest ones.
// A page is encoded once its rows can no longer move (they lie before the first row of the open timestamp group: SimT::fin_pub) and no
// order resting in the book can relabel one of them (SimT::mut_pub); its msg_type column is the only thing that can still change: an
// order's latest ORDER_EXECUTED row is relabelled when the order executes again. The engine stores the epoch of every relabelling in
// SimT::lastflip_mem (per page), and epoch_pub is the newest epoch whose relabellings are visible to a thread that read it before reading
// the rows. A page encoded after reading epoch E is still right at the end of the run if its lastflip is <= E. A page is written and
// digested (in file order) a little later (lag_com rows behind the limit); if its lastflip is newer than E by then, its msg_type column is
// encoded again first. A relabelling that arrives after the page was written is found at the end of the run (write_outputs), which then
// takes the digest and the file back to that page; the output never depends on timing.
template <class S>
struct TraceAhead {
  static constexpr size_t N = pq::kPageRows;
  std::thread th;
  std::atomic<bool> stop{false};
  size_t lag_enc = 2 * N, lag_com = 16 * N;
  size_t lag_pages = 40;             // pages are written and digested at the latest this many pages behind the newest finished row
  std::vector<uint32_t> ep;          // epoch read before the latest encode of each encoded page
  std::vector<Sha256Stream> snap;    // digest state before each written page
  std::vector<uint64_t> off;         // file offset of each written page
  size_t enc_done = 0, com_done = 0; // pages [0, enc_done) are encoded, [0, com_done) written and digested
  Sha256Stream hs;                   // the digest after the written pages
  int fd = -1;
  uint64_t pos = 0;
  bool failed = false;
  ~TraceAhead() {
    if (th.joinable()) { stop.store(true, std::memory_order_release); th.join(); }
    if (fd >= 0) close(fd);
  }
  void start(S& sim, std::vector<pq::Chunk>& tc, const std::string& out_dir) {
    if (T3_AA_SCAN) { lag_enc = N / 8; lag_com = N / 4; }  // (small margins: the scan already covers resting orders and messages in flight)
    else lag_com = N * std::min<size_t>(16, std::max<size_t>(6, size_t(sim.est_msgs / double(N) / 4)));
    if (T3_AA_COMLAG) lag_com = N * size_t(T3_AA_COMLAG);
    if (const char* e = T3_DEVENV("T3_AHEAD_ENC")) lag_enc = size_t(atoll(e));
    if (const char* e = T3_DEVENV("T3_AHEAD_COM")) lag_com = size_t(atoll(e));
    if constexpr (!S::NOLEDGER) lag_pages = T3_CC_LAG;
    if (const char* e = T3_DEVENV("T3_CC_LAG")) lag_pages = size_t(atoi(e));
    S::ahead_on = true;
    ep.reserve(4096); snap.reserve(4096); off.reserve(4096);
    th = std::thread([this, &sim, &tc, out_dir] {
#if defined(__linux__)
      if (T3_AA_NICE) setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), T3_AA_NICE);
#endif
      mkdirs(out_dir);
      fd = open((out_dir + "/trace.parquet").c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
      if (fd < 0) failed = true;
      auto encode_next = [&](uint64_t fin) {
        const size_t p = enc_done;
        if ((p + 1) * N + lag_enc > fin) return false;
        const uint32_t E = S::epoch_pub.load(std::memory_order_acquire);  // before the rows are read
        for (auto& c : tc) c.pages.emplace_back();
        encode_trace_page<S::WIDE>(sim.trace, p * N, N, tc, p);
        ep.push_back(E);
        enc_done++;
        return true;
      };
      auto commit_next = [&](uint64_t fin) {
        const size_t p = com_done;
        if (failed || p >= enc_done || (p + 1) * N + lag_com > fin) return false;
        if (__atomic_load_n(&S::lastflip_mem[(p + 1) * 16], __ATOMIC_RELAXED) > ep[p]) {  // the code column changed since it was encoded
          const uint32_t E = S::epoch_pub.load(std::memory_order_acquire);
          encode_trace_code(sim.trace, p * N, N, tc, p);
          ep[p] = E;
        }
        snap.push_back(hs);
        off.push_back(pos);
        std::vector<iovec> iov;
        if (p == 0) { hs.update(reinterpret_cast<const uint8_t*>("PAR1"), 4); iov.push_back({const_cast<char*>("PAR1"), 4}); pos = 4; }
        for (auto& c : tc) {
          hs.update(c.pages[p].p, c.pages[p].len);
          iov.push_back({c.pages[p].p, c.pages[p].len});
          pos += c.pages[p].len;
        }
        try { pq::detail::write_all(fd, iov, "trace.parquet"); } catch (...) { failed = true; fd = -1; return false; }
        com_done++;
        return true;
      };
      uint32_t jit = 0, rnd = 12345;  // test aid (T3_AHEAD_JITTER=us): random pauses between the steps, to shuffle the interleaving with the engine
      if (const char* e = T3_DEVENV("T3_AHEAD_JITTER")) jit = uint32_t(atoi(e));
      while (!stop.load(std::memory_order_acquire) && !S::run_done.load(std::memory_order_acquire)) {
        if (jit) {
          rnd = rnd * 1664525u + 1013904223u;
          struct timespec js = {0, long((rnd >> 8) % jit) * 1000};
          nanosleep(&js, nullptr);
        }
        const uint64_t fin = S::fin_pub.load(std::memory_order_acquire);
        uint64_t safe = fin;  // rows to write: those no resting order (or message in flight) can relabel; but never wait for more than 40 pages (an order resting that long is rarely hit again)
        if (T3_AA_SCAN) safe = std::max(S::mut_pub.load(std::memory_order_acquire), fin > lag_pages * N ? fin - lag_pages * N : 0);
        if (encode_next(fin) || commit_next(safe)) continue;
        struct timespec ts = {0, 50000};
        nanosleep(&ts, nullptr);
      }
    });
  }
  void finish() {
    stop.store(true, std::memory_order_release);
    th.join();
  }
};

template <class S>
static Result write_outputs(S& sim, double wall, const std::string& out_dir, OutChunks& oc, size_t msg_done, bool msgs, TraceAhead<S>* ta = nullptr);

// `simulate` (one scenario): the process ends as soon as its outputs are written
static bool g_exit_after_write = false;

// Profile-generating build (T3_PGO_GEN): its profile is written by an exit handler, which _exit skips, so it is dumped
// explicitly. Called through a pointer so both builds have the same control flow (the profile must match).
#ifdef T3_PGO_GEN
extern "C" void __gcov_dump(void);
static void (*volatile g_profile_dump)() = __gcov_dump;
#else
static void (*volatile g_profile_dump)() = nullptr;
#endif

// Report the run and end the process without freeing the simulation and output buffers (the kernel reclaims them).
[[noreturn]] static void finish_single_run(const Result& r) {
  phase();
  if (T3_DEVENV("T3_PHASES")) {
    static const char* nm[] = {"load", "parse", "setup", "threads", "simulate", "msg_encode", "trace_encode", "files", "events"};
    for (int k = 1; k < g_nph; k++)
      fprintf(stderr, "phase %s %.3f ms\n", nm[k - 1], std::chrono::duration<double>(g_ph[k] - g_ph[k - 1]).count() * 1e3);
  }
  printf("{\"scenario_id\": %s, \"n_events\": %lld, \"wall_clock_sec\": %s, \"events_per_sec\": %s}\n",
         json_str(r.scenario_id).c_str(), (long long)r.n_events, fmt_double(r.wall).c_str(),
         fmt_double(r.n_events / r.wall).c_str());
  fflush(stdout);
  fflush(stderr);
#if T3_ZB_PARREL
  pq::release_arena(g_pq_threads);
#endif
  if (void (*dump)() = g_profile_dump) dump();
  _exit(0);
}

template <class S>
static Result run_one_as(std::unique_ptr<S> sp, const std::string& config_path, const std::string& out_dir) {
  OutChunks oc;
  LedgerAhead<S> ahead;
  // a spare CPU encodes the message ledger while the simulation runs (when there is enough of it)
  const bool use_ahead = use_output_ahead(*sp);
  TraceAhead<S> tahead;
  if (use_ahead) {
    if (use_trace_ahead_late(*sp)) ahead.late_ta = &tahead;
    ahead.start(*sp, oc, out_dir);
    // the latency helper fills the file too when it has no draws to make (its chore list is read only after begin())
#if T3_ZB_PREFILL
    sp->feed.chores.push_back([&oc] { return oc.mf.fill(size_t(1) << 16); });
#endif
  }
  if (!S::NOLEDGER && !want_msgs(sp->cfg, false)) sp->led.sink_rows = 2048;  // no message file: the ledger rows are never read
  sp->feed.detach_at_stop = g_exit_after_write && !T3_DEVENV("T3_BENCH");
  bool use_tahead = use_trace_ahead(*sp);
  if constexpr (S::TA) {
    if (use_tahead) tahead.start(*sp, oc.tc, out_dir);
  }
  double wall = simulate_timed(*sp);
#ifdef RUSAGE_THREAD
  if (T3_DEVENV("T3_PHASES")) {
    struct rusage ru;
    getrusage(RUSAGE_THREAD, &ru);
    fprintf(stderr, "minflt main thread %ld\n", ru.ru_minflt);
  }
#endif
  if (use_tahead) tahead.finish();
  const size_t msg_done = use_ahead ? ahead.finish() : 0;
  if (tahead.th.joinable()) {  // started by the ledger encoder (now joined)
    tahead.finish();
    use_tahead = true;
  }
  if (const char* b = T3_DEVENV("T3_BENCH")) {
    int reps = atoi(b);
    double best = wall;
    for (int k = 0; k < reps; k++) {
      std::unique_ptr<S> s2 = prepare_as<S>(sp->cfg, true);
      best = std::min(best, simulate_timed(*s2));
    }
    fprintf(stderr, "bench %s first=%.1fus best=%.1fus events=%zu ns/ev first=%.1f best=%.1f\n",
            sp->cfg.scenario_id.c_str(), wall * 1e6, best * 1e6, sp->trace.size(), wall * 1e9 / sp->trace.size(),
            best * 1e9 / sp->trace.size());
  }
  if (T3_DEVENV("T3_NOWRITE")) {  // profiling aid only: report and skip output writing
    printf("{\"n_events\": %zu}\n", sp->trace.size());
    exit(0);
  }
  phase();
  return write_outputs(*sp, wall, out_dir, oc, msg_done, want_msgs(sp->cfg, false), use_tahead ? &tahead : nullptr);
}

static Result run_one(const std::string& config_path, const std::string& out_dir) {
  SimPtr sp = prepare(config_path, true, true);
  phase();
  return std::visit([&](auto& p) { return run_one_as(std::move(p), config_path, out_dir); }, sp);
}

// Runs the tasks on up to `threads` threads (the calling thread is one of them).
static void run_tasks(std::vector<std::function<void()>>& tasks, unsigned threads) {
  std::atomic<size_t> next{0};
  std::vector<std::exception_ptr> err(threads);
  auto work = [&](unsigned t) {
    try {
      for (size_t k; (k = next.fetch_add(1)) < tasks.size();) tasks[k]();
    } catch (...) {
      err[t] = std::current_exception();
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 1; t < threads && t < tasks.size(); t++) pool.emplace_back(work, t);
  work(0);
  for (auto& th : pool) th.join();
  for (auto& e : err)
    if (e) std::rethrow_exception(e);
}

// rows (trace + ledger) above which output encoding uses g_pq_threads threads
static constexpr size_t kParallelRows = 2 * pq::kPageRows;

// Encode and write the scenario's outputs; oc may already hold message pages [0, msg_done).
template <class S>
static Result write_outputs(S& sim, double wall, const std::string& out_dir, OutChunks& oc, size_t msg_done, bool msgs, TraceAhead<S>* ta) {
  const t3::Config& cfg = sim.cfg;
  mkdirs(out_dir);
  Result r;
  r.scenario_id = cfg.scenario_id;
  r.seed = cfg.seed;
  r.wall = wall;
  const auto& tr = sim.trace;
  const auto& led = sim.led;
  const size_t nt = tr.size(), nm = led.size();
  r.n_events = (int64_t)nt;
  r.n_messages = sim.n_messages();
  const size_t ntp = pq::page_count(nt), nmp = msgs ? pq::page_count(nm) : 0;
  std::vector<pq::Chunk>& tc = oc.tc;
  std::vector<pq::Chunk>& mc = oc.mc;
  for (auto& c : tc) c.pages.resize(ntp);
  for (auto& c : mc) c.pages.resize(nmp);
  auto trace_page = [&](size_t pg) {
    const size_t a = pg * pq::kPageRows, b = std::min(nt, a + pq::kPageRows);
    encode_trace_page<S::WIDE>(tr, a, b - a, tc, pg);
    release_rows(tr, a, b);
  };
  auto msg_page = [&](size_t pg) {
    const size_t a = pg * pq::kPageRows, b = std::min(nm, a + pq::kPageRows);
    encode_msg_page<S::WIDE>(led, a, b - a, mc, pg);
    release_rows(led, a, b);
  };
  const unsigned threads = (nt + (msgs ? nm : 0) > kParallelRows) ? g_pq_threads : 1;
  auto encode = [&](bool trace, size_t from, unsigned nthreads) {
    std::vector<std::function<void()>> tk;
    if (trace)
      for (size_t pg = from; pg < ntp; pg++) tk.push_back([&, pg] { trace_page(pg); });
    else
      for (size_t pg = from; pg < nmp; pg++) tk.push_back([&, pg] { msg_page(pg); });
    run_tasks(tk, nthreads);
  };
  auto gzip = [&](std::vector<pq::Chunk>& chunks, unsigned nthreads) {
    std::vector<std::function<void()>> tk;
    for (auto& c : chunks)
      for (auto& pc : c.pages) tk.push_back([&pc] { pq::compress(pc, 1); });
    run_tasks(tk, nthreads);
  };
  auto file_bytes = [](const std::vector<pq::Chunk>& c) {
    int64_t s = 0;
    for (auto& x : c) s += int64_t(x.len());
    return s;
  };
  // a file over the threshold gets GZIP-compressed pages (keeps a unit's outputs under 64 MiB)
  // message file first: with several threads it is written by one of them while the others encode the trace pages
  if (msgs) encode(false, msg_done, threads);
  if (msgs && file_bytes(mc) > g_gzip_above) gzip(mc, threads);
  const int64_t tgz = msgs ? g_gzip_above : 2 * g_gzip_above;  // without a message file the trace may use the whole tree budget
  phase();
  auto write_msgs = [&] { if (msgs) pq::write_file(out_dir + "/message_trace.parquet", (int64_t)nm, mc, pq::MESSAGE_PANDAS_META, false, true, oc.mf.fd.load(), oc.mf.claimed.load()); };
#if T3_ZB_CHAIN
  const unsigned tthreads = threads > 1 ? unsigned(std::max(1, int(threads) + T3_ZB_WDELTA)) : 1u;
#else
  std::thread mw;
  if (threads > 1) mw = std::thread(write_msgs);
  const unsigned tthreads = threads > 1 ? threads - 1 : 1;
#endif
  // trace.parquet: one row group per page (page-major), so with several threads the file is written and digested page by page
  // (in page order) while later pages are still being encoded
  const std::string tpath = out_dir + "/trace.parquet";
#if T3_ZB_CHAIN
  if (threads > 1) {
    std::unique_ptr<std::atomic<uint8_t>[]> ready(new std::atomic<uint8_t>[ntp]);
    for (size_t k = 0; k < ntp; k++) ready[k].store(0, std::memory_order_relaxed);
    // The digest and the file are two chains that follow the finished pages in page order. Whichever encoding thread finishes a page
    // runs them (a thread that finds a chain busy leaves the new page to the thread running it), so no thread has to sleep and wake up
    // to follow the encoders.
    // Pages [0, e0) were encoded during the run (TraceAhead), [0, c0) of them also written and digested. A page whose msg_type column
    // changed after it was encoded is encoded again (redo); if one of the written pages is among them (the first is p0), the digest
    // and the file go back to p0.
    size_t e0 = 0, c0 = 0;
    std::vector<size_t> redo;
    Sha256Stream hs;
    int tfd = -1;
    bool cut = false;
    if (ta) {
      e0 = std::min(ta->enc_done, ntp);
      c0 = std::min(ta->com_done, e0);
      if (ta->failed) c0 = 0;
      for (size_t t = 0; t < e0; t++) {
        if (S::lastflip_mem[(t + 1) * 16] > ta->ep[t]) redo.push_back(t);
        else ready[t].store(1, std::memory_order_relaxed);
      }
      if (c0 > 0 && !redo.empty() && redo[0] < c0) {
        c0 = redo[0];
        hs = ta->snap[c0];
        if (lseek(ta->fd, off_t(ta->off[c0]), SEEK_SET) < 0) c0 = 0;
        cut = true;
      } else if (c0 > 0) {
        hs = ta->hs;
      }
      if (c0 == 0 && ta->fd >= 0 && lseek(ta->fd, 0, SEEK_SET) != 0) { close(ta->fd); ta->fd = -1; }
      if (c0 == 0) { hs = Sha256Stream(); cut = true; }
      tfd = ta->fd;
      ta->fd = -1;
      if (T3_DEVENV("T3_AHEADSTAT"))
        fprintf(stderr, "ahead: est %.0f pages %zu encoded %zu written %zu (after cut %zu), %zu to redo%s\n", sim.est_msgs, ntp, e0, std::min(ta->com_done, e0), c0, redo.size(), cut ? ", cut" : "");
    }
    if (c0 == 0) hs.update(reinterpret_cast<const uint8_t*>("PAR1"), 4);
    size_t hnext = c0, wnext = c0;
    if (tfd < 0) tfd = open(tpath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (tfd < 0) throw std::runtime_error(T3_MSG("cannot write ", "E5 ") + tpath);
    std::vector<iovec> wiov;
    if (c0 == 0) wiov.push_back({const_cast<char*>("PAR1"), 4});
    std::atomic<bool> werr_set{false};
    std::exception_ptr werr;
    auto run_chain = [](std::atomic<int>& st, auto&& pass) {  // st: 0 idle, 1 running, 2 running and asked to run again
      int s = st.load();
      for (;;) {
        if (s == 0) {
          if (st.compare_exchange_weak(s, 1)) break;
        } else if (s == 1) {
          if (st.compare_exchange_weak(s, 2)) return;
        } else {
          return;
        }
      }
      for (;;) {
        pass();
        int e = 1;
        if (st.compare_exchange_strong(e, 0)) return;
        st.store(1);
      }
    };
    std::atomic<int> hst{0}, wst{0};
    auto hash_pass = [&] {
      for (; hnext < ntp && ready[hnext].load(); hnext++)
        for (auto& c : tc) hs.update(c.pages[hnext].p, c.pages[hnext].len);
    };
    auto write_pass = [&] {
      if (werr_set.load(std::memory_order_relaxed)) return;
      try {
        for (; wnext < ntp && ready[wnext].load(); wnext++)
          for (auto& c : tc) wiov.push_back({c.pages[wnext].p, c.pages[wnext].len});
        pq::detail::write_all(tfd, wiov, tpath);
        wiov.clear();
      } catch (...) {
        werr = std::current_exception();
        werr_set.store(true);
      }
    };
    std::vector<std::function<void()>> tk;
    tk.push_back(write_msgs);  // first: the message file takes one thread until it is written, the others encode trace pages
    for (size_t pg : redo)
      tk.push_back([&, pg] {
        const size_t a = pg * pq::kPageRows;
        encode_trace_code(tr, a, pq::kPageRows, tc, pg);
        release_rows(tr, a, a + pq::kPageRows);
        ready[pg].store(1);
        run_chain(wst, write_pass);
        run_chain(hst, hash_pass);
      });
    if (e0 > c0) {  // the pages encoded during the run but not yet written start both chains
      tk.push_back([&] { run_chain(wst, write_pass); });
      tk.push_back([&] { run_chain(hst, hash_pass); });
    }
    for (size_t pg = e0; pg < ntp; pg++)
      tk.push_back([&, pg] {
        trace_page(pg);
        ready[pg].store(1);
        run_chain(wst, write_pass);
        run_chain(hst, hash_pass);
      });
    {  // the rows of the pages encoded during the run (the redo pages release theirs after their msg_type column)
      size_t rd = 0;
      for (size_t pg = 0; pg < e0; pg++) {
        if (rd < redo.size() && redo[rd] == pg) { rd++; continue; }
        tk.push_back([&, pg] { release_rows(tr, pg * pq::kPageRows, (pg + 1) * pq::kPageRows); });
      }
    }
    run_tasks(tk, tthreads);
    phase();
    const std::vector<uint8_t> foot = pq::footer_for(tc, (int64_t)nt, pq::TRACE_PANDAS_META, false);
    hash_pass();  // every encoding thread has finished: the last pages (if a chain was left unfinished) and the footer
    hs.update(foot.data(), foot.size());
    const std::string sha = hs.hex();
    write_pass();
    if (!werr_set.load()) {
      wiov.push_back({const_cast<uint8_t*>(foot.data()), foot.size()});
      try {
        pq::detail::write_all(tfd, wiov, tpath);
      } catch (...) {
        werr = std::current_exception();
        werr_set.store(true);
      }
    }
    if (!werr_set.load()) {
      if (cut) {  // the file held bytes of an earlier version beyond this point
        int64_t total = 4 + int64_t(foot.size());
        for (auto& c : tc) total += int64_t(c.len());
        if (ftruncate(tfd, off_t(total)) != 0) { werr = std::make_exception_ptr(std::runtime_error(T3_MSG("short write ", "E25 ") + tpath)); werr_set.store(true); }
      }
      close(tfd);
    }
    if (werr) std::rethrow_exception(werr);
    r.trace_sha = sha;
    if (file_bytes(tc) > tgz) {  // too large: rewrite the file with GZIP-compressed pages
      gzip(tc, tthreads);
      r.trace_sha = pq::write_file(tpath, (int64_t)nt, tc, pq::TRACE_PANDAS_META, true, false);
    }
#else
  if (threads > 1) {
    std::unique_ptr<std::atomic<uint8_t>[]> ready(new std::atomic<uint8_t>[ntp + 1]);  // [ntp]: the footer
    for (size_t k = 0; k <= ntp; k++) ready[k].store(0, std::memory_order_relaxed);
    std::vector<uint8_t> foot;
    auto wait_ready = [&](size_t k) {
      while (!ready[k].load(std::memory_order_acquire)) {
        struct timespec ts = {0, 10000};
        nanosleep(&ts, nullptr);
      }
    };
    std::string sha;
    std::thread hasher([&] {
      Sha256Stream h;
      h.update(reinterpret_cast<const uint8_t*>("PAR1"), 4);
      for (size_t pg = 0; pg < ntp; pg++) {
        wait_ready(pg);
        for (auto& c : tc) h.update(c.pages[pg].p, c.pages[pg].len);
      }
      wait_ready(ntp);
      h.update(foot.data(), foot.size());
      sha = h.hex();
    });
    std::exception_ptr werr;
    std::thread writer([&] {
      try {
        int fd = open(tpath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) throw std::runtime_error(T3_MSG("cannot write ", "E5 ") + tpath);
        std::vector<iovec> iov{{const_cast<char*>("PAR1"), 4}};
        for (size_t pg = 0; pg <= ntp; pg++) {
          if (!ready[pg].load(std::memory_order_acquire)) {  // write what is ready, then wait for the next page
            pq::detail::write_all(fd, iov, tpath);
            iov.clear();
            wait_ready(pg);
          }
          if (pg < ntp)
            for (auto& c : tc) iov.push_back({c.pages[pg].p, c.pages[pg].len});
          else
            iov.push_back({foot.data(), foot.size()});
        }
        pq::detail::write_all(fd, iov, tpath);
        close(fd);
      } catch (...) {
        werr = std::current_exception();
      }
    });
    std::vector<std::function<void()>> tk;
    for (size_t pg = 0; pg < ntp; pg++)
      tk.push_back([&, pg] {
        trace_page(pg);
        ready[pg].store(1, std::memory_order_release);
      });
    run_tasks(tk, tthreads);
    phase();
    foot = pq::footer_for(tc, (int64_t)nt, pq::TRACE_PANDAS_META, false);
    ready[ntp].store(1, std::memory_order_release);
    writer.join();
    hasher.join();
    if (werr) std::rethrow_exception(werr);
    r.trace_sha = sha;
    if (file_bytes(tc) > tgz) {  // too large: rewrite the file with GZIP-compressed pages
      gzip(tc, tthreads);
      r.trace_sha = pq::write_file(tpath, (int64_t)nt, tc, pq::TRACE_PANDAS_META, true, false);
    }
#endif
  } else {
    if (ta && ta->fd >= 0) { close(ta->fd); ta->fd = -1; }
    encode(true, 0, tthreads);
    if (file_bytes(tc) > tgz) gzip(tc, tthreads);
    phase();
    r.trace_sha = pq::write_file(tpath, (int64_t)nt, tc, pq::TRACE_PANDAS_META, true, false);
  }
#if T3_ZB_CHAIN
  if (threads == 1) write_msgs();
#else
  if (threads > 1) mw.join();
  else write_msgs();
#endif
  phase();
  double eps = r.wall > 0 ? (double)r.n_events / r.wall : 0.0;
  std::string ev = "{\n";
  ev += "  \"scenario_id\": " + json_str(r.scenario_id) + ",\n";
  ev += "  \"seed\": " + std::to_string(r.seed) + ",\n";
  ev += "  \"n_events\": " + std::to_string(r.n_events) + ",\n";
  ev += "  \"wall_clock_sec\": " + fmt_double(r.wall) + ",\n";
  ev += "  \"events_per_sec\": " + fmt_double(eps) + ",\n";
  ev += "  \"trace_sha256\": " + json_str(r.trace_sha) + ",\n";
  ev += "  \"n_messages\": " + std::to_string(r.n_messages) + ",\n";
  ev += "  \"peak_memory_bytes\": " + std::to_string(peak_rss_bytes()) + ",\n";
  ev += "  \"gpu_seconds\": 0.0\n}\n";
  write_file(out_dir + "/events.json", ev.data(), ev.size());
  if (g_exit_after_write) finish_single_run(r);
  return r;
}

static int cmd_simulate(int argc, char** argv) {
  g_pq_threads = (unsigned)std::max<size_t>(1, std::min<size_t>(cpu_quota(), 8));
  std::string config, out;
  for (int i = 0; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) config = argv[++i];
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--seed" && i + 1 < argc) ++i;
  }
  if (config.empty() || out.empty()) {
    fprintf(stderr, T3_MSG("usage: simulate --config <scenario.json> --out <trace.parquet>\n", "usage\n"));
    return 2;
  }
  g_exit_after_write = true;
#if T3_ZB_ARENA
  pq::use_arena(size_t(4) << 30);
#endif
  run_one(config, dirname_of(out));  // does not return: finish_single_run ends the process
  return 0;
}

static int cmd_batch(int argc, char** argv) {
  g_phase_on = false;
  std::string bdir, odir;
  for (int i = 0; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--batch-dir" && i + 1 < argc) bdir = argv[++i];
    else if (a == "--out-dir" && i + 1 < argc) odir = argv[++i];
  }
  if (bdir.empty() || odir.empty()) {
    fprintf(stderr, T3_MSG("usage: simulate-batch --batch-dir <dir> --out-dir <dir>\n", "usage\n"));
    return 2;
  }
  std::vector<std::string> subs;
  if (DIR* d = opendir(bdir.c_str())) {
    while (dirent* e = readdir(d)) {
      std::string nm = e->d_name;
      if (nm.size() > 5 && nm.compare(nm.size() - 5, 5, ".json") == 0) subs.push_back(nm.substr(0, nm.size() - 5));
    }
    closedir(d);
  }
  std::sort(subs.begin(), subs.end());
  if (subs.empty()) {
    fprintf(stderr, T3_MSG("simulate-batch: no sub-scenarios (*.json) found in %s\n", "E4 %s\n"), bdir.c_str());
    return 1;
  }
  // Sub-scenarios are independent markets. Each worker takes the next sub and runs it from start to end: prepare (parse + seeding),
  // simulate, write its outputs. The batch wall clock spans the simulation work of every sub (first start to last end).
  const size_t n = subs.size();
  g_gzip_above = std::min<int64_t>(g_gzip_above, (int64_t(240) << 20) / int64_t(2 * std::max<size_t>(n, 1)));
  std::vector<SimPtr> sims(n);
  std::vector<double> walls(n, 0.0);
  std::vector<Result> res(n);
  std::vector<clk::time_point> sim_begin(n), sim_end(n);
  const size_t workers = std::min(n, cpu_quota());
  std::atomic<size_t> next{0};
  std::vector<std::exception_ptr> err(workers);
  auto work = [&](size_t w) {
    try {
      for (size_t i; (i = next.fetch_add(1)) < n;) {
        sims[i] = prepare(bdir + "/" + subs[i] + ".json", false);
        sim_begin[i] = clk::now();
        walls[i] = std::visit([](auto& sp) { return simulate_timed(*sp); }, sims[i]);
        sim_end[i] = clk::now();
        res[i] = std::visit([&](auto& sp) {
          OutChunks oc;
          return write_outputs(*sp, walls[i], odir + "/" + subs[i], oc, 0, want_msgs(sp->cfg, true));
        }, sims[i]);
      }
    } catch (...) {
      err[w] = std::current_exception();
    }
  };
  std::vector<std::thread> pool;
  for (size_t w = 1; w < workers; w++) pool.emplace_back(work, w);
  work(0);
  for (auto& th : pool) th.join();
  for (auto& e : err)
    if (e) std::rethrow_exception(e);
  clk::time_point first = sim_begin[0], last = sim_end[0];
  for (size_t i = 1; i < n; i++) {
    first = std::min(first, sim_begin[i]);
    last = std::max(last, sim_end[i]);
  }
  const double wall = std::chrono::duration<double>(last - first).count();
  int64_t total = 0;
  for (auto& r : res) total += r.n_events;
  std::string be = "{\n";
  be += "  \"n_scenarios\": " + std::to_string(subs.size()) + ",\n";
  be += "  \"total_events\": " + std::to_string(total) + ",\n";
  be += "  \"wall_clock_sec\": " + fmt_double(wall) + ",\n";
  be += "  \"events_per_sec\": " + fmt_double(wall > 0 ? (double)total / wall : 0.0) + ",\n";
  be += "  \"peak_memory_bytes\": " + std::to_string(peak_rss_bytes()) + ",\n";
  be += "  \"gpu_seconds\": 0.0,\n";
  be += "  \"per_scenario\": [\n";
  for (size_t i = 0; i < subs.size(); i++) {
    be += "    {\"sub\": " + json_str(subs[i]) + ", \"n_events\": " + std::to_string(res[i].n_events) +
          ", \"trace_sha256\": " + json_str(res[i].trace_sha) + "}";
    be += (i + 1 < subs.size()) ? ",\n" : "\n";
  }
  be += "  ]\n}\n";
  mkdirs(odir);
  write_file(odir + "/batch_events.json", be.data(), be.size());
  printf("{\"n_scenarios\": %zu, \"total_events\": %lld, \"wall_clock_sec\": %s}\n", subs.size(), (long long)total,
         fmt_double(wall).c_str());
  // all outputs are written: end the process without freeing the simulations (the kernel reclaims them)
  fflush(stdout);
  fflush(stderr);
  if (void (*dump)() = g_profile_dump) dump();
  _exit(0);
}

#if defined(__linux__)
extern "C" char __executable_start, etext;
#endif
// Program loading (analogue of the reference's untimed imports): map this binary's code pages and
// resolve the libm entry points before any timed work.
static void load_program() {
#if defined(__linux__)
  // map every code page (one read per page, nothing is executed): this binary and any shared libraries it runs
  // (code first-touched inside the run would otherwise take a file-map page fault)
  volatile uint64_t sink = 0;
  uint64_t acc = 0;
  if (FILE* m = fopen("/proc/self/maps", "r")) {
    char line[512];
    while (fgets(line, sizeof line, m)) {
      unsigned long a, b;
      char perm[8];
      if (sscanf(line, "%lx-%lx %7s", &a, &b, perm) != 3 || perm[0] != 'r' || perm[3] == 's') continue;
      if (!strchr(line, '/') || perm[1] == 'w') continue;  // file-backed, not writable (code and read-only data)
      for (unsigned long p = a; p < b; p += 4096) acc += *(const volatile unsigned char*)p;
    }
    fclose(m);
  } else {
    for (const char* p = &__executable_start; p < &etext; p += 4096) acc += *(const volatile unsigned char*)p;
  }
  sink = acc;
  (void)sink;
#endif
  volatile double x = 0.5;
  x = std::exp(x) + std::log(x + 1.0) + std::sqrt(x) + std::pow(x, 2.0) + std::nearbyint(x);
  (void)x;
}

int main(int argc, char** argv) {
#if defined(__linux__) && defined(PR_SET_THP_DISABLE)
  // no transparent huge pages (inherited by the threads): where the host enables them for every mapping, the first touch of each
  // thread's malloc arena and of the heap would zero 2 MiB, ~100 us of a thread's start for a few KiB of use
  prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
#endif
#if defined(__GLIBC__)
  // output page buffers (~130 KiB each) come from the heap instead of one mmap per buffer
  mallopt(M_MMAP_THRESHOLD, 64 << 20);
  mallopt(M_TOP_PAD, 16 << 20);
#endif
  phase();
  load_program();
  phase();
  try {
    std::string prog = argv[0];
    std::string base = prog.substr(prog.find_last_of('/') + 1);
    int i = 1;
    std::string verb;
    if (base == "simulate" || base == "simulate-batch") verb = base;
    if (argc > 1 && (std::string(argv[1]) == "simulate" || std::string(argv[1]) == "simulate-batch")) {
      verb = argv[1];
      i = 2;
    }
    if (verb == "simulate") return cmd_simulate(argc - i, argv + i);
    if (verb == "simulate-batch") return cmd_batch(argc - i, argv + i);
    fprintf(stderr, T3_MSG("usage: t3sim simulate|simulate-batch ...\n", "usage\n"));
    return 2;
  } catch (const std::exception& e) {
    fprintf(stderr, T3_MSG("t3sim error: %s\n", "error: %s\n"), e.what());
    return 1;
  }
}
