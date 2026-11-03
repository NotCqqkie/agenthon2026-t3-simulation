#pragma once
// Faithful C++ port of the Track-3 reference: pinned ABIDES (kernel, ExchangeAgent, OrderBook,
// SparseMeanRevertingOracle) + baselines/abides_fork (config, agents, latency model, trace and
// message-ledger extraction). Every RNG draw, message id, order id and queue tie-break follows the
// Python reference.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <tuple>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"
#include "queue.h"
#include "rng.h"

// AA switches (compile-time; each defaults to on): T3_AA_NL the no-ledger engine instance for single runs without a message file,
// T3_AA_ORD helper pre-faulting of the per-order arrays (needs NL), T3_AA_AHEAD trace pages encoded and written during the run (needs NL)
#ifndef T3_AA_NL
#define T3_AA_NL 1
#endif
#ifndef T3_AA_ORD
#define T3_AA_ORD 1
#endif
#ifndef T3_AA_AHEAD
#define T3_AA_AHEAD 1
#endif
#ifndef T3_AA_SCAN
#define T3_AA_SCAN 1  // trace-ahead: the engine also publishes the oldest row a resting order may still relabel (SimT::scan_mutable)
#endif
#ifndef T3_AA_COMLAG
#define T3_AA_COMLAG 0  // trace pages written during the run lag this many pages behind (0: a quarter of the estimated run, 6..16; 9999: none)
#endif
#ifndef T3_CC_LTA
#define T3_CC_LTA 1  // trace pages are also encoded and written during the run of a unit that writes a message ledger
#endif
#ifndef T3_CC_LTA_PAGES
#define T3_CC_LTA_PAGES 3  // ... started once the trace has this many pages
#endif
#ifndef T3_CC_LAG
#define T3_CC_LAG 3  // ... and writes and digests pages that many pages behind the newest finished row, unless an exact bound allows more (a late relabel is caught at the end and redone)
#endif
#ifndef T3_AA_NICE
#define T3_AA_NICE 0  // nice value of the trace-ahead thread (0: unchanged)
#endif

namespace t3 {

static constexpr int64_t DATE_NS = 1612483200000000000LL;  // pd.to_datetime("20210205").value
static constexpr int64_t H930 = 34200000000000LL;          // str_to_ns("09:30:00")
static constexpr int64_t H1600 = 57600000000000LL;         // str_to_ns("16:00:00")
static constexpr int64_t ONE_S = 1000000000LL;
static constexpr int64_t NIL = INT64_MIN;  // null marker

enum MsgType : uint8_t {
  M_WAKEUP,
  M_CLOSE_PRICE_REQ,
  M_HOURS_REQ,
  M_HOURS,
  M_CLOSE_PRICE,
  M_CLOSED,
  M_QSPREAD,
  M_QSPREAD_RESP,
  M_LIMIT,
  M_CANCEL,
  M_ACCEPTED,
  M_EXECUTED,
  M_CANCELLED,
  M_NTYPES
};
static const char* const MSG_NAMES[M_NTYPES] = {
    "AGENT_WAKEUP",        "MarketClosePriceRequestMsg", "MarketHoursRequestMsg", "MarketHoursMsg",
    "MarketClosePriceMsg", "MarketClosedMsg",            "QuerySpreadMsg",        "QuerySpreadResponseMsg",
    "LimitOrderMsg",       "CancelOrderMsg",             "OrderAcceptedMsg",      "OrderExecutedMsg",
    "OrderCancelledMsg"};

enum EvType : uint8_t { E_SUBMITTED, E_ACCEPTED, E_EXECUTED, E_CANCELLED, E_BEST_BID, E_BEST_ASK, E_FILLED, E_PARTIAL };
enum TraceCode : uint8_t { C_SUBMITTED, C_ACCEPTED, C_FILLED, C_PARTIAL, C_CANCELLED, C_QUOTE };
static const char* const TRACE_NAMES[6] = {"ORDER_SUBMITTED", "ORDER_ACCEPTED", "ORDER_FILLED",
                                           "PARTIAL_FILL",    "ORDER_CANCELLED", "QUOTE_UPDATE"};

enum AgentKind : uint8_t { A_EXCHANGE, A_NOISE, A_MM, A_VALUE, A_MOMENTUM };
enum LatModel : uint8_t { L_LOGNORMAL, L_UNIFORM, L_PARETO, L_DETERMINISTIC };

// Python's exact comparison of an int with a float: <0, 0, >0 for a<b, a==b, a>b.
static inline int cmp_int_float(int64_t a, double b) {
  if (b >= 9223372036854775808.0) return -1;
  if (b < -9223372036854775808.0) return 1;
  double fb = std::floor(b);
  int64_t bi = (int64_t)fb;
  if (a < bi) return -1;
  if (a > bi) return 1;
  return (b > fb) ? -1 : 0;
}
// Python round(float) -> int (half to even)
#if defined(__x86_64__) && !defined(__SSE4_1__)
// baseline x86-64 has no roundsd: nearbyint is a libm call. Adding and subtracting 1.5 * 2^52 rounds
// to nearest-even in default rounding mode for |x| < 2^51 (identical to nearbyint there).
static inline int64_t py_round(double x) {
  if (std::fabs(x) < 2251799813685248.0) {
    const double m = 6755399441055744.0;
    return (int64_t)((x + m) - m);
  }
  return (int64_t)std::nearbyint(x);
}
#else
static inline int64_t py_round(double x) { return (int64_t)std::nearbyint(x); }
#endif

// A Python number that may be an int or a float (oracle timestamps).
struct PyNum {
  bool isf = false;
  int64_t i = 0;
  double f = 0;
  static PyNum I(int64_t v) { PyNum n; n.i = v; return n; }
  static PyNum F(double v) { PyNum n; n.isf = true; n.f = v; return n; }
  double asf() const { return isf ? f : (double)i; }
};
static inline bool py_lt(const PyNum& a, const PyNum& b) {
  if (!a.isf && !b.isf) return a.i < b.i;
  if (a.isf && b.isf) return a.f < b.f;
  if (!a.isf) return cmp_int_float(a.i, b.f) < 0;
  return cmp_int_float(b.i, a.f) > 0;
}
static inline double py_sub_f(const PyNum& a, const PyNum& b) {
  if (!a.isf && !b.isf) return (double)(a.i - b.i);
  return a.asf() - b.asf();
}

struct AgentCfg {  // member order: the noise agents' hot fields first (see Agent in engine.h)
  int64_t interval_ns = 1000000000;
  double size_mean = 10, size_std = 2;
  int64_t reference_price = 100000;
  int64_t price_offset = 5;
  int64_t spread_ticks = 2, depth_levels = 3, size_per_level = 10;
  int64_t threshold = 2, lookback = 5;
  double sigma_n = 1000.0;
  AgentKind kind = A_NOISE;
};

struct Config {
  std::string scenario_id;
  // scenario_family "throughput-scale": the unit's card does not require message_trace.parquet (the organizers' family default)
  bool ledger_optional = false;
  int64_t seed = 0;
  int64_t horizon_ns = 0;
  int64_t reference_price = 100000;
  double kappa = 0, fund_vol = 0, ms_lambda = 0, ms_mean = 0, ms_var = 50000;
  bool has_jump = false;
  int64_t jump_time_rel = 0, jump_mag = 0;
  bool stp = false, stp_oldest = false;
  int64_t ack_delay = 0, compute_delay = 0;
  bool has_latency_cfg = true;
  LatModel lat = L_DETERMINISTIC;
  double lat_mean = 0, lat_sigma = 0, lat_min = 0, lat_max = 1e12, lat_alpha = 1.5, lat_mu = 0;
  std::vector<AgentCfg> agents;
};

static inline double fget(const js::V* o, const char* k, double dflt) {
  const js::V* v = o ? o->get(k) : nullptr;
  return v ? v->as_float() : dflt;
}
static inline int64_t iget(const js::V* o, const char* k, int64_t dflt) {
  const js::V* v = o ? o->get(k) : nullptr;
  return v ? v->as_int() : dflt;
}
// Python: int(max(lo, params.get(k, dflt)))
static inline int64_t int_max(const js::V* o, const char* k, int64_t dflt, int64_t lo) {
  const js::V* v = o ? o->get(k) : nullptr;
  if (!v) return std::max(lo, dflt);
  if (v->isint) return std::max(lo, v->i);
  return v->d > (double)lo ? (int64_t)std::trunc(v->d) : lo;
}

inline Config parse_config(const js::V& sc) {
  Config c;
  const js::V* sid = sc.get("scenario_id");
  c.scenario_id = (sid && sid->t == js::V::STR) ? sid->s : std::string();
  const js::V* fam = sc.get("scenario_family");
  c.ledger_optional = fam && fam->t == js::V::STR && fam->s == "throughput-scale";
  c.seed = sc.get("seed")->as_int();
  c.horizon_ns = sc.get("horizon_ns")->as_int();
  const js::V* ex = sc.get("exchange_config");
  bool proto = false;
  if (const js::V* pe = ex ? ex->get("protocol_enforcement") : nullptr) proto = pe->truthy();
  if (proto) {
    const js::V* sp = ex->get("stp_policy");
    if (sp && sp->truthy()) {
      c.stp = true;
      c.stp_oldest = (sp->t == js::V::STR && sp->s == "cancel_oldest");
    }
    c.ack_delay = iget(ex, "ack_delay_ns", 0);
    c.compute_delay = iget(ex, "compute_delay_ns", 0);
  }
  const js::V* oc = sc.get("oracle_config");
  const js::V* op = oc ? oc->get("params") : nullptr;
  c.reference_price = iget(op, "initial_price", 100000);
  double kps = fget(op, "kappa", 0.0);
  c.kappa = kps > 0 ? kps / 1e9 : 1.67e-16;
  c.fund_vol = fget(op, "sigma", 5e-5);
  double jr = fget(op, "jump_intensity", 0.0);
  c.ms_lambda = jr > 0 ? jr / 1e9 : 2.77778e-18;
  double jsig = fget(op, "jump_sigma", 0.0);
  c.ms_mean = jsig != 0.0 ? jsig : 1000.0;
  c.ms_var = 50000;
  if (const js::V* sj = op ? op->get("scheduled_jump") : nullptr) {
    if (sj->truthy()) {
      c.has_jump = true;
      c.jump_time_rel = sj->get("time_ns")->as_int();
      c.jump_mag = sj->get("magnitude")->as_int();
    }
  }
  const js::V* lc = sc.get("latency_config");
  if (lc && lc->truthy()) {
    const js::V* lp = lc->get("params");
    std::string model = "deterministic";
    if (const js::V* m = lc->get("model")) model = m->s;
    c.lat_mean = fget(lp, "mean_ns", 0.0);
    c.lat_sigma = fget(lp, "sigma", 0.0);
    c.lat_min = fget(lp, "min_ns", 0.0);
    c.lat_max = fget(lp, "max_ns", 1e12);
    c.lat_alpha = fget(lp, "alpha", 1.5);
    c.lat_mu = c.lat_mean > 0 ? std::log(c.lat_mean) : 0.0;
    if (model == "log_normal") c.lat = L_LOGNORMAL;
    else if (model == "uniform") c.lat = L_UNIFORM;
    else if (model == "pareto") c.lat = L_PARETO;
    else c.lat = L_DETERMINISTIC;
  } else {
    c.has_latency_cfg = false;
  }
  for (const js::V& ac : sc.get("agent_configs")->a) {
    std::string ty = ac.get("agent_type")->s;
    const js::V* p = ac.get("params");
    AgentCfg a;
    if (p && p->get("rebalance_interval_ns")) {
      a.interval_ns = p->get("rebalance_interval_ns")->as_int();
    } else {
      double hz = fget(p, "arrival_rate_hz", 1.0);
      a.interval_ns = hz > 0 ? (int64_t)(1e9 / hz) : (int64_t)1e9;
    }
    a.interval_ns = std::max<int64_t>(1, a.interval_ns);
    a.reference_price = c.reference_price;
    if (ty == "NoiseTrader") {
      a.kind = A_NOISE;
      a.size_mean = fget(p, "order_size_mean", 10);
      a.size_std = fget(p, "order_size_std", 2);
      a.price_offset = iget(p, "price_offset_ticks", 5);
    } else if (ty == "MarketMaker") {
      a.kind = A_MM;
      a.spread_ticks = int_max(p, "spread_ticks", 2, 2);
      a.depth_levels = int_max(p, "depth_levels", 3, 1);
      a.size_per_level = int_max(p, "size_per_level", 10, 1);
    } else if (ty == "ValueTrader") {
      a.kind = A_VALUE;
      a.size_mean = fget(p, "order_size_mean", 25);
      a.threshold = iget(p, "threshold_ticks", 2);
    } else if (ty == "MomentumTrader") {
      a.kind = A_MOMENTUM;
      a.size_mean = fget(p, "order_size_mean", 15);
      a.threshold = iget(p, "threshold_ticks", 2);
      a.lookback = int_max(p, "lookback", 5, 1);
    } else {
      throw std::runtime_error(T3_MSG("unsupported agent_type: ", "E30 ") + ty);
    }
    int64_t n = ac.get("count")->as_int();
    for (int64_t k = 0; k < n; k++) c.agents.push_back(a);
  }
  return c;
}

// ---------------------------------------------------------------------------------------------------------------------------
// Which bounce policy a scenario runs with (SimT::run_park when true, SimT::run_m12 otherwise), from the config only.
//
// A message popped for a recipient that is still busy (every trader is busy TDELAY after each delivery, the exchange compute_delay)
// is requeued once and parked the second time (run_m12), or, in run_park, parked at once when messages already wait for the
// recipient. Messages parked on one recipient are kept in key order: run_m12 shifts about half of them for an insertion, run_park
// does one binary search and one block move. run_park wins when a first bounce often finds waiters (about 8% of the deliveries or
// more) or when the parked sets get deep (a dozen messages on average per parking); elsewhere run_m12 is cheaper per parked message.
// Both come from bursts of messages for one recipient, which arrive spread over the latency model's range and are served one per
// busy window. In the agent mix only market makers send such bursts (a wakeup cancels and re-quotes 2 * depth_levels orders, so
// 4 * depth_levels messages go to the exchange and as many replies come back); with a compute_delay the exchange is a server too, and
// the agents that share a wakeup interval send their requests at the same instant.
// The estimate replays one burst through a toy of the loop's rule at one recipient (toy_server) with the arrival times drawn at fixed
// quantiles of the latency model, at the exchange (when it has a compute_delay) and at a market maker (the replies), and weighs the
// counts by the burst rate against the delivery rate. Calibrated on the measured share of first bounces that found waiters (public
// units and the fuzz corpus: the estimate is within a factor of ~2) and on the measured cycles of both loops (see the thresholds).
// A deterministic latency model never uses run_park (its time-grouped list already spaces the equal-time arrivals).
namespace park_rule {
// inverse of the standard normal CDF (Acklam's rational approximation)
inline double ndtri(double p) {
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
  if (p < 0.02425) {
    const double q = std::sqrt(-2 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - 0.02425) return -ndtri(1 - p);
  const double q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
// van der Corput sequence: fixed quasi-random numbers in (0, 1)
inline double vdc(unsigned i, unsigned base) {
  double f = 1, r = 0;
  for (i++; i; i /= base) {
    f /= base;
    r += f * (i % base);
  }
  return r;
}
// the latency model's quantile function (clamped like the draws)
inline double lat_q(const Config& c, double p) {
  double v;
  switch (c.lat) {
    case L_LOGNORMAL: v = std::exp(c.lat_mu + c.lat_sigma * ndtri(p)); break;
    case L_UNIFORM: v = c.lat_min + p * (c.lat_max - c.lat_min); break;
    case L_PARETO: v = (c.lat_min > 0 ? c.lat_min : 1.0) * std::pow(1.0 - p, -1.0 / c.lat_alpha); break;
    default: v = c.lat_mean;
  }
  return std::min(std::max(v, c.lat_min), c.lat_max);
}
struct Count {
  double waiters = 0;  // first bounces that found parked messages
  double depth = 0;    // sum of the parked set's size over the parkings
  Count& operator+=(const Count& o) {
    waiters += o.waiters;
    depth += o.depth;
    return *this;
  }
};
// B messages (keys 0..B-1) arrive at a[i] at one recipient that is busy s ns after each delivery; they take the loop's path (a
// message popped while the recipient is busy is requeued at its free time the first time and parked the second time, a parked message
// is delivered at the free time, smallest key first). d[i] = delivery time.
inline Count toy_server(const std::vector<double>& a, double s, std::vector<double>& d) {
  typedef std::tuple<double, int, int> E;  // time, key, bounces so far
  const size_t B = a.size();
  std::priority_queue<E, std::vector<E>, std::greater<E>> H;
  for (size_t i = 0; i < B; i++) H.emplace(a[i], (int)i, 0);
  std::priority_queue<int, std::vector<int>, std::greater<int>> park;
  double at = -1e30;
  Count n;
  d.assign(B, 0.0);
  while (!H.empty() || !park.empty()) {
    if (!park.empty() && (H.empty() || std::make_pair(at, park.top()) < std::make_pair(std::get<0>(H.top()), std::get<1>(H.top())))) {
      const int k = park.top();
      park.pop();
      d[(size_t)k] = at;
      at += s;
      continue;
    }
    const E e = H.top();
    H.pop();
    const double t = std::get<0>(e);
    const int k = std::get<1>(e);
    if (at > t) {
      if (std::get<2>(e) == 0) {
        n.waiters += !park.empty();
        H.emplace(at, k, 1);
      } else {
        n.depth += (double)park.size();
        park.push(k);
      }
    } else {
      d[(size_t)k] = t;
      at = t + s;
    }
  }
  return n;
}
// The bounces of one burst of B requests sent at one instant. They arrive at the exchange at quantiles of the latency model; when
// the exchange has a compute_delay it serves them one per busy window (its counts go to `ex`) and sends each reply as it delivers
// the request, else the replies are sent as the requests arrive. `count` of the requests, from `offset` on, are one market maker's:
// its replies come back after a second latency (the counts at the market maker go to `mm`).
inline void burst(const Config& c, int B, int offset, int count, Count& ex, Count& mm) {
  B = std::min(B, 512);
  std::vector<double> a((size_t)std::max(B, 0)), d;
  for (int i = 0; i < B; i++) a[(size_t)i] = lat_q(c, vdc((unsigned)i, 2));
  if (c.compute_delay > 0) {
    if (B > 1) ex += toy_server(a, (double)c.compute_delay, d);
  } else {
    d = a;
  }
  if (count > 1 && offset + count <= B) {
    std::vector<double> r((size_t)count), d2;
    for (int i = 0; i < count; i++) r[(size_t)i] = d[(size_t)(offset + i)] + lat_q(c, vdc((unsigned)(offset + i), 3));
    mm += toy_server(r, 50.0, d2);  // a trader is busy TDELAY after each delivery
  }
}
struct Estimate {
  double waiters = 0;  // first bounces that find waiters, per delivery
  double depth = 0;    // parked-set size summed over the parkings, per delivery
  int mm_together = 0; // the most market makers that wake at one instant (they park at the same time)
};
inline Estimate estimate(const Config& c) {
  struct Grp {  // the agents with one wakeup interval wake together
    int64_t interval;
    int traders = 0;
    std::vector<std::pair<int64_t, int>> mm;  // (depth_levels, agents)
  };
  std::vector<Grp> groups;
  double events = 0;
  int mm_together = 0;
  Count total;
  for (const AgentCfg& a : c.agents) {
    size_t k = 0;
    while (k < groups.size() && groups[k].interval != a.interval_ns) k++;
    if (k == groups.size()) {
      if (groups.size() >= 16) continue;  // (a pathological mix: no estimate for the rest)
      groups.push_back(Grp());
      groups[k].interval = a.interval_ns;
    }
    Grp& g = groups[k];
    const double rate = 1e9 / (double)a.interval_ns;
    if (a.kind == A_MM) {
      const int64_t D = std::max<int64_t>(1, a.depth_levels);
      size_t m = 0;
      while (m < g.mm.size() && g.mm[m].first != D) m++;
      if (m == g.mm.size()) g.mm.emplace_back(D, 0);
      g.mm[m].second++;
      events += rate * (1.0 + 8.0 * (double)D);
    } else {
      g.traders++;
      events += rate * (a.kind == A_NOISE ? 5.5 : 4.0);
    }
  }
  for (const Grp& g : groups) {
    const double rate = 1e9 / (double)g.interval;
    int B = g.traders;  // the requests that are sent at once: a quote request each, then the market makers' cancels and quotes
    for (const auto& m : g.mm) B += (int)std::min<int64_t>(4 * m.first, 512) * m.second;
    int offset = g.traders;
    Count ex, mm;
    for (const auto& m : g.mm) {  // each distinct depth: the replies at one of its market makers
      const int cnt = (int)std::min<int64_t>(4 * m.first, 512);
      Count e2, m2;
      burst(c, c.compute_delay > 0 ? B : cnt, c.compute_delay > 0 ? offset : 0, cnt, e2, m2);
      m2.waiters *= m.second;
      m2.depth *= m.second;
      mm += m2;
      offset += cnt * m.second;
    }
    if (c.compute_delay > 0) {  // the exchange's own bounces: all the group's requests, and the traders' orders after their quote replies
      Count none;
      burst(c, B, 0, 0, ex, none);
      if (g.traders > 1) burst(c, g.traders, 0, 0, ex, none);
    }
    total.waiters += rate * (ex.waiters + mm.waiters);
    total.depth += rate * (ex.depth + mm.depth);
    int together = 0;
    for (const auto& m : g.mm) together += m.second;
    mm_together = std::max(mm_together, together);
  }
  Estimate e;
  e.mm_together = mm_together;
  if (events > 0) {
    e.waiters = total.waiters / events;
    e.depth = total.depth / events;
  }
  return e;
}
// Thresholds. On ~175 public, fuzz-corpus and synthetic scenarios (PMU cycles of both loops, Zen 3) run_park gained 4.7% of the loop's
// cycles on average where the estimate is above them (worst +4%), and the two loops are equal within noise below (run_park would
// lose 3 to 8% in the scenarios with many shallow parked sets and no waiters at a first bounce, which the estimate keeps at ~0).
static constexpr double WAITERS_MIN = 0.10, DEPTH_MIN = 12.0;
// run_park keeps one release candidate per agent with parked messages in an array that it scans: for a handful of market makers
// that is cheap, but with 32 or 64 market makers waking together it costs run_park 4 to 16% (measured), so those stay on run_m12
static constexpr int MM_TOGETHER_MAX = 24;
}  // namespace park_rule

inline bool park_policy(const Config& c) {
  if (c.lat == L_DETERMINISTIC) return false;
  const park_rule::Estimate e = park_rule::estimate(c);
  return (e.waiters >= park_rule::WAITERS_MIN || e.depth >= park_rule::DEPTH_MIN) && e.mm_together <= park_rule::MM_TOGETHER_MAX;
}

}  // namespace t3
