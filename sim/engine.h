#pragma once
// Faithful C++ port of the Track-3 reference: pinned ABIDES (kernel, ExchangeAgent, OrderBook,
// SparseMeanRevertingOracle) + baselines/abides_fork (config, agents, latency model, trace and
// message-ledger extraction). Every RNG draw, message id, order id and queue tie-break follows the
// Python reference.
#include <sys/mman.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>

#if defined(T3_PROF_FEED) || defined(T3_PROF_TIMED)
#include <valgrind/cachegrind.h>
#endif

#include "config.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
static inline void cpu_relax() { _mm_pause(); }
#elif defined(__aarch64__)
static inline void cpu_relax() { asm volatile("yield"); }
#else
static inline void cpu_relax() {}
#endif

namespace t3 {

// hot functions: the distro's default -fstack-protector-strong adds a canary store/check to any function with an
// address-taken temporary (vector insert/erase), which costs instructions on every call
#define T3_NOSSP __attribute__((no_stack_protector))

// Append-only output stream in one reserved (untouched) address range. The helper thread pre-faults
// pages just ahead of the write cursor while the simulation runs, so the simulation thread rarely
// takes a page fault. Pre-faulting uses an atomic "+0" on each page, which can never lose a
// concurrent write by the simulation thread.
template <class T>
struct Stream {
  static constexpr size_t RESERVE = size_t(4) << 30;  // bytes of address space (not memory)
  T* base = nullptr;
  T* cur = nullptr;  // write cursor (next free element)
  T* lim = nullptr;  // next checkpoint: capacity check + cursor publication, one compare per push
  size_t cap = 0;
  size_t lead = 0;  // elements in front of `base` (inside the mapping): scratch rows that are written but never read
  // Without a helper thread the pages ahead of the cursor are populated in batches at the checkpoints (one madvise per
  // batch instead of a page fault per 4 KiB inside the run). Off when a helper pre-faults; disabled for good when the
  // kernel refuses (MADV_POPULATE_WRITE needs Linux 5.14).
  size_t popd = 0;  // bytes from base already populated or touched
  size_t pop_chunk = size_t(1) << 20;
  bool pop_on = false;
  // sink mode (a ledger nobody reads): at a checkpoint past sink_rows the cursor goes back to base, so the rows stay in a small
  // cache-resident window instead of growing memory; sunk counts the rows written over
  size_t sink_rows = 0, sunk = 0;
  alignas(64) std::atomic<size_t> pub{0};  // cursor published to the helper (elements)
  void (*partner)(void*) = nullptr;  // run at every checkpoint: serves a stream written without per-row checks (take_nc)
  void* partner_arg = nullptr;
  alignas(64) size_t touched = 0;          // helper-owned (own line: the simulation stores to the lines around): bytes pre-faulted
  size_t ahead = size_t(1) << 20;          // helper-owned: pre-fault distance past the cursor (bytes)
  char* hbase = nullptr;                   // helper-owned copy of base
  explicit Stream(size_t lead_ = 0) : lead(lead_) {
    void* p = mmap(nullptr, RESERVE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    base = cur = lim = (T*)p + lead;
    cap = RESERVE / sizeof(T) - lead;
#ifdef MADV_NOHUGEPAGE
    // plain 4 KiB pages: transparent huge pages can stall a first touch on zeroing/compaction
    madvise(p, RESERVE, MADV_NOHUGEPAGE);
#endif
  }
  Stream(const Stream&) = delete;
  // setup: fault in a fixed initial working set (independent of the run's size); the rest of the range is
  // faulted inside the timed run (by the helper thread ahead of the cursor, or by the simulation itself)
  void preface(size_t bytes) {
    volatile char* c = (volatile char*)base;
    for (size_t i = 0; i < bytes; i += 4096) c[i] = 0;
    touched = bytes;
    popd = bytes;
  }
  ~Stream() { munmap(base - lead, RESERVE); }
  __attribute__((noinline)) void populate_more() {
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
    if (popd + pop_chunk > RESERVE || madvise((char*)(base - lead) + popd, pop_chunk, MADV_POPULATE_WRITE) != 0) {
      pop_on = false;
      return;
    }
    popd += pop_chunk;
  }
  __attribute__((noinline)) void checkpoint() {
    size_t n = (size_t)(cur - base);
    if (sink_rows && n >= sink_rows) {
      sunk += n;
      cur = base;
      n = 0;
    }
    if (n > cap) throw std::runtime_error(T3_MSG("output stream capacity exceeded", "E40"));
    if (pop_on && n * sizeof(T) + (size_t(256) << 10) > popd) populate_more();
#ifdef PFW_BYTES
    { const char* pc = (const char*)cur; for (size_t k = 0; k < (size_t)PFW_BYTES; k += 64) __builtin_prefetch(pc + k, 1); }
#endif
    pub.store(n, std::memory_order_release);  // rows below n-1 are complete (the background ledger encoder reads them)
    lim = base + std::min((n | 127) + 1, cap + 1);
    if (partner) partner(partner_arg);
  }
  // reserve k elements; returns the first
  inline T* take(size_t k) {
    T* p = cur;
    cur = p + k;
    if (__builtin_expect(cur >= lim, 0)) checkpoint();
    return p;
  }
  // reserve k elements without the checkpoint test: the owner guarantees a checkpoint within a bounded number of elements
  // (cap leaves room for them) and the checked take(k) of the bulk writers re-syncs `lim`
  inline T* take_nc(size_t k) {
    T* p = cur;
    cur = p + k;
    return p;
  }
  inline T& push() { return *take(1); }
  inline T& operator[](size_t i) { return base[i]; }
  inline const T& operator[](size_t i) const { return base[i]; }
  size_t size() const { return (size_t)(cur - base); }
  // helper side: pre-fault up to AHEAD bytes past the cursor, at most `budget` pages per call
  bool prefault(size_t budget) {
    size_t want = pub.load(std::memory_order_relaxed) * sizeof(T) + ahead;
    if (want > RESERVE) want = RESERVE;
    if (touched >= want) return false;
    char* c = hbase ? hbase : (hbase = (char*)base);
    if (touched < want - ahead + (size_t(128) << 10)) budget *= 4;  // the cursor is close behind: catch up faster
    for (size_t k = 0; k < budget && touched < want; k++, touched += 4096)
      __atomic_fetch_add((uint64_t*)(c + touched), 0, __ATOMIC_RELAXED);
    return true;
  }
};

// The latency model's RandomState stream is consumed by exactly one draw per non-self message, in
// send order, independent of simulation state; a helper thread computes it ahead into a ring.
// The helper only starts drawing once the timed run has started.
struct LatencyFeed {
  static constexpr uint64_t CAP = 16384, MASK = CAP - 1, AHEAD = 8192;
#ifndef CHORE_GAP
#define CHORE_GAP 100000
#endif
  int64_t* ring = nullptr;
  alignas(64) std::atomic<uint64_t> produced{0};
  alignas(64) std::atomic<uint64_t> consumed{0};
  std::atomic<uint32_t> nq_pub{0};  // noise requests published (same line as `consumed`, which the consumer stores to at every checkpoint anyway)
  alignas(64) std::atomic<int> state{0};  // 0 = wait, 1 = go, 2 = stop
  alignas(64) uint64_t cons = 0;
  // requests of the noise decisions drawn ahead (SimT::NEnt): the agent ordinals of the entries the simulation has used up, written into the
  // consumer's own lines and published at the checkpoints (a store to a line the helper reads costs the consumer a store stall), after
  // padding the current line so that the helper never reads a line the consumer still writes
  uint16_t* nreq = nullptr;
  uint32_t nreq_mask = 0, nreq_n = 0, nreq_pubd = 0;
  uint64_t avail = 0, lim = 0;  // lim: the consumer's next checkpoint (<= avail); the consumer's position is lim whenever refill() runs
  const int64_t* rp = nullptr;
  const int64_t* rend = nullptr;
  std::thread th;
  alignas(64) nrng::RandomState rs;  // the helper writes it all the time: not on the consumer's line
  LatModel model = L_DETERMINISTIC;
  double mu = 0, sigma = 0, lo = 0, hi = 0, alpha = 0, base = 1;

  // one draw (also the tail of a batch, and the non-SSE2 build)
  inline int64_t draw1() {
    double v;
    switch (model) {
      case L_LOGNORMAL: v = rs.lognormal(mu, sigma); break;
      case L_UNIFORM: v = rs.uniform(lo, hi); break;
      default: v = base * (1.0 + rs.pareto(alpha)); break;
    }
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return py_round(v);
  }
#if defined(__SSE2__)
  // m <= 256 draws, m even. With |lo|, |hi| < 2^51 the clamped values are below 2^51 too, and adding 1.5 * 2^52 leaves the
  // rounded integer (nearest even, what py_round gives) in the low mantissa bits. A NaN passes through min/max unchanged,
  // like the scalar compares. Anything else takes the scalar path.
  void fill_even(int64_t* dst, int m) {
    if (!(std::fabs(lo) < 2251799813685248.0 && std::fabs(hi) < 2251799813685248.0)) {
      for (int i = 0; i < m; i++) dst[i] = draw1();
      return;
    }
    const __m128d vlo = _mm_set1_pd(lo), vhi = _mm_set1_pd(hi), mg = _mm_set1_pd(6755399441055744.0);
    const __m128i mgb = _mm_castpd_si128(mg);
    // clamp, round and store two values
    auto put = [&](int i, __m128d x) {
      x = _mm_min_pd(vhi, _mm_max_pd(vlo, x));
      _mm_storeu_si128((__m128i*)(dst + i), _mm_sub_epi64(_mm_castpd_si128(_mm_add_pd(x, mg)), mgb));
    };
    alignas(16) double v[264];
    if (model == L_LOGNORMAL) {
      rs.gauss_fill(v, m, mu, sigma);
      nrng::RandomState::map_exp(v, v, m);
      for (int i = 0; i < m; i += 2) put(i, _mm_loadu_pd(v + i));
    } else if (model == L_UNIFORM) {
      const __m128d vr = _mm_set1_pd(hi - lo);
      rs.each_u01x2(m, [&](int i, __m128d u) { put(i, _mm_add_pd(vlo, _mm_mul_pd(vr, u))); });
    } else {
      const __m128d one = _mm_set1_pd(1.0), sign = _mm_set1_pd(-0.0), va = _mm_set1_pd(alpha), vb = _mm_set1_pd(base);
      rs.each_u01x2(m, [&](int i, __m128d u) { _mm_storeu_pd(v + i, _mm_sub_pd(one, u)); });
      nrng::RandomState::map_log(v, v, m);
      for (int i = 0; i < m; i += 2) _mm_storeu_pd(v + i, _mm_div_pd(_mm_xor_pd(_mm_loadu_pd(v + i), sign), va));  // standard_exponential() / alpha
      nrng::RandomState::map_exp(v, v, m);
      for (int i = 0; i < m; i += 2) put(i, _mm_mul_pd(vb, _mm_add_pd(one, _mm_sub_pd(_mm_loadu_pd(v + i), one))));
    }
  }
  // n draws, in stream order
  void fill(int64_t* dst, uint64_t n) {
    while (n) {
      const int b = n > 256 ? 256 : (int)n;
      const int e = b & ~1;
      if (e) fill_even(dst, e);
      if (b & 1) dst[e] = draw1();
      dst += b;
      n -= (uint64_t)b;
    }
  }
#else
  void fill(int64_t* dst, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) dst[i] = draw1();
  }
#endif
  // draws into the ring at stream position p (wraps)
  inline void fill_ring(uint64_t p, uint64_t n) {
    const uint64_t first = CAP - (p & MASK);
    if (n > first) {
      fill(ring + (p & MASK), first);
      fill(ring, n - first);
    } else {
      fill(ring + (p & MASK), n);
    }
  }
  // one at a time, from small batches (the bench entry; the producer and the inline refill batch straight into the ring)
  int dpos = 64;
  inline int64_t draw() {
    if (dpos == 64) {
      if (!ring) ring = (int64_t*)malloc(sizeof(int64_t) * CAP);
      fill(ring, 64);
      dpos = 0;
    }
    return ring[dpos++];
  }
  std::vector<std::function<bool()>> chores;  // helper work besides latency draws
  bool draws = true;
  // nothing to do: back off instead of burning the container's CPU quota (the ring holds ~1 ms of draws)
  void idle() {
    struct timespec ts{0, 20000};
    nanosleep(&ts, nullptr);
  }
  // The chores read lines the simulation stores to (the streams' published cursors): every read costs it a coherence miss at its next
  // store, so they run at a bounded rate (the streams are pre-faulted far ahead of their cursors)
  uint64_t chore_tsc = 0;
  bool do_chores() {
#if defined(__x86_64__)
    const uint64_t now = __builtin_ia32_rdtsc();
    if (now - chore_tsc < CHORE_GAP) return false;
    chore_tsc = now;
#endif
    bool any = false;
    for (auto& f : chores) any |= f();
    return any;
  }
  std::atomic<int> ready{0};
  void producer() {
    ready.store(1, std::memory_order_release);  // running; waits (drawing nothing) for the timed start
    while (state.load(std::memory_order_acquire) == 0) cpu_relax();
    uint64_t p = 0;
    uint64_t chunk = 8;
    while (state.load(std::memory_order_relaxed) == 1) {
      uint64_t c = consumed.load(std::memory_order_acquire);
      if (!draws || p - c + chunk > AHEAD) {
        // the latency ring is far enough ahead: pre-fault output memory instead
        if (!do_chores()) idle();
        continue;
      }
      if (p - c > 1024 && chunk >= 64) do_chores();
      fill_ring(p, chunk);
      p += chunk;
      produced.store(p, std::memory_order_release);
      if (chunk < 256) chunk <<= 1;
    }
  }
  void start(const nrng::RandomState& src, const Config& c, bool with_draws) {
    draws = with_draws;
    rs = src;
    model = c.lat;
    mu = c.lat_mu; sigma = c.lat_sigma; lo = c.lat_min; hi = c.lat_max; alpha = c.lat_alpha;
    base = c.lat_min > 0 ? c.lat_min : 1.0;
    ring = (int64_t*)malloc(sizeof(int64_t) * CAP);
    memset(ring, 0, sizeof(int64_t) * CAP);  // fixed-size work buffer: allocate it fully at setup
    th = std::thread([this] {
#if defined(__linux__)
#ifndef T3_SEALED
      pthread_setname_np(pthread_self(), "t3helper");
#endif
#endif
      producer();
    });
    while (!ready.load(std::memory_order_acquire)) cpu_relax();  // thread start-up is setup, not run
  }
  void go() { state.store(1, std::memory_order_release); }
  bool detach_at_stop = false;  // the process ends right after the run: do not wait for the helper to notice (it only reads, and nothing is freed)
  void stop() {
    if (th.joinable()) {
      state.store(2, std::memory_order_release);
      if (detach_at_stop) {
        th.detach();
        return;
      }
      th.join();
    }
    free(ring);
    ring = nullptr;
  }
  bool inline_mode = false;  // no helper thread: the consumer refills the ring itself, a few draws at a time
  void start_inline(const nrng::RandomState& src, const Config& c) {
    rs = src;
    model = c.lat;
    mu = c.lat_mu; sigma = c.lat_sigma; lo = c.lat_min; hi = c.lat_max; alpha = c.lat_alpha;
    base = c.lat_min > 0 ? c.lat_min : 1.0;
    ring = (int64_t*)malloc(sizeof(int64_t) * CAP);
    memset(ring, 0, sizeof(int64_t) * CAP);
    inline_mode = true;
  }
  // consumer checkpoint, every <= 256 draws: publish the position, then obtain more draws if the ring ran dry
  __attribute__((noinline)) void publish_noise() {
    while (nreq_n & 31) nreq[nreq_n++ & nreq_mask] = 0xFFFF;
    nreq_pubd = nreq_n;
    nq_pub.store(nreq_n, std::memory_order_release);
  }
  __attribute__((noinline)) void refill() {
    cons = lim;
    consumed.store(cons, std::memory_order_release);
    if (nreq && nreq_n != nreq_pubd) publish_noise();
    if (cons == avail) {
      if (inline_mode) {
#ifdef T3_PROF_FEED
        CACHEGRIND_STOP_INSTRUMENTATION;  // profiling aid only: the draws stand for the helper thread's work
        constexpr uint64_t B = 8192;      // (toggling the instrumentation is expensive: few, large batches)
#else
        constexpr uint64_t B = 64;
#endif
        fill_ring(avail, B);
        avail += B;
#ifdef T3_PROF_FEED
        CACHEGRIND_START_INSTRUMENTATION;
#endif
      } else {
        while (true) {
          avail = produced.load(std::memory_order_acquire);
          if (cons != avail) break;
          cpu_relax();
        }
      }
    } else if (!inline_mode) {
      avail = produced.load(std::memory_order_acquire);
    }
    lim = std::min(avail, (cons | 255) + 1);
    rp = ring + (cons & MASK);
    rend = rp + (lim - cons);
  }
  __attribute__((always_inline)) inline int64_t next() {
    if (__builtin_expect(rp == rend, 0)) refill();
    return *rp++;
  }
};

// Final trace row (abides_fork.trace column order) and message-ledger row (delivery order = seq).
struct TraceRow {  // 32 bytes: order ids fit 32 bits (< trace capacity), agent ids 15 bits (the wide instance keeps bits 16.. in side >> 1)
  int64_t t, price, size;
  int32_t oid;
  int16_t agent;
  uint8_t code, side;
};
// A Msg starts with its LedgerRow, so recording a delivery is one 40-byte copy. For a wakeup the row has
// latency -1 and a stale causal/oid; for message types below M_LIMIT oid is stale (readers treat both as null).
struct LedgerRow {
  int64_t t_recv, latency;  // latency < 0 marks a wakeup (t_send null)
  uint32_t id, causal, oid;
  uint16_t src, dst;
  uint8_t type, oside, hb, ha, closed;  // oside/hb/ha/closed: message payload kept in the row's padding
  uint8_t srch, dsth;  // the wide instance: bits 16.. of src / dst (padding, never written or read, in the others)
};
static constexpr uint32_t NUL32 = 0xFFFFFFFFu;

// G: deterministic latency. The same code is compiled twice (G = false: general scenarios; G = true: deterministic
// latency); main.cpp picks the instantiation from the config when the Sim is created so neither pays for the other's mode checks.
// NL: a single run whose message ledger nobody reads (scenario family throughput-scale): no ledger row is written per delivery and
// the message fields only the ledger reads (latency, causal parent, src/dst) are not stored; the deliveries are counted instead.
// W: 32768 agents or more (the other instances keep agent ids in 15 bits). Agent ids up to 23 bits: bits 16.. of a trace row's agent
// go to side >> 1 and of a ledger row's src/dst to srch/dsth, and the queue key's sender/recipient fields hold one code per pair (mkkey).
template <bool G, bool NL = false, bool W = false>
struct SimT {
  static_assert(!(G && NL), "the no-ledger instance is for the general latency models");
  static_assert(!(NL && W), "the wide instances keep the ledger instance's layout");
  static constexpr bool NOLEDGER = NL;
  static constexpr bool WIDE = W;
  static constexpr bool TA = !G;  // the trace-ahead hooks (TraceAhead in main.cpp) exist: the no-ledger and the ledger instance of the general latency models
  // ------------------------------------------------------------ message pool
  // order fields: oprice/oqty (oid, oside in the row; an ORDER_EXECUTED carries the fill price in oprice); the order's
  // agent is the message's src (LIMIT, CANCEL) or dst (acks). QSPREAD_RESP: bid = oprice, ask = oqty. spare keeps the line full.
  struct Msg : LedgerRow {
    int64_t oprice, oqty, spare;
  };
  static_assert(sizeof(Msg) == 64, "Msg is one cache line");
  Stream<Msg> pool;  // stable addresses: a delivered message is read in place
  static constexpr uint32_t NOSLOT = 0xFFFFFFFFu;
  uint32_t free_head = NOSLOT;  // intrusive freelist: a free slot's `causal` holds the next free slot
  inline uint32_t new_slot() {
    if (free_head != NOSLOT) {
      uint32_t s = free_head;
      free_head = pool[s].causal;
      return s;
    }
    return (uint32_t)(pool.take(1) - pool.base);
  }
  inline void free_slot(uint32_t s) {
    pool[s].causal = free_head;
    free_head = s;
  }
  inline uint32_t alloc_msg(uint8_t type) {
    uint32_t s = new_slot();
    Msg& m = pool[s];
    m.type = type;
    m.id = (uint32_t)msg_counter++;
    return s;
  }

  // The request being delivered may become its own reply (same slot, new message id): the ledger row was copied
  // at delivery and the queue no longer references the slot. Handlers return true when they reused it.
  inline void remsg(uint32_t s, uint8_t type) {
    Msg& m = pool[s];
    m.type = type;
    m.id = (uint32_t)msg_counter++;
  }

  // ------------------------------------------------------------ event queue
  // Python: PriorityQueue of (time, (sender_id, recipient_id, message)); Message.__lt__ on id.
  // Key k = sender<<49 | recipient<<34 | msg_id. A message popped while its recipient is still
  // "in the future" is parked on the recipient instead of being requeued one by one; the parked set
  // is represented in the queue by one release entry carrying the set's minimum key at the
  // recipient's next free time. Requeues have no observable effect in the reference, so the
  // delivery sequence is identical.
  static constexpr uint32_t REL = 0x80000000u;
  std::vector<int> agent_lane;
  struct Park {
    std::vector<std::pair<uint64_t, uint32_t>> h;  // parked messages, descending by key
    uint32_t gen = 0;
    bool relq = false;  // the agent's release entry is queued in the wheel (node RELNODE(agent)), as `gen` or an older generation
  };
  std::vector<Park> park;
  static constexpr uint32_t RELNODE(int32_t r) { return (uint32_t)(EventQueue::MAX_SLOTS - 1 - (size_t)r); }
  static inline uint64_t mkkey(int32_t s, int32_t r, uint64_t id) {
    if constexpr (W) return (kpair(s, r) << 32) | id;
    return ((uint64_t)s << 49) | ((uint64_t)r << 34) | id;
  }
  // W: the only (sender, recipient) pairs are (0, r) (the exchange's messages and wakeup), (s, 0) and (s, s) (a trader's messages and
  // wakeup), so one code per pair in the order of the pairs fits beside the 32-bit message id: r, WB + 2s, WB + 2s + 1
  static constexpr uint64_t WB = uint64_t(1) << 23;
  static inline uint64_t kpair(int32_t s, int32_t r) { return s == 0 ? (uint64_t)r : WB + 2 * (uint64_t)s + (uint64_t)(r != 0); }
  static inline int32_t key_s(uint64_t k) {
    if constexpr (W) {
      const uint64_t v = k >> 32;
      return v < WB ? 0 : (int32_t)((v - WB) >> 1);
    }
    return (int32_t)(k >> 49);
  }
  static inline int32_t key_r(uint64_t k) {
    if constexpr (W) {
      const uint64_t v = k >> 32;
      return v < WB ? (int32_t)v : ((v & 1) ? (int32_t)((v - WB) >> 1) : 0);
    }
    return (int32_t)((k >> 34) & 0x7FFF);
  }
  __attribute__((always_inline)) inline void qput(int64_t t, int32_t s, int32_t r, uint32_t slot) {
    const QItem qi{(uint64_t)t, mkkey(s, r, pool[slot].id), slot, (uint32_t)r};
    if constexpr (G) gqp->push_g(q, qi);
    else q.push(qi, (uint64_t)(t - current_time));
  }
  // A release entry normally lives in the wheel (the agent's reserved node, recipient field = the phantom agent n_agents whose
  // free time is infinite, so the pop takes the busy path and recognises it there); when that node is occupied (by a superseded
  // older entry) or the time is far, it goes to the heap as a REL-flagged item.
  __attribute__((noinline)) void push_release(int32_t r) {
    Park& P = park[(size_t)r];
    const int64_t at = agent_time[(size_t)r];
    if constexpr (!G) {
      const uint64_t dt = (uint64_t)(at - current_time);
      if (!P.relq && dt < q.lim_dt) {
        P.relq = true;
        q.nodes[RELNODE(r)].pad2 = P.gen;
        q.push(QItem{(uint64_t)at, P.h.back().first, RELNODE(r), (uint32_t)n_agents}, dt);
        return;
      }
    }
    q.push_heap(QItem{(uint64_t)at, P.h.back().first, REL | (uint32_t)r, P.gen});
  }
  // The parked set is a short array in descending key order (the minimum at the back): a pop is a pop_back, an insertion
  // shifts the larger keys up one place (a sift in a binary heap mispredicts at every level)
  __attribute__((noinline)) static void park_push(Park& P, uint64_t k, uint32_t slot) {
    auto& h = P.h;
    h.emplace_back(k, slot);
    size_t j = h.size() - 1;
    while (j > 0 && h[j - 1].first < k) {
      h[j] = h[j - 1];
      j--;
    }
    h[j] = std::pair<uint64_t, uint32_t>(k, slot);
  }
  static inline std::pair<uint64_t, uint32_t> park_pop(Park& P) {
    auto top = P.h.back();
    P.h.pop_back();
    return top;
  }

  // ------------------------------------------------------------ state
  Config cfg;
  int n_agents = 0;
  int64_t mkt_open = 0, mkt_close = 0, start_time = 0, stop_time = 0;
  int64_t current_time = 0;
  int64_t msg_counter = 1, order_counter = 0;
  uint32_t causal_uid = NUL32;
  std::vector<int64_t> agent_time;
  int64_t xdelay = 50;  // computation delay of the exchange (agent 0); every trader's is the constant TDELAY
  static constexpr int64_t TDELAY = 50;  // default_computation_delay
#ifdef T3_DIAG_PHASES
  size_t diag_n = 0, diag_next = 1, diag_k = 0;
  std::chrono::steady_clock::time_point diag_t[16], diag_t0;
#endif

  nrng::RandomState global_rs, oracle_rs, latency_rs;
  std::vector<nrng::RandomState> agent_rs;
  LatencyFeed feed;
  bool helper_allowed = true;
  double est_msgs = 0, helper_threshold = 2000;
  bool use_feed = false, use_helper = false;
  size_t cpus_available = 1;

  // The first cache line holds everything a noise agent's wakeup / quote response / order reads and writes
  // (flags, known best prices, interval, size mean/std, reference price, offset mask), the rest of the config follows.
  struct alignas(64) Agent {
    AgentKind kind = A_EXCHANGE;
    bool first_wake = true, hours = false, mkt_closed = false;
    bool awaiting_spread = false;
    bool kb_has = false, ka_has = false;
    bool nq_on = false;      // noise agents whose decisions are drawn ahead by the helper thread (see NEnt)
    union {
      struct {
        uint32_t noff_mask;  // noise agents: rejection mask of randint(0, price_offset + 1), 0 when not the plain 32-bit case
        uint32_t noff_r;     // ... and its bound, price_offset
      };
      struct {
        uint32_t nq_base;    // nq_on: the agent's ordinal among the drawn-ahead noise agents
        uint32_t nq_n;       // ... and the number of decisions taken so far
      };
    };
    int64_t kb = 0, ka = 0;
    AgentCfg c;
    struct DEnt {
      int64_t oid, price, qty;  // oid < 0: deleted entry
      uint8_t side;
    };
    struct DVec {  // vector of DEnt with directly settable size (bulk appends write the entries in place)
      DEnt* p = nullptr;  // one scratch entry sits in front of p[0] (p[-1]): a dictionary lookup that finds nothing can address it
      size_t n = 0, cap = 0;
      static DEnt* alloc(size_t c) {
        DEnt* r = (DEnt*)malloc((c + 1) * sizeof(DEnt));
        r->oid = -1;
        return r + 1;
      }
      DVec() = default;
      DVec(const DVec& o) {
        if (o.cap) p = alloc(o.cap);
        if (o.n) memcpy((void*)p, (const void*)o.p, o.n * sizeof(DEnt));
        n = o.n; cap = o.cap;
      }
      DVec& operator=(const DVec& o) {
        DVec t(o);
        std::swap(p, t.p); std::swap(n, t.n); std::swap(cap, t.cap);
        return *this;
      }
      ~DVec() { if (p) free(p - 1); }
      size_t size() const { return n; }
      DEnt* data() const { return p; }
      DEnt& operator[](size_t i) const { return p[i]; }
      void ensure(size_t extra) {  // room for `extra` more entries
        if (__builtin_expect(cap - n < extra, 0)) {
          size_t nc = std::max<size_t>(n + extra, cap * 2);
          DEnt* q = alloc(nc);
          if (n) memcpy((void*)q, (const void*)p, n * sizeof(DEnt));
          if (p) free(p - 1);
          p = q; cap = nc;
        }
      }
    };
    DVec dict;
    uint32_t dict_live = 0;
    uint32_t dict_head = 0;  // entries below it are all deleted
    std::vector<double> hist;
    uint32_t hist_head = 0, hist_n = 0;
  };
  static_assert(sizeof(Agent) % 64 == 0, "Agent: whole cache lines");
  std::vector<Agent> agents;
  std::vector<int32_t> close_subs;

  struct BO {
    int64_t oid, qty, price;
    int32_t agent, prev, next;
    uint8_t side;
  };
  // One book side's price levels in ascending key order, best at the back. A level's key is its price xor a side constant: bids
  // price ^ 2^63, asks price ^ (2^63 - 1) (= 2^63 - 1 - price), so for both sides a higher unsigned key is a better level, and an
  // incoming order of key k crosses the opposite side's best level of key o exactly when k >= ~o (no overflow, nothing to branch on).
  struct Level {
    uint64_t key;
    int64_t total;
    int32_t head, tail;
  };
  static constexpr uint64_t KB = 0x8000000000000000ULL;
  static constexpr int LK = 8;  // levels scanned branch-free from the best
  // LK sentinel levels (key 0: below every key; head 0, so trim() stops there) sit in front of the first level, so a scan of the best
  // LK levels needs no bounds test. A level that loses its last order stays in the array as an empty level (head -1) and is reused by
  // the next order at its price; the best level is never empty (trim() drops empty levels from the back), so erasing a cancelled
  // order's level, whether the price exists, and shifting the levels behind it are not decisions the code has to make.
  struct LevVec {
    Level* raw = nullptr;
    Level* b = nullptr;  // first level
    Level* e = nullptr;  // one past the best level
    Level* lim = nullptr;
    LevVec() = default;
    LevVec(const LevVec&) = delete;
    ~LevVec() { free(raw); }
    void reserve(size_t c, bool touch = false) {
      const size_t n = (size_t)(e - b);
      if (raw && c <= (size_t)(lim - b)) return;
      Level* r = (Level*)malloc((c + LK) * sizeof(Level));
      if (touch) memset((void*)r, 0, (c + LK) * sizeof(Level));
      for (int i = 0; i < LK; i++) r[i] = Level{0, 0, 0, 0};
      if (n) memcpy((void*)(r + LK), (const void*)b, n * sizeof(Level));
      free(raw);
      raw = r; b = r + LK; e = b + n; lim = b + c;
    }
    bool empty() const { return e == b; }
    Level& back() const { return e[-1]; }
    void trim() {
      Level* x = e;
      while (x[-1].head < 0) x--;
      e = x;
    }
  };
  std::vector<BO> bo;
  int32_t bo_free_head = -1;  // intrusive free list of bo slots (a free slot's `next` links to the next free one)
  LevVec lv[2];  // [0] bids, [1] asks

  PyNum o_pt;
  int64_t o_pv = 0;
  PyNum o_mst;
  double o_msv = 0;
  bool o_jump_pending = false;
  int64_t o_jump_time = 0;

  // ------------------------------------------------------------ outputs
  Stream<LedgerRow> led;
  Stream<TraceRow> trace{1};  // one scratch row in front of the first (see emit_order_t)
  int64_t grp_t = INT64_MIN;

  // ------------------------------------------------------------ build_config analogue
  void setup(const Config& c) {
    cfg = c;
    // the trace is written without per-row checks: its checkpoint runs with the ledger's (every 128 deliveries, <= ~4 rows each
    // apart from market-maker ladders, which use the checked take); keep that much room below the capacity
    trace.cap -= size_t(1) << 20;
    if constexpr (!NL) {
      led.partner = [](void* t) { static_cast<SimT*>(t)->ledger_checkpoint(); };
      led.partner_arg = this;
    } else {
      ck = 128; ndeliv = 0;
      oi_touched = dpa_touched = size_t(16) * 4096;
      oi_base = dpa_base = nullptr;
    }
    if constexpr (TA) {
      scan_ctr = 0; scan_gap = 64; fin_last = 0; rm_head = 0;
      rm_window = (int64_t)std::min(1e15, c.lat_max) + c.ack_delay + c.compute_delay + 2 * 50 + 1000000;
    }
    n_agents = 1 + (int)cfg.agents.size();
    if (n_agents >= (W ? (int)WB : 32768)) throw std::runtime_error(T3_MSG("too many agents", "E41"));
    if (!cfg.has_latency_cfg) throw std::runtime_error(T3_MSG("scenario without latency_config is not supported", "E42"));
    mkt_open = DATE_NS + H930;
    mkt_close = mkt_open + cfg.horizon_ns;
    start_time = DATE_NS;
    stop_time = mkt_close + ONE_S;

    global_rs.seed((uint32_t)cfg.seed);
    oracle_rs.seed((uint32_t)global_rs.randint_seed_u32());
    o_pt = PyNum::I(mkt_open);
    o_pv = cfg.reference_price;
    {
      double ms_time_delta = global_rs.exponential(1.0 / cfg.ms_lambda);
      o_mst = PyNum::F((double)mkt_open + ms_time_delta);
      double msv = oracle_rs.normal(cfg.ms_mean, std::sqrt((double)cfg.ms_var));
      o_msv = oracle_rs.randint(0, 2) == 0 ? msv : -msv;
    }
    if (cfg.has_jump) {
      o_jump_pending = true;
      o_jump_time = mkt_open + cfg.jump_time_rel;
    }
    agent_rs.resize((size_t)n_agents);
    for (int a = 0; a < n_agents; a++) agent_rs[(size_t)a].seed((uint32_t)global_rs.randint_seed_u32());
    latency_rs.seed((uint32_t)global_rs.randint_seed_u32());
    (void)global_rs.randint_seed_u32();  // random_state_kernel (unused by the run)

    agents.assign((size_t)n_agents, Agent());
    agents[0].kind = A_EXCHANGE;
    for (int a = 1; a < n_agents; a++) {
      Agent& ag = agents[(size_t)a];
      ag.kind = cfg.agents[(size_t)a - 1].kind;
      ag.c = cfg.agents[(size_t)a - 1];
      if (ag.kind == A_MOMENTUM) ag.hist.assign((size_t)ag.c.lookback + 1, 0.0);
      if (ag.kind == A_NOISE && ag.c.price_offset >= 1 && ag.c.price_offset <= 0xFFFFFFFELL) {
        ag.noff_mask = 0xFFFFFFFFU >> __builtin_clz((uint32_t)ag.c.price_offset);
        ag.noff_r = (uint32_t)ag.c.price_offset;
      }
    }
    agent_time.assign((size_t)n_agents + 1, start_time);
    agent_time[(size_t)n_agents] = INT64_MAX;  // the phantom agent that wheel release entries are addressed to: always busy
    park.assign((size_t)n_agents, Park());
    {
      // wheel window (16384 buckets) should cover ~8x the 95th-percentile message latency
      double p95;
      switch (cfg.lat) {
        case L_LOGNORMAL: p95 = std::exp(cfg.lat_mu + 1.645 * cfg.lat_sigma); break;
        case L_UNIFORM: p95 = cfg.lat_max; break;
        case L_PARETO: p95 = (cfg.lat_min > 0 ? cfg.lat_min : 1.0) * std::pow(20.0, 1.0 / cfg.lat_alpha); break;
        default: p95 = cfg.lat_mean;
      }
      p95 = std::min(std::max(p95, cfg.lat_min), cfg.lat_max) + (double)(cfg.ack_delay + cfg.compute_delay) + 50.0;
      // bucket width ~ typical latency / 32..64: smaller buckets collide less (shorter sorted lists), larger ones make
      // the occupied-bucket scan jump further
      double lscale;
      switch (cfg.lat) {
        case L_UNIFORM: lscale = 0.5 * (cfg.lat_min + cfg.lat_max); break;
        case L_PARETO: lscale = 2.0 * (cfg.lat_min > 0 ? cfg.lat_min : 1.0); break;
        default: lscale = cfg.lat_mean;
      }
      int shift = 2;
      while (shift < 24 && (double)(32ull << (shift + 1)) <= lscale) shift++;
      while (shift < 24 && (double)(16384ull << shift) < 8.0 * p95) shift++;
      {  // finer buckets when many messages arrive close together: agents with the same interval wake at the same instants, so the messages of a
         // group (a market maker's wakeup sends ~4 per level) are in flight together, spread over the latency model's central range; the
         // bucket width is that range divided by (4 x the group's messages), so a push rarely finds its bucket occupied
        double burst = 1;
        std::vector<std::pair<int64_t, double>> grp;
        for (int a = 1; a < n_agents; a++) {
          const AgentCfg& ac = agents[(size_t)a].c;
          const double w = agents[(size_t)a].kind == A_MM ? 4.0 * (double)std::max<int64_t>(1, ac.depth_levels) : 1.0;
          size_t k = 0;
          while (k < grp.size() && grp[k].first != ac.interval_ns) k++;
          if (k == grp.size()) grp.emplace_back(ac.interval_ns, 0.0);
          grp[k].second += w;
          burst = std::max(burst, grp[k].second);
        }
        double q10, q90;
        switch (cfg.lat) {
          case L_LOGNORMAL: q10 = std::exp(cfg.lat_mu - 1.2816 * cfg.lat_sigma); q90 = std::exp(cfg.lat_mu + 1.2816 * cfg.lat_sigma); break;
          case L_UNIFORM: q10 = cfg.lat_min + 0.1 * (cfg.lat_max - cfg.lat_min); q90 = cfg.lat_min + 0.9 * (cfg.lat_max - cfg.lat_min); break;
          case L_PARETO: {
            const double b = cfg.lat_min > 0 ? cfg.lat_min : 1.0;
            q10 = b * std::pow(0.9, -1.0 / cfg.lat_alpha);
            q90 = b * std::pow(0.1, -1.0 / cfg.lat_alpha);
            break;
          }
          default: q10 = q90 = cfg.lat_mean;
        }
        q10 = std::min(std::max(q10, cfg.lat_min), cfg.lat_max);
        q90 = std::min(std::max(q90, cfg.lat_min), cfg.lat_max);
        const double wt = std::max(q90 - q10, 1.0) / (4.0 * burst);
        int sb = 0;
        while (sb < 24 && (double)(2ull << sb) <= wt) sb++;
        int sc = 0;
        while (sc < 24 && (double)(16384ull << sc) < 8.0 * p95) sc++;
        shift = std::max(std::min(shift, sb), sc);
      }
      q.init(shift);
      if constexpr (G) {  // the deterministic-latency instance: constant latency, messages in the group list
        if (cfg.lat != L_DETERMINISTIC) throw std::runtime_error(T3_MSG("SimT<true> needs a deterministic latency model", "E43"));
        double v = cfg.lat_mean;
        if (v < cfg.lat_min) v = cfg.lat_min;
        if (v > cfg.lat_max) v = cfg.lat_max;
        det_lat = py_round(v);
        gqp.reset(new GroupQ());
        gqp->init((uint64_t)(4.0 * p95 + 1000.0));
        gqp->xpre.assign((size_t)n_agents, GroupQ::XPre());
      }
    q.preface(size_t(64) << 10);
    if constexpr (!NL) led.preface(size_t(64) << 10);
    trace.preface(size_t(64) << 10);
    pool.preface(size_t(64) << 10);
    }
    {
      std::vector<int64_t> ivs;
      agent_lane.assign((size_t)n_agents, -1);
      for (int a = 1; a < n_agents; a++) {
        int64_t iv = agents[(size_t)a].c.interval_ns;
        size_t k = 0;
        while (k < ivs.size() && ivs[k] != iv) k++;
        if (k == ivs.size()) ivs.push_back(iv);
        agent_lane[(size_t)a] = (int)k;
      }
      q.lanes.assign(ivs.size(), Lane());
      std::vector<size_t> cnt(ivs.size(), 0);
      for (int a = 1; a < n_agents; a++) cnt[(size_t)agent_lane[(size_t)a]]++;
      for (size_t k = 0; k < ivs.size(); k++) q.lanes[k].init(2 * cnt[k] + 4);
      q.lanes_ready();
    }
    // fixed-size initial capacity for the growing containers, touched here at setup so their first pages
    // are not faulted inside the timed run (beyond these sizes they grow inside the run as usual)
    bo.resize(4096); bo.clear();
    for (int sd = 0; sd < 2; sd++) lv[sd].reserve(1024, true);
    for (auto& pk : park) { pk.h.resize(64); pk.h.clear(); }
    for (auto& ag : agents) if (ag.kind == A_MM) ag.dict.ensure(256);
    q.heap.reserve(256);
    // rough size of the run (wakeups x ~5 messages); tiny runs skip the helper thread entirely
    double est = 0;
    for (int a = 1; a < n_agents; a++) est += (double)cfg.horizon_ns / (double)agents[(size_t)a].c.interval_ns;
    est *= 5;
    est_msgs = est;
    // the helper needs a CPU of its own (quota >= 3) and only pays off beyond tiny runs
    use_helper = helper_allowed && cpus_available >= 3 && est > helper_threshold;
    if (const char* h = T3_DEVENV("T3_HELPER")) use_helper = helper_allowed && atoi(h) != 0;
    use_feed = use_helper && cfg.lat != L_DETERMINISTIC;
    label_live = est >= 20000;
    if constexpr (!G) {  // the event loop (see park_policy in config.h)
      park_on = park_policy(cfg);
      if (const char* e = T3_DEVENV("T3_PARK")) park_on = atoi(e) != 0;
      if (park_on) setup_park();
    }
  }
  // ------------------------------------------------------------ noise decisions drawn ahead
  // A noise trader's order size, side and price offset come from its own generator, in the same order at every act, whatever the market
  // does (nothing else uses that generator). The helper thread draws them ahead into a small ring per agent and the simulation reads the
  // next entry: it waits for the entry's tag (the number of the decision), and tells the helper which agent used one up; the helper then
  // draws the decision NQK acts later into the slot just read. Same draws in the same order, so the output cannot differ.
#ifndef NQ_K
#define NQ_K 32
#endif
#ifndef NQ_GAP
#define NQ_GAP 30000
#endif
  static constexpr uint32_t NQK = NQ_K, NQMASK = NQK - 1;
  struct alignas(64) NEnt {  // one line per entry: the helper's write to a used slot must not take the line of the next one away from the simulation
    int64_t size, off;
    uint32_t tag;
    uint8_t buy;
  };
  struct NGen {  // helper side: one noise agent's generator and parameters
    nrng::RandomState* rs;
    double mean, sd;
    int64_t poff;
    uint32_t mask, r32, nprod, pad;
  };
  NEnt* nring = nullptr;  // [ordinal][NQK]
  std::vector<NGen> ngen;
  bool noise_off = false;
  struct AuxCtx {  // what the auxiliary thread reads or writes, on lines of its own: the simulation stores to the lines around SimT's members
    NEnt* ring;
    NGen* gen;
    const uint16_t* rq;
    uint32_t mask;
    const std::atomic<uint32_t>* pub;
    alignas(64) uint32_t tail;
    alignas(64) uint64_t tsc_noise, tsc_chore;
  };
  AuxCtx* actx = nullptr;
  static inline void noise_draw(NGen& g, NEnt& en, uint32_t n) {
    nrng::RandomState& rs = *g.rs;
    en.size = std::max<int64_t>(1, py_round(rs.normal(g.mean, g.sd)));
    en.buy = rs.randbit();
    en.off = g.mask ? (int64_t)rs.masked_upto(g.mask, g.r32) : rs.randint_0_to(g.poff);
    __atomic_store_n(&en.tag, n, __ATOMIC_RELEASE);
  }
  static bool noise_serve(AuxCtx* c) {  // auxiliary thread: replace the decisions the simulation has used
    const uint32_t h = c->pub->load(std::memory_order_acquire);
    uint32_t t = c->tail;
    if (h == t) return false;
    do {
      const uint32_t j = c->rq[t & c->mask];
      t++;
      if (j == 0xFFFFu) continue;  // padding
      NGen& g = c->gen[j];
      noise_draw(g, c->ring[(size_t)j * NQK + (g.nprod & NQMASK)], g.nprod);
      g.nprod++;
    } while (t != h);
    c->tail = t;
    return true;
  }
  __attribute__((noinline)) void noise_wait(NEnt& en, uint32_t want) {
    // the entry is the replacement of one this agent used NQK acts ago: the auxiliary thread may not have seen that request yet
    if (feed.nreq_n != feed.nreq_pubd) feed.publish_noise();
    while (__atomic_load_n(&en.tag, __ATOMIC_ACQUIRE) != want) cpu_relax();
  }
  void setup_noise_offload() {  // before the helper thread starts
    double acts = 0;
    size_t nn = 0;
    for (int a = 1; a < n_agents; a++) {
      if (agents[(size_t)a].kind != A_NOISE) continue;
      acts += (double)cfg.horizon_ns / (double)agents[(size_t)a].c.interval_ns;
      nn++;
    }
    int mode = 1;  // T3_NOISEOFF: 0 off, 1 default, 2 forced on (small runs too: a test aid)
    if (const char* e = T3_DEVENV("T3_NOISEOFF")) mode = atoi(e);
    if (mode == 0 || nn == 0 || nn > 1024) return;
    // beyond the simulation thread and the latency helper (and the trace-ahead thread) the auxiliary thread needs a CPU of its own
    if (mode == 1 && (acts < 20000 || cpus_available < 4)) return;
    nring = (NEnt*)aligned_alloc(64, nn * NQK * sizeof(NEnt));
    if (!nring) throw std::bad_alloc();
    std::memset((void*)nring, 0, nn * NQK * sizeof(NEnt));
    // Room for every request the helper has not served yet: at most NQK per agent (an agent waits for its entry when it is NQK ahead), each
    // published interval holds at least one of them and at most 31 padding entries
    size_t cap = 4096;
    while (cap < 32 * nn * NQK + 64) cap <<= 1;
    feed.nreq = (uint16_t*)calloc(cap, sizeof(uint16_t));
    if (!feed.nreq) throw std::bad_alloc();
    feed.nreq_mask = (uint32_t)cap - 1;
    ngen.assign(nn, NGen());
    uint32_t j = 0;
    for (int a = 1; a < n_agents; a++) {
      Agent& ag = agents[(size_t)a];
      if (ag.kind != A_NOISE) continue;
      NGen& g = ngen[j];
      g.rs = &agent_rs[(size_t)a];
      g.mean = ag.c.size_mean; g.sd = ag.c.size_std; g.poff = ag.c.price_offset;
      g.mask = ag.noff_mask; g.r32 = ag.noff_r;
      for (uint32_t i = 0; i < NQK; i++) noise_draw(g, nring[(size_t)j * NQK + i], i);
      g.nprod = NQK;
      ag.nq_on = true;
      ag.nq_base = j;
      ag.nq_n = 0;
      j++;
    }
    actx = (AuxCtx*)aligned_alloc(64, (sizeof(AuxCtx) + 63) / 64 * 64);
    if (!actx) throw std::bad_alloc();
    std::memset((void*)actx, 0, sizeof(AuxCtx));
    actx->ring = nring; actx->gen = ngen.data(); actx->rq = feed.nreq; actx->mask = feed.nreq_mask; actx->pub = &feed.nq_pub;
    noise_off = true;
  }
  // The latency helper is close to the simulation thread's own load (draws, plus the pre-faulting of the output streams), so the decisions
  // are drawn by a thread of their own, which also takes over the pre-faulting
  std::thread aux_th;
  void aux_start(bool prefault) {
    aux_th = std::thread([this, prefault] {
#if defined(__linux__) && !defined(T3_SEALED)
      pthread_setname_np(pthread_self(), "t3aux");
#endif
      AuxCtx* const c = actx;
      while (feed.state.load(std::memory_order_acquire) == 0) cpu_relax();
      while (feed.state.load(std::memory_order_relaxed) == 1) {
        bool w = false;
        // every poll of a line the simulation stores to (the published counts, the streams' cursors) costs it a coherence miss at its next
        // store: bounded rates
#if defined(__x86_64__)
        const uint64_t now = __builtin_ia32_rdtsc();
        if (now - c->tsc_noise >= NQ_GAP) { c->tsc_noise = now; w |= noise_serve(c); }
        if (prefault && now - c->tsc_chore >= CHORE_GAP) { c->tsc_chore = now; w |= prefault_step(); }
#else
        w |= noise_serve(c);
        if (prefault) w |= prefault_step();
#endif
        if (!w) feed.idle();
      }
    });
  }
  // helper-thread creation is process setup; the helper draws nothing until begin()
  // pre-fault the output streams and the message pool ahead of their cursors (one bounded step; true if work was done)
  bool prefault_step() {
    bool any = false;
    if constexpr (!NL) any = led.prefault(8);
    any |= trace.prefault(8);
    any |= pool.prefault(8);
    if constexpr (NL && T3_AA_ORD) any |= prefault_orders();
    return any;
  }
  // NL: the per-order arrays (oi, dpa: indexed by order id) are first touched by the run in id order; this keeps their pages
  // ahead of the order counter (published at the checkpoints) so that the run does not take those page faults
  alignas(64) static inline std::atomic<uint64_t> ord_pub{0};
  // helper-owned, each on a line of its own (the simulation stores to the lines around): bytes touched (ZeroArr touches 16 pages at setup), and the arrays' addresses
  alignas(64) static inline size_t oi_touched = size_t(16) * 4096;
  alignas(64) static inline size_t dpa_touched = size_t(16) * 4096;
  alignas(64) static inline char* oi_base = nullptr;
  static inline char* dpa_base = nullptr;
  bool prefault_orders() {
    if (__builtin_expect(!oi_base, 0)) { oi_base = (char*)oi.p; dpa_base = (char*)dpa.p; }
    const size_t oc = (size_t)ord_pub.load(std::memory_order_relaxed);
    bool any = false;
    size_t want = std::min(oc * sizeof(OInfo) + (size_t(256) << 10), ZeroArr<OInfo>::N * sizeof(OInfo));
    for (int k = 0; k < 16 && oi_touched < want; k++, oi_touched += 4096) {
      __atomic_fetch_add((uint64_t*)(oi_base + oi_touched), 0, __ATOMIC_RELAXED);
      any = true;
    }
    want = std::min(oc * sizeof(uint32_t) + (size_t(128) << 10), ZeroArr<uint32_t>::N * sizeof(uint32_t));
    for (int k = 0; k < 16 && dpa_touched < want; k++, dpa_touched += 4096) {
      __atomic_fetch_add((uint32_t*)(dpa_base + dpa_touched), 0, __ATOMIC_RELAXED);
      any = true;
    }
    return any;
  }
  // chores_elsewhere: another thread calls prefault_step() during the run, the helper only draws latencies
  void prepare_threads(bool chores_elsewhere = false) {
    if (!use_helper && est_msgs > 32768) {  // large run without a helper (CPU quota < 3): populate pages in batches
      if constexpr (!NL) led.pop_on = true;
      trace.pop_on = true;
      pool.pop_on = true;
      pool.pop_chunk = size_t(256) << 10;
    }
    if (use_helper) {
      pool.ahead = size_t(128) << 10;  // slots are recycled: the pool stays small, a long pre-fault lead would be wasted work
      if (use_feed) setup_noise_offload();
      if (!chores_elsewhere && !noise_off) feed.chores.push_back([this] { return prefault_step(); });
      feed.start(latency_rs, cfg, use_feed);
      if (noise_off) aux_start(!chores_elsewhere);
    } else if (cfg.lat != L_DETERMINISTIC) {
      feed.start_inline(latency_rs, cfg);
    }
  }
  void begin() {
    if (use_helper) feed.go();
  }
  alignas(64) static inline std::atomic<bool> run_done{false};  // NL: the run is over (the trace-ahead thread winds down)
  void end_threads() {
    if constexpr (NL) run_done.store(true, std::memory_order_release);
    feed.stop();
    if (aux_th.joinable()) {
      if (feed.detach_at_stop) aux_th.detach();
      else aux_th.join();
    }
  }

  // ------------------------------------------------------------ ScenarioLatencyModel
  __attribute__((always_inline)) inline int64_t get_latency() {  // only non-self messages draw (the exchange and the traders never message themselves)
    if constexpr (G) return det_lat;  // deterministic model: no draw consumes the generator, every latency is the constant
    return feed.next();  // draws come from the helper thread, or from the consumer's own small batches (LatencyFeed::inline_mode)
  }

  // ------------------------------------------------------------ kernel API
  // exchange (agent 0) -> trader
  __attribute__((always_inline)) inline void send_x_at(int32_t recipient, uint32_t slot, int64_t sent_time) {
    int64_t lat = get_latency();
    Msg& m = pool[slot];
    m.t_recv = sent_time + lat;
    if constexpr (!NL) {
      m.latency = lat;
      m.causal = causal_uid;
      const uint32_t sd = (uint32_t)recipient << 16;  // src = 0, dst = recipient (adjacent fields, little endian)
      std::memcpy(&m.src, &sd, 4);
      if constexpr (W) { m.srch = 0; m.dsth = (uint8_t)(recipient >> 16); }
    }
    const uint64_t qt = (uint64_t)m.t_recv;
    const uint64_t qk = W ? (((uint64_t)recipient << 32) | m.id) : (((uint64_t)recipient << 34) | m.id);
    if constexpr (G) gqp->push_x_g(q, recipient, qt, qk, slot, (uint64_t)TDELAY);
    else q.push(QItem{qt, qk, slot, (uint32_t)recipient}, (uint64_t)(sent_time - current_time + lat));
  }
  __attribute__((always_inline)) inline void send_x(int32_t recipient, uint32_t slot, int64_t delay = 0) {
    send_x_at(recipient, slot, current_time + xdelay + delay);
  }
  // trader -> exchange
  __attribute__((always_inline)) inline void send_t(int32_t sender, uint32_t slot) {
    int64_t sent_time = current_time + TDELAY;
    int64_t lat = get_latency();
    Msg& m = pool[slot];
    m.t_recv = sent_time + lat;
    if constexpr (!NL) {
      m.latency = lat;
      m.causal = causal_uid;
      const uint32_t sd = (uint32_t)sender & 0xFFFFu;  // src = sender, dst = 0
      std::memcpy(&m.src, &sd, 4);
      if constexpr (W) { m.srch = (uint8_t)(sender >> 16); m.dsth = 0; }
    }
    const QItem qi{(uint64_t)m.t_recv, W ? (((WB + 2 * (uint64_t)sender) << 32) | m.id) : (((uint64_t)sender << 49) | m.id), slot, 0};
    if constexpr (G) gqp->push_msg_g(q, qi);
    else q.push(qi, (uint64_t)(TDELAY + lat));
  }
  void set_wakeup(int32_t agent, int64_t t) {
    if (current_time && t < current_time) throw std::runtime_error(T3_MSG("set_wakeup in past", "E44"));
    uint32_t s = alloc_msg(M_WAKEUP);
    Msg& m = pool[s];
    m.t_recv = t;
    if constexpr (!NL) { m.latency = -1; m.src = m.dst = (uint16_t)agent; }
    if constexpr (W) m.srch = m.dsth = (uint8_t)(agent >> 16);
    qput(t, agent, agent, s);
  }
  __attribute__((always_inline)) inline void rearm_wakeup(int32_t agent, int64_t t, uint32_t s) {  // s: a delivered wakeup of this agent (latency -1, src = dst = agent)
    Msg& m = pool[s];
    m.id = (uint32_t)msg_counter++;
    m.t_recv = t;
    q.push_lane(agent_lane[(size_t)agent], (uint64_t)t, mkkey(agent, agent, m.id), s);
  }

  // ------------------------------------------------------------ agent-log events -> trace rows
  // abides_fork.trace: rows sorted stably by (t_ns, order_id); per timestamp, BEST_BID/BEST_ASK
  // collapse to the last value per side, placed in order of first appearance (order_id -1 first).
  // Rows of the current timestamp group are written in place: quote rows (one per side, last value)
  // sit at the front in first-appearance order, then agent-log rows kept sorted by order_id (stable).
  // Rows of the current timestamp group are written in place: quote rows (one per side, updated in place)
  // sit at the front, then the agent rows from glo on. A quote arriving after agent rows shifts them right.
  TraceRow* glo = nullptr;  // first agent row of the group (= first row + number of quote rows)
  TraceRow* gq[2] = {nullptr, nullptr};
  // Small runs label the executions in one pass over the finished trace (finalize) instead of when each execution row
  // is written (emit_order_t): the data-dependent labelling branches mispredict, the closing pass over a few KB does not.
  bool label_live = true;

  inline void open_group() {
    grp_t = current_time;
    glo = trace.cur;
    gq[0] = gq[1] = nullptr;
  }
  __attribute__((always_inline)) inline void write_quote(TraceRow* pos, int sd, int64_t price, int64_t size) {
    pos->t = grp_t; pos->price = price; pos->size = size;
    const uint64_t tail = 0xFFFFFFFFull | ((uint64_t)C_QUOTE << 48) | ((uint64_t)sd << 56);  // oid -1, agent 0, code, side
    std::memcpy(&pos->oid, &tail, 8);
  }
  // both best quotes of the book after a LIMIT (a side with no level emits nothing)
  __attribute__((always_inline)) inline void emit_tops() {
#ifdef EXP_NOTRACE
    return;
#endif
    const bool hb = !lv[0].empty(), ha = !lv[1].empty();
    if (!(hb | ha)) return;
    if (current_time != grp_t) {  // a new group: no rows yet, so the quote rows are simply appended
      grp_t = current_time;
      TraceRow* p = trace.take_nc((size_t)hb + (size_t)ha);
      TraceRow* const pa = p + hb;
      if (hb) write_quote(p, 0, (int64_t)(lv[0].back().key ^ KB), lv[0].back().total);
      if (ha) write_quote(pa, 1, (int64_t)(lv[1].back().key ^ (KB - 1)), lv[1].back().total);
      gq[0] = hb ? p : nullptr;
      gq[1] = ha ? pa : nullptr;
      glo = pa + ha;
      return;
    }
    if (hb) quote_row(0, (int64_t)(lv[0].back().key ^ KB), lv[0].back().total);
    if (ha) quote_row(1, (int64_t)(lv[1].back().key ^ (KB - 1)), lv[1].back().total);
  }
  __attribute__((always_inline)) inline void quote_row(int sd, int64_t price, int64_t size) {
    if (TraceRow* r = gq[sd]) {
      r->price = price;
      r->size = size;
      return;
    }
    TraceRow* pos = glo;
    for (TraceRow* p = trace.take_nc(1); p > pos; --p) {
      *p = p[-1];
      if (p->code == C_FILLED) oi.p[(size_t)p->oid].le = (uint32_t)(p - trace.base) + 1;  // a labelled row moved
    }
    write_quote(pos, sd, price, size);
    gq[sd] = pos;
    glo = pos + 1;
  }
  // An ORDER_EXECUTED row is labelled when written: it is the order's latest execution, so it becomes FILLED and the
  // order's previous FILLED row goes back to PARTIAL (the last execution of each order ends up as ORDER_FILLED).
  // Rows of the open group that shift right (an insertion before them) keep the order info's last-execution index pointing at them.
  template <bool EX>
  __attribute__((always_inline)) inline void emit_order_t(uint8_t code, int32_t agent, const Msg& msg) {
    const uint8_t side = msg.oside;
    const int64_t oid = msg.oid;
    TraceRow* i;
    if (current_time != grp_t) {  // first row of the group: nothing to sort against
      open_group();
      i = trace.take_nc(1);
    } else {
      i = trace.take_nc(1);
      while (i > glo && i[-1].oid > oid) {
        *i = i[-1];
        if (i->code == C_FILLED) oi.p[(size_t)i->oid].le = (uint32_t)(i - trace.base) + 1;
        i--;
      }
    }
    i->t = grp_t; i->oid = oid; i->agent = agent; i->code = code; i->side = side;
    if constexpr (W) i->side = (uint8_t)(side | ((agent >> 16) << 1));
    std::memcpy(&i->price, &msg.oprice, 16);  // price and size: the message's oprice/oqty (an execution carries the fill price)
    if constexpr (EX) {
      if (label_live) {
        i->code = C_FILLED;
        uint32_t& le = oi.p[(size_t)oid].le;
        const int32_t pj = (int32_t)(le - 1);
        trace.base[pj].code = C_PARTIAL;  // the order's previous FILLED row; le 0 (it has none): the scratch row in front of the trace
        if constexpr (TA) __atomic_store_n(&lastflip_mem[((pj >> 14) + 1) * 16], epoch_now, __ATOMIC_RELAXED);  // (see TraceAhead in main.cpp)
        le = (uint32_t)(i - trace.base) + 1;
      }
    }
  }
  // Per-order state, indexed directly by order id (ids < 2^27 > trace capacity: every order has a SUBMITTED row) in
  // lazily zeroed pages: no bounds checks, 0 = none.
  template <class T>
  struct ZeroArr {
    static constexpr size_t N = size_t(1) << 27;
    T* p;
    ZeroArr() {
      void* m = mmap(nullptr, N * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      if (m == MAP_FAILED) throw std::bad_alloc();
#ifdef MADV_NOHUGEPAGE
      madvise(m, N * sizeof(T), MADV_NOHUGEPAGE);
#endif
      p = (T*)m;
      for (size_t i = 0; i < 16; i++) ((volatile char*)p)[i * 4096] = 0;  // fixed set-up touch: the first pages
    }
    ZeroArr(const ZeroArr&) = delete;
    ~ZeroArr() { munmap(p, N * sizeof(T)); }
  };
  struct OInfo {
    uint32_t slot1;  // 1 + the order's slot in the book (bo), 0 when it is not resting
    uint32_t le;     // 1 + trace row index of its latest ORDER_EXECUTED row
  };
  ZeroArr<OInfo> oi;
  ZeroArr<uint32_t> dpa;  // market makers: 1 + the order's index in its agent's dict, 0 when absent

  // ------------------------------------------------------------ SparseMeanRevertingOracle
  int64_t compute_fundamental(const PyNum& ts, double v_adj, bool adj_is_float, const PyNum& pt, int64_t pv) {
    double d = py_sub_f(ts, pt);
    const double mu = (double)cfg.reference_price;
    const double gamma = cfg.kappa;
    const double theta = cfg.fund_vol;
    double e1 = std::exp(-gamma * d);
    double loc = mu + (double)(pv - cfg.reference_price) * e1;
    double scale = std::sqrt((std::pow(theta, 2.0) / (2 * gamma)) * (1 - std::exp(-2 * gamma * d)));
    double v = oracle_rs.normal(loc, scale);
    if (adj_is_float) v += v_adj;
    int64_t vi = (v > 0) ? py_round(v) : 0;
    if (o_jump_pending) {
      bool ge = ts.isf ? (cmp_int_float(o_jump_time, ts.f) <= 0) : (ts.i >= o_jump_time);
      if (ge) {
        vi = std::max<int64_t>(0, vi + cfg.jump_mag);
        o_jump_pending = false;
      }
    }
    o_pt = ts;
    o_pv = vi;
    return vi;
  }
  int64_t advance_fundamental(int64_t now) {
    PyNum cur = PyNum::I(now);
    PyNum pt = o_pt;
    int64_t pv = o_pv;
    if (!py_lt(pt, cur)) return pv;
    while (py_lt(o_mst, cur)) {
      int64_t v = compute_fundamental(o_mst, o_msv, true, pt, pv);
      pt = o_mst;
      pv = v;
      double dt = global_rs.exponential(1.0 / cfg.ms_lambda);
      o_mst = PyNum::F(pt.asf() + std::trunc(dt));
      double msv = oracle_rs.normal(cfg.ms_mean, std::sqrt((double)cfg.ms_var));
      o_msv = oracle_rs.randint(0, 2) == 0 ? msv : -msv;
    }
    return compute_fundamental(cur, 0.0, false, pt, pv);
  }
  int64_t observe_price(int64_t now, nrng::RandomState& rs, double sigma_n) {
    int64_t r_t = now >= DATE_NS + H1600 ? advance_fundamental(DATE_NS + H1600 - 1) : advance_fundamental(now);
    return py_round(rs.normal((double)r_t, std::sqrt(sigma_n)));
  }

  // ------------------------------------------------------------ OrderBook
  inline int32_t bo_alloc() {
    if (bo_free_head >= 0) {
      const int32_t s = bo_free_head;
      bo_free_head = bo[(size_t)s].next;
      return s;
    }
    bo.emplace_back();
    return (int32_t)bo.size() - 1;
  }
  // how many of the LK best levels (ending at e) have a key below k (LT) / above k (!LT): a compare and an add-with-carry each
  template <bool LT>
  static inline int top_count(const Level* e, uint64_t k) {
    int c = 0;
#pragma GCC unroll 8
    for (int j = 1; j <= LK; j++) c += LT ? (e[-j].key < k) : (k < e[-j].key);
    return c;
  }
  // first level (of the m >= 1 from L) whose key is >= k, or L + m (branch-free binary search)
  static inline Level* lower_key(Level* L, uint64_t k, size_t m) {
    Level* b = L;
    while (m > 1) {
      const size_t h = m >> 1;
      b += h & (size_t)-(int64_t)(b[h - 1].key < k);
      m -= h;
    }
    return b + (b->key < k);
  }
  __attribute__((always_inline)) inline void enter_order(int S, int64_t oid, int32_t agent, uint64_t kin, int64_t price, int64_t qty) {
    int32_t sl = bo_alloc();
    BO& o = bo[(size_t)sl];
    o.oid = oid; o.qty = qty; o.price = price; o.agent = agent; o.side = (uint8_t)S; o.next = -1;
    oi.p[(size_t)oid].slot1 = (uint32_t)sl + 1;
    LevVec& L = lv[S];
    if (__builtin_expect(L.e == L.lim, 0)) L.reserve((size_t)(L.lim - L.b) * 2 + 16);
    Level* const e = L.e;
    // pos: the first level (from the worst) whose key is >= kin: LK minus the number of the best levels below kin back from the end
    const int lt = top_count<true>(e, kin);
    Level* pos = e - (LK - lt);
    if (__builtin_expect(lt == 0, 0)) pos = e - L.b > LK ? lower_key(L.b, kin, (size_t)(e - L.b) - LK) : L.b;
    if (pos != e && pos->key == kin) {
      const int32_t tl = pos->tail;  // -1 for an empty level: the order becomes its head
      o.prev = tl;
      *nb_or(tl, &(bo.data() + (ptrdiff_t)tl)->next, &pos->head) = sl;
      pos->tail = sl;
      pos->total += qty;
    } else {
      o.prev = -1;
      const Level nl{kin, qty, sl, sl};
      if (pos != e) std::memmove((void*)(pos + 1), (void*)pos, (size_t)((char*)e - (char*)pos));
      *pos = nl;
      L.e = e + 1;
    }
  }
  // pointer to bo[x].f when x >= 0, else to `alt`: a conditional move (whether an order has a neighbour is data-dependent)
  static inline int32_t* nb_or(int32_t x, int32_t* nbp, int32_t* alt) {
    __asm__("test %1, %1\n\tcmovns %2, %0" : "+r"(alt) : "r"(x), "r"(nbp) : "cc");
    return alt;
  }
  inline void unlink(int32_t sl, Level& lev) {
    BO& o = bo[(size_t)sl];
    const int32_t pv = o.prev, nx = o.next;
    BO* const B = bo.data();
    int32_t* const a = nb_or(pv, &(B + (ptrdiff_t)pv)->next, &lev.head);
    int32_t* const b = nb_or(nx, &(B + (ptrdiff_t)nx)->prev, &lev.tail);
    *a = nx;
    *b = pv;
    lev.total -= o.qty;
    oi.p[(size_t)o.oid].slot1 = 0;
    order_left(o.oid);
    o.next = bo_free_head;
    bo_free_head = sl;
  }
  // unlink of the first order of a level (what a match takes): it has no predecessor
  inline void unlink_front(int32_t sl, Level& lev) {
    BO& o = bo[(size_t)sl];
    const int32_t nx = o.next;
    lev.head = nx;
    *nb_or(nx, &(bo.data() + (ptrdiff_t)nx)->prev, &lev.tail) = -1;
    lev.total -= o.qty;
    oi.p[(size_t)o.oid].slot1 = 0;
    order_left(o.oid);
    o.next = bo_free_head;
    bo_free_head = sl;
  }
  // OrderBook.cancel_order (quiet=False): OrderCancelledMsg(cancelled_order) to the order's agent.
  T3_NOSSP bool book_cancel(int64_t oid, uint8_t side, int64_t price, uint32_t reuse = NOSLOT) {
    const uint32_t sl1 = oi.p[(size_t)oid].slot1;
    if (sl1 == 0) return false;
    const int32_t sl = (int32_t)sl1 - 1;
    if (bo[(size_t)sl].side != side || bo[(size_t)sl].price != price) return false;
    LevVec& L = lv[side];
    Level* const e = L.e;
    const uint64_t kin = (uint64_t)price ^ (KB - side);
    // the order's level: LK-1 back from the end minus the number of the best levels above it
    const int gt = top_count<false>(e, kin);
    Level* lev = e - 1 - gt;
    if (__builtin_expect(gt == LK, 0)) lev = lower_key(L.b, kin, (size_t)(e - L.b) - LK);
    int32_t agent = bo[(size_t)sl].agent;
    int64_t qty = bo[(size_t)sl].qty;
    unlink(sl, *lev);
    L.trim();
    if (reuse != NOSLOT) {  // reply in place of the CANCEL request (same oid/side/price)
      remsg(reuse, M_CANCELLED);
      pool[reuse].oqty = qty;
      send_x(agent, reuse, cfg.ack_delay);
      return true;
    }
    uint32_t m = alloc_msg(M_CANCELLED);
    Msg& mm = pool[m];
    mm.oid = oid; mm.oside = side; mm.oprice = price; mm.oqty = qty;
    send_x(agent, m, cfg.ack_delay);
    return true;
  }
  // The two ORDER_EXECUTED messages of one match: the resting (passive) order's, then the incoming (aggressor) order's.
  __attribute__((always_inline)) inline void send_fill2(int32_t pa, int64_t poid, uint8_t pside, int64_t fill, int32_t aa,
                                                        int64_t aoid, uint8_t aside, int64_t qty) {
    const int64_t st = current_time + xdelay + cfg.ack_delay;
    uint32_t m = alloc_msg(M_EXECUTED);
    Msg& mm = pool[m];
    mm.oid = poid; mm.oside = pside; mm.oprice = fill; mm.oqty = qty;
    send_x_at(pa, m, st);
    uint32_t m2 = alloc_msg(M_EXECUTED);
    Msg& m2r = pool[m2];
    m2r.oid = aoid; m2r.oside = aside; m2r.oprice = fill; m2r.oqty = qty;
    send_x_at(aa, m2, st);
  }
  __attribute__((always_inline)) inline bool handle_limit_s(int S, int64_t oid, int32_t agent, int64_t price, int64_t qty, uint32_t slot) {
    LevVec& OL = lv[S ^ 1];
    const uint64_t kin = (uint64_t)price ^ (KB - (uint64_t)S);
    bool reused = false;  // the LIMIT request's slot carries the cancel/accept reply
    while (true) {
      const bool cr = kin >= ~OL.back().key;  // an empty side's sentinel key 0 never crosses
      if (cfg.stp && cr) {
        const BO& resting = bo[(size_t)OL.back().head];
        if (resting.agent == agent) {
          if (cfg.stp_oldest) {
            if (book_cancel(resting.oid, resting.side, resting.price)) continue;
          } else {
            remsg(slot, M_CANCELLED);
            pool[slot].oqty = qty;
            send_x(agent, slot, cfg.ack_delay);
            reused = true;
            break;
          }
        }
      }
      if (!cr) {
        enter_order(S, oid, agent, kin, price, qty);
        remsg(slot, M_ACCEPTED);
        pool[slot].oqty = qty;
        send_x(agent, slot, cfg.ack_delay);
        reused = true;
        break;
      }
      Level& L = OL.back();
      int32_t hs = L.head;
      BO& front = bo[(size_t)hs];
      const int64_t m_oid = front.oid, m_price = front.price;
      const int32_t m_agent = front.agent;
      const uint8_t m_side = front.side;
      if (qty < front.qty) {  // the incoming order is used up (resting quantities are positive)
        front.qty -= qty;
        L.total -= qty;
        send_fill2(m_agent, m_oid, m_side, m_price, agent, oid, (uint8_t)S, qty);
        break;
      }
      const int64_t mq = front.qty;
      unlink_front(hs, L);
      OL.trim();
      send_fill2(m_agent, m_oid, m_side, m_price, agent, oid, (uint8_t)S, mq);
      qty -= mq;
      if (qty == 0) break;
    }
    emit_tops();
    return reused;
  }
  T3_NOSSP bool handle_limit_order(const Msg& msg, int32_t agent, uint32_t slot) {  // msg: the request, in `slot`
    const int64_t oid = msg.oid, price = msg.oprice, qty = msg.oqty;
    if (qty <= 0) return false;
    if (price < 0) return false;
    return handle_limit_s(msg.oside, oid, agent, price, qty, slot);
  }

  // ------------------------------------------------------------ ExchangeAgent
  void exchange_wakeup(int64_t t) {
    if (t >= mkt_close) {
      uint32_t m = alloc_msg(M_CLOSE_PRICE);  // one message object, delivered to every subscriber
      for (int32_t a : close_subs) {
        uint32_t s = new_slot();
        pool[s] = pool[m];
        send_x(a, s);
      }
      free_slot(m);
    }
  }
  __attribute__((always_inline)) inline bool exchange_receive(int64_t t, int32_t sender, uint32_t slot, const Msg& msg) {
    xdelay = cfg.compute_delay;
    const uint8_t ty = msg.type;
    if (t > mkt_close && ty != M_QSPREAD) {
      send_x(sender, alloc_msg(M_CLOSED));
      return false;
    }
    if (ty == M_QSPREAD) {  // (a level exists only while it holds resting quantity, so its total is positive)
      remsg(slot, M_QSPREAD_RESP);
      Msg& r = pool[slot];
      r.hb = !lv[0].empty();
      r.ha = !lv[1].empty();
      r.oprice = r.hb ? (int64_t)(lv[0].back().key ^ KB) : 0;
      r.oqty = r.ha ? (int64_t)(lv[1].back().key ^ (KB - 1)) : 0;
      r.closed = t > mkt_close;
      send_x(sender, slot);
      return true;
    }
    if (ty == M_LIMIT) return handle_limit_order(msg, sender, slot);
    switch (ty) {
      case M_CANCEL: return book_cancel(msg.oid, msg.oside, msg.oprice, slot);
      case M_CLOSE_PRICE_REQ: close_subs.push_back(sender); break;
      case M_HOURS_REQ:
        xdelay = 0;
        send_x(sender, alloc_msg(M_HOURS));
        break;
      default: break;
    }
    return false;
  }

  // ------------------------------------------------------------ trading agents
  // a new order's id is the largest so far, so its SUBMITTED row never sorts before another row of the group
  __attribute__((always_inline)) inline void emit_submitted(int32_t agent, uint8_t side, int64_t price, int64_t size, int64_t oid) {
#ifdef EXP_NOTRACE
    return;
#endif
    if (current_time != grp_t) open_group();
    TraceRow* i = trace.take_nc(1);
    i->t = grp_t; i->price = price; i->size = size; i->oid = oid; i->agent = agent; i->code = C_SUBMITTED; i->side = side;
    if constexpr (W) i->side = (uint8_t)(side | ((agent >> 16) << 1));
  }
  __attribute__((always_inline)) inline void place_limit_t(int32_t a, uint8_t side, int64_t price, int64_t qty) {
    int64_t oid = order_counter++;
    uint32_t m = alloc_msg(M_LIMIT);
    Msg& mm = pool[m];
    mm.oid = oid; mm.oside = side; mm.oprice = price; mm.oqty = qty;
    send_t(a, m);
    emit_submitted(a, side, price, qty, oid);
  }
  T3_NOSSP __attribute__((noinline)) void place_limit(int32_t a, uint8_t side, int64_t price, int64_t qty) {
    place_limit_t(a, side, price, qty);
  }
  // the market maker's quote ladder: bid/ask pairs from the inside out; the same operations as depth_levels x 2
  // place_limit_t calls plus the market maker's dict entry, with the dict and trace capacity checks and the group check done once
  T3_NOSSP __attribute__((noinline)) void place_ladder(int32_t a, Agent& ag, int64_t mid) {
    const AgentCfg& c = ag.c;
    const int64_t D = c.depth_levels;
    if (D <= 0) return;
    const size_t n = (size_t)D * 2;
    const int64_t half = c.spread_ticks / 2, qty = c.size_per_level;
    int64_t oid = order_counter;
    order_counter += (int64_t)n;
    uint32_t* const dp = dpa.p;
    ag.dict.ensure(n);
    typename Agent::DEnt* d = ag.dict.p + ag.dict.n;
    uint32_t di = (uint32_t)ag.dict.n + 1;
    ag.dict.n += n;
    ag.dict_live += n;
    if (current_time != grp_t) open_group();
    TraceRow* tr = trace.take(n);
    const int64_t gt = grp_t;
    for (int64_t l = 0; l < D; l++) {
      for (int sd = 0; sd < 2; sd++) {
        const int64_t price = sd == 0 ? mid - half - l : mid + half + l;
        dp[(size_t)oid] = di++;
        *d++ = typename Agent::DEnt{oid, price, qty, (uint8_t)sd};
        uint32_t m = alloc_msg(M_LIMIT);
        Msg& mm = pool[m];
        mm.oid = oid; mm.oside = (uint8_t)sd; mm.oprice = price; mm.oqty = qty;
        send_t(a, m);
        tr->t = gt; tr->price = price; tr->size = qty; tr->oid = oid; tr->agent = a; tr->code = C_SUBMITTED; tr->side = (uint8_t)sd;
        if constexpr (W) tr->side = (uint8_t)(sd | ((a >> 16) << 1));
        tr++;
        oid++;
      }
    }
  }
  inline bool dict_has(int64_t oid) const { return dpa.p[(size_t)oid] != 0; }
  __attribute__((noinline)) void dict_compact(Agent& ag) {  // squeeze the deleted entries out (the prefix below dict_head is all deleted)
    size_t w = 0;
    for (size_t r = ag.dict_head; r < ag.dict.n; r++) {
      if (ag.dict[r].oid < 0) continue;
      ag.dict[w] = ag.dict[r];
      dpa.p[(size_t)ag.dict[w].oid] = (uint32_t)w + 1;
      w++;
    }
    ag.dict.n = w;
    ag.dict_head = 0;
  }
  // Deleting only marks the entry; the wakeup that next walks the dictionary skips the deleted prefix and compacts (dict_trim)
  __attribute__((always_inline)) inline void dict_delete(Agent& ag, int64_t oid) {
    size_t p = (size_t)dpa.p[(size_t)oid] - 1;
    ag.dict[p].oid = -1;
    dpa.p[(size_t)oid] = 0;
    ag.dict_live--;
  }
  __attribute__((noinline)) void dict_trim(Agent& ag) {
    size_t h = ag.dict_head;
    const size_t n = ag.dict.n;
    while (h < n && ag.dict.p[h].oid < 0) h++;  // skip the deleted prefix (the cancels' replies mostly arrive in order)
    if (h == n) h = ag.dict.n = 0;
    ag.dict_head = h;
    if (ag.dict.size() > 64 && ag.dict_live * 4 < ag.dict.size()) dict_compact(ag);
  }

  __attribute__((always_inline)) inline void act_noise(int32_t a, Agent& ag, bool bid_t, bool ask_t) {
    const AgentCfg& c = ag.c;
    int64_t size, off;
    bool buy;
    if (ag.nq_on) {  // the decisions come from the helper thread's ring
      NEnt& en = nring[(size_t)ag.nq_base * NQK + (ag.nq_n & NQMASK)];
      const uint32_t want = ag.nq_n;
      if (__builtin_expect(__atomic_load_n(&en.tag, __ATOMIC_ACQUIRE) != want, 0)) noise_wait(en, want);
      size = en.size; off = en.off; buy = en.buy;
#ifdef NQ_DUMMY
      { static uint32_t ctr = 12345; ctr = ctr * 1664525u + 1013904223u; size = 5 + ((ctr >> 8) & 7); buy = (ctr >> 20) & 1; off = (ctr >> 24) % (ag.c.price_offset + 1); }
#endif
      ag.nq_n = want + 1;
#ifdef NQ_PF_PREV
      __builtin_prefetch(&nring[(size_t)ag.nq_base * NQK + (ag.nq_n & NQMASK)]);
#endif
      feed.nreq[feed.nreq_n++ & feed.nreq_mask] = (uint16_t)ag.nq_base;
    } else {
      nrng::RandomState& rs = agent_rs[(size_t)a];
      size = std::max<int64_t>(1, py_round(rs.normal(c.size_mean, c.size_std)));
      buy = rs.randbit();
      if (ag.noff_mask) {  // randint_0_to with the mask precomputed
        const uint32_t mask = ag.noff_mask, r32 = ag.noff_r;
        off = (int64_t)rs.masked_upto(mask, r32);
      } else {
        off = rs.randint_0_to(c.price_offset);
      }
    }
    const int64_t a_buy = ask_t ? ag.ka : (bid_t ? ag.kb : c.reference_price);
    const int64_t a_sell = bid_t ? ag.kb : (ask_t ? ag.ka : c.reference_price);
    const int64_t bm = -(int64_t)buy, sm = ~bm;  // masks: bm all ones for a buy, sm for a sell (branch-free: the side is a coin flip)
    place_limit_t(a, (uint8_t)(1 + bm), ((a_buy & bm) | (a_sell & sm)) + ((off ^ sm) - sm), size);
  }

  T3_NOSSP void act(int32_t a) {
    Agent& ag = agents[(size_t)a];
    const AgentCfg& c = ag.c;
    // Python truthiness of the known best bid/ask (None or 0 are falsy)
    const bool bid_t = ag.kb_has && ag.kb != 0;
    const bool ask_t = ag.ka_has && ag.ka != 0;
    switch (ag.kind) {
      case A_NOISE: act_noise(a, ag, bid_t, ask_t); break;
      case A_MM: {
        int64_t mid = (bid_t && ask_t) ? (ag.kb + ag.ka) / 2 : c.reference_price;
        dict_trim(ag);
        const typename Agent::DEnt* const dd = ag.dict.data();  // unchanged while the cancels are sent
        const size_t dn = ag.dict.size();
        for (size_t i = ag.dict_head; i < dn; i++) {
          const typename Agent::DEnt& de = dd[i];
          if (de.oid < 0) continue;
          uint32_t m = alloc_msg(M_CANCEL);
          Msg& mm = pool[m];
          mm.oid = de.oid; mm.oside = de.side;
          mm.oprice = de.price; mm.oqty = de.qty;
          send_t(a, m);
        }
        place_ladder(a, ag, mid);
        break;
      }
      case A_VALUE: {
        double mid;
        if (bid_t && ask_t) mid = (double)(ag.kb + ag.ka) / 2.0;
        else if (bid_t) mid = (double)ag.kb;
        else if (ask_t) mid = (double)ag.ka;
        else return;
        int64_t fundamental = observe_price(current_time, agent_rs[(size_t)a], c.sigma_n);
        int64_t size = std::max<int64_t>(1, py_round(c.size_mean));
        if (mid < (double)(fundamental - c.threshold) && ask_t) place_limit(a, 0, ag.ka, size);
        else if (mid > (double)(fundamental + c.threshold) && bid_t) place_limit(a, 1, ag.kb, size);
        break;
      }
      case A_MOMENTUM: {
        double mid;
        if (bid_t && ask_t) mid = (double)(ag.kb + ag.ka) / 2.0;
        else if (bid_t) mid = (double)ag.kb;
        else if (ask_t) mid = (double)ag.ka;
        else return;
        const size_t cap = (size_t)c.lookback + 1;
        if (ag.hist_n < cap) {
          size_t w = ag.hist_head + ag.hist_n;  // hist_head < cap and hist_n < cap here
          ag.hist[w >= cap ? w - cap : w] = mid;
          ag.hist_n++;
        } else {
          ag.hist[ag.hist_head] = mid;
          ag.hist_head = ag.hist_head + 1 == cap ? 0 : ag.hist_head + 1;
        }
        if (ag.hist_n <= (size_t)c.lookback) return;
        double past = ag.hist[ag.hist_head];
        int64_t size = std::max<int64_t>(1, py_round(c.size_mean));
        if (mid > past + (double)c.threshold && ask_t) place_limit(a, 0, ag.ka, size);
        else if (mid < past - (double)c.threshold && bid_t) place_limit(a, 1, ag.kb, size);
        break;
      }
      default: break;
    }
  }

  __attribute__((always_inline)) inline bool trader_wakeup(int32_t a, int64_t t, uint32_t slot) {
    Agent& ag = agents[(size_t)a];
#ifndef NQ_NO_PF_WAKE
    if (ag.nq_on) __builtin_prefetch(&nring[(size_t)ag.nq_base * NQK + (ag.nq_n & NQMASK)]);  // needed at the act, a round trip from now
#endif
    if (ag.first_wake) {
      ag.first_wake = false;
      send_t(a, alloc_msg(M_CLOSE_PRICE_REQ));
    }
    if (!ag.hours) send_t(a, alloc_msg(M_HOURS_REQ));
    if (!ag.hours || ag.mkt_closed) return false;
    rearm_wakeup(a, t + ag.c.interval_ns, slot);  // the delivered wakeup becomes the next one
    send_t(a, alloc_msg(M_QSPREAD));
    ag.awaiting_spread = true;
    return true;
  }

  // QSPREAD_RESP (the most frequent trader delivery; dispatched from deliver, outside trader_receive's frame)
  __attribute__((always_inline)) inline void trader_resp(int32_t a, const Msg& msg) {
    Agent& ag = agents[(size_t)a];
    if (msg.closed) ag.mkt_closed = true;
    ag.kb_has = msg.hb; ag.kb = msg.oprice;
    ag.ka_has = msg.ha; ag.ka = msg.oqty;
    if (ag.awaiting_spread) {
      if (!ag.mkt_closed) {
        if (ag.kind == A_NOISE) act_noise(a, ag, ag.kb_has && ag.kb != 0, ag.ka_has && ag.ka != 0);
        else act(a);
      }
      ag.awaiting_spread = false;
    }
  }

  __attribute__((always_inline)) inline void trader_receive(int32_t a, const Msg& msg) {
    Agent& ag = agents[(size_t)a];
    const uint8_t ty = msg.type;
    if (ty == M_ACCEPTED || ty == M_EXECUTED || ty == M_CANCELLED) {  // one emit_order copy for the three order events
      const bool ex = ty == M_EXECUTED;
      static constexpr uint8_t CODE_OF[3] = {C_ACCEPTED, C_PARTIAL, C_CANCELLED};  // by ty - M_ACCEPTED
      if (ex) emit_order_t<true>(C_PARTIAL, a, msg);
      else emit_order_t<false>(CODE_OF[ty - M_ACCEPTED], a, msg);
      if (ag.kind == A_MM) {
        if (ex) {
          if (dict_has(msg.oid)) {
            size_t p = (size_t)dpa.p[(size_t)msg.oid] - 1;
            if (msg.oqty >= ag.dict[p].qty) dict_delete(ag, msg.oid);
            else ag.dict[p].qty -= msg.oqty;
          }
        } else {
          // An accept changes nothing and a cancel deletes the order's entry; which one it is is a coin flip, so both do the same
          // few instructions (an accept, or an order the dictionary does not hold, only touches the scratch entry in front of it)
          const uint32_t idx = dpa.p[(size_t)msg.oid];
          const uint32_t d = (uint32_t)(ty == M_CANCELLED) & (uint32_t)(idx != 0);
          ag.dict.p[(size_t)idx - 1].oid |= -(int64_t)d;
          dpa.p[(size_t)msg.oid] = idx & (d - 1);
          ag.dict_live -= d;
        }
      }
      return;
    }
    if (ty == M_HOURS) {
      if (!ag.hours) {
        ag.hours = true;
        set_wakeup(a, mkt_open);
      }
    } else if (ty == M_CLOSED) {
      ag.mkt_closed = true;
    }
  }

  // Deterministic-latency loop, out of line (rare there): a release entry popped for recipient r is turned into
  // the parked message to deliver now (returned as sender << 32 | slot, or ~0 when there is nothing to deliver), and
  // a message popped while its recipient r is busy is parked on r (one release entry stands for the parked set).
  __attribute__((noinline)) uint64_t release_pop_cold(uint32_t eslot, uint32_t egen, int64_t* const AT) {
    const int32_t r = (int32_t)(eslot & ~REL);
    Park& P = park[(size_t)r];
    if (egen != P.gen || P.h.empty()) return ~0ULL;  // superseded release entry
    if (AT[(size_t)r] > current_time) {
      push_release(r);
      return ~0ULL;
    }
    auto pm = park_pop(P);
    if (!P.h.empty()) {  // re-arm the release for the rest at the agent's next free time (as deliver() will set it)
      AT[(size_t)r] = current_time + (r == 0 ? xdelay : TDELAY);
      push_release(r);
    }
    if constexpr (W) return ((uint64_t)key_s(pm.first) << 32) | pm.second;
    return ((pm.first >> 49) << 32) | pm.second;
  }
  __attribute__((noinline)) void park_busy_cold(uint64_t ek, uint32_t eslot, int32_t r) {
    Park& P = park[(size_t)r];
    const bool newmin = P.h.empty() || ek < P.h.back().first;
    park_push(P, ek, eslot);
    if (newmin) {
      P.gen++;
      push_release(r);
    }
  }
  // ------------------------------------------------------------ Kernel.initialize + runner
  // The event loop is compiled twice: run_m12 (below) and run_park (the park-first policy, at the end of the class), chosen at setup.
  T3_NOSSP void run() {
    if constexpr (!G) {
      if (park_on) {
        run_park();
        return;
      }
    }
    run_m12();
  }
  T3_NOSSP __attribute__((noinline)) void run_m12() {
    current_time = start_time;
    if constexpr (G) gqp->gnow = (uint64_t)start_time;  // the queue holds absolute times
    set_wakeup(0, mkt_close);                                          // ExchangeAgent.kernel_initializing
    for (int32_t a = 0; a < n_agents; a++) set_wakeup(a, start_time);  // Agent.kernel_starting
    GroupQ* const gl = gqp.get();  // G only: messages come from the time-grouped list (GroupQ), trader deliveries
                                   // already spaced by TDELAY at push time (see GroupQ::push_x_g)
    uint64_t cs = ((uint64_t)start_time >> q.SHIFT) & EventQueue::NBM;  // wheel scan start: bucket (mod NB) of the last popped time
    uint64_t ft = 0;                        // G: the started front group of the message list
    const GroupQ::SItem* fcur = nullptr;
    const GroupQ::SItem* fend = nullptr;
    int64_t* const AT = agent_time.data();  // sized at setup, never reallocated
    Msg* const PB = pool.base;              // reserved address range: never moves
    uint32_t ckl = ck;                      // NL: deliveries left in the block of 128 (a local, see nl_checkpoint)
    while (current_time <= stop_time) {  // current_time is never 0 (start_time = DATE_NS)
      QItem e;
      int32_t r, s;
      uint32_t slot;
      if constexpr (G) {
        bool from_slow = false;  // G: release entries only ever come out of the heap
        if (!gl->pop_fast_g(q, e, ft, fcur, fend)) {
          e = q.pop_slow();
          if (e.t == EventQueue::UMAX) break;
          gl->gnow = e.t;
          from_slow = true;
        }
        current_time = (int64_t)e.t;
        if (from_slow && (e.slot & REL)) {
          const uint64_t rp = release_pop_cold(e.slot, e.gen, AT);
          if (rp == ~0ULL) continue;
          r = (int32_t)(e.slot & ~REL);
          s = (int32_t)(rp >> 32);
          slot = (uint32_t)rp;
        } else {
          r = key_r(e.k);
          if (AT[(size_t)r] > current_time) {  // rare: the sends already enqueue most messages at their delivery time
            park_busy_cold(e.k, e.slot, r);
            continue;
          }
          s = key_s(e.k);
          slot = e.slot;
        }
      } else {
        if (q.pop_fast(e, cs)) {
          current_time = (int64_t)e.t;
          r = (int32_t)e.gen;  // the recipient, stored with the wheel node
        } else {  // lane / heap item; release entries only ever come out of here
          e = q.pop_slow();
          if (e.t == EventQueue::UMAX) break;
          current_time = (int64_t)e.t;
          cs = (e.t >> q.SHIFT) & EventQueue::NBM;
          if (e.slot & REL) {
            r = (int32_t)(e.slot & ~REL);
            Park& P = park[(size_t)r];
            if (e.gen != P.gen || P.h.empty()) continue;  // superseded release entry
            if (AT[(size_t)r] > current_time) {
              push_release(r);
              continue;
            }
            auto pm = park_pop(P);
            s = key_s(pm.first);
            slot = pm.second;
            if (!P.h.empty()) {  // re-arm the release for the rest at the agent's next free time (as deliver() will set it)
              AT[(size_t)r] = current_time + (r == 0 ? xdelay : TDELAY);
              push_release(r);
            }
            goto have_msg;
          }
          r = key_r(e.k);
        }
        if (AT[(size_t)r] > current_time) {  // recipient busy
          if (__builtin_expect(r == n_agents, 0)) {  // a release entry from the wheel: e.slot is the agent's reserved node
            const int32_t ra = (int32_t)(EventQueue::MAX_SLOTS - 1 - (size_t)e.slot);
            Park& P = park[(size_t)ra];
            P.relq = false;
            if (q.nodes[e.slot].pad2 != P.gen || P.h.empty()) continue;  // superseded release entry
            if (AT[(size_t)ra] > current_time) {
              push_release(ra);
              continue;
            }
            auto pm = park_pop(P);
            s = key_s(pm.first);
            slot = pm.second;
            r = ra;
            if (!P.h.empty()) {  // re-arm the release for the rest at the agent's next free time (as deliver() will set it)
              AT[(size_t)ra] = current_time + (ra == 0 ? xdelay : TDELAY);
              push_release(ra);
            }
            goto have_msg;
          }
          if (PB[e.slot].t_recv == current_time) {  // first time: requeue at its next free time (same key), as the reference does
            q.push(QItem{(uint64_t)AT[(size_t)r], e.k, e.slot, (uint32_t)r}, (uint64_t)(AT[(size_t)r] - current_time));
            continue;
          }  // requeued before: park (a burst of messages for one recipient would otherwise be requeued once per message ahead of it)
          Park& P = park[(size_t)r];
          const bool newmin = P.h.empty() || e.k < P.h.back().first;
          park_push(P, e.k, e.slot);
          if (newmin) {
            P.gen++;
            push_release(r);
          }
          continue;
        }
        s = key_s(e.k);
        slot = e.slot;
      }
    have_msg:
      deliver(AT, PB, s, r, slot, ckl);
#ifdef T3_DIAG_PHASES
      if (++diag_n == diag_next && diag_k < 16) { diag_t[diag_k++] = std::chrono::steady_clock::now(); diag_next *= 2; }
#endif
    }
    ck = ckl;
  }

  // NL: the deliveries are counted in blocks of 128; at the end of a block the trace stream's checkpoint runs (what the ledger's did)
  // (statics: the NL instance exists once per process, and the other instances keep their layout)
  static inline uint32_t ck = 128;
  static inline uint64_t ndeliv = 0;
  // NL with a trace-ahead encoder thread (TraceAhead in main.cpp): the thread encodes the pages of rows that no longer move while the
  // run goes on. Rows of the timestamp group that is still open move, so fin_pub (published at the checkpoints) is the first row of
  // that group. A finished row never changes except its msg_type code (an order's previous ORDER_FILLED row becomes PARTIAL_FILL
  // when the order executes again): every such change stores the current epoch in lastflip_mem[(page + 1) * 16] (slot 0: the scratch row), and
  // epoch_pub is the newest epoch whose changes are visible to a thread that read it. A page encoded after reading epoch E is
  // still right at the end of the run if its lastflip is <= E.
  alignas(64) static inline std::atomic<bool> ahead_on{false};  // (set once, possibly while the run is going on: the trace-ahead thread may start late)
  alignas(64) static inline uint32_t lastflip_mem[8195 * 16];  // one cache line per page (the engine writes it, the encoder thread reads it)
  alignas(64) static inline uint32_t epoch_now = 1;
  alignas(64) static inline std::atomic<uint64_t> fin_pub{0};
  alignas(64) static inline std::atomic<uint32_t> epoch_pub{0};
  // mut_pub: rows below it cannot be relabelled by an order that rests in the book or by an execution message already on its way (nor
  // have they moved). The latest ORDER_EXECUTED row of an order is the one its next execution relabels, so at a scan of the book the bound is
  // the oldest of these rows over the resting orders and over the orders that left the book less than rm_window ago, and no later than fin
  // (an order that enters the book after the scan gets its first execution row after the scan). Never decreases.
  alignas(64) static inline std::atomic<uint64_t> mut_pub{0};
  // Orders that left the book recently and had an ORDER_EXECUTED row before: an execution message already on its way relabels that row.
  // They keep bounding mut_pub for rm_window ns of simulated time (the longest a message takes).
  struct Rm { int64_t t; uint32_t row; };
  static inline Rm rm_ring[1024];
  static inline uint32_t rm_head = 0;
  static inline int64_t rm_window = 5000000;
  __attribute__((always_inline)) inline void note_removed(int64_t oid) {
    const uint32_t le = oi.p[(size_t)oid].le;
    if (__builtin_expect(le != 0, 0)) {
      Rm& r = rm_ring[rm_head++ & 1023];
      r.t = current_time;
      r.row = le - 1;
    }
  }
  static inline uint32_t scan_ctr = 0, scan_gap = 64;  // checkpoints since the last scan / until the next one (a deep book is scanned less often)
  static inline uint64_t fin_last = 0;
  __attribute__((noinline)) void scan_mutable() {
    uint64_t m = fin_last;
    uint32_t visited = 0;
    const BO* const B = bo.data();
    for (int sd = 0; sd < 2; sd++) {
      for (const Level* l = lv[sd].b; l != lv[sd].e; ++l) {
        for (int32_t sl = l->head; sl >= 0; sl = B[sl].next) {
          const uint32_t le = oi.p[(size_t)B[sl].oid].le;
          if (le && (uint64_t)le - 1 < m) m = (uint64_t)le - 1;
          visited++;
        }
      }
    }
    for (uint32_t k = 0; k < 1024 && k < rm_head; k++) {  // newest first
      const Rm& r = rm_ring[(rm_head - 1 - k) & 1023];
      if (r.t + rm_window < current_time) break;
      if (r.row < m) m = r.row;
    }
    mut_pub.store(m, std::memory_order_release);
    scan_ctr = 0;
    scan_gap = std::max<uint32_t>(64, visited / 2);
  }
  // the publications of a checkpoint for the trace-ahead thread
  __attribute__((always_inline)) inline void ahead_publish() {
    const TraceRow* p = trace.cur;
    size_t k = 0;
    while (p > trace.base && p[-1].t == current_time && k < 1024) { --p; ++k; }  // the rows of the open group
    if (k < 1024) {
      fin_last = (uint64_t)(p - trace.base);
      fin_pub.store(fin_last, std::memory_order_release);
    }
    if (T3_AA_SCAN && ++scan_ctr >= scan_gap) scan_mutable();
    epoch_pub.store(epoch_now, std::memory_order_release);
    epoch_now++;
  }
  __attribute__((noinline)) void nl_checkpoint(uint32_t& ckl) {
    ckl = 128;
    ndeliv += 128;
    trace.checkpoint();
    ord_pub.store((uint64_t)order_counter, std::memory_order_relaxed);
    if (ahead_on.load(std::memory_order_relaxed)) ahead_publish();
  }
  // the ledger instance: runs with the ledger stream's checkpoint (every 128 deliveries)
  __attribute__((noinline)) void ledger_checkpoint() {
    trace.checkpoint();
    if constexpr (TA) {
      if (ahead_on.load(std::memory_order_relaxed)) ahead_publish();
    }
  }
  // an order left the book (see note_removed): only the instances that can have a trace-ahead thread record it, the ledger instance only while one runs
  __attribute__((always_inline)) inline void order_left(int64_t oid) {
    if constexpr (NL) note_removed(oid);
    else if constexpr (TA) { if (__builtin_expect(ahead_on.load(std::memory_order_relaxed), 0)) note_removed(oid); }
  }
  // messages delivered so far (= ledger rows)
  int64_t n_messages() const {
    if constexpr (NL) return (int64_t)ndeliv + 128 - (int64_t)ck;
    else return (int64_t)(led.size() + led.sunk);
  }
  struct NoRow {  // stands for the ledger row of a delivery when there is no ledger: the stores to it vanish
    struct Sink {
      template <class T> Sink& operator=(const T&) { return *this; }
    } t_recv;
  };
  __attribute__((always_inline)) inline std::conditional_t<NL, NoRow, LedgerRow&> ledger_row(Msg& m, uint32_t& ckl) {
    if constexpr (NL) {
      if (__builtin_expect(--ckl == 0, 0)) nl_checkpoint(ckl);
      return NoRow();
    } else {
      causal_uid = m.id;
#ifdef EXP_NOLEDGER
      static LedgerRow Ldummy;
      LedgerRow& L = Ldummy;
#else
      LedgerRow& L = led.push();
#endif
      std::memcpy(&L, &m, sizeof(LedgerRow));
      return L;
    }
  }
  __attribute__((always_inline)) inline void deliver(int64_t* AT, Msg* PB, int32_t s, int32_t r, uint32_t slot, uint32_t& ckl) {
    Msg& m = PB[slot];
    auto&& L = ledger_row(m, ckl);
    bool reused;
    if (r == 0) {
      AT[0] = current_time + xdelay;  // xdelay may change below
      if (m.type == M_WAKEUP) {
        L.t_recv = current_time;
        exchange_wakeup(current_time);
        reused = false;
      } else {
        reused = exchange_receive(current_time, s, slot, m);
      }
    } else {
      AT[(size_t)r] = current_time + TDELAY;
      const uint8_t ty = m.type;
      if (ty == M_WAKEUP) {
        L.t_recv = current_time;
        reused = trader_wakeup(r, current_time, slot);
      } else {
        if (ty == M_QSPREAD_RESP) trader_resp(r, m);
        else trader_receive(r, m);
        reused = false;
      }
    }
    {  // free_slot(slot) after processing unless the slot was reused (sends made meanwhile never reuse this slot); branch-free
      const uint32_t fh = free_head;
      m.causal = reused ? m.causal : fh;
      free_head = reused ? fh : slot;
    }
  }

  // ------------------------------------------------------------ end of abides_fork.trace.extract_trace
  void finalize() {
    if (label_live) return;
    // same labelling as close_execs, applied once over the whole trace from the back: the last ORDER_EXECUTED row of
    // each order is the FILLED one, every earlier one stays PARTIAL (a group close never moves rows already written)
    OInfo* const seen = oi.p;
    for (TraceRow* p = trace.cur; p > trace.base;) {
      --p;
      if (p->code != C_PARTIAL) continue;
      uint32_t& s = seen[(size_t)p->oid].le;
      if (s) continue;
      s = 1;
      p->code = C_FILLED;
    }
  }

  int64_t det_lat = 0;  // G: the latency of every message
  std::unique_ptr<GroupQ> gqp;  // G: time-grouped message list (see queue.h)
  EventQueue q;  // declared last: the queue's size must not shift the offsets of the members above

  // ------------------------------------------------------------ park-first policy (see park_policy)
  // run_park differs from run_m12 only in how a message for a busy recipient is handled: it is requeued once (its first bounce) when
  // no message waits for the recipient, else parked at once; the parked set is a short ascending array, represented in the queue by
  // one release candidate (EventQueue::rel_*) carrying the set's smallest key at the recipient's next free time. Requeues have no
  // observable effect in the reference, so the delivery sequence is identical. Everything else is shared with run_m12.
  struct PE {
    uint64_t k;
    uint32_t slot, pad;
  };
  struct ParkA {  // the messages parked on one agent: a[head..n) ascending by key (the minimum first)
    PE* a = nullptr;
    uint32_t head = 0, n = 0, cap = 0;
    ParkA() = default;
    ParkA(const ParkA&) = delete;
    ~ParkA() { if (a) free(a - 1); }  // a[-1] is a sentinel entry with key 0 (below every key)
    bool empty() const { return head == n; }
  };
  std::vector<ParkA> parka;
  bool park_on = false;
  __attribute__((noinline)) void setup_park() {
    parka = std::vector<ParkA>((size_t)n_agents);
    q.rel_init((size_t)n_agents);
    for (auto& pk : parka) {
      pk.cap = 64;
      PE* const raw = (PE*)malloc(((size_t)pk.cap + 1) * sizeof(PE));
      if (!raw) throw std::bad_alloc();
      std::memset((void*)raw, 0, ((size_t)pk.cap + 1) * sizeof(PE));  // touched now, like the other initial capacities
      pk.a = raw + 1;
    }
  }
  // The parked set is a short array in ascending key order (the minimum at a[head]): a pop advances head. A message that arrives
  // later was mostly sent later, so it has one of the larger keys and its insertion shifts few entries; a new minimum takes the slot
  // in front of head when the set was popped before.
  __attribute__((noinline)) static void park_grow(ParkA& P) {
    const uint32_t live = P.n - P.head;
    if (P.head >= P.cap / 2 && P.head) {
      std::memmove((void*)P.a, (const void*)(P.a + P.head), live * sizeof(PE));
      P.head = 0;
      P.n = live;
    } else {
      P.cap = P.cap ? P.cap * 2 : 64;
      PE* const r = (PE*)realloc(P.a ? P.a - 1 : nullptr, ((size_t)P.cap + 1) * sizeof(PE));
      if (!r) throw std::bad_alloc();
      r[0] = PE{0, 0, 0};
      P.a = r + 1;
    }
  }
  // insert into the sorted array: the new entry goes to the end and moves down by compare-exchange steps without branches
  // (where it ends up is a coin flip); a longer move (a deep set, keys of other senders) is one binary search and one block move;
  // a[head - 1] is below every live key
  static inline void park_add(ParkA& P, uint64_t k, uint32_t slot) {  // the set is not empty
    if (__builtin_expect(P.n == P.cap, 0)) park_grow(P);
    PE* const a = P.a;
    int64_t j = P.n++;
    a[j].k = k;
    a[j].slot = slot;
#pragma GCC unroll 3
    for (int t = 0; t < 3; t++) {
      const uint64_t kx = a[j - 1].k, ky = a[j].k;
      const uint32_t sx = a[j - 1].slot, sy = a[j].slot;
      const uint64_t m = -(uint64_t)(kx > ky);
      const uint32_t m32 = (uint32_t)m;
      a[j - 1].k = (ky & m) | (kx & ~m);
      a[j].k = (kx & m) | (ky & ~m);
      a[j - 1].slot = (sy & m32) | (sx & ~m32);
      a[j].slot = (sx & m32) | (sy & ~m32);
      j -= (int64_t)(m & 1);
    }
    if (__builtin_expect(a[j - 1].k > a[j].k, 0)) {  // a deep move: binary search for the place, one block move
      const PE t = a[j];
      int64_t lo = (int64_t)P.head, hi = j - 1;  // a[hi].k > t.k; the place is in [lo, hi]
      while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (a[mid].k > t.k) hi = mid;
        else lo = mid + 1;
      }
      std::memmove((void*)(a + hi + 1), (const void*)(a + hi), (size_t)(j - hi) * sizeof(PE));
      a[hi] = t;
    }
  }
  // A message popped while its recipient r is busy is parked on r; the set is represented in the queue by one release candidate:
  // (r's free time, the set's smallest key)
  __attribute__((noinline)) void park_busy(uint64_t ek, uint32_t eslot, int32_t r, int64_t* const AT) {
    ParkA& P = parka[(size_t)r];
    if (P.empty()) {
      P.a[0].k = ek;
      P.a[0].slot = eslot;
      P.head = 0;
      P.n = 1;
      q.rel_add((uint32_t)r, (uint64_t)AT[(size_t)r], ek);
    } else if (ek < P.a[P.head].k) {  // the new smallest key
      if (P.head) {
        PE& x = P.a[--P.head];
        x.k = ek;
        x.slot = eslot;
      } else {
        park_add(P, ek, eslot);
      }
      q.rel_lower((uint32_t)r, (uint64_t)AT[(size_t)r], ek);
    } else {
      park_add(P, ek, eslot);
    }
  }
  // A release candidate for agent r was popped at current_time: the parked message to deliver now is returned in s/slot, or false
  // when there is nothing to deliver yet (r took another message at this time and is busy again).
  __attribute__((always_inline)) inline bool release_pop(int32_t r, int64_t* const AT, int32_t& s, uint32_t& slot) {
    ParkA& P = parka[(size_t)r];
    const PE pm = P.a[P.head];
    if (AT[(size_t)r] > current_time) {
      q.rel_set_best((uint64_t)AT[(size_t)r], pm.k);
      return false;
    }
    if (++P.head != P.n) {  // the candidate moves to the agent's next free time (as deliver() will set it) with the next key
      AT[(size_t)r] = current_time + (r == 0 ? xdelay : TDELAY);
      q.rel_set_best((uint64_t)AT[(size_t)r], P.a[P.head].k);
    } else {
      P.head = P.n = 0;
      q.rel_clear_best();
    }
    s = key_s(pm.k);
    slot = pm.slot;
    return true;
  }
  T3_NOSSP __attribute__((noinline)) void run_park() {
    static_assert(!G, "the park-first loop is for the general latency models");
    current_time = start_time;
    set_wakeup(0, mkt_close);                                          // ExchangeAgent.kernel_initializing
    for (int32_t a = 0; a < n_agents; a++) set_wakeup(a, start_time);  // Agent.kernel_starting
    uint64_t cs = ((uint64_t)start_time >> q.SHIFT) & EventQueue::NBM;  // wheel scan start: bucket (mod NB) of the last popped time
    int64_t* const AT = agent_time.data();  // sized at setup, never reallocated
    Msg* const PB = pool.base;              // reserved address range: never moves
    uint32_t ckl = ck;                      // NL: deliveries left in the block of 128 (a local, see nl_checkpoint)
    while (current_time <= stop_time) {  // current_time is never 0 (start_time = DATE_NS)
      QItem e;
      int32_t r, s;
      uint32_t slot;
      if (q.pop_fast(e, cs)) {
        current_time = (int64_t)e.t;
        r = (int32_t)e.gen;  // the recipient, stored with the wheel node
      } else {  // lane / heap item / release candidate
        if (__builtin_expect(q.slow_src != 2, 0)) {
          if (q.slow_src == 3) {  // a release candidate: the agent's smallest parked message
            current_time = (int64_t)q.slow.t;
            cs = (q.slow.t >> q.SHIFT) & EventQueue::NBM;
            r = (int32_t)(q.slow.slot & ~REL);
            if (!release_pop(r, AT, s, slot)) continue;
            goto have_msg;
          }
          e = q.pop_slow_cold_p();
        } else {
          e = q.pop_lane_p();
        }
        if (e.t == EventQueue::UMAX) break;
        current_time = (int64_t)e.t;
        cs = (e.t >> q.SHIFT) & EventQueue::NBM;
        r = key_r(e.k);
      }
      if (AT[(size_t)r] > current_time) {  // recipient busy
        if (parka[(size_t)r].empty() && PB[e.slot].t_recv == current_time) {  // first time and nobody waits: requeue at its next free time (same key), as the reference does
          q.push(QItem{(uint64_t)AT[(size_t)r], e.k, e.slot, (uint32_t)r}, (uint64_t)(AT[(size_t)r] - current_time));
          continue;
        }
        park_busy(e.k, e.slot, r, AT);
        continue;
      }
      s = key_s(e.k);
      slot = e.slot;
    have_msg:
      deliver(AT, PB, s, r, slot, ckl);
    }
    ck = ckl;
  }
};

using Sim = SimT<false>;     // general latency models
using SimDet = SimT<true>;   // deterministic latency (Config::lat == L_DETERMINISTIC)
using SimNL = SimT<false, true>;  // general latency models, no message ledger (see SimT)
using SimW = SimT<false, false, true>;  // 32768 agents or more (see SimT): general latency models
using SimDetW = SimT<true, false, true>;  // ... deterministic latency

}  // namespace t3
