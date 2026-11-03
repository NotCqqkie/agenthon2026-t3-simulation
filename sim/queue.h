#pragma once
// Exact min-priority queue over unique 128-bit keys (t = time offset, k = sender/recipient/message-id).
//  - calendar wheel (16384 buckets, width 2^SHIFT ns chosen per scenario) for message slots; nodes are indexed by slot and each
//    bucket is a singly linked list kept sorted by key;
//  - FIFO lanes for periodic re-arm wakeups pushed in non-decreasing key order;
//  - binary heap for everything else (far-future, release entries, out-of-order lane pushes).
// With a deterministic latency model (every message arrives exactly `latency` after it is sent) the wheel is
// replaced by GroupQ, a small ring of equal-time groups (see the end of this file).
// pop() returns the global minimum key, so the order equals a single binary heap's.
#include <sys/mman.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace t3 {

struct QItem {
  uint64_t t, k;
  uint32_t slot, gen;
};
static inline bool lt2(uint64_t at, uint64_t ak, uint64_t bt, uint64_t bk) {
#if defined(__x86_64__)  // cmp + sbb on the two halves (no __int128 register shuffling, no dead result store); the carry is the answer
  bool lt;
  __asm__("cmp %[bk], %[ak]\n\tsbb %[bt], %[at]" : "=@ccc"(lt), [at] "+r"(at) : [ak] "r"(ak), [bk] "rm"(bk), [bt] "rm"(bt));
  return lt;
#else
  return (((unsigned __int128)at << 64) | ak) < (((unsigned __int128)bt << 64) | bk);
#endif
}
static inline uint64_t ctz64(uint64_t m) {  // m != 0; a 64-bit result (no sign-extension of the int __builtin_ctzll)
#if defined(__x86_64__)
  uint64_t r;
  __asm__("rep bsfq %1, %0" : "=r"(r) : "r"(m) : "cc");
  return r;
#else
  return (uint64_t)__builtin_ctzll(m);
#endif
}
static inline bool qless(const QItem& a, const QItem& b) { return lt2(a.t, a.k, b.t, b.k); }

struct Lane {
  std::vector<QItem> ring;
  size_t head = 0, n = 0, mask = 0;
  uint64_t tail_t = 0, tail_k = 0;
  void init(size_t cap) {
    size_t c = 16;
    while (c < cap) c <<= 1;
    ring.assign(c, QItem{0, 0, 0, 0});
    mask = c - 1;
    head = n = 0;
  }
  void grow() {
    std::vector<QItem> r(ring.size() * 2);
    for (size_t i = 0; i < n; i++) r[i] = ring[(head + i) & mask];
    ring.swap(r);
    mask = ring.size() - 1;
    head = 0;
  }
};

struct EventQueue {
  int SHIFT = 4;  // bucket width 2^SHIFT ns, chosen per scenario from its latency scale
  static constexpr uint64_t NB = 16384;
  static constexpr uint64_t NBM = NB - 1;
  static constexpr size_t MAX_SLOTS = size_t(1) << 26;
  static constexpr uint64_t UMAX = ~0ULL;
  struct Node {
    uint64_t t, k;
    int32_t next;
    uint32_t rcp;  // recipient (QItem::gen of the pushed item): saves decoding it from the key at pop
    uint64_t pad2;  // 32 bytes: nodes never straddle a cache line and index with one shift
  };
  struct Bucket {
    int32_t head, tail;  // first / last node of the sorted list (tail valid while head >= 0)
  };
  Bucket bk[NB + 1];  // bk[NB] is the "wheel empty" bucket: its only node has key {UMAX, UMAX}, which never beats `slow`
  uint64_t bits[NB / 64];
  uint64_t sum[NB / 64 / 64];  // summary: bit w set whenever bits[w] may be non-zero (cleared lazily by scans)
  Node* nodes = nullptr;  // indexed by message slot (reserved address range)
  uint64_t lim_dt = 0;  // a message goes to the wheel when it arrives less than this many ns after the last popped time
  struct QHeap {  // raw storage: no vector accessors, and growth stays out of line
    QItem* p = nullptr;
    size_t n = 0, cap = 0;
    __attribute__((noinline)) void grow() {
      cap = cap ? cap * 2 : 256;
      p = (QItem*)realloc(p, cap * sizeof(QItem));
      if (!p) throw std::bad_alloc();
    }
    void reserve(size_t c) {
      while (cap < c) grow();
      memset((void*)p, 0, cap * sizeof(QItem));  // setup: touch the initial pages
    }
    ~QHeap() { free(p); }
    bool empty() const { return n == 0; }
    QItem& operator[](size_t i) { return p[i]; }
  } heap;
  std::vector<Lane> lanes;
  std::vector<QItem> lf;  // lane front items; {UMAX,UMAX} when the lane is empty
  // "slow" candidate: min over heap top and lane fronts (sentinel {UMAX,UMAX} when both are empty)
  QItem slow{UMAX, UMAX, 0, 0};
  int slow_src = 0;  // 0 none, 1 heap, 2 lane
  size_t slow_lane = 0;
  QItem lmin{UMAX, UMAX, 0, 0};  // min over lane fronts
  size_t lmin_i = 0;
  // second minimum over the lane fronts (the minimum excluding lane lmin_i), valid until a lane gets a new front
  // (then {0, 0}, which never beats anything): lets pop_slow() skip the lane rescan while the same lane goes on
  uint64_t l2t = 0, l2k = 0;

  EventQueue() {
    void* p = mmap(nullptr, (MAX_SLOTS + 1) * sizeof(Node), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    nodes = (Node*)p;
#ifdef MADV_NOHUGEPAGE
    madvise(p, (MAX_SLOTS + 1) * sizeof(Node), MADV_NOHUGEPAGE);
#endif
  }
  EventQueue(const EventQueue&) = delete;
  ~EventQueue() { munmap(nodes, (MAX_SLOTS + 1) * sizeof(Node)); }

  void preface(size_t bytes) {  // setup: fault in the first nodes (fixed size)
    volatile char* c = (volatile char*)nodes;
    for (size_t i = 0; i < bytes; i += 4096) c[i] = 0;
  }
  void init(int shift = 4) {
    SHIFT = shift;
    lim_dt = (NB - 1) << shift;  // dt < lim_dt  =>  bucket(t) - bucket(last popped) <= NB - 1
    for (uint64_t i = 0; i < NB; i++) bk[i].head = bk[i].tail = -1;
    bk[NB].head = bk[NB].tail = (int32_t)MAX_SLOTS;
    nodes[MAX_SLOTS].t = nodes[MAX_SLOTS].k = UMAX;
    nodes[MAX_SLOTS].next = -1;
    std::memset(bits, 0, sizeof bits);
    std::memset(sum, 0, sizeof sum);
    heap.reserve(256);
    slow = QItem{UMAX, UMAX, 0, 0};
    slow_src = 0;
    lmin = QItem{UMAX, UMAX, 0, 0};
    l2t = l2k = 0;
  }
  void lanes_ready() { lf.assign(lanes.size(), QItem{UMAX, UMAX, 0, 0}); }

  __attribute__((noinline)) void heap_push_grow(uint64_t et, uint64_t ek, uint32_t eslot, uint32_t egen) {
    heap.grow();  // out of line so that heap_push itself makes no call (no callee-saved register spills in its prologue)
    heap_push(et, ek, eslot, egen);
  }
  __attribute__((noinline)) void heap_push(uint64_t et, uint64_t ek, uint32_t eslot, uint32_t egen) {
    if (__builtin_expect(heap.n == heap.cap, 0)) {
      heap_push_grow(et, ek, eslot, egen);
      return;
    }
    QItem* const h = heap.p;
    size_t i = heap.n++;
    while (i > 0) {
      size_t p = (i - 1) >> 1;
      if (!lt2(et, ek, h[p].t, h[p].k)) break;
      h[i] = h[p];
      i = p;
    }
    h[i] = QItem{et, ek, eslot, egen};
    if (i == 0 && lt2(et, ek, slow.t, slow.k)) {  // new heap top beats the cached slow candidate
      slow = h[0];
      slow_src = 1;
    }
  }
  inline void heap_push(const QItem& e) { heap_push(e.t, e.k, e.slot, e.gen); }
  inline void heap_drop() {  // remove the top (caller has already copied it)
    QItem* const h = heap.p;
    const size_t m = --heap.n;
    const QItem last = h[m];
    if (m) {
      size_t i = 0;
      while (true) {
        size_t l = 2 * i + 1;
        if (l >= m) break;
        size_t r = l + 1;
        size_t c = (r < m && qless(h[r], h[l])) ? r : l;
        if (!qless(h[c], last)) break;
        h[i] = h[c];
        i = c;
      }
      h[i] = last;
    }
  }
  // lane-front minimum (lmin/lmin_i), recomputed only when a lane front is consumed
  void rescan_lanes() {
    lmin = QItem{UMAX, UMAX, 0, 0};
    lmin_i = 0;
    uint64_t st = UMAX, sk = UMAX;
    const size_t nl = lf.size();
    const QItem* f = lf.data();
    for (size_t i = 0; i < nl; i++) {
      if (lt2(f[i].t, f[i].k, lmin.t, lmin.k)) {
        st = lmin.t;
        sk = lmin.k;
        lmin = f[i];
        lmin_i = i;
      } else if (lt2(f[i].t, f[i].k, st, sk)) {
        st = f[i].t;
        sk = f[i].k;
      }
    }
    l2t = st;
    l2k = sk;
  }
  inline void refresh_slow() {
    if (!heap.empty() && qless(heap[0], lmin)) {
      slow = heap[0];
      slow_src = 1;
    } else if (lmin.t != UMAX) {
      slow = lmin;
      slow_src = 2;
      slow_lane = lmin_i;
    } else {
      slow = QItem{UMAX, UMAX, 0, 0};
      slow_src = 0;
    }
  }

  // release entries (flagged slots) always go to the heap
  __attribute__((always_inline)) inline void push_heap(const QItem& e) { heap_push(e); }

  // message slot -> wheel when near (hot path, inline); far events -> heap
  // dt = e.t - (last popped time) in ns (>= 0)
  __attribute__((always_inline)) inline void push(const QItem& e, uint64_t dt) {
    if (__builtin_expect(dt < lim_dt, 1)) {
      const uint64_t s = (e.t >> SHIFT) & NBM;
      Node& nd = nodes[e.slot];
      nd.t = e.t;
      nd.k = e.k;
      nd.rcp = e.gen;
      Bucket& B = bk[s];
      const int32_t h = B.head;
      if (h < 0) {
        nd.next = -1;
        B.head = B.tail = (int32_t)e.slot;
        bits[s >> 6] |= 1ULL << (s & 63);
        sum[s >> 12] |= 1ULL << ((s >> 6) & 63);  // (cleared lazily by scans)
        return;
      }
      if (lt2(e.t, e.k, nodes[h].t, nodes[h].k)) {
        nd.next = h;
        B.head = (int32_t)e.slot;
        return;
      }
      const int32_t tl = B.tail;
      if (lt2(nodes[tl].t, nodes[tl].k, e.t, e.k)) {
        nd.next = -1;
        nodes[tl].next = (int32_t)e.slot;
        B.tail = (int32_t)e.slot;
        return;
      }
      insert_after(h, e.slot, e.t, e.k);
      return;
    }
    heap_push(e.t, e.k, e.slot, e.gen);
  }
  // link slot into the sorted bucket list after node p (node p < new item)
  __attribute__((noinline)) void insert_after(int32_t p, uint32_t slot, uint64_t t, uint64_t k) {
    int32_t nx;
    while ((nx = nodes[p].next) >= 0 && !lt2(t, k, nodes[nx].t, nodes[nx].k)) p = nx;
    nodes[slot].next = nx;
    nodes[p].next = (int32_t)slot;
  }
  __attribute__((always_inline)) inline void push_lane(int li, uint64_t et, uint64_t ek, uint32_t eslot) {
    Lane& L = lanes[(size_t)li];
    if (L.n && lt2(et, ek, L.tail_t, L.tail_k)) {
      heap_push(et, ek, eslot, 0);
      return;
    }
    if (L.n == L.mask + 1) L.grow();
    QItem& r = L.ring[(L.head + L.n) & L.mask];
    r.t = et;
    r.k = ek;
    r.slot = eslot;
    r.gen = 0;
    if (!L.n) {
      const QItem e{et, ek, eslot, 0};
      lf[(size_t)li] = e;
      l2t = l2k = 0;
      if (qless(e, lmin)) {
        lmin = e;
        lmin_i = (size_t)li;
      }
      if (qless(e, slow)) {
        slow = e;
        slow_src = 2;
        slow_lane = (size_t)li;
      }
    }
    L.n++;
    L.tail_t = et;
    L.tail_k = ek;
  }

  // first non-empty wheel slot at or after s (the bucket of the last popped time; circular)
  inline uint64_t first_slot(uint64_t s) {
    const uint64_t w = s >> 6;
    const uint64_t m = bits[w] >> (s & 63);
    if (__builtin_expect(m != 0, 1)) return s + ctz64(m);
    const uint64_t w1 = (w + 1) & (NB / 64 - 1), w2 = (w + 2) & (NB / 64 - 1), w3 = (w + 3) & (NB / 64 - 1);
    if (bits[w1]) return (w1 << 6) | ctz64(bits[w1]);
    if (bits[w2]) return (w2 << 6) | ctz64(bits[w2]);
    if (bits[w3]) return (w3 << 6) | ctz64(bits[w3]);
    const uint64_t sm = sum[w >> 6] & ((~0ULL << (w & 63)) << 1);  // later words in the same summary word
    if (sm) {
      const uint64_t ww = ((w >> 6) << 6) | (uint64_t)__builtin_ctzll(sm);
      if (bits[ww]) return (ww << 6) | (uint64_t)__builtin_ctzll(bits[ww]);
    }
    return first_slot_far(w);
  }
  // later words (circular, ending with word w itself) via the summary bitmap
  __attribute__((noinline)) uint64_t first_slot_far(uint64_t w) {
    const uint64_t sw0 = w >> 6;
    for (uint64_t k = 0; k <= NB / 64 / 64; k++) {  // k = 0: summary word sw0 after w; then the following summary words, wrapping; k = last: sw0 again, bits up to w
      const uint64_t sw = (sw0 + k) & (NB / 64 / 64 - 1);
      uint64_t m = sum[sw];
      if (k == 0) m &= (~0ULL << (w & 63)) << 1;
      else if (k == NB / 64 / 64) m &= ~0ULL >> (63 - (w & 63));
      while (m) {
        const uint64_t ww = (sw << 6) | ctz64(m);
        if (bits[ww]) return (ww << 6) | ctz64(bits[ww]);
        sum[sw] &= ~(1ULL << (ww & 63));
        m &= m - 1;
      }
    }
    return NB;  // wheel empty
  }

  // slow path for the heap (or nothing): out of line
  __attribute__((noinline)) QItem pop_slow_cold() {
    const QItem e = slow;
    if (slow_src == 0) return e;
    heap_drop();
    refresh_slow();
    return e;
  }
  __attribute__((always_inline)) inline QItem pop_slow() {
    if (__builtin_expect(slow_src != 2, 0)) return pop_slow_cold();
    const QItem e = slow;
    Lane& L = lanes[slow_lane];
    L.head = (L.head + 1) & L.mask;
    const QItem nf = --L.n ? L.ring[L.head] : QItem{UMAX, UMAX, 0, 0};  // the lane's next item
    lf[slow_lane] = nf;
    if (lt2(nf.t, nf.k, l2t, l2k)) {  // still the minimum: same lane (slow_lane == lmin_i), l2 unchanged
      lmin = nf;
      if (!heap.empty() && qless(heap[0], nf)) {
        slow = heap[0];
        slow_src = 1;
      } else {
        slow = nf;  // slow_src and slow_lane are already this lane
      }
    } else {
      rescan_lanes();
      refresh_slow();
    }
    return e;
  }

  // Fast path: the next item comes from the wheel. Returns false when pop_slow() must supply it.
  // cs: bucket (mod NB) of the last popped time; updated when the wheel supplies the item
  __attribute__((always_inline)) inline bool pop_fast(QItem& out, uint64_t& cs) {
    const uint64_t wslot = first_slot(cs);
    const int32_t id = bk[wslot].head;
    const Node& nd = nodes[id];
    if (nd.t >= slow.t && (nd.t > slow.t || nd.k >= slow.k)) return false;
    const int32_t nx = nd.next;
    bk[wslot].head = nx;
    bits[wslot >> 6] ^= (uint64_t)(nx < 0) << (wslot & 63);  // the bucket's bit is set: it clears when the list ran out
    cs = wslot;
    out = QItem{nd.t, nd.k, (uint32_t)id, nd.rcp};
    return true;
  }

  // ---- Release candidates, for the park-first loop (SimT::run_park) only: nothing above reads this state, and the loop above
  // (pop_slow & co.) never sees slow_src 3. An agent whose parked messages wait for it to be free has one candidate, (the time it is
  // free, the smallest parked key); it is delivered like a queue item with slot REL | agent. rbest is the smallest candidate; the
  // cached slow candidate can be rbest (slow_src 3), which stays in place until the caller updates it (rel_set_best / rel_clear_best).
  static constexpr uint32_t REL = 0x80000000u;
  struct RelC {
    uint64_t t, k;
    uint32_t a, pad;
  };
  std::vector<RelC> rc;
  uint32_t nrc = 0, rbi = 0;  // number of candidates; index of rbest in rc
  QItem rbest{UMAX, UMAX, 0, 0};
  uint64_t l2pt = 0, l2pk = 0;  // the lane part of l2 (l2t/l2k = min(l2p, rbest) while valid; push_lane invalidates l2 to {0, 0})
  void rel_init(size_t n_agents) {
    rc.assign(n_agents, RelC{0, 0, 0, 0});
    nrc = rbi = 0;
    rbest = QItem{UMAX, UMAX, 0, 0};
  }
  inline void rescan_lanes_p() {
    rescan_lanes();
    l2pt = l2t;
    l2pk = l2k;
    if (lt2(rbest.t, rbest.k, l2t, l2k)) {
      l2t = rbest.t;
      l2k = rbest.k;
    }
  }
  inline void refresh_slow_p() {
    slow = lmin;
    slow_src = 2;
    slow_lane = lmin_i;
    if (!heap.empty() && qless(heap[0], slow)) {
      slow = heap[0];
      slow_src = 1;
    }
    if (qless(rbest, slow)) {
      slow = rbest;
      slow_src = 3;
    }
    if (slow.t == UMAX) slow_src = 0;
  }
  // the lane bound l2 follows rbest (while it is valid)
  inline void rel_bound() {
    if (l2t | l2k) {
      if (lt2(rbest.t, rbest.k, l2pt, l2pk)) {
        l2t = rbest.t;
        l2k = rbest.k;
      } else {
        l2t = l2pt;
        l2k = l2pk;
      }
    }
  }
  // rbest got smaller (or its slot changed): the lane bound and the slow candidate follow
  inline void rel_best_changed() {
    rel_bound();
    if (qless(rbest, slow)) {
      slow = rbest;
      slow_src = 3;
    }
  }
  // rbest changed (it was popped and then updated or removed)
  inline void rel_after() {
    rel_bound();
    refresh_slow_p();
  }
  // recompute rbest from the candidates (two or more)
  inline void rel_commit() {
    uint64_t bt = UMAX, bk = UMAX;
    uint32_t bi = 0;
    for (uint32_t i = 0; i < nrc; i++) {
      if (lt2(rc[i].t, rc[i].k, bt, bk)) {
        bt = rc[i].t;
        bk = rc[i].k;
        bi = i;
      }
    }
    rbi = bi;
    rbest = QItem{bt, bk, REL | rc[bi].a, 0};
    rel_after();
  }
  // agent a has no candidate yet
  inline void rel_add(uint32_t a, uint64_t t, uint64_t k) {
    const uint32_t i = nrc++;
    rc[i] = RelC{t, k, a, 0};
    if (lt2(t, k, rbest.t, rbest.k)) {
      rbi = i;
      rbest = QItem{t, k, REL | a, 0};
      rel_best_changed();
    }
  }
  // agent a has a candidate and its smallest parked key dropped to k (its free time is t)
  inline void rel_lower(uint32_t a, uint64_t t, uint64_t k) {
    uint32_t i = 0;
    while (rc[i].a != a) i++;
    rc[i].t = t;
    rc[i].k = k;
    if (i == rbi) {
      rel_commit();
    } else if (lt2(t, k, rbest.t, rbest.k)) {
      rbi = i;
      rbest = QItem{t, k, REL | a, 0};
      rel_best_changed();
    }
  }
  // the candidate that was just popped (rbest) moves to a new time and key
  inline void rel_set_best(uint64_t t, uint64_t k) {
    rc[rbi].t = t;
    rc[rbi].k = k;
    if (nrc == 1) {
      rbest.t = t;
      rbest.k = k;
      rel_after();
    } else {
      rel_commit();
    }
  }
  // the candidate that was just popped (rbest) is gone
  inline void rel_clear_best() {
    rc[rbi] = rc[--nrc];
    if (nrc == 0) {
      rbi = 0;
      rbest.t = rbest.k = UMAX;
      rel_after();
    } else {
      rel_commit();
    }
  }
  // slow path for the heap (or nothing), for the loop that sees release candidates: out of line
  __attribute__((noinline)) QItem pop_slow_cold_p() {
    const QItem e = slow;
    if (slow_src == 1) {
      heap_drop();
      refresh_slow_p();
    }
    return e;  // none (the end), or a release candidate: the caller updates it (rel_set_best / rel_clear_best)
  }
  // slow_src == 2: the next item is a lane's front
  __attribute__((always_inline)) inline QItem pop_lane_p() {
    const QItem e = slow;
    Lane& L = lanes[slow_lane];
    L.head = (L.head + 1) & L.mask;
    const QItem nf = --L.n ? L.ring[L.head] : QItem{UMAX, UMAX, 0, 0};  // the lane's next item
    lf[slow_lane] = nf;
    if (lt2(nf.t, nf.k, l2t, l2k)) {  // still the minimum: same lane (slow_lane == lmin_i), l2 unchanged
      lmin = nf;
      if (!heap.empty() && qless(heap[0], nf)) {
        slow = heap[0];
        slow_src = 1;
      } else {
        slow = nf;  // slow_src and slow_lane are already this lane
      }
    } else {
      rescan_lanes_p();
      refresh_slow_p();
    }
    return e;
  }
};

// Time-grouped message list used instead of the wheel when the latency model is deterministic: a small ring of
// groups of equal time, in time order; each group keeps its items sorted by key (insertion at push). The front
// group becomes "started" when it is first consumed; a started group takes no more pushes. Anything that does not
// fit (earlier than or equal to a started front, far away, deep walk, ring full) goes to the heap, so the global
// minimum is still popped. Periodic wakeups keep using the lanes.
struct GroupQ {
  struct SItem {
    uint64_t k;
    uint32_t slot, pad;
  };
  struct Grp {
    SItem* a;
    uint32_t n, cap;
  };
  static constexpr int GR = 256, GM = GR - 1;
  bool gstarted = false;  // the front group is partly consumed
  int rh = 0, rn = 0;     // ring head and size; position p is (rh + p) & GM
  int nfree = 0;
  uint64_t ghz = 0;       // items further than this beyond the last popped time go to the heap
  uint64_t gnow = 0;      // time of the last popped item
  uint64_t rt[GR];        // group times by ring slot
  uint8_t rg[GR];         // group storage id by ring slot
  uint8_t gfree[GR];      // free group storage ids (stack)
  Grp gs[GR];
  GroupQ() = default;
  GroupQ(const GroupQ&) = delete;
  ~GroupQ() {
    for (auto& g : gs) std::free(g.a);
  }
  void init(uint64_t horizon) {
    ghz = horizon;
    for (int i = 0; i < GR; i++) {
      gs[i].cap = 64;
      gs[i].n = 0;
      gs[i].a = (SItem*)std::malloc(sizeof(SItem) * gs[i].cap);
      std::memset(gs[i].a, 0, sizeof(SItem) * gs[i].cap);
      gfree[nfree++] = (uint8_t)i;
    }
  }

  __attribute__((noinline)) void grp_grow(Grp& g) {
    g.cap *= 2;
    g.a = (SItem*)std::realloc(g.a, sizeof(SItem) * g.cap);
  }
  inline void grp_add(Grp& g, uint32_t slot, uint64_t k) {
    uint32_t i = g.n;
    if (__builtin_expect(i == g.cap, 0)) grp_grow(g);
    SItem* const a = g.a;
    while (i > 0 && a[i - 1].k > k) {
      a[i] = a[i - 1];
      i--;
    }
    a[i].k = k;
    a[i].slot = slot;
    g.n++;
  }
  // new group holding e at ring position p (0..rn); false when the ring or the storage is full
  inline bool grp_insert(int p, const QItem& e) {
    if (__builtin_expect(rn == GR || nfree == 0, 0)) return false;
    if (p == 0) {
      rh = (rh + GM) & GM;
    } else {
      for (int i = rn; i > p; i--) {
        rt[(rh + i) & GM] = rt[(rh + i - 1) & GM];
        rg[(rh + i) & GM] = rg[(rh + i - 1) & GM];
      }
    }
    const uint8_t id = gfree[--nfree];
    rt[(rh + p) & GM] = e.t;
    rg[(rh + p) & GM] = id;
    rn++;
    Grp& g = gs[id];
    g.n = 1;
    g.a[0].k = e.k;
    g.a[0].slot = e.slot;
    return true;
  }
  // message sends: arrival within the horizon by construction
  struct XPre {
    uint64_t t = ~0ULL, maxk = 0;
    uint32_t cnt = 0;
  };
  std::vector<XPre> xpre;
  // exchange -> trader. Every trader is busy TDELAY after each delivery, so the n-th message (in key order) for a
  // trader among those arriving at the same time is delivered no earlier than arrival + n*TDELAY: enqueue it there
  // directly instead of re-queueing it n times. Only used while keys arrive in increasing order.
  __attribute__((always_inline)) inline void push_x_g(EventQueue& q, int32_t recipient, uint64_t qt, uint64_t qk, uint32_t slot, uint64_t tdelay) {
    XPre& x = xpre[(size_t)recipient];
    if (x.t != qt) {
      x.t = qt;
      x.cnt = 0;
      x.maxk = qk;
    } else if (qk > x.maxk) {
      x.maxk = qk;
      qt += tdelay * ++x.cnt;
    }
    push_msg_g(q, QItem{qt, qk, slot, 0});
  }
  // message sends: arrival within the horizon by construction
  __attribute__((always_inline)) inline void push_msg_g(EventQueue& q, const QItem& e) {
    int p = rn;  // e belongs after ring position p-1: walk back from the tail past later groups
    for (int steps = 0; p > 0; steps++) {
      const uint64_t t = rt[(rh + p - 1) & GM];
      if (t < e.t) break;
      if (t == e.t) {
        if (p == 1 && gstarted) {
          q.heap_push(e);
          return;
        }
        grp_add(gs[rg[(rh + p - 1) & GM]], e.slot, e.k);
        return;
      }
      if (steps == 5) {
        q.heap_push(e);
        return;
      }
      p--;
    }
    if ((p == 0 && gstarted) || !grp_insert(p, e)) q.heap_push(e);
  }
  inline void push_g(EventQueue& q, const QItem& e) {
    if (__builtin_expect(e.t - gnow > ghz, 0)) {
      q.heap_push(e);
      return;
    }
    push_msg_g(q, e);
  }
  __attribute__((always_inline)) inline bool pop_fast_g(EventQueue& q, QItem& out, uint64_t& ft, const SItem*& fcur,
                                                        const SItem*& fend) {
    if (__builtin_expect(!gstarted, 0)) {
      // start the front group once it can be next (a started group takes no more pushes)
      if (rn == 0 || rt[rh] > q.slow.t) return false;
      const Grp& g = gs[rg[rh]];
      ft = rt[rh];
      fcur = g.a;
      fend = g.a + g.n;
      gstarted = true;
      gnow = ft;
    }
    if (!lt2(ft, fcur->k, q.slow.t, q.slow.k)) return false;
    out = QItem{ft, fcur->k, fcur->slot, 0};
    if (++fcur == fend) {
      gfree[nfree++] = rg[rh];
      rh = (rh + 1) & GM;
      rn--;
      gstarted = false;
    }
    return true;
  }

};

}  // namespace t3
