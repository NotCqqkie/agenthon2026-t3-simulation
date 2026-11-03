#pragma once
// Dependency-free header-only Parquet writer (C++17, little-endian hosts).
// Output: 1 row group, DataPage V1 pages of kPageRows rows, all leaves OPTIONAL, definition levels RLE/bit-packed
// hybrid (bit width 1), DELTA_BINARY_PACKED for INT32/INT64, PLAIN dictionary page + RLE_DICTIONARY data pages for
// string columns. Pages are encoded uncompressed; compress() GZIP-compresses one afterwards (chosen per file by the caller).
// The caller streams each page's rows into IntPage/DictPage encoders (pages are independent, so they can be
// encoded on several threads); the file is written with one writev. Deterministic: identical inputs give
// byte-identical output, whatever the thread count.
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

#include <smmintrin.h>

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <zlib.h>

#include "sha256.h"

// ZB end-phase switches (compile-time; each defaults to on, -DT3_ZB_x=0 builds the variant without it):
//   T3_ZB_CHAIN   the digest and the trace file follow the pages on the encoder threads (no hasher/writer threads); the message file
//                 write is a task of the same queue; otherwise a hasher, a writer and a message-writer thread as in M12
//   T3_ZB_PREFILL message_trace.parquet is opened during the run and its page cache is pre-filled with zeros by idle helpers
//   T3_ZB_ARENA   page buffers come from a dense NOHUGEPAGE arena instead of malloc
//   T3_ZB_PARREL  the arena is released on several threads before _exit (needs T3_ZB_ARENA)
//   T3_ZB_READ    the scenario file is read with read(2) instead of an ifstream
//   T3_ZB_WDELTA  encoder threads of the end phase = threads + T3_ZB_WDELTA (with T3_ZB_CHAIN)
#ifndef T3_ZB_CHAIN
#define T3_ZB_CHAIN 1
#endif
#ifndef T3_ZB_PREFILL
#define T3_ZB_PREFILL 1
#endif
#ifndef T3_ZB_ARENA
#define T3_ZB_ARENA 1
#endif
#ifndef T3_ZB_PARREL
#define T3_ZB_PARREL 1
#endif
#ifndef T3_ZB_READ
#define T3_ZB_READ 1
#endif
#ifndef T3_ZB_WDELTA
#define T3_ZB_WDELTA 0
#endif

// T3_SEALED (release build): no developer switches, terse messages. T3_MSG(dev, sealed) picks the text;
// T3_DEVENV(name) reads a developer switch (always unset in a sealed build).
#ifndef T3_MSG
#ifdef T3_SEALED
#define T3_MSG(dev, sealed) sealed
#define T3_DEVENV(name) static_cast<const char*>(nullptr)
#else
#define T3_MSG(dev, sealed) dev
#define T3_DEVENV(name) getenv(name)
#endif
#endif

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "parquet.h requires a little-endian host");

namespace pq {

// pandas metadata ("pandas" key) copied from pandas 1.5.3 / pyarrow 15.0.2 reference files.
inline constexpr const char* TRACE_PANDAS_META = R"PQMETA({"index_columns": [], "column_indexes": [], "columns": [{"name": "t_ns", "field_name": "t_ns", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "agent_id", "field_name": "agent_id", "pandas_type": "int32", "numpy_type": "int32", "metadata": null}, {"name": "msg_type", "field_name": "msg_type", "pandas_type": "unicode", "numpy_type": "string", "metadata": null}, {"name": "side", "field_name": "side", "pandas_type": "unicode", "numpy_type": "string", "metadata": null}, {"name": "price", "field_name": "price", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "size", "field_name": "size", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "order_id", "field_name": "order_id", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}], "creator": {"library": "pyarrow", "version": "15.0.2"}, "pandas_version": "1.5.3"})PQMETA";
inline constexpr const char* MESSAGE_PANDAS_META = R"PQMETA({"index_columns": [], "column_indexes": [], "columns": [{"name": "seq", "field_name": "seq", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "t_recv_ns", "field_name": "t_recv_ns", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "t_send_ns", "field_name": "t_send_ns", "pandas_type": "int64", "numpy_type": "Int64", "metadata": null}, {"name": "latency_ns", "field_name": "latency_ns", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "src_id", "field_name": "src_id", "pandas_type": "int32", "numpy_type": "int32", "metadata": null}, {"name": "dst_id", "field_name": "dst_id", "pandas_type": "int32", "numpy_type": "int32", "metadata": null}, {"name": "message_id", "field_name": "message_id", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "msg_type", "field_name": "msg_type", "pandas_type": "unicode", "numpy_type": "string", "metadata": null}, {"name": "order_id", "field_name": "order_id", "pandas_type": "int64", "numpy_type": "Int64", "metadata": null}, {"name": "causal_parent", "field_name": "causal_parent", "pandas_type": "int64", "numpy_type": "Int64", "metadata": null}], "creator": {"library": "pyarrow", "version": "15.0.2"}, "pandas_version": "1.5.3"})PQMETA";

namespace detail {

// ---- Thrift compact protocol (write-only, raw pointer, caller guarantees capacity) ----
enum { T_I32 = 5, T_I64 = 6, T_BINARY = 8, T_LIST = 9, T_STRUCT = 12 };

struct TW {
  uint8_t* p;
  int last[8];
  int depth = 0;
  explicit TW(uint8_t* q) : p(q) { last[0] = 0; }
  void varint(uint64_t v) {
    while (v >= 0x80) { *p++ = uint8_t(v | 0x80); v >>= 7; }
    *p++ = uint8_t(v);
  }
  static uint64_t zz32(int32_t v) { return (uint32_t(v) << 1) ^ uint32_t(v >> 31); }
  static uint64_t zz64(int64_t v) { return (uint64_t(v) << 1) ^ uint64_t(v >> 63); }
  void field(int type, int id) {
    int d = id - last[depth];
    if (d > 0 && d <= 15) *p++ = uint8_t((d << 4) | type);
    else { *p++ = uint8_t(type); varint(zz32(int16_t(id))); }
    last[depth] = id;
  }
  void i32(int id, int32_t v) { field(T_I32, id); varint(zz32(v)); }
  void i64(int id, int64_t v) { field(T_I64, id); varint(zz64(v)); }
  void str(int id, const char* s, size_t n) { field(T_BINARY, id); varint(n); std::memcpy(p, s, n); p += n; }
  void str(int id, const std::string& s) { str(id, s.data(), s.size()); }
  void begin_struct(int id) { field(T_STRUCT, id); last[++depth] = 0; }
  void end_struct() { *p++ = 0; --depth; }
  void begin_list(int id, int elem, size_t n) {
    field(T_LIST, id);
    if (n < 15) *p++ = uint8_t((n << 4) | elem);
    else { *p++ = uint8_t(0xF0 | elem); varint(n); }
  }
  void begin_elem() { last[++depth] = 0; }  // struct element inside a list
  void end_elem() { *p++ = 0; --depth; }
  void list_str(const std::string& s) { varint(s.size()); std::memcpy(p, s.data(), s.size()); p += s.size(); }
};

// parquet.thrift enums
enum { TYPE_INT32 = 1, TYPE_INT64 = 2, TYPE_BYTE_ARRAY = 6 };
enum { ENC_PLAIN = 0, ENC_RLE = 3, ENC_DELTA_BINARY_PACKED = 5, ENC_RLE_DICTIONARY = 8 };
enum { PAGE_DATA = 0, PAGE_DICTIONARY = 2 };

inline uint8_t* put_varint(uint8_t* p, uint64_t v) {
  while (v >= 0x80) { *p++ = uint8_t(v | 0x80); v >>= 7; }
  *p++ = uint8_t(v);
  return p;
}
inline void store64(void* p, uint64_t x) { std::memcpy(p, &x, 8); }

// Transposed miniblock layout used by the dense encoders: value i (0..31) of a miniblock sits at q[(i & 7) * 4 + (i >> 3)],
// so the same-numbered values of two neighbouring groups of 8 share one 16-byte vector and one shift count.
// Two groups at a time (SSE2): the W bytes of group g go to d + g * W. A group's 8-byte stores run past its W bytes (zero
// bytes), so groups are stored in order and the next group's stores overwrite the overshoot.
template <int W>
inline __attribute__((always_inline)) void pack16_t(const uint64_t* q, uint8_t* d0) {
  constexpr int NW = (W + 7) / 8;
  __m128i a[NW];
  for (int j = 0; j < NW; j++) a[j] = _mm_setzero_si128();
#pragma GCC unroll 8
  for (int k = 0; k < 8; k++) {
    const __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(q + 4 * k));
    const int o = k * W, wi = o >> 6, sh = o & 63;
    a[wi] = _mm_or_si128(a[wi], _mm_slli_epi64(v, sh));
    if (sh + W > 64) a[wi + 1] = _mm_or_si128(a[wi + 1], _mm_srli_epi64(v, 64 - sh));
  }
  for (int j = 0; j < NW; j++) _mm_storel_epi64(reinterpret_cast<__m128i*>(d0 + 8 * j), a[j]);
  for (int j = 0; j < NW; j++) _mm_storel_epi64(reinterpret_cast<__m128i*>(d0 + W + 8 * j), _mm_unpackhi_epi64(a[j], a[j]));
}
template <int W>
uint8_t* pack32_tw(const uint64_t* q, uint8_t* d) {
  pack16_t<W>(q, d);
  pack16_t<W>(q + 2, d + 2 * W);
  return d + 4 * W;
}
using PackFn = uint8_t* (*)(const uint64_t*, uint8_t*);
template <int... Ws>
constexpr std::array<PackFn, sizeof...(Ws)> make_pack_t_table(std::integer_sequence<int, Ws...>) {
  return {{(Ws == 0 ? nullptr : &pack32_tw<(Ws == 0 ? 1 : Ws)>)...}};
}
inline constexpr int kPackMaxW = 32;  // wider miniblocks take the generic packer
inline constexpr auto kPackT = make_pack_t_table(std::make_integer_sequence<int, kPackMaxW + 1>());

// Bit-pack 32 values of width w (33..64), LSB first, into 4*w bytes.
inline uint8_t* pack32_wide(const uint64_t* v, int w, uint8_t* d) {
  unsigned __int128 acc = 0;
  int bits = 0;
  for (int i = 0; i < 32; i++) {
    acc |= (unsigned __int128)v[i] << bits;
    bits += w;
    if (bits >= 64) {
      store64(d, uint64_t(acc));
      d += 8;
      acc >>= 64;
      bits -= 64;
    }
  }
  for (; bits > 0; bits -= 8) { *d++ = uint8_t(acc); acc >>= 8; }
  return d;
}
// the same for a transposed miniblock
inline uint8_t* pack32_t(const uint64_t* q, int w, uint8_t* d) {
  if (w == 0) return d;
  if (w <= kPackMaxW) return kPackT[w](q, d);
  uint64_t v[32];
  for (int i = 0; i < 32; i++) v[i] = q[(i & 7) * 4 + (i >> 3)];
  return pack32_wide(v, w, d);
}

// Transposed miniblock of 32-bit values (the minimum mn32 is subtracted from each): value i at q[(i & 7) * 4 + (i >> 3)],
// so one vector holds the same-numbered value of the four groups of 8. Group g's W bytes go to d + g * W (stored in group order: a group's last 4-byte store may
// run past its W bytes into the next group's area, which is stored afterwards).
constexpr int kLaneMaxW = 32;
template <int W>
__attribute__((target("sse4.1"))) uint8_t* pack32_n(const int32_t* q, uint8_t* d, int32_t mn32) {
  constexpr int NW = (W + 3) / 4;
  const __m128i mv = _mm_set1_epi32(mn32);
  __m128i a[NW];
  for (int j = 0; j < NW; j++) a[j] = _mm_setzero_si128();
#pragma GCC unroll 8
  for (int k = 0; k < 8; k++) {
    const __m128i v = _mm_sub_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(q + 4 * k)), mv);
    const int o = k * W, wi = o >> 5, sh = o & 31;
    a[wi] = _mm_or_si128(a[wi], _mm_slli_epi32(v, sh));
    if (sh + W > 32) a[wi + 1] = _mm_or_si128(a[wi + 1], _mm_srli_epi32(v, 32 - sh));
  }
  for (int j = 0; j < NW; j++) { const int32_t x = _mm_cvtsi128_si32(a[j]); std::memcpy(d + 4 * j, &x, 4); }
  for (int j = 0; j < NW; j++) { const int32_t x = _mm_extract_epi32(a[j], 1); std::memcpy(d + W + 4 * j, &x, 4); }
  for (int j = 0; j < NW; j++) { const int32_t x = _mm_extract_epi32(a[j], 2); std::memcpy(d + 2 * W + 4 * j, &x, 4); }
  for (int j = 0; j < NW; j++) { const int32_t x = _mm_extract_epi32(a[j], 3); std::memcpy(d + 3 * W + 4 * j, &x, 4); }
  return d + 4 * W;
}
using PackNFn = uint8_t* (*)(const int32_t*, uint8_t*, int32_t);
template <int... Ws>
constexpr std::array<PackNFn, sizeof...(Ws)> make_pack_n_table(std::integer_sequence<int, Ws...>) {
  return {{(Ws == 0 ? nullptr : &pack32_n<(Ws == 0 ? 1 : Ws)>)...}};
}
inline constexpr auto kPackN = make_pack_n_table(std::make_integer_sequence<int, kLaneMaxW + 1>());

constexpr size_t kDeltaBlock = 128, kDeltaMini = 4, kDeltaMiniVals = kDeltaBlock / kDeltaMini;

// Upper bound of a DELTA_BINARY_PACKED stream of m values (a partly filled miniblock is still written at full length).
inline size_t delta_bound(size_t m) { return 64 + ((m + kDeltaBlock - 1) / kDeltaBlock) * (10 + kDeltaMini + kDeltaBlock * 8); }

inline int dict_bit_width(size_t dict_size) {
  int bw = 1;
  while ((size_t(1) << bw) < dict_size) bw++;
  return bw;
}

// Page header with a DataPageHeader (type 0) or DictionaryPageHeader (type 2); returns end pointer.
inline uint8_t* put_page_header(uint8_t* p, int page_type, uint32_t body, int32_t num_values, int encoding,
                                uint32_t compressed_body) {
  TW w(p);
  w.i32(1, page_type);
  w.i32(2, int32_t(body));             // uncompressed_page_size
  w.i32(3, int32_t(compressed_body));  // compressed_page_size
  if (page_type == PAGE_DATA) {
    w.begin_struct(5);
    w.i32(1, num_values);
    w.i32(2, encoding);
    w.i32(3, ENC_RLE);
    w.i32(4, ENC_RLE);
    w.end_struct();
  } else {
    w.begin_struct(7);
    w.i32(1, num_values);
    w.i32(2, encoding);
    w.end_struct();
  }
  *w.p++ = 0;
  return w.p;
}

constexpr size_t kPageHdrMax = 64;

// gzip (RFC 1952) member, as the parquet GZIP codec expects; deterministic for a fixed level.
inline void gzip_append(std::vector<uint8_t>& out, const uint8_t* src, size_t n, int level) {
  z_stream zs;
  std::memset(&zs, 0, sizeof zs);
  if (deflateInit2(&zs, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error(T3_MSG("pq: deflateInit2 failed", "E20"));
  size_t bound = deflateBound(&zs, (uLong)n) + 64;
  size_t at = out.size();
  out.resize(at + bound);
  zs.next_in = const_cast<Bytef*>(src);
  zs.avail_in = (uInt)n;
  zs.next_out = out.data() + at;
  zs.avail_out = (uInt)bound;
  int rc = deflate(&zs, Z_FINISH);
  if (rc != Z_STREAM_END) {
    deflateEnd(&zs);
    throw std::runtime_error(T3_MSG("pq: deflate failed", "E21"));
  }
  out.resize(at + zs.total_out);
  deflateEnd(&zs);
}

}  // namespace detail

// Page buffers of a single-scenario run come from one reserved range (use_arena) instead of malloc: each thread bump-allocates inside
// its own 2 MiB slabs and gives back the unused tail of every page (a buffer is sized for the worst case, a page uses a fraction), so the
// pages of a run lie densely in memory, are faulted in as 4 KiB pages and are never freed (the process ends instead). Without the arena
// (batch runs, or the range is used up) buffers come from new[].
namespace detail {
inline uint8_t* g_ar_base = nullptr;
inline size_t g_ar_size = 0;
inline std::atomic<size_t> g_ar_next{0};  // slab frontier
inline std::atomic<size_t> g_ar_live{0};  // bytes handed out and not given back: what the pages really occupy
struct ArSlab { uint8_t *cur = nullptr, *end = nullptr, *last = nullptr; };
inline thread_local ArSlab t_slab;
inline bool in_arena(const uint8_t* p) { return p >= g_ar_base && p < g_ar_base + g_ar_size; }
struct BufDel { void operator()(uint8_t* p) const noexcept { if (!in_arena(p)) delete[] p; } };
using Buf = std::unique_ptr<uint8_t[], BufDel>;
inline Buf alloc_buf(size_t n) {
  constexpr size_t kSlab = size_t(2) << 20;
  if (g_ar_base) {
    ArSlab& s = t_slab;
    const size_t m = (n + 63) & ~size_t(63);
    if (size_t(s.end - s.cur) < m) {
      const size_t sz = (std::max(m, kSlab) + 4095) & ~size_t(4095);
      const size_t off = g_ar_next.fetch_add(sz, std::memory_order_relaxed);
      if (off + sz > g_ar_size) return Buf(new uint8_t[n]);
      s.cur = g_ar_base + off;
      s.end = s.cur + sz;
    }
    s.last = s.cur;
    s.cur += m;
    g_ar_live.fetch_add(m, std::memory_order_relaxed);
    return Buf(s.last);
  }
  return Buf(new uint8_t[n]);
}
// the page in buf ends at used_end: if buf is this thread's latest allocation, its unused tail is free again
inline void ar_trim(const uint8_t* buf, const uint8_t* used_end) {
  ArSlab& s = t_slab;
  if (buf == s.last && in_arena(buf)) {
    uint8_t* const nc = const_cast<uint8_t*>(used_end) + 63 - (uintptr_t(used_end) + 63) % 64;
    g_ar_live.fetch_sub(size_t(s.cur - nc), std::memory_order_relaxed);
    s.cur = nc;
  }
}
}  // namespace detail
// one reserved range of `bytes` of address space for the page buffers of this process
inline void use_arena(size_t bytes) {
  void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (m == MAP_FAILED) return;
#ifdef MADV_NOHUGEPAGE
  madvise(m, bytes, MADV_NOHUGEPAGE);  // plain 4 KiB pages: no first-touch zeroing/compaction stalls of huge pages
#endif
  detail::g_ar_size = bytes;
  detail::g_ar_base = static_cast<uint8_t*>(m);
}

// The page buffers are dead once the outputs are written: give their memory back on `threads` threads, so that the exit of the process
// has less to unmap (freeing a page costs about as much on one thread as the exit's single thread would pay for it).
inline void release_arena(unsigned threads) {
  const size_t used = std::min(detail::g_ar_next.load(), detail::g_ar_size);
  if (!detail::g_ar_base || !used) return;
  // a thread must have a few MiB of pages to free to pay for its start (the slabs are 2 MiB each but mostly untouched in a small run)
  threads = unsigned(std::min<size_t>(threads, detail::g_ar_live.load() / (size_t(3) << 20)));
  if (threads < 2) return;  // (and one thread frees no faster than the exit does)
  const size_t chunk = ((used + threads - 1) / threads + 4095) & ~size_t(4095);
  auto rel = [&](unsigned i) {
    const size_t a = size_t(i) * chunk;
    if (a < used) madvise(detail::g_ar_base + a, std::min(chunk, used - a), MADV_DONTNEED);
  };
  std::vector<std::thread> ths;
  for (unsigned i = 1; i < threads; i++) ths.emplace_back(rel, i);
  rel(0);
  for (auto& t : ths) t.join();
}

// One encoded piece of a column chunk: a data page, preceded by the dictionary page for a dictionary column's
// first page. Bytes [p, p + len) inside buf.
struct Piece {
  detail::Buf buf;
  uint8_t* p = nullptr;
  size_t len = 0;
  size_t dict_page_len = 0;
  int64_t uncompressed = 0;
  int codec = 0;  // 0 UNCOMPRESSED, 2 GZIP
  // uncompressed pages only (what compress() needs): header lengths and page parameters
  size_t dict_hdr_len = 0, data_hdr_len = 0;
  int32_t num_values = 0, dict_n = 0;
  int encoding = 0;
};

namespace detail {

// Complete a page whose value bytes are [lo, hi), with free space in front of lo: writes backwards the definition
// levels (RLE run when all or no rows are present, else the bit-packed defbits), the 4-byte level length, the page
// header and, for a dictionary column's first page, the dictionary page.
inline void finish_page(Piece& pc, uint8_t* lo, uint8_t* hi, size_t n, size_t m, const uint8_t* defbits, int encoding,
                        const std::vector<uint8_t>* dict_body, int32_t dict_n) {
  uint8_t* p = lo;
  size_t dlen;
  if (m == n || m == 0 || !defbits) {
    uint8_t def[16];
    uint8_t* e = put_varint(def, uint64_t(n) << 1);
    *e++ = (m == n) ? 1 : 0;
    dlen = size_t(e - def);
    p -= dlen;
    std::memcpy(p, def, dlen);
  } else {
    const size_t groups = (n + 7) / 8;
    p -= groups;
    std::memcpy(p, defbits, groups);
    uint8_t h[10];
    uint8_t* e = put_varint(h, uint64_t(groups) << 1 | 1);
    p -= (e - h);
    std::memcpy(p, h, size_t(e - h));
    dlen = groups + size_t(e - h);
  }
  p -= 4;
  const uint32_t dl = uint32_t(dlen);
  std::memcpy(p, &dl, 4);
  const size_t blen = size_t(hi - p);
  uint8_t h[kPageHdrMax];
  uint8_t* e = put_page_header(h, PAGE_DATA, uint32_t(blen), int32_t(n), encoding, uint32_t(blen));
  p -= (e - h);
  std::memcpy(p, h, size_t(e - h));
  pc.data_hdr_len = size_t(e - h);
  pc.dict_page_len = pc.dict_hdr_len = 0;
  if (dict_body) {
    uint8_t dh[kPageHdrMax];
    uint8_t* de = put_page_header(dh, PAGE_DICTIONARY, uint32_t(dict_body->size()), dict_n, ENC_PLAIN, uint32_t(dict_body->size()));
    p -= dict_body->size();
    std::memcpy(p, dict_body->data(), dict_body->size());
    p -= (de - dh);
    std::memcpy(p, dh, size_t(de - dh));
    pc.dict_hdr_len = size_t(de - dh);
    pc.dict_page_len = pc.dict_hdr_len + dict_body->size();
  }
  pc.p = p;
  pc.len = size_t(hi - p);
  pc.uncompressed = int64_t(pc.len);
  pc.codec = 0;
  pc.num_values = int32_t(n);
  pc.dict_n = dict_n;
  pc.encoding = encoding;
}

// Free bytes kept in front of a page's values: page header, level length, definition levels, stream header and
// a dictionary page.
inline size_t page_front(size_t n, size_t dict_bytes) {
  return 2 * kPageHdrMax + 4 + 10 + (n + 7) / 8 + 32 + dict_bytes;
}

}  // namespace detail

// Encoder of one DELTA_BINARY_PACKED data page (block 128, 4 miniblocks of 32, int64 wraparound deltas) of an
// INT32/INT64 column of n rows: one encode() (or push_sequence()), then finish().
// A block's 128 deltas are collected in r_ in the transposed miniblock layout (see pack16_t), then flush()ed.
class IntPage {
 public:
  explicit IntPage(size_t n) : n_(n) {
    using namespace detail;
    front_ = page_front(n, 0);
    buf_ = detail::alloc_buf(front_ + delta_bound(n) + 16);
    d_ = buf_.get() + front_;
  }
  // non-nullable column: row i has value get(i)
  template <typename Get>
  void encode(Get get) {
    using namespace detail;
    const size_t n = n_;
    m_ = rows_ = n;
    if (!n) return;
    first_ = get(0);
    size_t i = 1;
    for (; i + kDeltaBlock <= n; i += kDeltaBlock) write_block(get, i);
    tail_deltas(get, i, n - i, get(i - 1));
  }
  // nullable column: row i is present when bit (i & 7) of pbits[i >> 3] is set (m rows, bits past row n clear; pbits
  // must stay valid until finish()), with value get(i). The present values are collected in xs_ (xs_[0] = the value
  // before them) and turned into deltas a block at a time.
  template <typename Get>
  void encode_present(const uint8_t* pbits, size_t m, Get get) {
    using namespace detail;
    const size_t n = n_;
    if (m == n) { encode(get); return; }
    defbits_ = pbits;
    rows_ = n;
    m_ = m;
    if (!m) return;
    size_t g = 0;
    while (!pbits[g]) g++;
    const size_t f = g * 8 + size_t(__builtin_ctz(pbits[g]));
    xs_[0] = first_ = get(f);
    size_t k = 0;
    uint32_t b = pbits[g] & ~((2u << (f & 7)) - 1u);  // present rows after row f
    const size_t gfull = n >> 3;
    auto xget = [this](size_t j) { return xs_[j]; };
    // every row stores its value at xs_[1 + k]; k advances (bit test, add with carry) only for a present row
#define PQ_ROW(J)                                                                                              \
  {                                                                                                            \
    xs_[1 + k] = get(base + J);                                                                                \
    __asm__("bt %[j], %[b]\n\tadc $0, %[k]" : [k] "+r"(k) : [b] "r"(b), [j] "i"(J) : "cc");                    \
  }
    for (; g < gfull; g++) {
      const size_t base = g * 8;
      PQ_ROW(0) PQ_ROW(1) PQ_ROW(2) PQ_ROW(3) PQ_ROW(4) PQ_ROW(5) PQ_ROW(6) PQ_ROW(7)
      if (k >= kDeltaBlock) k = present_block(xget, k);
      b = pbits[g + 1];
    }
#undef PQ_ROW
    for (size_t j = 0, base = g * 8; j < (n & 7); j++) {
      xs_[1 + k] = get(base + j);
      k += (b >> j) & 1;
    }
    if (k >= kDeltaBlock) k = present_block(xget, k);
    tail_deltas(xget, 1, k, xs_[0]);
  }
  // the n values start, start + 1, ... (non-nullable column only): every block has min delta 1 and bit width 0
  void push_sequence(int64_t start, size_t n) {
    if (!n) return;
    first_ = start;
    m_ = n;
    rows_ = n;
    for (size_t blocks = (n - 1 + detail::kDeltaBlock - 1) / detail::kDeltaBlock; blocks; blocks--) {
      *d_++ = 2;  // zigzag(1)
      for (size_t b = 0; b < detail::kDeltaMini; b++) *d_++ = 0;
    }
  }
  void finish(Piece& pc) {
    using namespace detail;
    uint8_t h[40];
    uint8_t* e = put_varint(h, kDeltaBlock);
    e = put_varint(e, kDeltaMini);
    e = put_varint(e, m_);
    e = put_varint(e, TW::zz64(first_));
    uint8_t* lo = buf_.get() + front_ - (e - h);
    std::memcpy(lo, h, size_t(e - h));
    finish_page(pc, lo, d_, n_, m_, defbits_, ENC_DELTA_BINARY_PACKED, nullptr, 0);
    pc.buf = std::move(buf_);  // the page lies inside buf_
    detail::ar_trim(pc.buf.get(), pc.p + pc.len);
  }

 private:
  // Writes the block of the 128 deltas get(at0 + j) - get(at0 + j - 1): from 32-bit lanes when the block's values lie
  // within 2^30 of the value before it (checked as they are gathered), else from 64-bit deltas.
  template <typename Get>
  void write_block(Get& get, size_t at0) {
    if (lanes_ok_) {
      if (block32(get, at0)) return;
      if (++lane_misses_ == 2) lanes_ok_ = false;  // this column does not fit: stop trying on this page
    }
    mn_ = delta_block(get, at0);
    k_ = detail::kDeltaBlock;
    flush();
  }
  // The 128 values as offsets y = get(at0 + j) - get(at0 - 1) + 2^30 in r32_ (transposed); when every y lies in
  // [0, 2^31) the block is written by flush32 and true is returned.
  template <typename Get>
  bool block32(Get& get, size_t at0) {
    using namespace detail;
    const uint64_t base = uint64_t(get(at0 - 1)) - (uint64_t(1) << 30);
    uint64_t acc = 0;
    for (size_t b = 0; b < kDeltaMini; b++) {
#pragma GCC unroll 4
      for (size_t g = 0; g < 4; g++) {
        int32_t* r = r32_ + b * kDeltaMiniVals + g;
        const size_t at = at0 + b * kDeltaMiniVals + g * 8;
#pragma GCC unroll 8
        for (size_t k = 0; k < 8; k++) {
          uint64_t y = uint64_t(get(at + k)) - base;
          __asm__("" : "+r"(y));  // keeps the compiler from vectorising this into shuffles and spills
          acc |= y;
          r[4 * k] = int32_t(uint32_t(y));
        }
        __asm__("" : "+r"(acc));
      }
    }
    if (acc >> 31) return false;
    flush32(deltas32());
    return true;
  }
  // r32_ holds 4 transposed miniblocks of offsets y; replaces them by the deltas y[j] - y[j - 1] (the value before the
  // block is 2^30), records each miniblock's largest in mx32_ and returns the smallest of all.
  __attribute__((target("sse4.1"), noinline)) int32_t deltas32() {
    __m128i mn = _mm_set1_epi32(INT32_MAX);
    __m128i pv = _mm_set1_epi32(1 << 30);  // previous miniblock's last vector
    for (size_t b = 0; b < detail::kDeltaMini; b++) {
      __m128i* q = reinterpret_cast<__m128i*>(r32_ + b * detail::kDeltaMiniVals);
      const __m128i v7 = _mm_load_si128(q + 7);
      __m128i mx = _mm_set1_epi32(INT32_MIN);
      // vector k holds y[8g + k] for g = 0..3, so its predecessor is vector k - 1, and for k = 0 it is lane g - 1 of
      // vector 7 (lane 3 of the previous miniblock's for g = 0). Descending k: the vectors read are still unchanged.
#pragma GCC unroll 8
      for (int k = 7; k >= 1; k--) {
        const __m128i d = _mm_sub_epi32(_mm_load_si128(q + k), _mm_load_si128(q + k - 1));
        mn = _mm_min_epi32(mn, d);
        mx = _mm_max_epi32(mx, d);
        _mm_store_si128(q + k, d);
      }
      const __m128i d = _mm_sub_epi32(_mm_load_si128(q), _mm_alignr_epi8(v7, pv, 12));
      mn = _mm_min_epi32(mn, d);
      mx = _mm_max_epi32(mx, d);
      _mm_store_si128(q, d);
      pv = v7;
      mx = _mm_max_epi32(mx, _mm_shuffle_epi32(mx, 0x4E));
      mx = _mm_max_epi32(mx, _mm_shuffle_epi32(mx, 0xB1));
      mx32_[b] = _mm_cvtsi128_si32(mx);
    }
    mn = _mm_min_epi32(mn, _mm_shuffle_epi32(mn, 0x4E));
    mn = _mm_min_epi32(mn, _mm_shuffle_epi32(mn, 0xB1));
    return _mm_cvtsi128_si32(mn);
  }
  // flush() for the full block of deltas in r32_ (transposed; smallest mn32, miniblock maxima in mx32_)
  __attribute__((target("sse4.1"), noinline)) void flush32(int32_t mn32) {
    using namespace detail;
    d_ = put_varint(d_, TW::zz64(int64_t(mn32)));
    uint8_t* widths = d_;
    d_ += kDeltaMini;
    for (size_t b = 0; b < kDeltaMini; b++) {
      const uint32_t range = uint32_t(mx32_[b]) - uint32_t(mn32);  // largest offset in the miniblock
      const int w = range ? 32 - __builtin_clz(range) : 0;
      if (w) d_ = kPackN[w](r32_ + b * kDeltaMiniVals, d_, mn32);
      widths[b] = uint8_t(w);
    }
  }
  // The 128 deltas get(at + j) - get(at + j - 1) into r_ (the previous value is re-read instead of carried in a
  // register); returns their minimum.
  template <typename Get>
  int64_t delta_block(Get& get, size_t at0) {
    using namespace detail;
    int64_t mn = INT64_MAX;
    for (size_t b = 0; b < kDeltaMini; b++) {
#pragma GCC unroll 4
      for (size_t g = 0; g < 4; g++) {
        uint64_t* r = r_ + b * kDeltaMiniVals + g;  // transposed: delta 32b + 8g + k at r[4k]
        const size_t at = at0 + b * kDeltaMiniVals + g * 8;
#pragma GCC unroll 8
        for (size_t k = 0; k < 8; k++) {
          const int64_t dl = int64_t(uint64_t(get(at + k)) - uint64_t(get(at + k - 1)));
          r[4 * k] = uint64_t(dl);
          mn = dl < mn ? dl : mn;
        }
        __asm__("" : "+r"(mn));  // keeps the compiler from deferring the minima (it spills the deltas)
      }
    }
    return mn;
  }
  // the last cnt < 128 deltas get(at + j) - get(at + j - 1), the value before get(at) being prev
  template <typename Get>
  void tail_deltas(Get& get, size_t at, size_t cnt, int64_t prev) {
    using namespace detail;
    if (!cnt) return;
    int64_t mn = INT64_MAX;
    for (size_t j = 0; j < cnt; j++) {
      const int64_t x = get(at + j);
      const int64_t dl = int64_t(uint64_t(x) - uint64_t(prev));
      prev = x;
      r_[tpos(j)] = uint64_t(dl);
      mn = dl < mn ? dl : mn;
    }
    k_ = cnt;
    mn_ = mn;
    flush();
  }
  // xs_ holds k >= 128 present values after xs_[0]: writes the first block of 128, moves the rest down; returns the rest
  template <typename Get>
  size_t present_block(Get& xget, size_t k) {
    using namespace detail;
    write_block(xget, 1);
    for (size_t j = 0; j < 8; j++) xs_[j] = xs_[kDeltaBlock + j];
    return k - kDeltaBlock;
  }
  static size_t tpos(size_t j) { return (j & ~size_t(31)) + (j & 7) * 4 + ((j >> 3) & 3); }
  // Writes the block of k_ deltas in r_: min delta, miniblock widths, packed values.
  void flush() {
    using namespace detail;
    const int64_t mn = mn_;
    const size_t used = (k_ + kDeltaMiniVals - 1) / kDeltaMiniVals;
    for (size_t j = k_; j < used * kDeltaMiniVals; j++) r_[tpos(j)] = uint64_t(mn);  // padding packs as 0
    d_ = put_varint(d_, TW::zz64(mn));
    uint8_t* widths = d_;
    d_ += kDeltaMini;
    const __m128i mv = _mm_set1_epi64x(mn);
    for (size_t b = 0; b < kDeltaMini; b++) {
      int w = 0;
      if (b < used) {
        uint64_t* q = r_ + b * kDeltaMiniVals;
        __m128i o0 = _mm_setzero_si128(), o1 = o0;
        if (mn == 0) {  // nothing to subtract
#pragma GCC unroll 16
          for (size_t j = 0; j < kDeltaMiniVals; j += 4) {
            o0 = _mm_or_si128(o0, _mm_load_si128(reinterpret_cast<const __m128i*>(q + j)));
            o1 = _mm_or_si128(o1, _mm_load_si128(reinterpret_cast<const __m128i*>(q + j + 2)));
          }
        } else {
#pragma GCC unroll 16
          for (size_t j = 0; j < kDeltaMiniVals; j += 4) {
            const __m128i a0 = _mm_sub_epi64(_mm_load_si128(reinterpret_cast<const __m128i*>(q + j)), mv);
            const __m128i a1 = _mm_sub_epi64(_mm_load_si128(reinterpret_cast<const __m128i*>(q + j + 2)), mv);
            _mm_store_si128(reinterpret_cast<__m128i*>(q + j), a0);
            _mm_store_si128(reinterpret_cast<__m128i*>(q + j + 2), a1);
            o0 = _mm_or_si128(o0, a0);
            o1 = _mm_or_si128(o1, a1);
          }
        }
        o0 = _mm_or_si128(o0, o1);
        const uint64_t o = uint64_t(_mm_cvtsi128_si64(o0)) | uint64_t(_mm_cvtsi128_si64(_mm_unpackhi_epi64(o0, o0)));
        w = o ? 64 - __builtin_clzll(o) : 0;
        d_ = pack32_t(q, w, d_);
      }
      widths[b] = uint8_t(w);
    }
    k_ = 0;
    mn_ = INT64_MAX;
  }
  size_t n_, front_, m_ = 0, k_ = 0, rows_ = 0;
  detail::Buf buf_;
  const uint8_t* defbits_ = nullptr;
  uint8_t* d_;
  int64_t first_ = 0, mn_ = INT64_MAX;
  bool lanes_ok_ = true;
  int lane_misses_ = 0;
  alignas(16) uint64_t r_[detail::kDeltaBlock];
  alignas(16) int32_t r32_[detail::kDeltaBlock];
  int32_t mx32_[detail::kDeltaMini];
  int64_t xs_[detail::kDeltaBlock + 16];
};

// Encoder of one RLE_DICTIONARY data page of a string column (dictionary <= 255 entries) of n rows: encode(code),
// code(i) = 0xFF for null, then finish(). A chunk's first page also carries the dictionary page (with_dict).
class DictPage {
 public:
  DictPage(size_t n, const std::vector<std::string>& dict, bool with_dict) : n_(n), dsize_(dict.size()) {
    using namespace detail;
    if (dict.size() > 255) throw std::invalid_argument(T3_MSG("pq: dict > 255 entries", "E22"));
    bw_ = dict_bit_width(dict.size());
    if (with_dict) {
      db_.reset(new std::vector<uint8_t>());
      for (auto& e : dict) {
        uint32_t l = uint32_t(e.size());
        db_->insert(db_->end(), reinterpret_cast<uint8_t*>(&l), reinterpret_cast<uint8_t*>(&l) + 4);
        db_->insert(db_->end(), e.begin(), e.end());
      }
    }
    front_ = page_front(n, db_ ? db_->size() : 0);
    buf_ = detail::alloc_buf(front_ + ((n + 7) / 8) * size_t(bw_) + 16);
    d_ = buf_.get() + front_;
  }
  template <typename Code>
  void encode(Code code) {
    switch (bw_) {  // all rows present: the common case
      case 1: if (encode_all<1>(code)) return; break;
      case 2: if (encode_all<2>(code)) return; break;
      case 3: if (encode_all<3>(code)) return; break;
      case 4: if (encode_all<4>(code)) return; break;
      case 5: if (encode_all<5>(code)) return; break;
      case 6: if (encode_all<6>(code)) return; break;
      case 7: if (encode_all<7>(code)) return; break;
      default: if (encode_all<8>(code)) return; break;
    }
    encode_slow(code);
  }
  void finish(Piece& pc) {
    using namespace detail;
    if (vk_) { store64(d_, vacc_); d_ += bw_; }
    if (bits_ && (rows_ & 7)) bits_[rows_ >> 3] = acc_;
    uint8_t h[12];
    uint8_t* e = h;
    *e++ = uint8_t(bw_);
    if (m_) e = put_varint(e, uint64_t((m_ + 7) / 8) << 1 | 1);
    uint8_t* lo = buf_.get() + front_ - (e - h);
    std::memcpy(lo, h, size_t(e - h));
    finish_page(pc, lo, d_, n_, m_, bits_.get(), ENC_RLE_DICTIONARY, db_.get(), int32_t(dsize_));
    pc.buf = std::move(buf_);  // the page lies inside buf_
    detail::ar_trim(pc.buf.get(), pc.p + pc.len);
  }

 private:
  // 8 codes (bytes of x, each < 2^BW) packed into BW bytes (BW <= 8) by three pair-merging steps
  template <int BW>
  static inline uint64_t compact8(uint64_t x) {
    if (BW == 8) return x;
    constexpr uint64_t f1 = (uint64_t(1) << (2 * BW)) - 1, f2 = (uint64_t(1) << (4 * BW)) - 1;
    constexpr uint64_t M1 = f1 | f1 << 16 | f1 << 32 | f1 << 48;
    constexpr uint64_t M2 = f2 | f2 << 32;
    constexpr uint64_t M3 = BW == 8 ? ~uint64_t(0) : (uint64_t(1) << (8 * BW)) - 1;
    if (BW <= 4) {  // each shift is at least the field width: the shifted-down low halves stay outside the masks
      x = (x | (x >> (8 - BW))) & M1;
      x = (x | (x >> (16 - 2 * BW))) & M2;
      return (x | (x >> (32 - 4 * BW))) & M3;
    }
    x = (x & 0x00FF00FF00FF00FFull) | ((x & 0xFF00FF00FF00FF00ull) >> (8 - BW));
    x = (x & 0x0000FFFF0000FFFFull) | ((x & 0xFFFF0000FFFF0000ull) >> (16 - 2 * BW));
    return (x & 0xFFFFFFFFull) | ((x & 0xFFFFFFFF00000000ull) >> (32 - 4 * BW));
  }
  // All rows present: gathers the codes 64 at a time, checks their range, packs 8 codes per step. Returns false
  // (nothing committed) when some code is null (0xFF) or out of range.
  template <int BW, typename Code>
  bool encode_all(Code code) {
    const size_t n = n_;
    uint8_t* d = d_;
    alignas(16) uint8_t cb[72];
    const __m128i lim = _mm_set1_epi8(char(dsize_ - 1));  // dsize_ <= 255: codes above are invalid
    __m128i bad = _mm_setzero_si128();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
#pragma GCC unroll 8
      for (size_t j = 0; j < 64; j++) cb[j] = code(i + j);
      __m128i mx = _mm_load_si128(reinterpret_cast<const __m128i*>(cb));
      for (int q = 16; q < 64; q += 16) mx = _mm_max_epu8(mx, _mm_load_si128(reinterpret_cast<const __m128i*>(cb + q)));
      bad = _mm_or_si128(bad, _mm_subs_epu8(mx, lim));
      if (BW == 1) {  // codes 0/1: bit 0 of each byte up to the byte's top bit, then one movemask per 16 codes
        uint64_t y = 0;
        for (int q = 0; q < 4; q++) {
          const __m128i v = _mm_slli_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(cb + 16 * q)), 7);
          y |= uint64_t(unsigned(_mm_movemask_epi8(v))) << (16 * q);
        }
        std::memcpy(d, &y, 8);
        d += 8;
      } else if (BW == 4) {  // two codes per byte: a pair of bytes folded into one, 16 bytes from 32 codes
        const __m128i lo = _mm_set1_epi16(0x00FF);
        for (int q = 0; q < 4; q += 2) {
          const __m128i v0 = _mm_load_si128(reinterpret_cast<const __m128i*>(cb + 16 * q));
          const __m128i v1 = _mm_load_si128(reinterpret_cast<const __m128i*>(cb + 16 * q + 16));
          const __m128i t0 = _mm_and_si128(_mm_or_si128(v0, _mm_srli_epi16(v0, 4)), lo);
          const __m128i t1 = _mm_and_si128(_mm_or_si128(v1, _mm_srli_epi16(v1, 4)), lo);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(d), _mm_packus_epi16(t0, t1));
          d += 16;
        }
      } else {
#pragma GCC unroll 8
        for (size_t j = 0; j < 64; j += 8) {
          uint64_t x;
          std::memcpy(&x, cb + j, 8);
          const uint64_t y = compact8<BW>(x);
          std::memcpy(d, &y, 8);
          d += BW;
        }
      }
    }
    if (i < n) {
      std::memset(cb, 0, 64);
      const size_t r = n - i;
      uint8_t mx = 0;
      for (size_t j = 0; j < r; j++) { cb[j] = code(i + j); mx = cb[j] > mx ? cb[j] : mx; }
      if (mx >= dsize_) return false;
      for (size_t j = 0; j < r; j += 8) {
        uint64_t x;
        std::memcpy(&x, cb + j, 8);
        const uint64_t y = compact8<BW>(x);
        std::memcpy(d, &y, 8);
        d += BW;
      }
    }
    if (_mm_movemask_epi8(_mm_cmpeq_epi8(bad, _mm_setzero_si128())) != 0xFFFF) return false;
    d_ = d;
    m_ = rows_ = n;
    vk_ = 0;
    return true;
  }
  template <typename Code>
  void encode_slow(Code code) {
    if (!bits_) bits_.reset(new uint8_t[(n_ + 7) / 8 + 8]);
    const size_t n = n_;
    const int bw = bw_;
    uint8_t* d = d_;
    uint64_t vacc = 0;
    int vk = 0;
    size_t m = 0;
    for (size_t g = 0; g * 8 < n; g++) {
      uint8_t b = 0;
      for (size_t i = g * 8, e = std::min(n, i + 8); i < e; i++) {
        const uint8_t c = code(i);
        if (c == 0xFF) continue;
        if (__builtin_expect(c >= dsize_, 0)) throw std::invalid_argument(T3_MSG("pq: dict code out of range", "E23"));
        b |= uint8_t(1u << (i & 7));
        m++;
        vacc |= uint64_t(c) << (bw * vk);
        if (++vk == 8) { detail::store64(d, vacc); d += bw; vacc = 0; vk = 0; }
      }
      bits_[g] = b;
    }
    d_ = d;
    vacc_ = vacc;
    vk_ = vk;
    m_ = m;
    rows_ = n;
    acc_ = (n & 7) ? bits_[n >> 3] : 0;
  }

  size_t n_, dsize_, front_, m_ = 0, rows_ = 0;
  int bw_;
  std::unique_ptr<std::vector<uint8_t>> db_;
  detail::Buf buf_;
  std::unique_ptr<uint8_t[]> bits_;
  uint8_t* d_;
  uint8_t acc_ = 0;
  uint64_t vacc_ = 0;
  int vk_ = 0;
};

// GZIP-compress an uncompressed page in place (a new buffer; deterministic for a fixed level).
inline void compress(Piece& pc, int level) {
  using namespace detail;
  if (pc.codec != 0) return;
  std::vector<uint8_t> out;
  int64_t unc = 0;
  const uint8_t* q = pc.p;
  size_t dict_len = 0;
  if (pc.dict_page_len) {
    const uint8_t* body = q + pc.dict_hdr_len;
    const size_t bl = pc.dict_page_len - pc.dict_hdr_len;
    std::vector<uint8_t> z;
    gzip_append(z, body, bl, level);
    uint8_t h[kPageHdrMax];
    uint8_t* e = put_page_header(h, PAGE_DICTIONARY, uint32_t(bl), pc.dict_n, ENC_PLAIN, uint32_t(z.size()));
    out.insert(out.end(), h, e);
    out.insert(out.end(), z.begin(), z.end());
    unc += int64_t(e - h) + int64_t(bl);
    dict_len = out.size();
    q += pc.dict_page_len;
  }
  const uint8_t* body = q + pc.data_hdr_len;
  const size_t bl = size_t(pc.p + pc.len - body);
  std::vector<uint8_t> z;
  gzip_append(z, body, bl, level);
  uint8_t h[kPageHdrMax];
  uint8_t* e = put_page_header(h, PAGE_DATA, uint32_t(bl), pc.num_values, pc.encoding, uint32_t(z.size()));
  out.insert(out.end(), h, e);
  out.insert(out.end(), z.begin(), z.end());
  unc += int64_t(e - h) + int64_t(bl);
  pc.buf.reset(new uint8_t[out.size()]);
  std::memcpy(pc.buf.get(), out.data(), out.size());
  pc.p = pc.buf.get();
  pc.len = out.size();
  pc.dict_page_len = dict_len;
  pc.uncompressed = unc;
  pc.codec = 2;
}

// A column chunk: its pages in row order.
struct Chunk {
  const char* name = "";
  int type = 0;  // detail::TYPE_*
  bool dict = false;
  std::vector<Piece> pages;
  size_t len() const { size_t s = 0; for (auto& p : pages) s += p.len; return s; }
};
inline constexpr int TYPE_INT32 = detail::TYPE_INT32, TYPE_INT64 = detail::TYPE_INT64, TYPE_STRING = detail::TYPE_BYTE_ARRAY;

// Rows per data page: fixed, so the file bytes do not depend on how many threads encode it.
inline constexpr size_t kPageRows = size_t(1) << 14;
inline size_t page_count(size_t nrows) { return nrows ? (nrows + kPageRows - 1) / kPageRows : 1; }

// Footer description: the columns, and per row group its rows and per column chunk its placement and sizes.
struct ColRef { const char* name; int type; bool dict; };
struct ChunkPlace { int64_t start = 0, len = 0, uncompressed = 0; size_t dict_page_len = 0; int codec = 0; };
struct RowGroupPlace { int64_t rows = 0; std::vector<ChunkPlace> cols; };

namespace detail {

// FileMetaData + its length + "PAR1".
inline std::vector<uint8_t> make_footer(const std::vector<ColRef>& cols, int64_t nrows, const std::vector<RowGroupPlace>& rgs,
                                        const std::string& meta) {
  size_t fbound = 1024 + meta.size();
  for (auto& c : cols) fbound += 64 + 3 * strlen(c.name);
  fbound += rgs.size() * (64 + cols.size() * (128 + strlen("causal_parent")));
  for (auto& c : cols) fbound += rgs.size() * strlen(c.name);
  std::vector<uint8_t> fbuf(fbound + 8);
  TW w(fbuf.data());
  w.i32(1, 1);  // version
  w.begin_list(2, T_STRUCT, cols.size() + 1);
  w.begin_elem();  // root
  w.i32(3, 0);     // REQUIRED
  w.str(4, "schema", 6);
  w.i32(5, int32_t(cols.size()));
  w.end_elem();
  for (auto& c : cols) {
    w.begin_elem();
    w.i32(1, c.type);
    w.i32(3, 1);  // OPTIONAL
    w.str(4, c.name, strlen(c.name));
    if (c.type == TYPE_BYTE_ARRAY) {
      w.i32(6, 0);  // ConvertedType UTF8
      w.begin_struct(10);  // LogicalType
      w.begin_struct(1);   // STRING
      w.end_struct();
      w.end_struct();
    }
    w.end_elem();
  }
  w.i64(3, nrows);
  w.begin_list(4, T_STRUCT, rgs.size());
  for (const RowGroupPlace& rg : rgs) {
    w.begin_elem();  // RowGroup
    w.begin_list(1, T_STRUCT, cols.size());
    int64_t rg_size = 0, rg_comp = 0;
    for (size_t ci = 0; ci < cols.size(); ci++) {
      const ColRef& c = cols[ci];
      const ChunkPlace& k = rg.cols[ci];
      rg_size += k.uncompressed;
      rg_comp += k.len;
      w.begin_elem();  // ColumnChunk
      w.i64(2, k.start);
      w.begin_struct(3);  // ColumnMetaData
      w.i32(1, c.type);
      if (c.dict) {
        w.begin_list(2, T_I32, 3);
        w.varint(TW::zz32(ENC_PLAIN)); w.varint(TW::zz32(ENC_RLE)); w.varint(TW::zz32(ENC_RLE_DICTIONARY));
      } else {
        w.begin_list(2, T_I32, 2);
        w.varint(TW::zz32(ENC_RLE)); w.varint(TW::zz32(ENC_DELTA_BINARY_PACKED));
      }
      w.begin_list(3, T_BINARY, 1);
      w.varint(strlen(c.name));
      std::memcpy(w.p, c.name, strlen(c.name));
      w.p += strlen(c.name);
      w.i32(4, k.codec);
      w.i64(5, rg.rows);
      w.i64(6, k.uncompressed);
      w.i64(7, k.len);
      w.i64(9, k.start + int64_t(k.dict_page_len));
      if (c.dict) w.i64(11, k.start);
      w.end_struct();
      w.end_elem();
    }
    w.i64(2, rg_size);
    w.i64(3, rg.rows);
    w.i64(5, rg.cols.empty() ? 4 : rg.cols[0].start);
    w.i64(6, rg_comp);
    w.end_elem();
  }
  w.begin_list(5, T_STRUCT, 1);
  w.begin_elem();
  w.str(1, "pandas", 6);
  w.str(2, meta);
  w.end_elem();
  w.str(6, "t3sim", 5);
  *w.p++ = 0;
  const uint32_t flen = uint32_t(w.p - fbuf.data());
  std::memcpy(w.p, &flen, 4);
  std::memcpy(w.p + 4, "PAR1", 4);
  fbuf.resize(flen + 8);
  return fbuf;
}

// writev all of iov (any count) to fd
inline void write_all(int fd, std::vector<iovec>& iov, const std::string& path) {
  size_t i = 0;
  while (i < iov.size()) {
    ssize_t r = writev(fd, iov.data() + i, int(std::min<size_t>(iov.size() - i, 1024)));
    if (r < 0) { close(fd); throw std::runtime_error(T3_MSG("short write ", "E25 ") + path); }
    size_t got = size_t(r);
    while (i < iov.size() && got >= iov[i].iov_len) got -= iov[i++].iov_len;
    if (got) { iov[i].iov_base = static_cast<uint8_t*>(iov[i].iov_base) + got; iov[i].iov_len -= got; }
  }
}

}  // namespace detail

// Placement of a file's column chunks after "PAR1". one_rg: one row group holding every page of each column, in column
// order; else one row group per page index (page pg of every column), so a writer can emit the file page by page.
inline std::vector<RowGroupPlace> place_chunks(const std::vector<Chunk>& cols, int64_t nrows, bool one_rg) {
  std::vector<RowGroupPlace> rgs;
  int64_t off = 4;
  auto place = [&](const Piece& pc, ChunkPlace& k) {
    k.start = off;
    k.len += int64_t(pc.len);
    k.uncompressed += pc.uncompressed;
    off += int64_t(pc.len);
  };
  if (one_rg) {
    RowGroupPlace rg;
    rg.rows = nrows;
    for (auto& c : cols) {
      ChunkPlace k;
      const int64_t start = off;
      for (auto& pg : c.pages) place(pg, k);
      k.start = start;
      k.codec = c.pages.empty() ? 0 : c.pages[0].codec;
      k.dict_page_len = c.pages.empty() ? 0 : c.pages[0].dict_page_len;
      rg.cols.push_back(k);
    }
    rgs.push_back(std::move(rg));
    return rgs;
  }
  const size_t np = cols.empty() ? 0 : cols[0].pages.size();
  for (size_t pg = 0; pg < np; pg++) {
    RowGroupPlace rg;
    rg.rows = std::min<int64_t>(nrows - int64_t(pg * kPageRows), int64_t(kPageRows));
    for (auto& c : cols) {
      ChunkPlace k;
      place(c.pages[pg], k);
      k.codec = c.pages[pg].codec;
      k.dict_page_len = c.pages[pg].dict_page_len;
      rg.cols.push_back(k);
    }
    rgs.push_back(std::move(rg));
  }
  return rgs;
}

// Footer (FileMetaData + length + "PAR1") for that placement.
inline std::vector<uint8_t> footer_for(const std::vector<Chunk>& cols, int64_t nrows, const std::string& meta, bool one_rg) {
  std::vector<ColRef> refs;
  for (auto& c : cols) refs.push_back({c.name, c.type, c.dict});
  return detail::make_footer(refs, nrows, place_chunks(cols, nrows, one_rg), meta);
}

// Write a complete parquet file from encoded chunks with one writev; returns the SHA-256 hex of the file bytes when
// want_sha, else "". one_rg: one row group (column-major), else one row group per page (page-major; every page of a
// string column must then carry its dictionary page). The digest of a large file is computed on another thread while
// the same bytes are written.
// prefd >= 0: the file was opened by the caller (O_TRUNC) and its first `prefilled` bytes already hold filler (page cache populated
// ahead of time); the bytes are overwritten from offset 0 and the file is cut to its final size when the filler was longer.
inline std::string write_file(const std::string& path, int64_t nrows, const std::vector<Chunk>& cols,
                              const std::string& meta, bool want_sha, bool one_rg = true, int prefd = -1, size_t prefilled = 0) {
  using namespace detail;
  std::vector<uint8_t> foot = footer_for(cols, nrows, meta, one_rg);
  std::vector<iovec> iov;
  iov.push_back({const_cast<char*>("PAR1"), 4});
  if (one_rg) {
    for (auto& c : cols)
      for (auto& pg : c.pages) iov.push_back({pg.p, pg.len});
  } else {
    const size_t np = cols.empty() ? 0 : cols[0].pages.size();
    for (size_t pg = 0; pg < np; pg++)
      for (auto& c : cols) iov.push_back({c.pages[pg].p, c.pages[pg].len});
  }
  iov.push_back({foot.data(), foot.size()});
  std::string sha;
  std::thread hasher;
  auto digest = [&sha](const std::vector<iovec>& iv) {
    Sha256Stream h;
    for (auto& v : iv) h.update(static_cast<const uint8_t*>(v.iov_base), v.iov_len);
    sha = h.hex();
  };
  size_t bytes = 0;
  for (auto& v : iov) bytes += v.iov_len;
  if (want_sha) {
    if (bytes > (size_t(1) << 18)) hasher = std::thread(digest, iov);
    else digest(iov);
  }
  int fd = prefd >= 0 ? prefd : open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    if (hasher.joinable()) hasher.join();
    throw std::runtime_error(T3_MSG("cannot write ", "E24 ") + path);
  }
  try {
    if (prefd >= 0 && lseek(fd, 0, SEEK_SET) != 0) throw std::runtime_error(T3_MSG("short write ", "E25 ") + path);
    write_all(fd, iov, path);
    if (prefilled > bytes && ftruncate(fd, off_t(bytes)) != 0) throw std::runtime_error(T3_MSG("short write ", "E25 ") + path);
  } catch (...) {
    if (hasher.joinable()) hasher.join();
    throw;
  }
  close(fd);
  if (hasher.joinable()) hasher.join();
  return sha;
}

}  // namespace pq
