#pragma once
// Header-only SHA-256. Portable C++17 core; ARMv8 SHA2 and x86 SHA-NI block
// functions are selected at runtime (CPU feature check) and are verified once
// against the portable implementation before use.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif
#define SHA256_HAVE_ARM 1
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__clang__) || defined(__GNUC__))
#include <cpuid.h>
#include <immintrin.h>
#define SHA256_HAVE_X86 1
#endif

namespace sha256_detail {

alignas(16) inline constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline void compress_portable(uint32_t st[8], const uint8_t* p, size_t nblocks) {
  while (nblocks--) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) |
             uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; i++) {
      uint32_t a = w[i - 15], b = w[i - 2];
      uint32_t s0 = ((a >> 7) | (a << 25)) ^ ((a >> 18) | (a << 14)) ^ (a >> 3);
      uint32_t s1 = ((b >> 17) | (b << 15)) ^ ((b >> 19) | (b << 13)) ^ (b >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i++) {
      uint32_t S1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7));
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + S1 + ch + K[i] + w[i];
      uint32_t S0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10));
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
    p += 64;
  }
}

#ifdef SHA256_HAVE_ARM
#if defined(__clang__)
#define SHA256_ARM_TARGET __attribute__((target("sha2")))
#else
#define SHA256_ARM_TARGET __attribute__((target("+crypto")))
#endif
SHA256_ARM_TARGET inline void compress_arm(uint32_t st[8], const uint8_t* p, size_t nblocks) {
  uint32x4_t s0 = vld1q_u32(st), s1 = vld1q_u32(st + 4);
  while (nblocks--) {
    uint32x4_t a0 = s0, a1 = s1;
    uint32x4_t w[4];
    for (int i = 0; i < 4; i++) w[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 16 * i)));
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (int i = 0; i < 16; i++) {
      uint32x4_t wk = vaddq_u32(w[i & 3], vld1q_u32(K + 4 * i));
      if (i < 12)
        w[i & 3] = vsha256su1q_u32(vsha256su0q_u32(w[i & 3], w[(i + 1) & 3]), w[(i + 2) & 3], w[(i + 3) & 3]);
      uint32x4_t t = s0;
      s0 = vsha256hq_u32(s0, s1, wk);
      s1 = vsha256h2q_u32(s1, t, wk);
    }
    s0 = vaddq_u32(s0, a0);
    s1 = vaddq_u32(s1, a1);
    p += 64;
  }
  vst1q_u32(st, s0);
  vst1q_u32(st + 4, s1);
}
inline bool cpu_has_arm_sha2() {
#if defined(__APPLE__)
  int v = 0;
  size_t n = sizeof v;
  return sysctlbyname("hw.optional.arm.FEAT_SHA256", &v, &n, nullptr, 0) == 0 && v == 1;
#elif defined(__linux__)
  return (getauxval(AT_HWCAP) & (1UL << 6)) != 0;  // HWCAP_SHA2
#elif defined(__ARM_FEATURE_SHA2)
  return true;
#else
  return false;
#endif
}
#endif  // SHA256_HAVE_ARM

#ifdef SHA256_HAVE_X86
__attribute__((target("sha,sse4.1,ssse3"))) inline void compress_shani(uint32_t st[8], const uint8_t* p,
                                                                     size_t nblocks) {
  const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
  __m128i tmp = _mm_loadu_si128((const __m128i*)&st[0]);
  __m128i s1 = _mm_loadu_si128((const __m128i*)&st[4]);
  tmp = _mm_shuffle_epi32(tmp, 0xB1);
  s1 = _mm_shuffle_epi32(s1, 0x1B);
  __m128i s0 = _mm_alignr_epi8(tmp, s1, 8);
  s1 = _mm_blend_epi16(s1, tmp, 0xF0);
  while (nblocks--) {
    __m128i a0 = s0, a1 = s1;
    __m128i w[4];
    for (int i = 0; i < 4; i++) w[i] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(p + 16 * i)), MASK);
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (int i = 0; i < 16; i++) {
      __m128i msg = _mm_add_epi32(w[i & 3], _mm_load_si128((const __m128i*)(K + 4 * i)));
      s1 = _mm_sha256rnds2_epu32(s1, s0, msg);
      msg = _mm_shuffle_epi32(msg, 0x0E);
      s0 = _mm_sha256rnds2_epu32(s0, s1, msg);
      if (i < 12) {
        __m128i n = _mm_sha256msg1_epu32(w[i & 3], w[(i + 1) & 3]);
        n = _mm_add_epi32(n, _mm_alignr_epi8(w[(i + 3) & 3], w[(i + 2) & 3], 4));
        w[i & 3] = _mm_sha256msg2_epu32(n, w[(i + 3) & 3]);
      }
    }
    s0 = _mm_add_epi32(s0, a0);
    s1 = _mm_add_epi32(s1, a1);
    p += 64;
  }
  tmp = _mm_shuffle_epi32(s0, 0x1B);
  s1 = _mm_shuffle_epi32(s1, 0xB1);
  s0 = _mm_blend_epi16(tmp, s1, 0xF0);
  s1 = _mm_alignr_epi8(s1, tmp, 8);
  _mm_storeu_si128((__m128i*)&st[0], s0);
  _mm_storeu_si128((__m128i*)&st[4], s1);
}
inline bool cpu_has_shani() {
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
  bool ssse3 = c & (1u << 9), sse41 = c & (1u << 19);
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
  return ssse3 && sse41 && (b & (1u << 29));
}
#endif  // SHA256_HAVE_X86

typedef void (*CompressFn)(uint32_t*, const uint8_t*, size_t);

inline bool selftest(CompressFn f) {
  uint8_t buf[64 * 5];
  uint32_t x = 12345;
  for (size_t i = 0; i < sizeof buf; i++) { x = x * 1664525u + 1013904223u; buf[i] = uint8_t(x >> 24); }
  static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint32_t a[8], b[8];
  std::memcpy(a, init, 32);
  std::memcpy(b, init, 32);
  compress_portable(a, buf, 5);
  f(b, buf, 2);
  f(b, buf + 128, 3);
  return std::memcmp(a, b, 32) == 0;
}

inline CompressFn pick() {
#ifdef SHA256_HAVE_ARM
  if (cpu_has_arm_sha2() && selftest(compress_arm)) return compress_arm;
#endif
#ifdef SHA256_HAVE_X86
  if (cpu_has_shani() && selftest(compress_shani)) return compress_shani;
#endif
  return compress_portable;
}

inline CompressFn compress_fn() {
  static const CompressFn f = pick();
  return f;
}

inline void digest(CompressFn f, const uint8_t* p, size_t n, uint8_t out[32]) {
  uint32_t st[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  size_t full = n / 64;
  if (full) f(st, p, full);
  uint8_t tail[128] = {0};
  size_t r = n - full * 64;
  if (r) std::memcpy(tail, p + full * 64, r);
  tail[r] = 0x80;
  size_t tl = (r < 56) ? 64 : 128;
  uint64_t bits = uint64_t(n) * 8;
  for (int i = 0; i < 8; i++) tail[tl - 1 - i] = uint8_t(bits >> (8 * i));
  f(st, tail, tl / 64);
  for (int i = 0; i < 8; i++) {
    out[4 * i] = uint8_t(st[i] >> 24);
    out[4 * i + 1] = uint8_t(st[i] >> 16);
    out[4 * i + 2] = uint8_t(st[i] >> 8);
    out[4 * i + 3] = uint8_t(st[i]);
  }
}

inline std::string hex(const uint8_t d[32]) {
  static const char* H = "0123456789abcdef";
  std::string s(64, '0');
  for (int i = 0; i < 32; i++) { s[2 * i] = H[d[i] >> 4]; s[2 * i + 1] = H[d[i] & 15]; }
  return s;
}

}  // namespace sha256_detail

// Incremental SHA-256 over a sequence of byte ranges (same result as hashing their concatenation).
struct Sha256Stream {
  uint32_t st[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf[64];
  size_t nbuf = 0;
  uint64_t total = 0;
  sha256_detail::CompressFn f = sha256_detail::compress_fn();
  void update(const uint8_t* p, size_t n) {
    total += n;
    if (nbuf) {
      size_t t = std::min(n, 64 - nbuf);
      std::memcpy(buf + nbuf, p, t);
      nbuf += t; p += t; n -= t;
      if (nbuf < 64) return;
      f(st, buf, 1);
      nbuf = 0;
    }
    if (n >= 64) { f(st, p, n / 64); p += n / 64 * 64; n %= 64; }
    std::memcpy(buf, p, n);
    nbuf = n;
  }
  std::string hex() {
    uint8_t tail[128] = {0};
    std::memcpy(tail, buf, nbuf);
    tail[nbuf] = 0x80;
    size_t tl = (nbuf < 56) ? 64 : 128;
    uint64_t bits = total * 8;
    for (int i = 0; i < 8; i++) tail[tl - 1 - i] = uint8_t(bits >> (8 * i));
    f(st, tail, tl / 64);
    uint8_t d[32];
    for (int i = 0; i < 8; i++) {
      d[4 * i] = uint8_t(st[i] >> 24); d[4 * i + 1] = uint8_t(st[i] >> 16);
      d[4 * i + 2] = uint8_t(st[i] >> 8); d[4 * i + 3] = uint8_t(st[i]);
    }
    return sha256_detail::hex(d);
  }
};

// Raw 32-byte digest.
inline void sha256_raw(const uint8_t* p, size_t n, uint8_t out[32]) {
  sha256_detail::digest(sha256_detail::compress_fn(), p, n, out);
}

// Lowercase hex digest (64 chars). Uses hardware SHA when available.
inline std::string sha256_hex(const uint8_t* p, size_t n) {
  uint8_t d[32];
  sha256_raw(p, n, d);
  return sha256_detail::hex(d);
}

// Always uses the portable implementation (for cross-checking).
inline std::string sha256_hex_portable(const uint8_t* p, size_t n) {
  uint8_t d[32];
  sha256_detail::digest(sha256_detail::compress_portable, p, n, d);
  return sha256_detail::hex(d);
}

// Name of the implementation selected at runtime: "arm-sha2", "x86-shani" or "portable".
inline const char* sha256_impl() {
  using namespace sha256_detail;
  CompressFn f = compress_fn();
#ifdef SHA256_HAVE_ARM
  if (f == compress_arm) return "arm-sha2";
#endif
#ifdef SHA256_HAVE_X86
  if (f == compress_shani) return "x86-shani";
#endif
  return "portable";
}
