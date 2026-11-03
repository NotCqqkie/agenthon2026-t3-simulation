#pragma once
#include <cstdint>
#include <cmath>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

extern "C" double __ieee754_exp(double) __attribute__((weak));
extern "C" double __ieee754_log(double) __attribute__((weak));

namespace nrng {

struct RandomState {
  uint32_t mt[624];
  uint32_t out[48];  // tempered words of the chunk generated last
  int pos = 48, gen = 0;
  bool has_gauss = false;
  double cached_gauss = 0.0;

  void seed(uint32_t s) {
    for (int i = 0; i < 624; i++) {
      mt[i] = s;
      s = 1812433253U * (s ^ (s >> 30)) + (uint32_t)i + 1U;
    }
    pos = 48;
    gen = 0;
    has_gauss = false;
    cached_gauss = 0.0;
  }

  // The 624-word twist is generated lazily, CH words at a time, in index order. Word i only reads
  // the old mt[i], mt[i+1] and (i < 227 ? old mt[i+397] : new mt[i-227]); in-order chunks therefore
  // give exactly the values of a full regeneration, but a stream that is only partly consumed
  // never pays for the whole block, and each chunk is computed four words at a time.
  static constexpr int CH = 48;
  typedef uint32_t v4u __attribute__((vector_size(16)));

  static inline v4u twist4(v4u a, v4u b, v4u c) {
    const v4u UP = {0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U};
    const v4u LO = {0x7fffffffU, 0x7fffffffU, 0x7fffffffU, 0x7fffffffU};
    const v4u ONE = {1U, 1U, 1U, 1U};
    const v4u MA = {0x9908b0dfU, 0x9908b0dfU, 0x9908b0dfU, 0x9908b0dfU};
    v4u y = (a & UP) | (b & LO);
    return c ^ (y >> 1) ^ (((v4u){0, 0, 0, 0} - (y & ONE)) & MA);
  }

  static inline v4u temper4(v4u y) {
    y ^= (y >> 11);
    y ^= (y << 7) & (v4u){0x9d2c5680U, 0x9d2c5680U, 0x9d2c5680U, 0x9d2c5680U};
    y ^= (y << 15) & (v4u){0xefc60000U, 0xefc60000U, 0xefc60000U, 0xefc60000U};
    y ^= (y >> 18);
    return y;
  }

  __attribute__((noinline)) void refill() {
    if (gen >= 624) gen = 0;  // 624 = 13 * CH: chunks never straddle the end
    const int g = gen, e = g + CH;
    gen = e;
    pos = 0;
    if (e <= 227 || (g >= 227 && e <= 623)) {
      const int src = g < 227 ? 397 : -227;
      for (int i = g; i < e; i += 4) {
        v4u a, b, c;
        __builtin_memcpy(&a, mt + i, 16);
        __builtin_memcpy(&b, mt + i + 1, 16);
        __builtin_memcpy(&c, mt + i + src, 16);
        a = twist4(a, b, c);
        __builtin_memcpy(mt + i, &a, 16);
        a = temper4(a);
        __builtin_memcpy(out + (i - g), &a, 16);
      }
      return;
    }
    const uint32_t UPPER = 0x80000000U, LOWER = 0x7fffffffU, MATRIX_A = 0x9908b0dfU;
    for (int i = g; i < e; i++) {
      uint32_t y = (mt[i] & UPPER) | (mt[i + 1 < 624 ? i + 1 : 0] & LOWER);
      mt[i] = mt[i < 227 ? i + 397 : i - 227] ^ (y >> 1) ^ (-(int32_t)(y & 1U) & MATRIX_A);
    }
    for (int i = 0; i < CH; i += 4) {
      v4u y;
      __builtin_memcpy(&y, mt + g + i, 16);
      y = temper4(y);
      __builtin_memcpy(out + i, &y, 16);
    }
  }

  inline uint32_t next_u32() {
    if (pos == CH) refill();
    return out[pos++];
  }

  // the first word of the stream whose low bits (mask) are <= r32, each word consumed until then: the rejection loop
  // `while ((v = next_u32() & mask) > r32) {}` with two words looked at at once (a rejection is a coin flip, so the loop's exit branch mispredicts)
  __attribute__((always_inline)) inline uint32_t masked_upto(uint32_t mask, uint32_t r32) {
    if (__builtin_expect(pos <= CH - 2, 1)) {
      const uint32_t v0 = out[pos] & mask, v1 = out[pos + 1] & mask;
      const bool a0 = v0 <= r32;
      if (__builtin_expect(a0 | (v1 <= r32), 1)) {
        pos += 2 - (int)a0;
        return a0 ? v0 : v1;
      }
      pos += 2;
    }
    uint32_t v;
    while ((v = next_u32() & mask) > r32) {}
    return v;
  }

  inline uint64_t next_u64() {
    uint64_t hi = next_u32();
    uint64_t lo = next_u32();
    return (hi << 32) | lo;
  }

  inline double next_double() {
    int32_t a = (int32_t)(next_u32() >> 5), b = (int32_t)(next_u32() >> 6);
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  inline double gauss() {
    if (has_gauss) {
      has_gauss = false;
      return cached_gauss;
    }
    return gauss_pair();
  }
#if defined(__SSE2__)
  // Marsaglia polar pair, both uniforms of an attempt computed together (the same IEEE operations as the scalar form below)
  __attribute__((noinline)) double gauss_pair() {
    __m128d x;
    double r2;
    do {
      uint32_t tmp[4];
      const uint32_t* wp;
      if (__builtin_expect(pos <= CH - 4, 1)) {  // four draws from the current chunk: one bounds test
        wp = out + pos;
        pos += 4;
      } else {
        tmp[0] = next_u32(); tmp[1] = next_u32(); tmp[2] = next_u32(); tmp[3] = next_u32();
        wp = tmp;
      }
      const __m128i w = _mm_loadu_si128((const __m128i*)wp);
      const __m128d hi = _mm_cvtepi32_pd(_mm_shuffle_epi32(_mm_srli_epi32(w, 5), _MM_SHUFFLE(3, 3, 2, 0)));  // w0>>5, w2>>5
      const __m128d lo = _mm_cvtepi32_pd(_mm_shuffle_epi32(_mm_srli_epi32(w, 6), _MM_SHUFFLE(3, 3, 3, 1)));  // w1>>6, w3>>6
      const __m128d u = _mm_mul_pd(_mm_add_pd(_mm_mul_pd(hi, _mm_set1_pd(67108864.0)), lo), _mm_set1_pd(1.0 / 9007199254740992.0));
      x = _mm_sub_pd(_mm_add_pd(u, u), _mm_set1_pd(1.0));  // 2.0 * u - 1.0
      const __m128d sq = _mm_mul_pd(x, x);
      r2 = _mm_cvtsd_f64(sq) + _mm_cvtsd_f64(_mm_unpackhi_pd(sq, sq));
    } while (r2 >= 1.0 || r2 == 0.0);
    double f = std::sqrt(-2.0 * std::log(r2) / r2);
    cached_gauss = f * _mm_cvtsd_f64(x);
    has_gauss = true;
    return f * _mm_cvtsd_f64(_mm_unpackhi_pd(x, x));
  }

  // ---- batch generation (the same words and the same IEEE operations as next_double() / gauss() one at a time)
  // the two draws made of the words (w0,w1) and (w2,w3): lanes [next_double(), next_double()]
  static inline __m128d u01x2(__m128i w) {
    const __m128d hi = _mm_cvtepi32_pd(_mm_shuffle_epi32(_mm_srli_epi32(w, 5), _MM_SHUFFLE(3, 3, 2, 0)));
    const __m128d lo = _mm_cvtepi32_pd(_mm_shuffle_epi32(_mm_srli_epi32(w, 6), _MM_SHUFFLE(3, 3, 3, 1)));
    return _mm_mul_pd(_mm_add_pd(_mm_mul_pd(hi, _mm_set1_pd(67108864.0)), lo), _mm_set1_pd(1.0 / 9007199254740992.0));
  }

  __attribute__((noinline)) void words4_straddle(uint32_t* t) {  // four words across a chunk boundary
    for (int k = 0; k < 4; k++) t[k] = next_u32();
  }
  // the next four words as four next_u32() calls would return them; p is the caller's copy of pos
  __attribute__((always_inline)) inline __m128i words4(uint32_t& p) {
    if (__builtin_expect(p > (uint32_t)CH - 4, 0)) {
      if (p == (uint32_t)CH) {
        refill();
        p = 4;
        return _mm_loadu_si128((const __m128i*)out);
      }
      pos = (int)p;
      uint32_t t[4];
      words4_straddle(t);
      p = (uint32_t)pos;
      return _mm_loadu_si128((const __m128i*)t);
    }
    const __m128i w = _mm_loadu_si128((const __m128i*)(out + p));
    p += 4;
    return w;
  }

  // f(i, lanes) for the n (even) next_double() values, two at a time
  template <class F>
  __attribute__((always_inline)) inline void each_u01x2(int n, F f) {
    uint32_t p = (uint32_t)pos;
    for (int i = 0; i < n; i += 2) f(i, u01x2(words4(p)));
    pos = (int)p;
  }

  // n gauss() values in draw order, as loc + scale * gauss(), into g (n <= 256; g needs room for n + 4 doubles)
  void gauss_fill(double* g, int n, double loc, double scale) {
    int i = 0;
    if (n > 0 && has_gauss) {
      has_gauss = false;
      g[0] = loc + scale * cached_gauss;
      i = 1;
    }
    const int r = n - i, P = (r + 1) >> 1;
    if (P == 0) return;
    __m128d xs[130];
    double r2s[130];
    uint32_t p = (uint32_t)pos;
    int k = 0;
    // only accepted attempts advance k, so exactly the words of the sequential rejection loop are consumed
    while (k < P) {
      const __m128d u = u01x2(words4(p));
      const __m128d x = _mm_sub_pd(_mm_add_pd(u, u), _mm_set1_pd(1.0));
      const __m128d sq = _mm_mul_pd(x, x);
      const double r2 = _mm_cvtsd_f64(sq) + _mm_cvtsd_f64(_mm_unpackhi_pd(sq, sq));
      xs[k] = _mm_shuffle_pd(x, x, 1);  // [x2, x1]: the returned value first, then the cached one
      r2s[k] = r2;
      uint64_t b;
      __builtin_memcpy(&b, &r2, 8);
      k += (b - 1) < 0x3FEFFFFFFFFFFFFFULL;  // 0 < r2 < 1 (r2 is never negative or NaN)
    }
    pos = (int)p;
    const int Pe = (P + 1) & ~1;  // pairs processed two at a time; an odd P gets a padding pair
    r2s[P] = 0.5;
    xs[P] = _mm_setzero_pd();
    double f[130];
    map_log(f, r2s, Pe);
    for (int j = 0; j < Pe; j += 2)
      _mm_storeu_pd(f + j, _mm_sqrt_pd(_mm_div_pd(_mm_mul_pd(_mm_set1_pd(-2.0), _mm_loadu_pd(f + j)), _mm_loadu_pd(r2s + j))));
    const __m128d vl = _mm_set1_pd(loc), vs = _mm_set1_pd(scale);
    double* o = g + i;
    for (int j = 0; j < Pe; j++)
      _mm_storeu_pd(o + 2 * j, _mm_add_pd(vl, _mm_mul_pd(vs, _mm_mul_pd(xs[j], _mm_set1_pd(f[j])))));
    if (r & 1) {  // the last pair's second value stays cached (raw, not transformed)
      cached_gauss = _mm_cvtsd_f64(_mm_unpackhi_pd(xs[P - 1], xs[P - 1])) * f[P - 1];
      has_gauss = true;
    }
  }

  // dst[i] = log(src[i]) / exp(src[i]) for i < n: libm calls in a tight loop (the calls are independent, so they overlap)
  static double libm_log(double x) { return std::log(x); }
  static __attribute__((noinline)) void map_log(double* dst, const double* src, int n) {
    double (*f)(double) = &__ieee754_log ? __ieee754_log : libm_log;  // see map_exp
    for (int i = 0; i < n; i++) dst[i] = f(src[i]);
  }
  static double libm_exp(double x) { return std::exp(x); }
  static __attribute__((noinline)) void map_exp(double* dst, const double* src, int n) {
    // a static glibc's exp() is a wrapper (errno handling) around __ieee754_exp, which returns the same value; call that
    // directly when the link provides it (weak: a build without it uses exp())
    double (*f)(double) = &__ieee754_exp ? __ieee754_exp : libm_exp;
    for (int i = 0; i < n; i++) dst[i] = f(src[i]);
  }
#else
  __attribute__((noinline)) double gauss_pair() {
    double x1, x2, r2;
    do {
      x1 = 2.0 * next_double() - 1.0;
      x2 = 2.0 * next_double() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    double f = std::sqrt(-2.0 * std::log(r2) / r2);
    cached_gauss = f * x1;
    has_gauss = true;
    return f * x2;
  }
#endif

  inline double normal(double loc, double scale) { return loc + scale * gauss(); }
  inline double lognormal(double mean, double sigma) { return std::exp(normal(mean, sigma)); }
  inline double uniform(double low, double high) { return low + (high - low) * next_double(); }
  inline double standard_exponential() { return -std::log(1.0 - next_double()); }
  inline double exponential(double scale) { return scale * standard_exponential(); }
  inline double pareto(double a) { return std::exp(standard_exponential() / a) - 1.0; }

  static inline uint64_t gen_mask(uint64_t max) { return ~0ULL >> __builtin_clzll(max | 1); }

  // RandomState.randint(low, high), dtype int64, legacy masked rejection
  inline int64_t randint(int64_t low, int64_t high) {
    uint64_t rng = (uint64_t)(high - 1) - (uint64_t)low;
    uint64_t off = (uint64_t)low;
    if (rng == 0) return low;
    if (rng <= 0xFFFFFFFFULL) {
      if (rng == 0xFFFFFFFFULL) return (int64_t)(off + next_u32());
      uint32_t mask = (uint32_t)gen_mask(rng), v;
      while ((v = next_u32() & mask) > rng) {}
      return (int64_t)(off + v);
    }
    if (rng == 0xFFFFFFFFFFFFFFFFULL) return (int64_t)(off + next_u64());
    uint64_t mask = gen_mask(rng), v;
    while ((v = next_u64() & mask) > rng) {}
    return (int64_t)(off + v);
  }

  // randint(0, 2) and randint(0, hi + 1): the 32-bit masked-rejection case, inlined
  __attribute__((always_inline)) inline bool randbit() { return (next_u32() & 1U) != 0; }
  __attribute__((always_inline)) inline int64_t randint_0_to(int64_t hi) {  // == randint(0, hi + 1)
    const uint64_t rng = (uint64_t)hi;
    if (rng - 1 < 0xFFFFFFFEULL) {
      const uint32_t mask = 0xFFFFFFFFU >> __builtin_clz((uint32_t)rng), r32 = (uint32_t)rng;
      uint32_t v;
      while ((v = next_u32() & mask) > r32) {}
      return (int64_t)v;
    }
    return randint(0, hi + 1);
  }

  // np.random.randint(0, 2**32, dtype="uint64")
  inline uint64_t randint_seed_u32() { return next_u32(); }
};

}  // namespace nrng
