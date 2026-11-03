#pragma once
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

namespace js {

struct V {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false;
  double d = 0;
  bool isint = false;
  int64_t i = 0;
  std::string s;
  std::vector<V> a;
  std::vector<std::pair<std::string, V>> o;

  const V* get(const char* k) const {
    if (t != OBJ) return nullptr;
    for (auto& kv : o)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
  bool truthy() const {
    switch (t) {
      case NUL: return false;
      case BOOL: return b;
      case NUM: return isint ? i != 0 : d != 0.0;
      case STR: return !s.empty();
      case ARR: return !a.empty();
      case OBJ: return !o.empty();
    }
    return false;
  }
  // Python float(x)
  double as_float() const {
    if (t == BOOL) return b ? 1.0 : 0.0;
    if (t == STR) {  // float("2e7"), float(" 0.5 ")
      const std::string r = py_num(s);
      char* e = nullptr;
      const double v = std::strtod(r.c_str(), &e);
      if (r.find_first_of("xX") != std::string::npos || e != r.c_str() + r.size()) throw std::runtime_error(T3_MSG("not a number", "E10"));
      return v;
    }
    if (t != NUM) throw std::runtime_error(T3_MSG("not a number", "E10"));
    return isint ? (double)i : d;
  }
  // Python int(x)
  int64_t as_int() const {
    if (t == BOOL) return b ? 1 : 0;
    if (t == STR) {  // int("1000"), int(" +1_000 ")
      const std::string r = py_num(s);
      const size_t k = (r[0] == '+' || r[0] == '-') ? 1 : 0;
      if (r.size() == k || r.find_first_not_of("0123456789", k) != std::string::npos) throw std::runtime_error(T3_MSG("not a number", "E10"));
      return std::strtoll(r.c_str(), nullptr, 10);
    }
    if (t != NUM) throw std::runtime_error(T3_MSG("not a number", "E10"));
    return isint ? i : (int64_t)std::trunc(d);
  }
  // a numeric string as Python reads it: surrounding whitespace dropped, single underscores between digits removed
  static std::string py_num(const std::string& x) {
    const char* ws = " \t\n\r\f\v";
    const size_t b = x.find_first_not_of(ws);
    if (b == std::string::npos) throw std::runtime_error(T3_MSG("not a number", "E10"));
    const size_t e = x.find_last_not_of(ws);
    std::string r;
    for (size_t k = b; k <= e; k++) {
      if (x[k] == '_') {
        if (k == b || k == e || !isdigit((unsigned char)x[k - 1]) || !isdigit((unsigned char)x[k + 1])) throw std::runtime_error(T3_MSG("not a number", "E10"));
        continue;
      }
      r += x[k];
    }
    return r;
  }
};

struct Parser {
  const char* p;
  const char* e;
  void ws() {
    while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
  }
  [[noreturn]] void fail() { throw std::runtime_error(T3_MSG("bad json", "E11")); }
  V parse() {
    ws();
    if (p >= e) fail();
    V v;
    char c = *p;
    if (c == '{') {
      v.t = V::OBJ;
      p++;
      ws();
      if (*p == '}') { p++; return v; }
      while (true) {
        ws();
        V k = parse();
        if (k.t != V::STR) fail();
        ws();
        if (*p != ':') fail();
        p++;
        V val = parse();
        v.o.emplace_back(std::move(k.s), std::move(val));
        ws();
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; break; }
        fail();
      }
    } else if (c == '[') {
      v.t = V::ARR;
      p++;
      ws();
      if (*p == ']') { p++; return v; }
      while (true) {
        v.a.push_back(parse());
        ws();
        if (*p == ',') { p++; continue; }
        if (*p == ']') { p++; break; }
        fail();
      }
    } else if (c == '"') {
      v.t = V::STR;
      p++;
      while (p < e && *p != '"') {
        if (*p == '\\') {
          p++;
          char x = *p++;
          switch (x) {
            case 'n': v.s += '\n'; break;
            case 't': v.s += '\t'; break;
            case 'r': v.s += '\r'; break;
            case 'b': v.s += '\b'; break;
            case 'f': v.s += '\f'; break;
            case 'u': {
              unsigned cp = (unsigned)strtoul(std::string(p, 4).c_str(), nullptr, 16);
              p += 4;
              if (cp < 0x80) v.s += (char)cp;
              else if (cp < 0x800) { v.s += (char)(0xC0 | (cp >> 6)); v.s += (char)(0x80 | (cp & 0x3F)); }
              else { v.s += (char)(0xE0 | (cp >> 12)); v.s += (char)(0x80 | ((cp >> 6) & 0x3F)); v.s += (char)(0x80 | (cp & 0x3F)); }
              break;
            }
            default: v.s += x;
          }
        } else {
          v.s += *p++;
        }
      }
      if (p >= e) fail();
      p++;
    } else if (c == 't' && e - p >= 4 && !strncmp(p, "true", 4)) {
      v.t = V::BOOL; v.b = true; p += 4;
    } else if (c == 'f' && e - p >= 5 && !strncmp(p, "false", 5)) {
      v.t = V::BOOL; v.b = false; p += 5;
    } else if (c == 'n' && e - p >= 4 && !strncmp(p, "null", 4)) {
      v.t = V::NUL; p += 4;
    } else {
      const char* st = p;
      bool isint = true;
      if (*p == '-') p++;
      while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) {
        if (*p == '.' || *p == 'e' || *p == 'E') isint = false;
        p++;
      }
      if (p == st) fail();
      std::string num(st, p);
      v.t = V::NUM;
      v.isint = isint;
      if (isint) {
        v.i = strtoll(num.c_str(), nullptr, 10);
        v.d = (double)v.i;
      } else {
        v.d = strtod(num.c_str(), nullptr);
      }
    }
    return v;
  }
};

inline V parse(const std::string& text) {
  Parser ps{text.data(), text.data() + text.size()};
  return ps.parse();
}

}  // namespace js
