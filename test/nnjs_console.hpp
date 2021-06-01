#pragma once
#ifndef NNJS_CONSOLE_HPP
#define NNJS_CONSOLE_HPP

// Console simulator
// ----------------------------------------------------

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace NN { namespace console {

  // toString

  template<typename T1> std::string toString(const T1& x) { std::stringstream s; s << x; return s.str(); }
  template<typename T1> std::string toString(const std::vector<T1>& x)
  {
    std::stringstream s;
    int i = 0;
    s << "[";
    for (auto& item : x)
    {
      if (i > 0) { s << ","; }
      s << toString(item);
      i++;
    }
    s << "]";
    return s.str();
  }

  #if 0 // Will grap const char * as array
  template<typename T1, int n> std::string toString(const T1 (&x)[n])
  {
    std::stringstream s;
    int i = 0;
    s << "[";
    for (int i = 0; i < n; i++)
    {
      auto& item = x[i];
      if (i > 0) { s << ","; }
      s << toString(item);
      i++;
    }
    s << "]";
    return s.str();
  }
  #endif

  // log

  template<typename T1>
  void log(T1 a1) { std::cout << toString(a1) << "\n"; };

  template<typename T1, typename T2> 
  void log(const T1& a1, const T2& a2) { std::cout << toString(a1) << " " << toString(a2) << "\n"; };

  template<typename T1, typename T2, typename T3> 
  void log(const T1& a1, const T2& a2, const T3& a3) { std::cout << toString(a1) << " " << toString(a2) << " " << toString(a3) << "\n"; };

  template<typename T1, typename T2, typename T3, typename T4>
  void log(const T1& a1, const T2& a2, const T3& a3, const T4& a4) { std::cout << toString(a1) << " " << toString(a2) << " " << toString(a3) << " " << toString(a4) << "\n"; };

  template<typename T1, typename T2, typename T3, typename T4, typename T5>
  void log(const T1& a1, const T2& a2, const T3& a3, const T4& a4, const T5& a5) { std::cout << toString(a1) << " " << toString(a2) << " " << toString(a3) << " " << toString(a4) << " " << toString(a5) << "\n"; };

  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
  void log(const T1& a1, const T2& a2, const T3& a3, const T4& a4, const T5& a5, const T6& a6) { std::cout << toString(a1) << " " << toString(a2) << " " << toString(a3) << " " << toString(a4) << " " << toString(a5) << " " << toString(a6) << "\n"; };

  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
  void log(const T1& a1, const T2& a2, const T3& a3, const T4& a4, const T5& a5, const T6& a6, const T7& a7) { std::cout << toString(a1) << " " << toString(a2) << " " << toString(a3) << " " << toString(a4) << " " << toString(a5) << " " << toString(a6) << " " << toString(a7) << "\n"; };
} }

#endif
