#pragma once

// Dependencies
#include <type_traits>
#include <cmath>
#include <Eigen/Core>
#include <cppad/cppad.hpp>

// Forward-declare CG<Base> so we can specialize traits without including cg.hpp
namespace CppAD { namespace cg { template<class Base> class CG; } }

template<class T>
struct Sdf2D {
    T d;                                  // signed distance (>=0 outside)
    Eigen::Matrix<T,2,1> n;               // unit outward normal (‖n‖ = 1)
};

template<class T>
struct Sdf2DFull { 
    T d; 
    Eigen::Matrix<T,2,1> grad; 
    Eigen::Matrix<T,2,2> H;
};


// Detect CppAD AD types (cover AD<Base> and CG<Base>)
template<class T> struct is_ad : std::false_type {};
template<class Base> struct is_ad< CppAD::AD<Base> >     : std::true_type {};
template<class Base> struct is_ad< CppAD::cg::CG<Base> > : std::true_type {};

// Conditional expression wrappers: use CppAD when T is AD, else plain ternary.
template<class T>
inline T cexp_gt(const T& x, const T& y, const T& a, const T& b) {
    if constexpr (is_ad<T>::value) return CppAD::CondExpGt(x, y, a, b);
    else                           return (x > y) ? a : b;
}
template<class T>
inline T cexp_ge(const T& x, const T& y, const T& a, const T& b) {
    if constexpr (is_ad<T>::value) return CppAD::CondExpGe(x, y, a, b);
    else                           return (x >= y) ? a : b;
}
template<class T>
inline T cexp_lt(const T& x, const T& y, const T& a, const T& b) {
    if constexpr (is_ad<T>::value) return CppAD::CondExpLt(x, y, a, b);
    else                           return (x < y) ? a : b;
}

// Math wrappers: CppAD for AD types, std for arithmetic.
template<class T>
inline T ad_abs(const T& x){
    if constexpr (is_ad<T>::value)            return CppAD::abs(x);
    else if constexpr (std::is_arithmetic_v<T>) return T(std::abs(static_cast<double>(x)));
    else                                       return CppAD::abs(x);
}

template<class T>
inline T ad_sqrt(const T& x){
    if constexpr (is_ad<T>::value)            return CppAD::sqrt(x);
    else if constexpr (std::is_arithmetic_v<T>) return T(std::sqrt(static_cast<double>(x)));
    else                                       return CppAD::sqrt(x);
}

template<class T>
inline T clamp_ad(const T& x, const T& lo, const T& hi) {
    T x1 = cexp_lt(x, lo, lo, x);
    return cexp_gt(x1, hi, hi, x1);
}

template<class T>
inline T max_ad(const T& a, const T& b) {
    return cexp_gt(a, b, a, b);
}

template<class T>
inline T min_ad(const T& a, const T& b){ 
    return cexp_lt(a,b,a,b); 
}

template<class T>
inline T safe_norm2(const T& x, const T& y, const T& eps) {
    return ad_sqrt(x*x + y*y + eps*eps);
}