#pragma once

// Rounded-rectangle signed distance with unit outward normal.
// Works for both T = double and T = CppAD::AD<double>.
//
// SDF is for a box centered at the origin with half-sizes `half = (hx, hy)`
// and uniform corner radius `r`. When r=0, it reduces to a sharp box.
//
// Returns: Sdf2D<T> { d, n }  where d is signed distance (>=0 outside),
// and n is the unit outward normal.

// Dependencies
#include <type_traits>
#include <Eigen/Core>
#include <cppad/cppad.hpp>

namespace CRISP { namespace sdf {
// ------------------------ Small utilities ------------------------------------
// Detect CppAD AD types
template<class T> struct is_ad : std::false_type {};
template<class Base> struct is_ad< CppAD::AD<Base> > : std::true_type {};

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
inline T ad_abs(const T& x) {
    if constexpr (is_ad<T>::value) return CppAD::abs(x);
    else                           return T(std::abs(double(x)));
}
template<class T>
inline T ad_sqrt(const T& x) {
    if constexpr (is_ad<T>::value) return CppAD::sqrt(x);
    else                           return T(std::sqrt(double(x)));
}

template<class T>
struct Sdf2D {
    T d;                                  // signed distance (>=0 outside)
    Eigen::Matrix<T,2,1> n;               // unit outward normal (‖n‖ = 1)
};

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
inline T safe_norm2(const T& x, const T& y, const T& eps) {
    return ad_sqrt(x*x + y*y + eps*eps);
}

// ------------------------ Rounded Box SDF + normal ----------------------------
// p    : query point (x,y)
// half : box half-sizes (hx,hy)
// r_in : corner radius (clamped to [0, min(hx,hy)])
// eps  : small number to avoid 0/0 at corners/edges
template<class T>
inline Sdf2D<T> sdgBoxRounded(const Eigen::Matrix<T,2,1>& p,
                              const Eigen::Matrix<T,2,1>& half,
                              const T& r_in,
                              const T eps = T(1e-12)){
    const T zero = T(0), one = T(1);

    // Clamp r
    T rmax = cexp_lt(half.x(), half.y(), half.x(), half.y());
    T r    = clamp_ad(r_in, zero, rmax);

    // |p| and reduced half-sizes (half - r)
    T ax = ad_abs(p.x());
    T ay = ad_abs(p.y());
    T hx = half.x() - r;
    T hy = half.y() - r;

    // distances to the rounded rectangle
    T wx = ax - hx;
    T wy = ay - hy;

    // g = max(wx, wy); q = max(w, 0)
    T g  = max_ad(wx, wy);
    T qx = cexp_gt(wx, zero, wx, zero);
    T qy = cexp_gt(wy, zero, wy, zero);
    T l  = safe_norm2(qx, qy, eps);

    // Signed distance: outside uses circle (|q|-r), inside uses slab (g-r)
    T d = cexp_gt(g, zero, (l - r), (g - r));

    // Unit outward normal.
    // signs to mirror into correct quadrant
    T sx = cexp_ge(p.x(), zero, one, T(-1));
    T sy = cexp_ge(p.y(), zero, one, T(-1));

    // Outside: normal ~ q / |q|
    T nx_out = sx * (qx / l);
    T ny_out = sy * (qy / l);

    // Inside: pick dominant slab normal
    T pickX  = cexp_gt(wx, wy, one, zero);
    T pickY  = cexp_gt(wx, wy, zero, one);
    T nx_in  = sx * pickX;
    T ny_in  = sy * pickY;

    Eigen::Matrix<T,2,1> n;
    n.x() = cexp_gt(g, zero, nx_out, nx_in);
    n.y() = cexp_gt(g, zero, ny_out, ny_in);

    // Final guard/renormalization
    T nn = safe_norm2(n.x(), n.y(), eps);
    n.x() /= nn; n.y() /= nn;

    return { d, n };
}
}} // namespace CRISP::sdf

