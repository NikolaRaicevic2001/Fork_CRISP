#pragma once

#include "sdf/utils.h"

namespace CRISP { namespace sdf {
template<class T> inline Sdf2D<T> sdgCircle(const Eigen::Matrix<T,2,1>& p,
                                            const T& r,
                                            const T eps = T(1e-12))
{
    // robust length so the normal is defined even at the center
    const T len = safe_norm2(p.x(), p.y(), eps);

    // avoid division-by-zero at the centre
    T inv = CppAD::CondExpGt(len, T(0), T(1)/len, T(0));

    Sdf2D<T> out;
    out.d    = len - r;            // signed distance
    out.n.x() = p.x() * inv;      // unit outward normal
    out.n.y() = p.y() * inv;
    return out;
}

// Signed distance function for circle
template<class T>
inline T sdfCircle(const Eigen::Matrix<T,2,1>& p,  T radius)
{
    return CppAD::sqrt(p.x()*p.x() + p.y()*p.y()) - radius;
}

// Gradient of the SDF using analytical formula
template<class T>
inline Eigen::Matrix<T,2,1> sdfCircle_Grad(const Eigen::Matrix<T,2,1>& p, T radius)
{
    /* distance from centre ------------------------------------ */
    T len = CppAD::sqrt(p.x()*p.x() + p.y()*p.y());

    /* avoid division-by-zero at the centre -------------------- */
    T inv = CppAD::CondExpGt(len, T(0), T(1)/len, T(0));

    Eigen::Matrix<T,2,1> n;
    n << p.x()*inv, p.y()*inv;          // p / |p|
    return -n;
}
}}