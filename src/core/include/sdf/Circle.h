#pragma once

#include "sdf/utils.h"

// Circle SDF, gradient, Hessian (all analytical, eps-smoothed at the origin).
namespace CRISP { namespace sdf {

template<class T>
inline T circle_len(const Eigen::Matrix<T,2,1>& p, const T& eps) 
{
    return CppAD::sqrt(p.x()*p.x() + p.y()*p.y() + eps*eps);
}

// ---- SDF: d(p) = ||p|| - r ------------------------------------------------
template<class T>
inline T sdfCircle(const Eigen::Matrix<T,2,1>& p, const T& R, const T& eps=T(0)) 
{
    const T L = circle_len(p, eps);
    return L - R;
}

// ---- Gradient: ∇d(p) = p/||p|| (unit outward normal) ----------------------
template<class T>
inline Eigen::Matrix<T,2,1> gradCircle(const Eigen::Matrix<T,2,1>& p, const T& /*R*/, const T& eps=T(0)) 
{
    const T L = circle_len(p, eps);
    Eigen::Matrix<T,2,1> g( T(0), T(0) );
    T inv = CppAD::CondExpGt(L, T(0), T(1)/L, T(0));
    g.x() = p.x() * inv;
    g.y() = p.y() * inv;
    return g;
}

// ---- Hessian: H = (I/||p||) - (p p^T)/||p||^3 -----------------------------
template<class T>
inline Eigen::Matrix<T,2,2> hessianCircle(const Eigen::Matrix<T,2,1>& p, const T& /*R*/, const T& eps=T(0)) 
{
    const T L = circle_len(p, eps);
    Eigen::Matrix<T,2,2> H; H.setZero();
    T invL   = CppAD::CondExpGt(L, T(0), T(1)/L, T(0));
    T invL3  = invL*invL*invL;

    // H = (I / L) - (p p^T) / L^3
    H(0,0) = invL - p.x()*p.x()*invL3;
    H(1,1) = invL - p.y()*p.y()*invL3;
    H(0,1) = - p.x()*p.y()*invL3;
    H(1,0) = H(0,1);
    return H;
}

// ---- Full wrapper returning {d, grad, H} ----------------------------------
template<class T>
inline Sdf2DFull<T> sdgCircleFull(const Eigen::Matrix<T,2,1>& p, const T& r, const T eps = T(1e-12))
{
    Sdf2DFull<T> out;
    out.d    = sdfCircle<T>(p, r, eps);
    out.grad = gradCircle<T>(p, r, eps);
    out.H    = hessianCircle<T>(p, r, eps);
    return out;
}

// ---- Wrapper returning {d, grad} only ----------------------------------
// Returns signed distance and unit outward normal for a circle of radius r.
// p   : query point (world or local — you decide upstream)
// r   : circle radius
// eps : small smoothing to keep the normal well-defined at p = 0
template<class T>
inline Sdf2D<T> sdgCircle(const Eigen::Matrix<T,2,1>& p, const T& r, const T eps = T(1e-12))
{
    // robust length so the normal is defined even at the center
    const T len = safe_norm2(p.x(), p.y(), eps);

    // avoid division by zero at the center
    const T inv = CppAD::CondExpGt(len, T(0), T(1) / len, T(0));

    Sdf2D<T> out;
    out.d     = len - r;       // signed distance
    out.n.x() = p.x() * inv;   // unit outward normal
    out.n.y() = p.y() * inv;
    return out;
}

}} // namespace CRISP::sdf
