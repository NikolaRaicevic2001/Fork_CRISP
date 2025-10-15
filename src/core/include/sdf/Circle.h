#pragma once

#include "sdf/utils.h"

// Circle SDF, gradient, Hessian (all analytical, eps-smoothed at the origin).
namespace CRISP { namespace sdf {

// ---- SDF: d(p) = ||p|| - r ------------------------------------------------
template<class T>
inline T sdfCircle(const Eigen::Matrix<T,2,1>& p, const T& r, const T eps = T(1e-12))
{
    const T len = safe_norm2(p.x(), p.y(), eps);   // strictly > 0 if eps>0
    return len - r;
}

// ---- Gradient: ∇d(p) = p/||p|| (unit outward normal) ----------------------
template<class T>
inline Eigen::Matrix<T,2,1> gradCircle(const Eigen::Matrix<T,2,1>& p, const T& /*r*/, const T eps = T(1e-12))
{
    const T len = safe_norm2(p.x(), p.y(), eps);
    const T inv = CppAD::CondExpGt(len, T(0), T(1)/len, T(0));

    Eigen::Matrix<T,2,1> g;
    g.x() = p.x() * inv;
    g.y() = p.y() * inv;
    return g;
}

// ---- Hessian: H = (I/||p||) - (p p^T)/||p||^3 -----------------------------
template<class T>
inline Eigen::Matrix<T,2,2> hessianCircle(const Eigen::Matrix<T,2,1>& p,
                                          const T& /*r*/,
                                          const T eps = T(1e-12))
{
    const T len = safe_norm2(p.x(), p.y(), eps);
    const T inv  = CppAD::CondExpGt(len, T(0), T(1)/len,  T(0));     // 1/||p||
    const T inv3 = CppAD::CondExpGt(len, T(0), inv*inv*inv, T(0));   // 1/||p||^3

    Eigen::Matrix<T,2,2> H = Eigen::Matrix<T,2,2>::Zero();
    H(0,0) = inv - inv3 * p.x()*p.x();
    H(0,1) =     - inv3 * p.x()*p.y();
    H(1,0) = H(0,1);
    H(1,1) = inv - inv3 * p.y()*p.y();
    return H;
}

// ---- Full wrapper returning {d, grad, H} ----------------------------------
template<class T>
struct Sdf2DFull { T d; Eigen::Matrix<T,2,1> grad; Eigen::Matrix<T,2,2> H;};

template<class T>
inline Sdf2DFull<T> sdgCircleFull(const Eigen::Matrix<T,2,1>& p, const T& r, const T eps = T(1e-12))
{
    Sdf2DFull<T> out;
    out.d    = sdfCircle<T>(p, r, eps);
    out.grad = gradCircle<T>(p, r, eps);
    out.H    = hessianCircle<T>(p, r, eps);
    return out;
}

// ---- Compatibility wrapper (keeps your previous API) ----------------------
template<class T>
inline Sdf2D<T> sdgCircle(const Eigen::Matrix<T,2,1>& p, const T& r, const T eps = T(1e-12))
{
    Sdf2D<T> out;
    out.d = sdfCircle<T>(p, r, eps);
    out.n = gradCircle<T>(p, r, eps);   
    return out;
}

}} // namespace CRISP::sdf
