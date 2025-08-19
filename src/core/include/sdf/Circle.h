#pragma once

#include "sdf/utils.h"

namespace CRISP { namespace sdf {

template<class T> inline Sdf2D<T> sdfCircle(const Eigen::Matrix<T,2,1>& p,
                                            const Eigen::Matrix<T,2,1>& c,
                                            const T& r_in,
                                            const T eps = T(1e-12)){
    const T zero = T(0);

    // clamp radius to be nonnegative (safe for optimization / AD)
    const T r = cexp_lt(r_in, zero, zero, r_in);

    // shift to circle-centered coordinates
    const T qx = p.x() - c.x();
    const T qy = p.y() - c.y();

    // robust length so the normal is defined even at the center
    const T len = safe_norm2(qx, qy, eps);

    Sdf2D<T> out;
    out.d    = len - r;            // signed distance
    out.n.x() = qx / len;          // unit outward normal
    out.n.y() = qy / len;
    return out;
}
}}