#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>   // std::replace
#include <sstream>     // std::ostringstream
#include <string>
#include <Eigen/Core>

#include "sdf/RoundBox.h"
#include "sdf/Circle.h"

int main(int argc, char** argv) {
    using T  = double;
    using V2 = Eigen::Matrix<T,2,1>;
    namespace fs = std::filesystem;

    // ---------------------- CLI ----------------------
    // Usage:
    //   ./sdf_viewer <shape> [params...]
    //
    // Shapes and params (defaults in brackets):
    //   rounded        hx  hy  r        [0.5  0.3  0.1]
    //   roundedsmooth  hx  hy  r        [0.5  0.3  0.1]
    //   round          hx  hy  r        [0.5  0.3  0.1]
    //   circle         cx  cy  R        [0.0  0.0  0.3]
    //
    std::string shape = (argc > 1) ? argv[1] : "rounded";

    // defaults for box-like shapes
    T hx = 0.5, hy = 0.3, r = 0.1;

    // defaults for circle
    T cx = 0.0, cy = 0.0, R = 0.3;

    if (shape == "circle") {
        if (argc > 2) cx = std::atof(argv[2]);
        if (argc > 3) cy = std::atof(argv[3]);
        if (argc > 4) R  = std::atof(argv[4]);
    } else {
        if (argc > 2) hx = std::atof(argv[2]);
        if (argc > 3) hy = std::atof(argv[3]);
        if (argc > 4) r  = std::atof(argv[4]);
    }

    // ---------------------- Grid ----------------------
    const int W = 400, H = 400;
    const T xmin = -1.2, xmax = 1.2, ymin = -1.2, ymax = 1.2;

    // ------------------ Output path -------------------
    const fs::path out_dir = fs::path(__FILE__).parent_path().parent_path() / "results";
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    std::cout << "Output directory: " << out_dir << "\n";

    auto num = [](double v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << v;
        std::string s = oss.str();
        std::replace(s.begin(), s.end(), '.', 'p'); // 0.5 -> 0p500
        return s;
    };

    std::string fname;
    if (shape == "circle") {
        fname = "circle_sdf_cx" + num(cx) + "_cy" + num(cy) + "_R" + num(R) + ".csv";
    } else { // rounded / roundedsmooth / round
        fname = shape + "_sdf_hx" + num(hx) + "_hy" + num(hy) + "_r" + num(r) + ".csv";
    }
    const fs::path out_csv = out_dir / fname;

    std::ofstream csv(out_csv);
    if (!csv) {
        std::cerr << "ERROR: failed to open " << out_csv << " for writing.\n";
        return 1;
    }
    csv << std::setprecision(17);
    csv << "x,y,d,nx,ny\n";

    // ----------------- Precompute params --------------
    const V2 half(hx, hy);
    const V2 center(cx, cy);

    // ----------------- Sample SDF ---------------------
    Sdf2D<T> sdg; // reuse this each loop

    for (int j = 0; j < H; ++j) {
        const T y = ymin + (ymax - ymin) * (T(j) / (H - 1));
        for (int i = 0; i < W; ++i) {
            const T x = xmin + (xmax - xmin) * (T(i) / (W - 1));
            const V2 p(x, y);

            if (shape == "roundedsmooth") {
                sdg = CRISP::sdf::sdfBoxRoundedSmooth<T>(p, half, r);
            } else if (shape == "rounded") {
                sdg = CRISP::sdf::sdfBoxRounded<T>(p, half, r);
            } else if (shape == "round") {
                sdg = CRISP::sdf::sdfBoxRound<T>(p, half, r);
            } else if (shape == "circle") {
                sdg = CRISP::sdf::sdfCircle<T>(p, center, R);
            } else {
                std::cerr << "ERROR: unknown shape '" << shape << "'\n";
                return 1;
            }

            csv << x << "," << y << "," << sdg.d << "," << sdg.n.x() << "," << sdg.n.y() << "\n";
        }
    }

    std::cerr << "Wrote " << out_csv << "\n";
    return 0;
}
