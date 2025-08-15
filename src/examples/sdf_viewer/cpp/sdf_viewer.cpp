#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <Eigen/Core>
#include "sdf/RoundBox.h"  


int main(int argc, char** argv) {
    using T  = double;
    using V2 = Eigen::Matrix<T,2,1>;
    namespace fs = std::filesystem;

    // --- CLI params ----------------------------------------------------------
    // Usage: ./sdf_viewer [shape] [hx] [hy] [r]
    // shape ∈ {"rounded_box","box"}; "box" is rounded with r=0
    std::string shape = (argc > 1) ? argv[1] : "rounded_box";
    T hx = 0.5, hy = 0.3, r = 0.1;    
    if (argc > 3) { hx = std::atof(argv[2]); hy = std::atof(argv[3]); }
    if (argc > 4) { r  = std::atof(argv[4]); }

    // --- Sampling grid -------------------------------------------------------
    const int W = 400, H = 400;
    const T xmin = -1.2, xmax = 1.2, ymin = -1.2, ymax = 1.2;

    // --- Output path: --------------------------------------------------------
    const fs::path out_dir = fs::path(__FILE__).parent_path().parent_path() / "results";
    std::cout << "Output directory: " << out_dir << "\n";

    // number -> filename-friendly string (e.g., 0.5 -> "0p500")
    auto num = [](double v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << v;
        std::string s = oss.str();
        std::replace(s.begin(), s.end(), '.', 'p');
        return s;
    };

    std::string fname = shape + "_sdf_" + num(hx) + "_" + num(hy);
    if (shape != "box") fname += "_r_" + num(r);
    fname += ".csv";
    const fs::path out_csv = out_dir / fname;

    std::ofstream csv(out_csv);
    if (!csv) {
        std::cerr << "ERROR: failed to open " << out_csv << " for writing.\n";
        return 1;
    }
    csv << std::setprecision(17);
    csv << "x,y,d,nx,ny\n";

    // --- Sample SDF ----------------------------------------------------------
    for (int j = 0; j < H; ++j) {
        const T y = ymin + (ymax - ymin) * (T(j) / (H - 1));
        for (int i = 0; i < W; ++i) {
            const T x = xmin + (xmax - xmin) * (T(i) / (W - 1));
            const V2 p(x, y);
            const V2 half(hx, hy);

            // Use rounded SDF; "box" is just r=0
            const T rr = (shape == "box") ? T(0) : r;
            const auto sdg = CRISP::sdf::sdgBoxRounded<T>(p, half, rr);

            csv << x << "," << y << "," << sdg.d << "," << sdg.n.x() << "," << sdg.n.y() << "\n";
        }
    }

    std::cerr << "Wrote " << out_csv << "\n";
    return 0;
}
