extern "C" {
    __device__ const double *cescor_b;
    __device__ const int *cescor_iz;
    __device__ const double *cescor_fz;
}
__device__ void add_corrector(int node, double x, double y, int width, double grid_min,
                              double inverse_step, double radius, double& bx, double& by, double& bz) {
    const double u = (x - grid_min) * inverse_step;
    const double v = (y - grid_min) * inverse_step;
    if (!isfinite(u + v) || u < 0 || v < 0 || u >= width - 1 || v >= width - 1 ||
        x * x + y * y > radius * radius) {
        bx = nan("");
        return;
    }
    const int ix = static_cast<int>(u), iy = static_cast<int>(v), iz = cescor_iz[node];
    const double wx[] = {1 - (u - ix), u - ix};
    const double wy[] = {1 - (v - iy), v - iy};
    const double wz[] = {1 - cescor_fz[node], cescor_fz[node]};
    double field[3] = {};
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                const double weight = wz[k] * wy[j] * wx[i];
                const int offset = (((iz + k) * width + iy + j) * width + ix + i) * 3;
                for (int component = 0; component < 3; ++component) {
                    field[component] += weight * cescor_b[offset + component];
                }
            }
        }
    }
    bx += field[0];
    by += field[1];
    bz += field[2];
}
