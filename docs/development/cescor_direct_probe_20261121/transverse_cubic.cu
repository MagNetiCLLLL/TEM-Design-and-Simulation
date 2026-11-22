    // At fixed z, interpolate the same four x and y coordinates for each B component.
    // The upper and lower z tables retain the native linear weights.
    if (ix < 1 || ix + 2 >= width || iy < 1 || iy + 2 >= width) {
        bx = nan("");
        return;
    }
    const double fx = u - ix, fy = v - iy;
    double weight[2][4];
    for (int dim = 0; dim < 2; ++dim) {
        const double f = dim == 0 ? fx : fy;
        weight[dim][0] = -f * (f - 1) * (f - 2) / 6;
        weight[dim][1] = (f + 1) * (f - 1) * (f - 2) / 2;
        weight[dim][2] = -(f + 1) * f * (f - 2) / 2;
        weight[dim][3] = (f + 1) * f * (f - 1) / 6;
    }
    double field[3] = {};
    const double wz[] = {1 - cescor_fz[node], cescor_fz[node]};
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                const double w = wz[k] * weight[1][j] * weight[0][i];
                const int offset = (((iz + k) * width + iy + j - 1) * width + ix + i - 1) * 3;
                for (int component = 0; component < 3; ++component) {
                    field[component] += w * cescor_b[offset + component];
                }
            }
        }
    }
    bx += field[0];
    by += field[1];
    bz += field[2];
}
