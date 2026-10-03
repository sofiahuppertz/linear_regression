#include "ml_operations.h"

gradient        compute_gradient(const dataset *ds, weight_and_bias wb){

    double f_wb = 0;
    double dj_w_i = 0;
    double dj_b_i = 0;
    gradient    g = {0, 0};

    for (size_t i = 0; i < ds->m; i++){
        f_wb = linreg_predict(ds->x[i], wb);
        dj_w_i = (f_wb - ds->y[i]) * ds->x[i];
        dj_b_i = f_wb - ds->y[i];
        g.dw += dj_w_i;
        g.db += dj_b_i;
    }
    g.dw /= ds->m;
    g.db /= ds->m;
    return g;
}
