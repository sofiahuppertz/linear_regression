#include "ml_operations.h"

double squared_error_cost(const dataset *ds, weight_and_bias wb){

    double  cost = 0;

    for (size_t i = 0; i < ds->m; i++) {
        double err = linreg_predict(ds->x[i], wb) - ds->y[i];

        cost += err * err;
    }
    cost = cost / (2 * ds->m);

    return cost;
}
