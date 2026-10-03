#include "ml_operations.h"

double r_squared(const dataset *ds, weight_and_bias wb)
{
    double sum_of_squared_errors = 0.0;
    double sum_of_squared_difference_from_mean = 0.0;
    double mean = 0.0;
    size_t i;

    if (ds->m == 0)
        return 0.0;

    for (i = 0; i < ds->m; i++)
        mean += ds->y[i];
    mean /= ds->m;

    for (i = 0; i < ds->m; i++) {
        double error = linreg_predict(ds->x[i], wb) - ds->y[i];
        double deviation = ds->y[i] - mean;

        sum_of_squared_errors += error * error;
        sum_of_squared_difference_from_mean += deviation * deviation;
    }
    if (sum_of_squared_difference_from_mean == 0.0)   // every y identical; nothing to explain
        return 0.0;
    return 1.0 - sum_of_squared_errors / sum_of_squared_difference_from_mean;
}
