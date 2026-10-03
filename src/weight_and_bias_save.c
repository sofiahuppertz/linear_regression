#include "ml_operations.h"

// variables.csv format: a "theta0,theta1" header plus one data row,
// where theta0 is the bias and theta1 the weight.
int weight_and_bias_save(const char *path, weight_and_bias wb)
{
    FILE *f;

    f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f, "theta0,theta1\n");
    fprintf(f, "%.12g,%.12g\n", wb.b, wb.w);
    if (fclose(f) != 0)
        return -1;
    return 0;
}
