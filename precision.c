#include "ml_operations.h"
#include <stdio.h>

#define DATA_PATH      "data.csv"
#define VARIABLES_PATH "variables.csv"

int main(void)
{
    dataset         ds;
    weight_and_bias wb;

    if (weight_and_bias_load(VARIABLES_PATH, &wb) != 0) {
        fprintf(stderr, "precision: no %s, run ./train first\n", VARIABLES_PATH);
        return 1;
    }
    if (dataset_load(DATA_PATH, &ds) != 0) {
        fprintf(stderr, "precision: cannot read %s\n", DATA_PATH);
        return 1;
    }

    printf("Model:   price = %.6f * km + %.4f\n", wb.w, wb.b);
    printf("Samples: %zu\n\n", ds.m);

    printf("R^2   %8.4f   fraction of the variance explained (1.0 = perfect)\n",
           r_squared(&ds, wb));

    dataset_free(&ds);
    return 0;
}
