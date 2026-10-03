#include "ml_operations.h"
#include <stdio.h>

#define DATA_PATH        "data.csv"
#define VARIABLES_PATH   "variables.csv"
#define LEARNING_RATE    0.1
#define ITERATIONS       5000
#define PATIENCE         20

int main(void)
{
    dataset         ds;
    scaling         s;
    weight_and_bias init = {0.0, 0.0};
    training_params params = {
        .learning_rate  = LEARNING_RATE,
        .num_iterations = ITERATIONS,
        .patience       = PATIENCE
    };
    weight_and_bias scaled;
    weight_and_bias wb;

    if (dataset_load(DATA_PATH, &ds) != 0) {
        fprintf(stderr, "train: cannot read %s\n", DATA_PATH);
        return 1;
    }
    printf("Loaded %zu samples from %s\n", ds.m, DATA_PATH);

    s      = z_score_normalisation(&ds);
        scaled = gradient_descent(&ds, init, params);
    wb     = weight_and_bias_denormalize(scaled, s);

    printf("\ntheta0 (bias)   = %.6f\n", wb.b);
    printf("theta1 (weight) = %.6f\n", wb.w);

    dataset_free(&ds);

    if (weight_and_bias_save(VARIABLES_PATH, wb) != 0) {
        fprintf(stderr, "train: cannot write %s\n", VARIABLES_PATH);
        return 1;
    }
    printf("Saved to %s\n", VARIABLES_PATH);
    return 0;
}
