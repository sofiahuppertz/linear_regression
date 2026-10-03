#include "ml_operations.h"
#include <stdio.h>

// params.patience stops the loop once w and b have both been unchanged for that
// many consecutive iterations; 0 disables it and the full count always runs.
weight_and_bias gradient_descent(const dataset *ds, weight_and_bias init, training_params params){

    weight_and_bias wb = init;
    weight_and_bias prev = init;
    gradient  g = {0,0};
    size_t    unchanged = 0;

    for (size_t i = 0; i < params.num_iterations; i++ ){
        g = compute_gradient(ds, wb);

        wb.w -= params.learning_rate * g.dw;
        wb.b -= params.learning_rate * g.db;

        if (i % 10 == 0)
            printf("Iteration %6zu: w = %14.6g  b = %14.6g  R^2 = %10.4g\n",
                   i, wb.w, wb.b, r_squared(ds, wb));

        if (wb.w == prev.w && wb.b == prev.b)
            unchanged++;
        else
            unchanged = 0;
        prev = wb;

        if (params.patience != 0 && unchanged >= params.patience) {
            printf("Stopped at iteration %zu: w and b unchanged for %zu iterations\n",
                   i, params.patience);
            break;
        }
    }
    return wb;
}
