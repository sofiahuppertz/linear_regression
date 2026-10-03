#include "ml_operations.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VARIABLES_PATH "variables.csv"

int main(void)
{
    weight_and_bias wb;
    char            line[CSV_LINE_MAX];
    char           *end;
    double          mileage;

    if (weight_and_bias_load(VARIABLES_PATH, &wb) != 0)
        fprintf(stderr, "predict: no %s yet, using untrained model\n", VARIABLES_PATH);

    printf("Enter a mileage: ");
    fflush(stdout);
    if (!fgets(line, sizeof line, stdin))
        return 1;

    line[strcspn(line, "\n")] = '\0';

    errno = 0;
    mileage = strtod(line, &end);

    if (end == line || *end != '\0') {
        fprintf(stderr, "predict: '%s' is not a number\n", line);
        return 1;
    }
    if (errno == ERANGE || !isfinite(mileage)) {
        fprintf(stderr, "predict: '%s' is out of range\n", line);
        return 1;
    }
    if (mileage < 0) {
        fprintf(stderr, "predict: mileage cannot be negative\n");
        return 1;
    }
    printf("Estimated price: %.2f\n", linreg_predict(mileage, wb));
    return 0;
}
