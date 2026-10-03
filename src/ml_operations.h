#ifndef ML_OPERATIONS_H
#define ML_OPERATIONS_H

#include <stddef.h>

#include <math.h>
#include <stdio.h>

#define CSV_LINE_MAX 256

typedef struct {
    double *x;
    double *y;
    size_t  m;
} dataset;

typedef struct {
    double w;
    double b;
} weight_and_bias;

typedef struct {
    double dw;
    double db;
} gradient;

typedef struct {
    double mean;
    double spread;
} scaling;

typedef struct {
    double learning_rate;
    size_t num_iterations;
    size_t patience;        // consecutive unchanged iterations before stopping; 0 disables
} training_params;

int  dataset_load(const char *path, dataset *ds);
void dataset_free(dataset *ds);

double          squared_error_cost(const dataset *ds, weight_and_bias wb);
gradient        compute_gradient(const dataset *ds, weight_and_bias wb);
weight_and_bias gradient_descent(const dataset *ds, weight_and_bias init, training_params params);
double          linreg_predict(double x, weight_and_bias wb);

scaling         z_score_normalisation(dataset *ds);
weight_and_bias weight_and_bias_denormalize(weight_and_bias wb, scaling s);

double r_squared(const dataset *ds, weight_and_bias wb);

int weight_and_bias_save(const char *path, weight_and_bias wb);
int weight_and_bias_load(const char *path, weight_and_bias *wb);

#endif
