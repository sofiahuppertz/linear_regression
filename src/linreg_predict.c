#include "ml_operations.h"

double          linreg_predict(const double x, weight_and_bias wb) {
    return (wb.w * x) + wb.b;
}
