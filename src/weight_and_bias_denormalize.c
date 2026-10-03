#include "ml_operations.h"

// Undo a dataset_normalize, so the thetas describe the raw data:
//   y = w'*(x - mean)/spread + b'  =  (w'/spread)*x + (b' - w'*mean/spread)
weight_and_bias weight_and_bias_denormalize(weight_and_bias wb, scaling s)
{
    weight_and_bias raw;

    raw.w = wb.w / s.spread;
    raw.b = wb.b - wb.w * s.mean / s.spread;
    return raw;
}
