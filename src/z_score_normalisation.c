#include "ml_operations.h"

// Z-score standardization of the features, in place: x' = (x - mean) / spread.
// Returns the factors used, so the trained thetas can be mapped back to the
// original units with weight_and_bias_denormalize.
scaling z_score_normalisation(dataset *ds)
{
    scaling s = {0.0, 1.0};
    double  sum = 0.0;
    double  var = 0.0;
    size_t  i;

    if (ds->m == 0)
        return s;

    for (i = 0; i < ds->m; i++)
        sum += ds->x[i];
    s.mean = sum / ds->m;

    for (i = 0; i < ds->m; i++) {
        double d = ds->x[i] - s.mean;

        var += d * d;
    }
    s.spread = sqrt(var / ds->m);
    if (s.spread == 0.0)          // every x identical; nothing to scale
        s.spread = 1.0;

    for (i = 0; i < ds->m; i++)
        ds->x[i] = (ds->x[i] - s.mean) / s.spread;
    return s;
}
