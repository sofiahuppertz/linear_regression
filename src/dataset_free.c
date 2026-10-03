#include "ml_operations.h"
#include <stdlib.h>

void dataset_free(dataset *ds)
{
    if (!ds)
        return;
    free(ds->x);      // free(NULL) is a no-op, so this is safe either way
    free(ds->y);
    ds->x = NULL;     // leaving these set would make a second call a double free
    ds->y = NULL;
    ds->m = 0;
}
