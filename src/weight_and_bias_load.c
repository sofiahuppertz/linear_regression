#include "ml_operations.h"
#include <stdlib.h>

// Reads the thetas written by weight_and_bias_save. Leaves *wb at {0, 0} and returns -1
// when the file is missing or malformed, which is the untrained case.
int weight_and_bias_load(const char *path, weight_and_bias *wb)
{
    FILE *f;
    char  line[CSV_LINE_MAX];
    char *cur;
    char *end;

    wb->w = 0.0;
    wb->b = 0.0;

    f = fopen(path, "r");
    if (!f)
        return -1;

    // first line is the header, second holds the values
    if (!fgets(line, sizeof line, f) || !fgets(line, sizeof line, f)) {
        fclose(f);
        return -1;
    }
    fclose(f);

    cur = line;
    wb->b = strtod(cur, &end);
    if (end == cur || *end != ',') {
        wb->b = 0.0;
        return -1;
    }
    cur = end + 1;
    wb->w = strtod(cur, &end);
    if (end == cur) {
        wb->w = 0.0;
        wb->b = 0.0;
        return -1;
    }
    return 0;
}
