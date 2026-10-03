#include "ml_operations.h"
#include <stdlib.h>

static int parse_row(const char *line, double *x, double *y)
{
    char *end;

    *x = strtod(line, &end);
    if (end == line || *end != ',')
        return -1;
    line = end + 1;
    *y = strtod(line, &end);
    if (end == line)
        return -1;
    return 0;
}

// First pass: how many lines actually parse as data?
static size_t count_rows(FILE *f)
{
    char   line[CSV_LINE_MAX];
    size_t rows = 0;
    double x;
    double y;

    while (fgets(line, sizeof line, f))
        if (parse_row(line, &x, &y) == 0)
            rows++;
    return rows;
}

int dataset_load(const char *path, dataset *ds)
{
    FILE   *f;
    char    line[CSV_LINE_MAX];
    size_t  rows;
    size_t  i = 0;
    double  x;
    double  y;

    ds->x = NULL;
    ds->y = NULL;
    ds->m = 0;

    f = fopen(path, "r");
    if (!f)
        return -1;

    rows = count_rows(f);
    if (rows == 0) {
        fclose(f);
        return -1;
    }
    rewind(f);

    ds->x = malloc(rows * sizeof *ds->x);
    ds->y = malloc(rows * sizeof *ds->y);
    if (!ds->x || !ds->y) {
        dataset_free(ds);
        fclose(f);
        return -1;
    }

    // Second pass: fill. Same filter as the count, so i lands on rows.
    while (i < rows && fgets(line, sizeof line, f)) {
        if (parse_row(line, &x, &y) == 0) {
            ds->x[i] = x;
            ds->y[i] = y;
            i++;
        }
    }
    fclose(f);
    ds->m = i;
    return 0;
}
