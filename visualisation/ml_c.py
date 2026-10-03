"""ctypes bindings to the C code in src/.

Nothing here reimplements the maths: every number comes back from the same
functions train.c and predict.c call."""

import ctypes
import glob
import os
import subprocess
import sys

import numpy as np

# This file lives in visualisation/, so the project root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
HEADER = os.path.join(SRC, "ml_operations.h")
_EXT = ".dylib" if sys.platform == "darwin" else ".so"
LIB_PATH = os.path.join(ROOT, "libml" + _EXT)
DATA_CSV = os.path.join(ROOT, "data.csv")
VARIABLES_CSV = os.path.join(ROOT, "variables.csv")

_lib = None


def build(force=False, verbose=True):
    """Compile src/*.c as a shared library, which is what ctypes can load.
    Rebuilds only when a source or the header is newer than the library."""
    srcs = sorted(glob.glob(os.path.join(SRC, "*.c")))
    newest = max(os.path.getmtime(p) for p in srcs + [HEADER])
    stale = (not os.path.exists(LIB_PATH)
             or os.path.getmtime(LIB_PATH) < newest)
    if force or stale:
        cmd = (["cc", "-shared", "-fPIC", "-O2", "-I", SRC]
               + srcs + ["-lm", "-o", LIB_PATH])
        if verbose:
            print("cc -shared -fPIC -O2 -Isrc src/*.c -lm -o "
                  + os.path.basename(LIB_PATH))
        subprocess.run(cmd, check=True)
    elif verbose:
        print(f"{os.path.basename(LIB_PATH)} up to date")
    return LIB_PATH


class Dataset(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("m", ctypes.c_size_t)]


class WeightAndBias(ctypes.Structure):
    _fields_ = [("w", ctypes.c_double), ("b", ctypes.c_double)]

    def __repr__(self):
        return f"WeightAndBias(w={self.w!r}, b={self.b!r})"


class Gradient(ctypes.Structure):
    _fields_ = [("dw", ctypes.c_double), ("db", ctypes.c_double)]

    def __repr__(self):
        return f"Gradient(dw={self.dw!r}, db={self.db!r})"


class Scaling(ctypes.Structure):
    _fields_ = [("mean", ctypes.c_double), ("spread", ctypes.c_double)]

    def __repr__(self):
        return f"Scaling(mean={self.mean!r}, spread={self.spread!r})"


class TrainingParams(ctypes.Structure):
    _fields_ = [("learning_rate", ctypes.c_double),
                ("num_iterations", ctypes.c_size_t),
                ("patience", ctypes.c_size_t)]

    def __repr__(self):
        return (f"TrainingParams(learning_rate={self.learning_rate!r}, "
                f"num_iterations={self.num_iterations!r}, "
                f"patience={self.patience!r})")


_DP = ctypes.POINTER(ctypes.c_double)


def load(force_build=False, verbose=False):
    global _lib
    if _lib is not None and not force_build:
        return _lib
    build(force=force_build, verbose=verbose)
    lib = ctypes.CDLL(LIB_PATH)

    lib.dataset_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(Dataset)]
    lib.dataset_load.restype = ctypes.c_int
    lib.dataset_free.argtypes = [ctypes.POINTER(Dataset)]
    lib.dataset_free.restype = None

    lib.squared_error_cost.argtypes = [ctypes.POINTER(Dataset), WeightAndBias]
    lib.squared_error_cost.restype = ctypes.c_double
    lib.compute_gradient.argtypes = [ctypes.POINTER(Dataset), WeightAndBias]
    lib.compute_gradient.restype = Gradient
    lib.gradient_descent.argtypes = [ctypes.POINTER(Dataset), WeightAndBias,
                                     TrainingParams]
    lib.gradient_descent.restype = WeightAndBias
    lib.linreg_predict.argtypes = [ctypes.c_double, WeightAndBias]
    lib.linreg_predict.restype = ctypes.c_double

    lib.z_score_normalisation.argtypes = [ctypes.POINTER(Dataset)]
    lib.z_score_normalisation.restype = Scaling
    lib.weight_and_bias_denormalize.argtypes = [WeightAndBias, Scaling]
    lib.weight_and_bias_denormalize.restype = WeightAndBias

    lib.r_squared.argtypes = [ctypes.POINTER(Dataset), WeightAndBias]
    lib.r_squared.restype = ctypes.c_double

    lib.weight_and_bias_save.argtypes = [ctypes.c_char_p, WeightAndBias]
    lib.weight_and_bias_save.restype = ctypes.c_int
    lib.weight_and_bias_load.argtypes = [ctypes.c_char_p,
                                         ctypes.POINTER(WeightAndBias)]
    lib.weight_and_bias_load.restype = ctypes.c_int

    _lib = lib
    return _lib


class CDataset:
    """A C `dataset` whose x/y point straight at two numpy buffers. The arrays
    stay owned by numpy, so dataset_free is never called on it."""

    def __init__(self, x, y):
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        if self.x.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be 1-D and the same length")
        self._c = Dataset(self.x.ctypes.data_as(_DP),
                          self.y.ctypes.data_as(_DP),
                          ctypes.c_size_t(self.x.size))

    @property
    def ref(self):
        return ctypes.byref(self._c)

    def __len__(self):
        return self.x.size


def as_dataset(x, y=None):
    return x if isinstance(x, CDataset) else CDataset(x, y)


def dataset_load(path):
    """C dataset_load: the same CSV parser train.c uses. -> (x, y)"""
    lib = load()
    ds = Dataset()
    if lib.dataset_load(str(path).encode(), ctypes.byref(ds)) != 0:
        raise IOError(f"dataset_load failed on {path}")
    x = np.ctypeslib.as_array(ds.x, shape=(ds.m,)).copy()
    y = np.ctypeslib.as_array(ds.y, shape=(ds.m,)).copy()
    lib.dataset_free(ctypes.byref(ds))
    return x, y


def compute_cost(x, y, w, b):
    """C squared_error_cost. Same signature as the Coursera labs' compute_cost,
    so the plotting helpers work unchanged."""
    ds = as_dataset(x, y)
    return load().squared_error_cost(ds.ref, WeightAndBias(float(w), float(b)))


def compute_gradient(x, y, w, b):
    """C compute_gradient -> (dj_dw, dj_db)"""
    ds = as_dataset(x, y)
    g = load().compute_gradient(ds.ref, WeightAndBias(float(w), float(b)))
    return g.dw, g.db


def predict(x, w, b):
    """C linreg_predict, over a scalar or an array."""
    lib = load()
    wb = WeightAndBias(float(w), float(b))
    if np.isscalar(x):
        return lib.linreg_predict(float(x), wb)
    xs = np.asarray(x, dtype=np.float64)
    out = [lib.linreg_predict(float(v), wb) for v in xs.ravel()]
    return np.array(out).reshape(xs.shape)


def gradient_descent(x, y, w_init=0.0, b_init=0.0, learning_rate=0.1,
                     num_iterations=1000, patience=0):
    """The C gradient_descent loop in one shot. Prints a line every 10
    iterations on stdout, exactly like ./train. patience=0 disables early
    stopping."""
    ds = as_dataset(x, y)
    params = TrainingParams(float(learning_rate), int(num_iterations),
                            int(patience))
    wb = load().gradient_descent(
        ds.ref, WeightAndBias(float(w_init), float(b_init)), params)
    return wb.w, wb.b


def gradient_descent_history(x, y, w_init=0.0, b_init=0.0, learning_rate=0.1,
                             num_iterations=1000, patience=0, verbose=True):
    """The same descent driven one C step at a time, so the (w, b) path and the
    cost history stay available for the plots.

    patience mirrors src/gradient_descent.c: stop once w and b have both been
    unchanged for that many consecutive iterations. 0 disables it."""
    ds = as_dataset(x, y)
    w, b = float(w_init), float(b_init)
    prev_w, prev_b = w, b
    unchanged = 0
    p_hist, j_hist = [], []
    every = 10 if patience else max(num_iterations // 10, 1)
    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient(ds, None, w, b)
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db
        cost = compute_cost(ds, None, w, b)
        p_hist.append([w, b])
        j_hist.append(cost)
        if verbose and i % every == 0:
            print(f"Iteration {i:4}: cost {cost:0.4e}  dj_dw {dj_dw: 0.3e}  "
                  f"dj_db {dj_db: 0.3e}  w {w: 0.5e}  b {b: 0.5e}")
        unchanged = unchanged + 1 if (w == prev_w and b == prev_b) else 0
        prev_w, prev_b = w, b
        if patience and unchanged >= patience:
            if verbose:
                print(f"Stopped at iteration {i}: w and b unchanged "
                      f"for {patience} iterations")
            break
    return w, b, np.array(j_hist), np.array(p_hist)


def z_score_normalisation(x):
    """C z_score_normalisation, on a copy -> (x_norm, Scaling)"""
    ds = CDataset(np.array(x, dtype=np.float64), np.zeros(np.size(x)))
    s = load().z_score_normalisation(ds.ref)
    return ds.x, s


def weight_and_bias_denormalize(w, b, scaling):
    wb = load().weight_and_bias_denormalize(
        WeightAndBias(float(w), float(b)), scaling)
    return wb.w, wb.b


def r_squared(x, y, w, b):
    ds = as_dataset(x, y)
    return load().r_squared(ds.ref, WeightAndBias(float(w), float(b)))


def weight_and_bias_save(path, w, b):
    rc = load().weight_and_bias_save(str(path).encode(),
                                     WeightAndBias(float(w), float(b)))
    if rc != 0:
        raise IOError(f"weight_and_bias_save failed on {path}")


def weight_and_bias_load(path):
    wb = WeightAndBias()
    rc = load().weight_and_bias_load(str(path).encode(), ctypes.byref(wb))
    return wb.w, wb.b, rc == 0
