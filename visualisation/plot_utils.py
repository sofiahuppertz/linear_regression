"""Plotting routines for ft_linear_regression.

Copied from the Coursera C1W1 optional labs (lab_utils_uni.py and
lab_utils_common.py) and adapted on two points: every cost value comes
from the C squared_error_cost through ml_c, and the w/b ranges are derived
from the trained parameters instead of being hard-coded for the housing
example."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import ml_c

dlblue, dlorange, dldarkred = '#0096ff', '#FF9300', '#C00000'
dlmagenta, dlpurple = '#FF40FF', '#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
dlc = dict(dlblue=dlblue, dlorange=dlorange, dldarkred=dldarkred,
           dlmagenta=dlmagenta, dlpurple=dlpurple)
dlcm = LinearSegmentedColormap.from_list('dl_map', dlcolors, N=5)

_HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(_HERE, 'deeplearning.mplstyle')
OUTDIR = os.path.join(_HERE, 'images')


def save(fig: plt.Figure, name: str, dpi: int = 150) -> str:
    """Write fig to visualisation/images/<name>.png and return the path.

    Call it before plt.show(): the inline backend closes the figure on show,
    so a savefig afterwards writes a blank canvas."""
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return path


def use_lab_style():
    if os.path.exists(STYLE):
        plt.style.use(STYLE)


def plt_data(x, y, f_wb=None, ax=None, title="Car price vs mileage",
             xlabel="Mileage (km)", ylabel="Price"):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.scatter(x, y, marker='x', c='r', label="Actual Value")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if f_wb is not None:
        order = np.argsort(x)
        ax.plot(np.asarray(x)[order], np.asarray(f_wb)[order],
                c=dlblue, label="Our Prediction")
    ax.legend()
    return ax


def plt_normalisation(x, x_norm, y):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plt_data(x, y, ax=ax[0], title="Before normalisation",
             xlabel="Mileage (km)")
    plt_data(x_norm, y, ax=ax[1], title="After z-score normalisation",
             xlabel=r"$(km - \mu)\ /\ \sigma$")
    for a in ax:
        a.legend().remove()
    fig.suptitle("Same data, rescaled feature")
    return fig, ax


def mk_cost_lines(x, y, w, b, ax):
    """Vertical error bars between each point and the model, plus the
    total cost. The lab annotated every term; with 24 points only the
    total stays readable."""
    label = 'cost for point'
    total = 0.0
    for xi, yi in zip(x, y):
        f_wb = w * xi + b
        total += (f_wb - yi) ** 2 / 2
        ax.vlines(xi, yi, f_wb, lw=2, color=dlpurple, ls='dotted', label=label)
        label = ''
    total /= len(x)
    ax.text(0.05, 0.05, f"cost = (1/m)*sum = {total:0.0f}",
            transform=ax.transAxes, color=dlpurple)


def cost_grid(x, y, w_space, b_space):
    """J(w,b) over a mesh, one C squared_error_cost call per node."""
    ds = ml_c.CDataset(x, y)
    tmp_b, tmp_w = np.meshgrid(b_space, w_space)
    z = np.zeros_like(tmp_b)
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            z[i, j] = ml_c.compute_cost(ds, None, tmp_w[i, j], tmp_b[i, j])
            if z[i, j] == 0:
                z[i, j] = 1e-6
    return tmp_w, tmp_b, z


def _spans(w_final, b_final, w_span, b_span):
    if w_span is None:
        w_span = 2.5 * max(abs(w_final), 1.0)
    if b_span is None:
        b_span = w_span
    return (np.array([w_final - w_span, w_final + w_span]),
            np.array([b_final - b_span, b_final + b_span]))


def soup_bowl():
    """The idealised J(w,b) = w^2 + b^2 bowl, straight from the lab."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(45, -120)

    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)
    z = np.zeros((len(w), len(b)))
    for j, xw in enumerate(w):
        for i, yb in enumerate(b):
            z[i, j] = xw ** 2 + yb ** 2

    W, B = np.meshgrid(w, b)
    ax.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("$J(w,b)$", rotation=90)
    ax.set_title("$J(w,b)$\n [You can rotate this figure]", size=15)
    return fig, ax


def soup_bowl_data(x_train, y_train, w_final=0.0, b_final=0.0,
                   w_span=None, b_span=None, log_scale=False):
    """The same bowl, but the real J(w,b) of this dataset
    (C cost on a mesh)."""
    w_range, b_range = _spans(w_final, b_final, w_span, b_span)
    tmp_w, tmp_b, z = cost_grid(x_train, y_train,
                                np.linspace(*w_range, 100),
                                np.linspace(*b_range, 100))
    if log_scale:
        z = np.log(z)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(45, -120)
    ax.plot_surface(tmp_w, tmp_b, z, cmap="Spectral_r", alpha=0.7,
                    antialiased=False)
    ax.plot_wireframe(tmp_w, tmp_b, z, color='k', alpha=0.1)
    j_final = ml_c.compute_cost(x_train, y_train, w_final, b_final)
    ax.scatter3D(w_final, b_final, j_final,
                 marker='X', s=120, color=dldarkred)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel(("$\\log J(w,b)$" if log_scale else "$J(w,b)$"), rotation=90)
    ax.set_title("$J(w,b)$ for data.csv\n", size=15)
    return fig, ax


def inbounds(a, b, xlim, ylim):
    xlow, xhigh = xlim
    ylow, yhigh = ylim
    ax, ay = a
    bx, by = b
    return (xlow < ax < xhigh and xlow < bx < xhigh
            and ylow < ay < yhigh and ylow < by < yhigh)


def plt_contour_wgrad(x, y, hist, ax=None, w_range=None, b_range=None,
                      contours=None, resolution=None, w_final=0.0,
                      b_final=0.0, step=10):
    """Contour of J(w,b) with the gradient descent path drawn on top."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 6))
    wr, br = _spans(w_final, b_final, None, None)
    w_range = np.linspace(*wr, 80) if w_range is None else np.asarray(w_range)
    b_range = np.linspace(*br, 80) if b_range is None else np.asarray(b_range)
    w0, b0, z = cost_grid(x, y, w_range, b_range)

    if contours is None:
        j_end = max(ml_c.compute_cost(x, y, w_final, b_final), z.min(), 1e-6)
        contours = np.unique(np.geomspace(j_end, z.max(), 8))
    CS = ax.contour(w0, b0, z, contours, linewidths=2,
                    colors=[dlblue, dlorange, dldarkred, dlmagenta, dlpurple])
    ax.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title('Cost J(w,b) with the path of gradient descent')
    ax.hlines(b_final, ax.get_xlim()[0], w_final, lw=2, color=dlpurple,
              ls='dotted')
    ax.vlines(w_final, ax.get_ylim()[0], b_final, lw=2, color=dlpurple,
              ls='dotted')

    if resolution is None:
        resolution = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    base = list(hist[0])
    for point in list(hist)[0::step]:
        point = list(point)
        edist = np.sqrt((base[0] - point[0]) ** 2 + (base[1] - point[1]) ** 2)
        if edist > resolution or point == list(hist[-1]):
            if inbounds(point, base, ax.get_xlim(), ax.get_ylim()):
                ax.annotate('', xy=point, xytext=base, xycoords='data',
                            arrowprops={'arrowstyle': '->', 'color': 'r',
                                        'lw': 3},
                            va='center', ha='center')
            base = point
    return ax


def plt_convergence(j_hist):
    """Left: the cost curve. Right: how far the cost still is above its final
    value, on a log axis.

    A plain zoom on the last iterations is useless once the run has converged:
    the remaining spread is a few units in the last place of a double, so the
    axis autoscales onto rounding noise. The excess J - J_final decays
    geometrically (a straight line here) until it reaches that floor, which is
    the point where more iterations buy nothing."""
    j_hist = np.asarray(j_hist, dtype=float)
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(11, 4))

    ax[0].plot(j_hist, color=dlblue)
    ax[0].set_title("Cost vs. iteration")
    ax[0].set_ylabel('Cost')

    excess = j_hist - j_hist.min()
    steps = np.arange(len(j_hist))
    moving = excess > 0
    ax[1].semilogy(steps[moving], excess[moving], color=dlblue)
    if moving.sum() > 10:
        # the plateau is rounding noise: take its level from the last fifth of
        # the curve, and mark where the decay first reaches it
        tail = excess[moving][-max(len(excess) // 5, 5):]
        floor = float(np.median(tail))
        reached = int(steps[moving][np.argmax(excess[moving] <= floor * 3)])
        ax[1].axhline(floor, color=dldarkred, ls='--', lw=1)
        ax[1].annotate(f'rounding-error floor\nreached at iteration {reached}',
                       xy=(reached, floor), xytext=(10, 28),
                       textcoords='offset points', fontsize=9, color=dldarkred)
    ax[1].set_title("Cost above its final value (log)")
    ax[1].set_ylabel(r'$J - J_{final}$')

    for a in ax:
        a.set_xlabel('iteration step')
    return fig, ax


def plt_r_squared(x, y, w, b, p_hist=None):
    """R^2 as a decomposition of variance.

    Left  : the total variation in y, measured as deviations from its mean.
            Squared and summed, that is SS_tot = m * Var(y).
    Middle: the variation still left once the model has had its say - the
            residuals, squaring to SS_res.
    Right  (optional, needs p_hist): R^2 at each step of training.

    SS_res/SS_tot is the fraction of the variance the model does NOT account
    for, so R^2 = 1 - SS_res/SS_tot is the fraction it does. The m in both sums
    cancels, which is why this is a statement about variance and not about the
    sample size. Every number comes from the C r_squared / linreg_predict.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_mean = y.mean()
    y_hat = np.asarray(ml_c.predict(x, w, b), dtype=float)

    ss_tot = float(((y - y_mean) ** 2).sum())
    ss_res = float(((y - y_hat) ** 2).sum())
    r2 = ml_c.r_squared(x, y, w, b)

    n = 3 if p_hist is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 4.2),
                           constrained_layout=True)

    order = np.argsort(x)
    for a, pred, ss, name, colour in (
            (ax[0], np.full_like(y, y_mean), ss_tot, r'$SS_{tot}$', dlorange),
            (ax[1], y_hat, ss_res, r'$SS_{res}$', dlblue)):
        a.vlines(x, y, pred, color=dldarkred, alpha=0.55, lw=1.2, zorder=1)
        a.scatter(x, y, marker='x', s=45, color=dldarkred, zorder=3)
        a.plot(x[order], pred[order], color=colour, lw=2, zorder=2)
        a.set_xlabel('km')
        a.set_ylabel('price')
        a.text(0.97, 0.95, f'{name} = {ss:,.0f}', transform=a.transAxes,
               ha='right', va='top', fontsize=11)

    ax[0].set_title('Total variance: spread of price around its mean')
    ax[1].set_title('Unexplained variance: what the line leaves behind')

    if p_hist is not None:
        p_hist = np.asarray(p_hist, dtype=float)
        r2_hist = [ml_c.r_squared(x, y, wi, bi) for wi, bi in p_hist]
        ax[2].plot(r2_hist, color=dlblue, lw=2)
        ax[2].axhline(r2, color=dldarkred, ls='--', lw=1)
        # the first iterations are far below 0 (a fresh model is much
        # worse than the mean); clamping keeps the part anyone reads legible
        ax[2].set_ylim(0.0, 1.02)
        ax[2].text(0.97, 0.06,
                   f'starts at $R^2$ = {r2_hist[0]:.1f}, off scale',
                   transform=ax[2].transAxes, ha='right', fontsize=9,
                   color=dldarkred)
        ax[2].text(0.97, 0.86, f'final $R^2$ = {r2:.4f}',
                   transform=ax[2].transAxes, ha='right', fontsize=10)
        ax[2].set_title('$R^2$ during training')
        ax[2].set_xlabel('iteration step')
        ax[2].set_ylabel('$R^2$')

    fig.suptitle(
        f'$R^2 = 1 - SS_{{res}}/SS_{{tot}} = 1 - {ss_res:,.0f}/{ss_tot:,.0f}'
        f' = {r2:.4f}$   -   {1 - r2:.1%} of the variance is left '
        f'unexplained', fontsize=13)
    return fig, ax


def plt_cost_surface(x_train, y_train, w_final, b_final, w_span=None,
                     b_span=None, resolution=160, elev=32, azim=-125,
                     figsize=(11, 8.5), p_hist=None):
    """The J(w,b) bowl on its own, at plotting resolution, with the trained
    minimum marked.

    resolution is the mesh side, so the cost grid is resolution^2 C calls -
    160 gives a smooth surface in a couple of seconds. p_hist, if given, draws
    the descent path down the wall of the bowl.
    """
    w_range, b_range = _spans(w_final, b_final, w_span, b_span)
    W, B, Z = cost_grid(x_train, y_train,
                        np.linspace(*w_range, resolution),
                        np.linspace(*b_range, resolution))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev, azim)

    ax.plot_surface(W, B, Z, cmap=dlcm, alpha=0.30, antialiased=True,
                    rstride=1, cstride=1, linewidth=0)
    step = max(resolution // 40, 1)
    ax.plot_wireframe(W, B, Z, color='k', alpha=0.10,
                      rstride=step, cstride=step)

    j_min = ml_c.compute_cost(x_train, y_train, w_final, b_final)

    if p_hist is not None:
        p_hist = np.asarray(p_hist, dtype=float)
        j_path = [ml_c.compute_cost(x_train, y_train, wi, bi)
                  for wi, bi in p_hist]
        ax.plot(p_hist[:, 0], p_hist[:, 1], j_path, color=dlorange, lw=1.8,
                alpha=0.95, label='descent path', zorder=5)

    # a dropline from the surface minimum to the contour floor, so the marker
    # reads as a position in (w, b) and not just a floating dot
    ax.plot([w_final, w_final], [b_final, b_final], [Z.min(), j_min],
            color=dldarkred, ls='--', lw=1.2, zorder=6)
    ax.scatter([w_final], [b_final], [j_min], s=170, color=dldarkred,
               edgecolor='white', linewidth=1.6, depthshade=False, zorder=10,
               label='minimum after training')
    ax.text(w_final, b_final, j_min + 0.10 * (Z.max() - Z.min()),
            f'$w$ = {w_final:,.1f}\n$b$ = {b_final:,.1f}\n$J$ = {j_min:,.0f}',
            fontsize=10, color=dldarkred, weight='bold')

    # the style file suppresses the automatic 1e7 offset on 3D axes, so fold
    # the exponent into the label instead of leaving bare 0.00-1.75 ticks
    exponent = int(np.floor(np.log10(max(abs(Z).max(), 1e-30))))
    if exponent:
        ax.zaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v / 10 ** exponent:g}'))
        title_units = f'  (in units of $10^{{{exponent}}}$)'
    else:
        title_units = ''

    ax.set_xlabel('$w$', labelpad=12)
    ax.set_ylabel('$b$', labelpad=12)
    ax.set_zlabel('$J(w, b)$', rotation=90, labelpad=20)
    ax.set_title('Cost surface $J(w, b)$, normalised features' + title_units,
                 size=14, pad=18)
    ax.legend(loc='upper left', framealpha=0.85)
    return fig, ax
