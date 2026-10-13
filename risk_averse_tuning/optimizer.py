import warnings

import torch
from torch.optim import Optimizer

from .risk import RAGMMBounds


class GMM(Optimizer):
    """
    Generalized Momentum Method (GMM) optimizer.

    Implements the paper's recursion

    .. math::

        y_k       = x_k + \\gamma \\, d_k, \\qquad d_k = x_k - x_{k-1}

        d_{k+1}   = \\beta \\, d_k - \\alpha \\, \\nabla f(y_k)

        x_{k+1}   = x_k + d_{k+1}

    which equals ``x_{k+1} = (1+beta) x_k - beta x_{k-1} - alpha grad f(y_k)`` with
    ``y_k = (1+gamma) x_k - gamma x_{k-1}``. Special cases: gradient descent
    (``beta = gamma = 0``), heavy ball (``gamma = 0``), Nesterov / AGD
    (``gamma = beta``).

    **The gradient must be evaluated at y_k, not x_k, whenever gamma != 0.**
    Two ways to do that:

    * pass a closure to :meth:`step` (recommended): ``step`` moves the parameters
      to ``y_k``, calls the closure (which must zero the gradients, compute the
      loss, call ``backward()`` and return the loss), restores ``x_k`` and updates;
    * call :meth:`apply_lookahead` before the forward/backward pass, then
      ``step()`` without a closure.

    Calling ``step()`` without either, with ``gamma != 0``, evaluates the
    gradient at ``x_k`` and a ``RuntimeWarning`` is issued.

    Args:
        params: iterable of parameters to optimize
        lr:     step size alpha > 0 (default: 1e-3)
        beta:   momentum coefficient beta >= 0 (default: 0.9)
        gamma:  lookahead coefficient gamma >= 0; 0 disables lookahead (default: 0.0).
                RA-GMM can choose gamma > 1; this is allowed by the paper's Theorem 2.
    """

    def __init__(self, params, lr=1e-3, beta=0.9, gamma=0.0):
        if not lr > 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta: {beta}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        defaults = dict(lr=lr, beta=beta, gamma=gamma)
        super().__init__(params, defaults)
        self._warned_lookahead = False

    @torch.no_grad()
    def apply_lookahead(self):
        """Move the parameters to y_k = x_k + gamma d_k, saving x_k; ``step`` restores x_k."""
        for group in self.param_groups:
            gamma = group['gamma']
            if gamma == 0.0:
                continue
            for p in group['params']:
                state = self.state[p]
                if 'x_k_backup' in state:
                    raise RuntimeError("apply_lookahead() was already called since the last step()")
                buf = state.get('momentum_buffer')
                if buf is None:           # d_0 = 0, so y_0 = x_0
                    continue
                state['x_k_backup'] = p.data.clone()
                p.data.add_(buf, alpha=gamma)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one generalized momentum step.

        With ``closure``: moves to ``y_k``, evaluates the closure there (it must
        zero the gradients, compute the loss, call ``backward()`` and return the
        loss), restores ``x_k`` and applies the update. Returns the closure's loss.

        Without ``closure``: uses the gradients already in ``.grad``. These must
        have been computed at ``y_k`` (see :meth:`apply_lookahead`) when
        ``gamma != 0``.
        """
        loss = None
        if closure is not None:
            self.apply_lookahead()
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            gamma = group['gamma']
            for p in group['params']:
                state = self.state[p]
                looked_ahead = 'x_k_backup' in state
                if looked_ahead:                       # always restore x_k, even without a gradient
                    p.data.copy_(state.pop('x_k_backup'))
                if p.grad is None:
                    continue
                if (closure is None and gamma != 0.0 and not looked_ahead
                        and state.get('momentum_buffer') is not None and not self._warned_lookahead):
                    warnings.warn(
                        "GMM.step() without a closure and without apply_lookahead(): with gamma != 0 "
                        "the gradient was evaluated at x_k instead of y_k = x_k + gamma d_k, so the "
                        "lookahead is ignored. Pass a closure to step() or call apply_lookahead() "
                        "before the forward pass.", RuntimeWarning, stacklevel=2)
                    self._warned_lookahead = True
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                d = state['momentum_buffer']
                d_new = beta * d - lr * p.grad
                p.data.add_(d_new)
                state['momentum_buffer'] = d_new

        return loss


class RAGMM:
    """
    Risk-Averse Generalized Momentum Method (RA-GMM, Can & Gürbüzbalaban 2025).

    Chooses ``lr`` (alpha), ``beta`` and ``gamma`` of a wrapped :class:`GMM` by
    solving the paper's problem (51) by grid search: minimize the asymptotic EV@R
    bound of Theorem 3 over ``(vartheta, psi)`` in ``S_c U S_0`` (Theorem 2)
    subject to ``rho^2 / rho_*^2 <= 1 + rate_slack`` with ``rho_*^2 = 1 - sqrt(mu/L)``.
    The choice depends only on ``(L, mu, d, zeta, rate_slack, phi)`` and is made
    once, when the wrapper is created.

    The theory assumes a strongly convex ``mu``-``L`` smooth objective and
    unbiased gradients with additive noise. ``L`` and ``mu`` must be supplied by
    the caller, with ``L > mu > 0``.

    Args:
        optimizer:   :class:`GMM` instance to control
        L:           gradient-Lipschitz constant of the objective
        mu:          strong-convexity constant of the objective
        zeta:        tail probability, ``P{f(x_k) - f* >= EV@R_{1-zeta}} <= zeta``
                     (default 0.05, a 95% guarantee). Note the paper's experiments
                     pass ``zeta = 0.95``, a very weak guarantee.
        grid_size:   bins per axis of the (vartheta, psi) grid (default 100, as in
                     the authors' reference code)
        rate_slack:  the trade-off parameter ``eps`` in (51b): how much slower than
                     the fastest certified rate ``rho_*^2`` the method may be
                     (default 0.05; the paper uses 0.25 and 0.05). If
                     ``1 + rate_slack >= 1 / rho_*^2`` the constraint is vacuous and
                     the selected step size depends on ``grid_size``.
        phi:         the constant ``phi`` in (0, 1) of Theorem 3 (default 0.99)
        dimension:   dimension ``d`` of the optimization variable; defaults to the
                     number of parameters of ``optimizer`` that require gradients

    Usage::

        gmm = GMM(model.parameters(), lr=1 / (L + mu))
        opt = RAGMM(gmm, L=L, mu=mu)          # tunes (alpha, beta, gamma) here

        def closure():
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            return loss

        stats = opt.step(closure)             # gradient is taken at y_k

    ``stats['evar_bound']`` is the EV@R bound for gradient-noise variance
    ``sigma^2 = 1``; multiply by ``sigma^2`` for other noise levels.
    """

    _DEPRECATED = ('alpha_range', 'psi_range', 'eps')

    def __init__(self, optimizer, L, mu, zeta=0.05, *, grid_size=100, rate_slack=0.05,
                 phi=0.99, dimension=None, **deprecated):
        unknown = set(deprecated) - set(self._DEPRECATED)
        if unknown:
            raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
        if deprecated:
            warnings.warn(f"{sorted(deprecated)} are ignored: RA-GMM now searches (vartheta, psi) over "
                          "the stable sets of the paper's Theorem 2.", DeprecationWarning, stacklevel=2)
        if not isinstance(optimizer, GMM):
            raise TypeError(
                f"optimizer must be a GMM instance, got {type(optimizer).__name__}"
            )

        self.optimizer = optimizer
        self.L = float(L)
        self.mu = float(mu)
        self.zeta = zeta
        self.phi = phi
        self.grid_size = grid_size
        self.rate_slack = rate_slack
        self.step_count = 0

        if dimension is None:
            dimension = sum(p.numel() for g in optimizer.param_groups for p in g['params']
                            if p.requires_grad)
        self.dimension = int(dimension)

        self.bounds = RAGMMBounds(zeta=zeta, phi=phi)
        self.bounds.d = self.dimension
        self.solution = self.bounds.solve(self.L, self.mu, self.dimension,
                                          rate_slack=rate_slack, grid_size=grid_size)
        self._apply_hyperparams(self.solution.alpha, self.solution.beta, self.solution.gamma)
        self._announce_tuned = True

    # ------------------------------------------------------------------ #
    # Convenience proxies                                                  #
    # ------------------------------------------------------------------ #

    @property
    def alpha(self):
        """Current step size alpha (``lr`` of the wrapped GMM)."""
        return self.optimizer.param_groups[0]['lr']

    @property
    def beta(self):
        """Current momentum coefficient beta."""
        return self.optimizer.param_groups[0]['beta']

    @property
    def gamma(self):
        """Current lookahead coefficient gamma."""
        return self.optimizer.param_groups[0]['gamma']

    @property
    def psi(self):
        """The paper's psi of the selected parameters (gamma = psi * beta)."""
        return self.solution.psi

    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)

    def apply_lookahead(self):
        """See :meth:`GMM.apply_lookahead`."""
        self.optimizer.apply_lookahead()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _apply_hyperparams(self, alpha, beta, gamma):
        for group in self.optimizer.param_groups:
            group['lr'] = alpha
            group['beta'] = beta
            group['gamma'] = gamma

    def _snapshot(self):
        """Copy of x_k for every parameter (the pre-lookahead value if lookahead is active)."""
        snap = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    backup = self.optimizer.state[p].get('x_k_backup')
                    snap.append((p, (backup if backup is not None else p.data).clone()))
        return snap

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one GMM step with the risk-averse parameters.

        If a non-finite parameter appears, ``x_k`` is restored for that parameter
        and its momentum buffer is reset (``nan_detected`` is reported).

        Returns a dict of diagnostics: ``alpha, beta, gamma, psi, evar_bound,
        sigma_hat, step, tuned, nan_detected`` (plus ``loss`` when a closure is
        given). ``sigma_hat`` is the standard deviation of the entries of the
        current gradient; it is a diagnostic, not an estimate of the noise level.
        """
        self.step_count += 1
        snapshot = self._snapshot()
        loss = self.optimizer.step(closure)

        grads = [p.grad.detach().reshape(-1) for g in self.optimizer.param_groups
                 for p in g['params'] if p.grad is not None]
        if not grads:
            return {}
        g = torch.cat(grads)
        sigma_hat = float(g.std()) if g.numel() > 1 else 0.0

        nan_detected = False
        for p, x_k in snapshot:
            if not torch.isfinite(p.data).all():
                nan_detected = True
                p.data.copy_(x_k)
                buf = self.optimizer.state[p].get('momentum_buffer')
                if buf is not None:
                    buf.zero_()

        tuned = self._announce_tuned
        self._announce_tuned = False
        stats = {
            'alpha':        self.alpha,
            'beta':         self.beta,
            'gamma':        self.gamma,
            'psi':          self.psi,
            'sigma_hat':    sigma_hat,
            'evar_bound':   self.solution.evar_bound,
            'step':         self.step_count,
            'tuned':        tuned,
            'nan_detected': nan_detected,
        }
        if closure is not None:
            stats['loss'] = loss.detach() if isinstance(loss, torch.Tensor) else loss
        return stats
