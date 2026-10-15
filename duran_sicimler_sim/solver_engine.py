"""Duran Sicimler Teoremi V - Sayisal Cozucu Motoru.
Eylul 2026 Navier-Stokes ic-cekirdek (Denk. 3.2, 4.2-4.14) + elastik sicim gerilimi.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar
from numpy.polynomial.legendre import leggauss as _leggauss

# Faz 2 Gauss-Legendre cizelgesi (denge-korumali genlik integrali icin, M = 256)
_GL_XI, _GL_W = _leggauss(256)

# ---------------- Parametreler ----------------
DEFAULTS = dict(
    F0=1.0, X0=1.0, alpha_eta=0.3, U0=1.0,
    kappa_r=1.0, gamma_rel=0.1, ell0=1.0,
    X_star=1.0, c_s=5.0, ell_light0=1.0, beta0=0.5,
    U_even=0.5, bmax=1e16,
)
H_VALUES = [0.002, 0.005, 0.008, 0.01]
MU_VALUES = [0.02, 0.05, 0.1, 0.2]

# ---------------- Analitik profiller ----------------
def L_eta(eta, h):
    return 1.0 - 2.0 * h * eta ** 2

def d_eta(eta):
    return 1.0 - eta ** 2

def F_prof(X, eta, h, F0=1.0, X0=1.0, alpha=0.3):
    X = np.asarray(X, dtype=float)
    return F0 * (1.0 - alpha * eta ** 2) / (1.0 + X / X0) ** (1.0 + h)

def E_prof(X, eta, h, F0=1.0, X0=1.0, alpha=0.3):
    X = np.asarray(X, dtype=float)
    return np.sqrt(2.0 * np.maximum(X, 0.0)) * F_prof(X, eta, h, F0, X0, alpha)

def U_prof(X, eta, h, U0=1.0, X0=1.0):
    X = np.asarray(X, dtype=float)
    return U0 * eta / (1.0 + X / X0) ** (1.0 + h)

def _G_int(X, h, X0=1.0):
    """int_0^X (1+x/X0)^{-(1+h)} dx = X0/h * (1-(1+X/X0)^{-h}). h->0 limiti log.

    Kucuk-X sadeleşmesine karsi expm1/log1p formu kullanilir (1e-15 mertebesinde tam).
    """
    X = np.asarray(X, dtype=float)
    if abs(h) < 1e-12:
        return X0 * np.log1p(X / X0)
    return X0 / h * (-np.expm1(-h * np.log1p(X / X0)))

def A_X_U(X, eta, h, U0=1.0, X0=1.0):
    X = np.asarray(X, dtype=float)
    eta = np.asarray(eta, dtype=float)
    Xb, eb = np.broadcast_arrays(X, eta)
    G = _G_int(Xb, h, X0)
    out = np.zeros_like(Xb, dtype=float)
    nz = Xb > 1e-300
    out[nz] = U0 * eb[nz] * G[nz] / Xb[nz]
    out[~nz] = U0 * eb[~nz]
    return out

def dA_deta(X, eta, h, U0=1.0, X0=1.0):
    X = np.asarray(X, dtype=float)
    Xb = np.broadcast_arrays(X, np.asarray(eta, dtype=float))[0]
    G = _G_int(Xb, h, X0)
    out = np.zeros_like(Xb, dtype=float)
    nz = Xb > 1e-300
    out[nz] = U0 * G[nz] / Xb[nz]
    out[~nz] = U0
    return out

def V0_prof(X, eta, h, U0=1.0, X0=1.0):
    Xb, eb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(eta, dtype=float))
    L = L_eta(eb, h)
    d = d_eta(eb)
    U = U_prof(Xb, eb, h, U0, X0)
    A = A_X_U(Xb, eb, h, U0, X0)
    Ap = dA_deta(Xb, eb, h, U0, X0)
    D = 0.5 - h
    return Xb / L * (2.0 * eb * U - 2.0 * D * eb * A - d * Ap)

def Pi_X0(X, eta, h, F0=1.0, X0=1.0, alpha=0.3):
    F = F_prof(X, eta, h, F0, X0, alpha)
    return F ** 2

def Pi0_int(X, eta, h, F0=1.0, X0=1.0, alpha=0.3):
    X = np.asarray(X, dtype=float)
    return -(F0 * (1.0 - alpha * eta ** 2)) ** 2 * X0 / (1.0 + 2.0 * h) * (1.0 + X / X0) ** (-(1.0 + 2.0 * h))

def a_prof(X, eta, h, X0=1.0):
    X = np.asarray(X, dtype=float)
    return 2.0 * X * (1.0 + h) / (X0 + X)

def l_prof(X, eta, h, X0=1.0):
    return 1.0 - a_prof(X, eta, h, X0) / 2.0

def Sigma_rtheta(X, eta, q, h, F0=1.0, X0=1.0, alpha=0.3):
    F = F_prof(X, eta, h, F0, X0, alpha)
    a = a_prof(X, eta, h, X0)
    return -(q ** (-1.0 - h)) * F * a

def u_theta_0(X, s, h, F0=1.0, X0=1.0, alpha=0.3):
    E = E_prof(X, 0.0, h, F0, X0, alpha)
    return np.exp((0.5 + h) * s) * E

def u_r_0(X, eta, tau, h, U0=1.0, X0=1.0):
    X = np.asarray(X, dtype=float)
    V0 = V0_prof(X, eta, h, U0, X0)
    q = tau
    return V0 / np.sqrt(2.0 * np.maximum(X, 1e-300) * q)

# ---------------- Gorev 1: sicim ----------------
def ell_explicit_analytic(s, X, h, p=None):
    """Acik cozumun analitik ifadesi (dogrulama icin). Vektorize."""
    if p is None:
        p = DEFAULTS
    F = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    a = np.asarray(np.abs(a_prof(X, 0.0, h, p['X0'])), dtype=float)
    kr, gr, l0 = p['kappa_r'], p['gamma_rel'], p['ell0']
    s = np.asarray(s, dtype=float)
    if abs(h) < 1e-14:
        grow = kr * F * a * s
    else:
        grow = kr * F * a * (np.exp(h * s) - 1.0) / h
    damp = gr * (1.0 - np.exp(-s))
    return l0 * np.exp(grow - damp)

def ell_ode_rhs(s, y, X, h, p):
    F = float(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']))
    a = float(a_prof(X, 0.0, h, p['X0']))
    kr, gr = p['kappa_r'], p['gamma_rel']
    return [kr * F * abs(a) * np.exp(h * s) * y[0] - gr * np.exp(-s) * y[0]]

def solve_ell_explicit(X, h, s_max, p=None, npts=2000):
    if p is None:
        p = DEFAULTS
    sol = solve_ivp(ell_ode_rhs, (0.0, s_max), [p['ell0']],
                    args=(X, h, p), method='Radau', rtol=1e-12, atol=1e-14,
                    dense_output=True)
    s = np.linspace(0, s_max, npts)
    return s, sol.sol(s)[0], sol

def Pi_eff_X(s, ell, X, h, mu_s, p=None):
    if p is None:
        p = DEFAULTS
    F = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    return F ** 2 - mu_s * np.exp(-(1.0 + 2.0 * h) * np.asarray(s)) * np.asarray(ell) ** 2 / (2.0 * np.asarray(X))

def E_eff_of_s(s, ell, X, h, mu_s, p=None):
    Pi = Pi_eff_X(s, ell, X, h, mu_s, p)
    return np.sqrt(np.maximum(0.0, 2.0 * X * Pi))

def u_eff_of_s(s, ell, X, h, mu_s, p=None):
    return np.exp((0.5 + h) * np.asarray(s)) * E_eff_of_s(s, ell, X, h, mu_s, p)

def _f_arrest(s, X, h, mu_s, p):
    ell = ell_explicit_analytic(s, X, h, p)
    return float(Pi_eff_X(s, ell, X, h, mu_s, p))

def find_s_kritik(X, h, mu_s, p=None, s_hi=200.0):
    """brentq ile Pi_eff=0 koku. (s_lo=0, s_hi) araliginda."""
    if p is None:
        p = DEFAULTS
    f0 = _f_arrest(0.0, X, h, mu_s, p)
    fhi = _f_arrest(s_hi, X, h, mu_s, p)
    if f0 <= 0:
        return 0.0
    # s_hi'yi buyut
    while fhi > 0 and s_hi < 2000:
        s_hi *= 2.0
        fhi = _f_arrest(s_hi, X, h, mu_s, p)
    if fhi > 0:
        raise RuntimeError('Arrest bulunamadi')
    s_c = brentq(lambda s: _f_arrest(s, X, h, mu_s, p), 0.0, s_hi, xtol=1e-14, rtol=1e-14, maxiter=1000)
    return float(s_c)

def compute_u_max(X, h, mu_s, p=None, s_crit=None):
    if p is None:
        p = DEFAULTS
    if s_crit is None:
        s_crit = find_s_kritik(X, h, mu_s, p)
    if s_crit <= 0:
        return 0.0, 0.0
    def neg_u(s):
        ell = ell_explicit_analytic(s, X, h, p)
        return -float(u_eff_of_s(s, ell, X, h, mu_s, p))
    res = minimize_scalar(neg_u, bounds=(0.0, s_crit), method='bounded',
                          options={'xatol': 1e-14})
    return float(-res.fun), float(res.x)

# ---------------- Teorem 2: X -> 0 Kartezyen eksen duzenlemesi ----------------
def l_shear_reg(X, h, X0=1.0):
    """Duzenli makaslama cekirdegi l(X,0) = (X0 - h X)/(X0 + X).

    Not: l_prof(X,0,h) = 1 - a/2 ile cebirsel olarak ozdestir
    (1 - X(1+h)/(X0+X) = (X0 - hX)/(X0+X)); ayri implementasyon capraz dogrulama icindir.
    """
    X = np.asarray(X, dtype=float)
    return (X0 - h * X) / (X0 + X)

def G_drive(X, h, p=None):
    """Cekirdek gerilme hizi G(X) = F(X,0) sqrt(a(X,0)^2 + beta0^2 l(X,0)^2).

    X = 0'da a = 0, l = 1, F = F0 oldugundan G(0) = F0*beta0: sonlu, tekillik yok.
    """
    if p is None:
        p = DEFAULTS
    F = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    a = np.asarray(a_prof(X, 0.0, h, p['X0']), dtype=float)
    rl = np.asarray(l_shear_reg(X, h, p['X0']), dtype=float)
    b0 = p['beta0']
    return F * np.sqrt(a ** 2 + b0 ** 2 * rl ** 2)

def ltilde_explicit_analytic(s, X, h, p=None):
    """Duzenli sicim degiskeni: d l_tilde/ds = kappa G(X) e^{hs} l_tilde - gamma e^{-s} l_tilde."""
    if p is None:
        p = DEFAULTS
    G = np.asarray(G_drive(X, h, p), dtype=float)
    kr, gr, l0 = p['kappa_r'], p['gamma_rel'], p['ell0']
    s = np.asarray(s, dtype=float)
    if abs(h) < 1e-14:
        grow = kr * G * s
    else:
        grow = kr * G * (np.exp(h * s) - 1.0) / h
    damp = gr * (1.0 - np.exp(-s))
    return l0 * np.exp(grow - damp)

def ell_theta_reg(X, s, l_tilde, p=None):
    """ell_theta(X, s) = sqrt(X / X_*) l_tilde(X, s). Eksen davranisi E ~ sqrt(2X) F ile uyumludur."""
    if p is None:
        p = DEFAULTS
    return np.sqrt(np.maximum(np.asarray(X, dtype=float), 0.0) / p['X_star']) * np.asarray(l_tilde, dtype=float)

def Pi_eff_X_reg(s, l_tilde, X, h, mu_s, p=None):
    """Duzenlenmis etkin basinc gradyani: F^2 - (mu_s/(2 X_*)) e^{-(1+2h)s} l_tilde^2.

    ell_theta = sqrt(X/X_*) l_tilde yazilinca 2X carpanlari sadeleşir; hicbir 1/X tekilligi kalmaz.
    """
    if p is None:
        p = DEFAULTS
    F = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    return F ** 2 - (mu_s / (2.0 * p['X_star'])) * np.exp(-(1.0 + 2.0 * h) * np.asarray(s)) * np.asarray(l_tilde, dtype=float) ** 2

def Pi_center_reg(s, h, mu_s, X_max=50.0, p=None):
    """Kumulatif merkez basinci: Pi_eff_reg(0,s) = -int_0^{X_max} Pi_eff_X_reg dx (quad)."""
    if p is None:
        p = DEFAULTS
    def integrand(x):
        lt = float(np.asarray(ltilde_explicit_analytic(s, x, h, p)))
        return float(np.asarray(Pi_eff_X_reg(s, lt, x, h, mu_s, p)))
    val, err = quad(integrand, 0.0, X_max, limit=200)
    return float(-val), float(err)

# Tam geri-tepkili (self-consistent) ODE
def solve_selfconsistent(X, h, mu_s, s_max, p=None, npts=2000):
    if p is None:
        p = DEFAULTS
    F0v = float(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']))
    av = abs(float(a_prof(X, 0.0, h, p['X0'])))
    kr = p['kappa_r']

    def rhs(s, y):
        ell = y[0]
        disc = F0v ** 2 - mu_s * np.exp(-(1.0 + 2.0 * h) * s) * ell ** 2 / (2.0 * X)
        Feff = np.sqrt(max(0.0, disc))
        return [kr * Feff * av * np.exp(h * s) * ell]

    def event_arrest(s, y):
        ell = y[0]
        disc = F0v ** 2 - mu_s * np.exp(-(1.0 + 2.0 * h) * s) * ell ** 2 / (2.0 * X)
        return disc
    event_arrest.terminal = True
    event_arrest.direction = -1

    sol = solve_ivp(rhs, (0.0, s_max), [p['ell0']], method='Radau',
                    rtol=1e-10, atol=1e-13, events=event_arrest, dense_output=True)
    s_end = float(sol.t_events[0][0]) if sol.t_events[0].size > 0 else float(sol.t[-1])
    s = np.linspace(0, s_end, npts)
    ell = sol.sol(s)[0]
    return s, ell, sol, s_end

# ---------------- Gorev 2: akustik ufuk ----------------
def q_horizon(X, eta, h, p=None):
    """|u_r|=c_s kosulundan q_ufuk. (0,1] araligina kirpilir."""
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    V0 = V0_prof(X, eta, h, p['U0'], p['X0'])
    cs = p['c_s']
    qh = (V0 ** 2) / (2.0 * np.maximum(X, 1e-300) * cs ** 2)
    return np.minimum(np.maximum(qh, 0.0), 1.0)

def omega_prime_ratio(X, eta, tau, h, p=None):
    if p is None:
        p = DEFAULTS
    ur = np.abs(u_r_0(X, eta, tau, h, p['U0'], p['X0']))
    return 1.0 - ur / p['c_s']

# ---------------- Gorev 2.3: osilator ----------------
def solve_string_modes(n_list, h, s_max=6.0, p=None, npts=2000):
    """Q''+Q'+w0^2 e^{-2(1+h)s} Q=0, Q(0)=1,Q'(0)=0. w0=n*cs/l0."""
    if p is None:
        p = DEFAULTS
    cs, l0 = p['c_s'], p['ell_light0']
    s_eval = np.linspace(0, s_max, npts)
    out = {}
    for n in n_list:
        w0 = n * cs / l0
        def rhs(s, y):
            return [y[1], -y[1] - w0 ** 2 * np.exp(-2.0 * (1.0 + h) * s) * y[0]]
        sol = solve_ivp(rhs, (0.0, s_max), [1.0, 0.0], method='DOP853',
                        rtol=1e-13, atol=1e-15, t_eval=s_eval)
        Q = sol.y[0]
        Qd = sol.y[1]
        tau = np.exp(-s_eval)
        omega_n = w0 * tau ** h
        Llight = l0 * tau ** (-h)
        I_n = omega_n * Llight  # analitik olarak n*cs sabit
        # Enerji: E=1/2[(dq/dt)^2 + w^2 q^2], dq/dt = -e^s Q'
        dqdt = -np.exp(s_eval) * Qd
        E = 0.5 * (dqdt ** 2 + omega_n ** 2 * Q ** 2)
        # P_yakit = -dE/dtau (tau azaldikca); tau grid ters sirali oldugu icin gradient
        P = -np.gradient(E, tau)
        out[n] = dict(s=s_eval, tau=tau, Q=Q, omega_n=omega_n,
                      Llight=Llight, I_n=I_n, E=E, P=P, w0=w0)
    return out

# ================= FAZ 2: tam cekirdek cozucu =================
# ---- Gorev 3: denge-cikarmali (equilibrium-subtracted) global model ----
def delta_global(s, X, h, p=None):
    """Denge-korumali genlik sapmasi delta = l_tilde - l0.

    Dinamik: d l/ds = kappa G e^{hs} l - gamma e^{-s} (l - l0)  =>  G = 0'da l = l0 korunur,
    boylece (l^2 - l0^2) ~ O(G) ~ O(X^{-1-h}) soner ve [0, inf) integrali yakinsar.
    Cozum (integrating factor, vektorize Gauss-Legendre M = 256):
    delta(s) = e^{A(s)+B(s)} int_0^s e^{-A-B} kappa G e^{hs'} l0 ds'.
    """
    if p is None:
        p = DEFAULTS
    s = float(s)
    G0 = np.asarray(G_drive(X, h, p), dtype=float)
    if s == 0.0:
        return np.zeros_like(G0)
    kr, gr, l0 = p['kappa_r'], p['gamma_rel'], p['ell0']
    sp = 0.5 * s * (_GL_XI + 1.0)
    w = 0.5 * s * _GL_W
    ehs = np.exp(h * sp)
    ems = np.exp(-sp)
    K = sp if abs(h) < 1e-14 else (ehs - 1.0) / h
    G = np.atleast_1d(G0)
    A = kr * G[:, None] * K[None, :]
    B = gr * (ems - 1.0)
    F = np.exp(-A - B[None, :]) * (kr * G[:, None] * ehs[None, :] * l0)
    integ = F @ w
    A_s = kr * G * (s if abs(h) < 1e-14 else (np.exp(h * s) - 1.0) / h)
    B_s = gr * (np.exp(-s) - 1.0)
    out = np.exp(A_s + B_s) * integ
    return np.reshape(out, np.shape(G0))

def ltilde_global(s, X, h, p=None):
    """Global model genligi: l_tilde(X, s) = l0 + delta_global(s, X)."""
    if p is None:
        p = DEFAULTS
    return p['ell0'] + delta_global(s, X, h, p)

def ltsq_minus_l02(s, X, h, p=None):
    """(l_tilde^2 - l0^2) = 2 l0 delta + delta^2: delta'dan dogrudan, sadeleşmesiz."""
    if p is None:
        p = DEFAULTS
    d = np.asarray(delta_global(s, X, h, p), dtype=float)
    return 2.0 * p['ell0'] * d + d * d

def Pi_eff_X_global(X, s, h, mu_s, p=None):
    """Global etkin basinc gradyani: F^2 - (mu_s/(2 X_*)) e^{-(1+2h)s} (l_tilde^2 - l0^2)."""
    if p is None:
        p = DEFAULTS
    F = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    dm = np.asarray(ltsq_minus_l02(s, X, h, p), dtype=float)
    return F ** 2 - (mu_s / (2.0 * p['X_star'])) * np.exp(-(1.0 + 2.0 * h) * float(s)) * dm

def _rho_factor(s, h, p):
    """Relaksasyon baglasim carpanı: rho(s) = e^{B(s)} int_0^s e^{-B(s')+hs'} ds' (GL-256)."""
    s = float(s)
    sp = 0.5 * s * (_GL_XI + 1.0)
    wq = 0.5 * s * _GL_W
    Bm = p['gamma_rel'] * (np.exp(-sp) - 1.0)
    Bs = p['gamma_rel'] * (np.exp(-s) - 1.0)
    return float(np.exp(Bs) * np.sum(np.exp(-Bm + h * sp) * wq))

def Pi_center_global(s, h, mu_s, p=None, eps=1e-13, Xc=50.0):
    """Kumulatif kuresel merkez basinci: -int_0^inf Pi_eff_X_global dx. Kesme yok.

    Yavas cebirsel kuyruk (X^{-1-h}) yuzunden integral ikiye bolunur:
    [0, Xc] cizgisel araligi + kuyruk donusumu x = Xc u^{-1/h}, u in (0, 1].
    Donusum altinda kuyruk integrandi duzgunlesir (u^0 mertebesi), tekillik kalmaz.
    """
    if p is None:
        p = DEFAULTS
    s = float(s)

    def f(x):
        return float(np.asarray(Pi_eff_X_global(x, s, h, mu_s, p)))

    v1, e1 = quad(f, 0.0, Xc, epsabs=eps, epsrel=eps, limit=500)

    # Kuyruk sabiti g(0): (l^2-l0^2) ~ A_t X^{-1-h}  =>  g(u) -> g0 (duzgun limit).
    kr, l0 = p['kappa_r'], p['ell0']
    rho = _rho_factor(s, h, p)
    S_G = float(np.sqrt(4.0 * (1.0 + h) ** 2 + p['beta0'] ** 2 * h ** 2))
    A_t = 2.0 * l0 ** 2 * kr * rho * p['F0'] * S_G * p['X0'] ** (1.0 + h)
    C0e = (mu_s / (2.0 * p['X_star'])) * np.exp(-(1.0 + 2.0 * h) * s)
    g0 = -C0e * A_t * Xc ** (-h) / h

    U_MIN = 0.05  # alti analitik sabit (duzeltmeler ~1e-300 mertebesinde)

    def g(u):
        x = Xc * u ** (-1.0 / h)
        return f(x) * (Xc / h) * u ** (-1.0 / h - 1.0)

    v2, e2 = quad(g, U_MIN, 1.0, epsabs=eps, epsrel=eps, limit=500)
    return float(-(v1 + v2 + U_MIN * g0)), float(e1 + e2)

# ---- Gorev 4: navier-stokes.pdf s. 26-27 (Onerme 4.2, Denk. 4.8-4.14) ----
# Not: Faz 1'deki U_prof/V0 (U_even = 0) aynen korunur (Gorev 2 tablolari degismez).
# Asagidaki _full ailesi s. 8 / Proposition B.2'deki cift parcayi (U_even = 0.5) icerir.
def U_full(X, eta, h, U_even=0.5, U0=1.0, X0=1.0, alpha=0.3):
    Xb, eb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(eta, dtype=float))
    return (U_even * (1.0 - alpha * eb ** 2) + U0 * eb) / (1.0 + Xb / X0) ** (1.0 + h)

def A_X_full(X, eta, h, U_even=0.5, U0=1.0, X0=1.0, alpha=0.3):
    Xb, eb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(eta, dtype=float))
    G = _G_int(Xb, h, X0)
    br = U_even * (1.0 - alpha * eb ** 2) + U0 * eb
    out = np.zeros_like(Xb, dtype=float)
    nz = Xb > 1e-300
    out[nz] = br[nz] * G[nz] / Xb[nz]
    out[~nz] = br[~nz]
    return out

def dA_full_deta(X, eta, h, U_even=0.5, U0=1.0, X0=1.0, alpha=0.3):
    Xb, eb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(eta, dtype=float))
    G = _G_int(Xb, h, X0)
    brp = U_even * (-2.0 * alpha * eb) + U0
    out = np.zeros_like(Xb, dtype=float)
    nz = Xb > 1e-300
    out[nz] = brp[nz] * G[nz] / Xb[nz]
    out[~nz] = brp[~nz]
    return out

def W_full(X, eta, h, p=None, U_even=0.5):
    """Tasinim katsayisi (Denk. 4.8): W = 1 - 2D eta A_X(U) - d(eta) dA/deta."""
    if p is None:
        p = DEFAULTS
    Xb, eb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(eta, dtype=float))
    D = 0.5 - h
    A = A_X_full(Xb, eb, h, U_even, p['U0'], p['X0'], p['alpha_eta'])
    Ap = dA_full_deta(Xb, eb, h, U_even, p['U0'], p['X0'], p['alpha_eta'])
    return 1.0 - 2.0 * D * eb * A - d_eta(eb) * Ap

def Hc_full(X, eta, h, p=None, U_even=0.5):
    """Tasinim katsayisi (Denk. 4.8): H_c = D eta + d(eta) U."""
    if p is None:
        p = DEFAULTS
    Xb, eb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(eta, dtype=float))
    return (0.5 - h) * eb + d_eta(eb) * U_full(Xb, eb, h, U_even, p['U0'], p['X0'], p['alpha_eta'])

def F_eta_over_F(eta, alpha=0.3):
    eb = np.asarray(eta, dtype=float)
    return -2.0 * alpha * eb / (1.0 - alpha * eb ** 2)

def Sq_full(X, eta, h, p=None, U_even=0.5):
    """Acisal momentum kaynagi (Denk. 4.9)."""
    if p is None:
        p = DEFAULTS
    Xb, eb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(eta, dtype=float))
    W = W_full(Xb, eb, h, p, U_even)
    rl = l_prof(Xb, eb, h, p['X0'])
    U = U_full(Xb, eb, h, U_even, p['U0'], p['X0'], p['alpha_eta'])
    Hc = Hc_full(Xb, eb, h, p, U_even)
    return -W * rl - h * (1.0 - 2.0 * eb * U) - Hc * F_eta_over_F(eb, p['alpha_eta'])

def Qs_full(X, eta, h, p=None, U_even=0.5):
    """Radyal inviskid integral (Denk. 4.10): (1/F) int_0^1 sig F(sig X) S_q(sig X) dsig.

    X = 0 dali limit formuludur: Q_s(0, eta) = S_q(0, eta)/2 (tam).
    """
    if p is None:
        p = DEFAULTS
    scalar = np.ndim(X) == 0 and np.ndim(eta) == 0
    Xb, eb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(eta, dtype=float))
    out = np.zeros_like(Xb, dtype=float)
    for idx in np.ndindex(Xb.shape):
        xv, ev = float(Xb[idx]), float(eb[idx])
        if xv == 0.0:
            out[idx] = float(np.asarray(Sq_full(0.0, ev, h, p, U_even))) / 2.0
        else:
            Fxv = float(np.asarray(F_prof(xv, ev, h, p['F0'], p['X0'], p['alpha_eta'])))
            def ig(sig, _xv=xv, _ev=ev):
                xs = sig * _xv
                return sig * float(np.asarray(F_prof(xs, _ev, h, p['F0'], p['X0'], p['alpha_eta']))) * float(np.asarray(Sq_full(xs, _ev, h, p, U_even)))
            v, _ = quad(ig, 0.0, 1.0, limit=200)
            out[idx] = v / Fxv
    return out[()] if scalar else out

def T0_full(X, eta, h, p=None, U_even=0.5):
    """Tegetsel stres kalintisi (Denk. 4.11): T0 = F (X Q_s/L - a)."""
    if p is None:
        p = DEFAULTS
    F = F_prof(X, eta, h, p['F0'], p['X0'], p['alpha_eta'])
    Qs = Qs_full(X, eta, h, p, U_even)
    L = L_eta(np.asarray(eta, dtype=float), h)
    a = a_prof(X, eta, h, p['X0'])
    return np.asarray(F) * (np.asarray(X) * np.asarray(Qs) / L - np.asarray(a))

def T0_eff_global(X, s, h, mu_s, p=None, U_even=0.5):
    """Etkin kalinti: ayni kinematik parantez, genlik F -> F_eff = sqrt(max(0, Pi_global))."""
    if p is None:
        p = DEFAULTS
    Qs = np.asarray(Qs_full(X, 0.0, h, p, U_even), dtype=float)
    L = L_eta(0.0, h)
    a = np.asarray(a_prof(X, 0.0, h, p['X0']), dtype=float)
    Pi = np.asarray(Pi_eff_X_global(X, s, h, mu_s, p), dtype=float)
    Fe = np.sqrt(np.maximum(Pi, 0.0))
    return Fe * (np.asarray(X, dtype=float) * Qs / L - a)

def T0_eff_global(X, s, h, mu_s, p=None, U_even=0.5):
    """Etkin kalinti: ayni kinematik parantez, genlik F -> F_eff = sqrt(max(0, Pi_global))."""
    if p is None:
        p = DEFAULTS
    Qs = np.asarray(Qs_full(X, 0.0, h, p, U_even), dtype=float)
    L = L_eta(0.0, h)
    a = np.asarray(a_prof(X, 0.0, h, p['X0']), dtype=float)
    Pi = np.asarray(Pi_eff_X_global(X, s, h, mu_s, p), dtype=float)
    Fe = np.sqrt(np.maximum(Pi, 0.0))
    return Fe * (np.asarray(X, dtype=float) * Qs / L - a)

# ---- Gorev T10 (istege bagli): rasyonel egrilik carpani ----
def K_curv(X, p=None):
    """Rasyonel egrilik carpani: K(X) = (X0+X_*)/(2 X_* (X0+X)).

    K(0) = (X0+X_*)/(2 X_* X0) (varsayilanlarda 1.0, sonlu), K(X_*) = 1/(2X_*) (tam),
    X -> inf: K ~ (X0+X_*)/(2 X_* X) ~ 1/X. PDE basincta sabit 1/(2X_*) yerine gecer;
    ize ek X^{-1} baskisi vererek kuyruk ussunu ~1 artirir.
    """
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    return (p['X0'] + p['X_star']) / (2.0 * p['X_star'] * (p['X0'] + X))

# ---- Gorev 5: (X, s) tam PDE + FENE-P (Method of Lines, logaritmik X izgarasi, Radau) ----
def pde_grid(Nx=80, Xmax=50.0, xmin=1e-6):
    """X = 0 + logaritmik kuyruk. X = 0'da X d_X l = 0 dogal kosul."""
    return np.concatenate(([0.0], np.logspace(np.log10(xmin), np.log10(Xmax), Nx - 1)))

def _upwind_DX(X, l, W):
    """W isaretine gore upwind X turevi. X = 0'da turev terimi zaten X carpani ile sifirlanir."""
    N = X.size
    D = np.zeros_like(l)
    db = (l[1:] - l[:-1]) / (X[1:] - X[:-1])
    D[1] = db[0] if W[1] >= 0 else (l[2] - l[1]) / (X[2] - X[1])
    for i in range(2, N - 1):
        D[i] = db[i - 1] if W[i] >= 0 else (l[i + 1] - l[i]) / (X[i + 1] - X[i])
    D[N - 1] = db[N - 2]
    return D

def make_l_rhs(X, h, mu_s, p, bmax, transport, curv=False):
    """Paylasilan sag-taraf: FENE-P + denge-korumali relaksasyon + (istege bagli) tasinim.

    curv=True: sabit mu_s/(2X_*) yerine rasyonel egrilik carpani mu_s*K_curv(X) (Test T10).
    """
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    if curv:
        Carr = mu_s * np.asarray(K_curv(X, p), dtype=float)
    else:
        Carr = np.full(X.shape, mu_s / (2.0 * p['X_star']))
    l0 = p['ell0']
    F2 = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float) ** 2
    a = np.asarray(a_prof(X, 0.0, h, p['X0']), dtype=float)
    ls = np.asarray(l_shear_reg(X, h, p['X0']), dtype=float)
    W = np.asarray(W_full(X, 0.0, h, p, p['U_even']), dtype=float)
    kr, gr, b0 = p['kappa_r'], p['gamma_rel'], p['beta0']

    def rhs(s, y):
        l = np.asarray(y, dtype=float)
        den = np.maximum(1.0 - l * l / bmax, 1e-12)
        Pi = F2 - Carr * np.exp(-(1.0 + 2.0 * h) * s) * (l * l - l0 * l0) / den
        Fe = np.sqrt(np.maximum(Pi, 0.0))
        Ge = Fe * np.sqrt(a * a + b0 * b0 * ls * ls)
        R = kr * Ge * np.exp(h * s) * l * (1.0 - l * l / bmax) - gr * np.exp(-s) * (l - l0)
        if transport:
            return R - W * X * _upwind_DX(X, l, W)
        return R
    return rhs

def solve_l_field(X, h, mu_s, s_max=40.0, p=None, bmax=None, npts=121, transport=True, curv=False):
    """Tasinimli (PDE) veya tasinimsiz (ODE) l(X, s) alani. Donus: dict(s, L, Pi, Uth)."""
    if p is None:
        p = DEFAULTS
    if bmax is None:
        bmax = p['bmax']
    X = np.asarray(X, dtype=float)
    l0 = p['ell0']
    rhs = make_l_rhs(X, h, mu_s, p, bmax, transport, curv)
    s_eval = np.linspace(0.0, s_max, npts)
    sol = solve_ivp(rhs, (0.0, s_max), np.full(X.size, l0), method='Radau',
                    rtol=1e-8, atol=1e-10, t_eval=s_eval)
    L = sol.y
    if curv:
        Carr = mu_s * np.asarray(K_curv(X, p), dtype=float)
    else:
        Carr = np.full(X.shape, mu_s / (2.0 * p['X_star']))
    F2 = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float) ** 2
    S, Xg = np.meshgrid(s_eval, X)
    den = np.maximum(1.0 - L * L / bmax, 1e-12)
    Pi = F2[:, None] - Carr[:, None] * np.exp(-(1.0 + 2.0 * h) * S) * (L * L - l0 * l0) / den
    Uth = np.exp((0.5 + h) * S) * np.sqrt(np.maximum(0.0, 2.0 * Xg * Pi))
    return dict(s=s_eval, X=X, L=L, Pi=Pi, Uth=Uth, sol=sol)

def tail_F_analytic(Xmax, h, p):
    """F-parcasi kuyrugu (tam analitik): int_{Xmax}^inf F^2 dX."""
    return (p['F0'] ** 2) * p['X0'] * (1.0 + Xmax / p['X0']) ** (-(1.0 + 2.0 * h)) / (1.0 + 2.0 * h)

def tail_S_rho(s, h, mu_s, Xmax, p):
    """Sicim-parcasi kuyrugu (analitik genlik): -C0e A_t Xmax^{-h}/h.

    Uzak alanda Geff ~ G, FENE ~ 1 gecerli oldugundan tasinimsiz modelin kuyrugunu verir.
    """
    rho = _rho_factor(s, h, p)
    S_G = float(np.sqrt(4.0 * (1.0 + h) ** 2 + p['beta0'] ** 2 * h ** 2))
    A_t = 2.0 * p['ell0'] ** 2 * p['kappa_r'] * rho * p['F0'] * S_G * p['X0'] ** (1.0 + h)
    C0e = (mu_s / (2.0 * p['X_star'])) * np.exp(-(1.0 + 2.0 * h) * float(s))
    return float(-C0e * A_t * Xmax ** (-h) / h)

def wake_exponent(X, S, npts=8):
    """Iz kuyruk ussu: son npts noktada ln|S| ~ lnA - q lnX dogrusal uydurma. q <= 1 iraksar."""
    Xe = np.asarray(X[-npts:], dtype=float)
    Se = np.asarray(S[-npts:], dtype=float)
    if np.any(~np.isfinite(Se)) or np.any(Se >= 0) or np.max(np.abs(Se)) < 1e-25:
        return float('nan'), float('nan')
    Ye = np.log(-Se)
    Le = np.log(Xe)
    M = np.column_stack([np.ones_like(Le), -Le])
    coef, *_ = np.linalg.lstsq(M, Ye, rcond=None)
    return float(np.exp(coef[0])), float(coef[1])

def field_metrics(X, tg, L, h, mu_s, bmax, p, curv=False):
    """Tablo 8 metrikleri: Pi(X_*,40), Pi(0,40), I_50, I_inf (ODE), u_max ve konumu, FENE siniri, iz ussu."""
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    tg = np.asarray(tg, dtype=float)
    l0 = p['ell0']
    if curv:
        Carr = mu_s * np.asarray(K_curv(X, p), dtype=float)
    else:
        Carr = np.full(X.shape, mu_s / (2.0 * p['X_star']))
    F2 = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float) ** 2
    S, Xg = np.meshgrid(tg, X)
    den = np.maximum(1.0 - L * L / bmax, 1e-12)
    Pi = F2[:, None] - Carr[:, None] * np.exp(-(1.0 + 2.0 * h) * S) * (L * L - l0 * l0) / den
    Uth = np.exp((0.5 + h) * S) * np.sqrt(np.maximum(0.0, 2.0 * Xg * Pi))
    i_star = int(np.argmin(np.abs(X - p['X_star'])))
    I50 = np.trapezoid(Pi, X, axis=0)
    Xmax = float(X[-1])
    tF = tail_F_analytic(Xmax, h, p)
    tS = tail_S_rho(float(tg[-1]), h, mu_s, Xmax, p)
    A_s, q = wake_exponent(X, Pi[:, -1] - F2)
    iu = np.unravel_index(int(np.argmax(Uth)), Uth.shape)
    return dict(Pi_star_40=float(Pi[i_star, -1]), Pi_axis_40=float(Pi[0, -1]),
                I50_40=float(I50[-1]), tail_F_40=float(tF), tail_S_40=float(tS),
                Iinf_40=float(I50[-1] + tF + tS),
                u_max=float(Uth[iu]), s_umax=float(tg[iu[1]]), X_umax=float(X[iu[0]]),
                fene_max=float(np.max(L * L) / bmax), wake_A=A_s, wake_q=q,
                Pi=Pi, Uth=Uth, I50=I50)

def compute_table8(mu_list=(0.05, 0.10, 0.20), h=0.005, s_max=40.0, p=None, Nx=80):
    """Tablo 8: her mu icin tasinimsiz ODE ve tam PDE karsilastirmasi (cozum alanlari dahil)."""
    if p is None:
        p = DEFAULTS
    X = pde_grid(Nx=Nx)
    out = {}
    for mu in mu_list:
        row = {}
        for tag, tr in (('ODE', False), ('PDE', True)):
            fld = solve_l_field(X, h, mu, s_max, p, p['bmax'], 121, transport=tr)
            met = field_metrics(X, fld['s'], fld['L'], h, mu, p['bmax'], p)
            met['success'] = bool(fld['sol'].success)
            met['L'] = fld['L']
            met['s'] = fld['s']
            row[tag] = met
        out[float(mu)] = row
    out['X'] = X
    return out

# ---------------- Dogrulama testleri ----------------
def run_verification():
    p = DEFAULTS
    rep = {}
    # T1: Radau vs analitik (acik)
    s, num, _ = solve_ell_explicit(p['X_star'], 0.005, 12.0, p, npts=500)
    ana = ell_explicit_analytic(s, p['X_star'], 0.005, p)
    rep['T1_radau_vs_analytic'] = float(np.max(np.abs(num - ana) / np.maximum(np.abs(ana), 1e-300)))
    # T2: A+h? A+D=1
    rep['T2_AD_sum'] = [(0.5 + h) + (0.5 - h) for h in H_VALUES]
    # T3: Pi0 turev tutarliligi: dPi0/dX = F^2
    Xs = np.linspace(0.1, 5, 9)
    errs = []
    for X in Xs:
        dnum = (Pi0_int(X + 1e-6, 0.0, 0.005) - Pi0_int(X - 1e-6, 0.0, 0.005)) / 2e-6
        errs.append(abs(dnum - Pi_X0(X, 0.0, 0.005)))
    rep['T3_dPi_consistency_maxerr'] = float(max(errs))
    # T4: invariant
    modes = solve_string_modes([1, 2, 3, 5], 0.005, s_max=6.0, npts=2000)
    rep['T4_invariant_maxerr'] = {n: float(np.max(np.abs(m['I_n'] - n * p['c_s']))) for n, m in modes.items()}
    # T5: eksen duzgunlugu E~sqrt(2X)F0
    Xe = 1e-8
    E0 = float(E_prof(Xe, 0.0, 0.005))
    rep['T5_axis_ratio'] = float(E0 / np.sqrt(2 * Xe))
    # T6: Teorem 2 eksen duzenlemesi (h = 0.005, mu_s = 0.1, s = 10)
    h6, mu6, s6, Xmax6 = 0.005, 0.1, 10.0, 50.0
    t6 = {}
    Xg = np.array([0.0, 1e-12, 0.05, 0.5, 1.0, 5.0])
    t6['l_identity_maxerr'] = float(np.max(np.abs(l_shear_reg(Xg, h6, p['X0']) - l_prof(Xg, 0.0, h6, p['X0']))))
    G0 = float(np.asarray(G_drive(0.0, h6, p)))
    Gt = float(np.asarray(G_drive(1e-12, h6, p)))
    t6['G_axis'] = G0
    t6['G_1e-12'] = Gt
    t6['G_expected_F0_beta0'] = p['F0'] * p['beta0']
    t6['G_continuity_err'] = float(abs(G0 - Gt))
    lt0 = float(np.asarray(ltilde_explicit_analytic(s6, 0.0, h6, p)))
    ltt = float(np.asarray(ltilde_explicit_analytic(s6, 1e-12, h6, p)))
    P0 = float(np.asarray(Pi_eff_X_reg(s6, lt0, 0.0, h6, mu6, p)))
    Pt = float(np.asarray(Pi_eff_X_reg(s6, ltt, 1e-12, h6, mu6, p)))
    t6['Pi_reg_axis'] = P0
    t6['Pi_reg_1e-12'] = Pt
    t6['Pi_reg_continuity_err'] = float(abs(P0 - Pt))
    # eski (duzenlenmemis) formulun eksendeki davranisi ile karsitlik
    ell_old = float(np.asarray(ell_explicit_analytic(s6, 1e-12, h6, p)))
    t6['old_correction_1e-12'] = float(mu6 * np.exp(-(1.0 + 2.0 * h6) * s6) * ell_old ** 2 / (2.0 * 1e-12))
    t6['new_correction_1e-12'] = float((mu6 / (2.0 * p['X_star'])) * np.exp(-(1.0 + 2.0 * h6) * s6) * ltt ** 2)
    Pc, Perr = Pi_center_reg(s6, h6, mu6, X_max=Xmax6, p=p)
    t6['Pi_center_Xmax50'] = Pc
    t6['Pi_center_quad_err'] = Perr
    # ayristirma capraz kontrolu: -int F^2 (analitik) + sicim parcasi (quad)
    Fpart = -(p['F0'] ** 2) * p['X0'] / (1.0 + 2.0 * h6) * (1.0 - (1.0 + Xmax6 / p['X0']) ** (-(1.0 + 2.0 * h6)))
    def _str2(x):
        return float(np.asarray(ltilde_explicit_analytic(s6, x, h6, p))) ** 2
    Sval, _ = quad(_str2, 0.0, Xmax6, limit=200)
    Sval = (mu6 / (2.0 * p['X_star'])) * np.exp(-(1.0 + 2.0 * h6) * s6) * Sval
    t6['Pi_center_decomp'] = float(Fpart + Sval)
    t6['Pi_center_decomp_err'] = float(abs(Pc - (Fpart + Sval)))
    t6['finite'] = bool(np.isfinite(P0) and np.isfinite(Pt) and np.isfinite(Pc))
    rep['T6_axis_regularization'] = t6
    # T7: Teorem 3 kesmesiz kuresel yakinsaklik (h = 0.005, mu_s = 0.1)
    h7, mu7 = 0.005, 0.1
    t7 = {'Pi_center': {}, 'quad_err': {}}
    for s7 in [5, 10, 15, 20]:
        v7, e7 = Pi_center_global(float(s7), h7, mu7, p)
        t7['Pi_center'][str(s7)] = v7
        t7['quad_err'][str(s7)] = e7
    t7['quad_err_max'] = float(max(t7['quad_err'].values()))
    t7['tail'] = {}
    for s7 in [10, 20]:
        R4 = float(1e4 ** (1 + h7) * float(np.asarray(ltsq_minus_l02(float(s7), 1e4, h7, p))))
        R6 = float(1e6 ** (1 + h7) * float(np.asarray(ltsq_minus_l02(float(s7), 1e6, h7, p))))
        t7['tail'][str(s7)] = {'R_1e4': R4, 'R_1e6': R6, 'rel_diff': float(abs(R6 - R4) / abs(R4))}
    t7['tail_rel_diff_max'] = float(max(v['rel_diff'] for v in t7['tail'].values()))
    # capraz kontrol: delta_global vs dogrudan Radau (denge-korumali ODE)
    def _dg_rhs(s, y, Xc):
        Gc = float(np.asarray(G_drive(Xc, h7, p)))
        return [p['kappa_r'] * Gc * np.exp(h7 * s) * y[0] - p['gamma_rel'] * np.exp(-s) * (y[0] - p['ell0'])]
    xchk = []
    for Xc in [0.0, 0.5, 1.0, 2.0, 10.0, 100.0]:
        sol7 = solve_ivp(_dg_rhs, (0.0, 10.0), [p['ell0']], args=(Xc,), method='Radau', rtol=1e-12, atol=1e-14)
        ref7 = float(sol7.y[0, -1]) - p['ell0']
        got7 = float(np.asarray(delta_global(10.0, Xc, h7, p)))
        xchk.append(abs(got7 - ref7) / max(abs(ref7), 1e-300))
    t7['delta_crosscheck_maxrelerr'] = float(max(xchk))
    t7['pass'] = bool(t7['quad_err_max'] < 1e-11 and t7['tail_rel_diff_max'] < 1e-2
                      and t7['delta_crosscheck_maxrelerr'] < 1e-10)
    rep['T7_global_convergence'] = t7
    # T8: Onerme 4.2 operatorleri ve T0 kalintisi (h = 0.005, eta = 0, X in [0, 10])
    h8, mu8, s8 = 0.005, 0.1, 10.0
    t8 = {}
    Sq00 = float(np.asarray(Sq_full(0.0, 0.0, h8, p)))
    t8['Sq00'] = Sq00
    t8['Sq00_expected_-h'] = -h8
    q_tiny = float(np.asarray(Qs_full(1e-15, 0.0, h8, p)))
    t8['Qs_identity_err'] = float(abs(q_tiny - Sq00 / 2.0))
    X8 = np.linspace(0.0, 10.0, 400)
    W8 = np.asarray(W_full(X8, 0.0, h8, p), dtype=float)
    S8 = np.asarray(Sq_full(X8, 0.0, h8, p), dtype=float)
    Q8 = np.asarray(Qs_full(X8, 0.0, h8, p), dtype=float)
    T08 = np.asarray(T0_full(X8, 0.0, h8, p), dtype=float)
    Te8 = np.asarray(T0_eff_global(X8, s8, h8, mu8, p), dtype=float)
    t8['W_range'] = [float(W8.min()), float(W8.max())]
    t8['Sq_range'] = [float(S8.min()), float(S8.max())]
    t8['Qs_range'] = [float(Q8.min()), float(Q8.max())]
    t8['T0_range'] = [float(T08.min()), float(T08.max())]
    n2 = float(np.sqrt(np.trapezoid(T08 ** 2, X8)))
    n2e = float(np.sqrt(np.trapezoid(Te8 ** 2, X8)))
    ni = float(np.max(np.abs(T08)))
    nie = float(np.max(np.abs(Te8)))
    t8['L2_0'], t8['L2_eff'] = n2, n2e
    t8['Linf_0'], t8['Linf_eff'] = ni, nie
    t8['L2_reduction'] = float(1.0 - n2e / n2)
    t8['Linf_reduction'] = float(1.0 - nie / ni)
    t8['pass'] = bool(t8['Qs_identity_err'] < 1e-14 and np.isfinite(n2e)
                      and t8['L2_reduction'] > 0 and t8['Linf_reduction'] > 0)
    rep['T8_proposition42'] = t8
    # T9: tam PDE + FENE-P (h = 0.005, mu_s = 0.1, s in [0, 40])
    h9, mu9 = 0.005, 0.1
    t9 = {}
    X9 = pde_grid(Nx=80)
    fld_o = solve_l_field(X9, h9, mu9, 40.0, p, p['bmax'], 121, transport=False)
    fld_p = solve_l_field(X9, h9, mu9, 40.0, p, p['bmax'], 121, transport=True)
    met_o = field_metrics(X9, fld_o['s'], fld_o['L'], h9, mu9, p['bmax'], p)
    met_p = field_metrics(X9, fld_p['s'], fld_p['L'], h9, mu9, p['bmax'], p)
    t9['ode_success'] = bool(fld_o['sol'].success)
    t9['pde_success'] = bool(fld_p['sol'].success)
    t9['fene_max_ode'] = met_o['fene_max']
    t9['fene_max_pde'] = met_p['fene_max']
    t9['Pi_star_40_ode'] = met_o['Pi_star_40']
    t9['Pi_star_40_pde'] = met_p['Pi_star_40']
    t9['Pi_axis_40'] = met_p['Pi_axis_40']
    t9['u_max_ode'] = met_o['u_max']
    t9['u_max_pde'] = met_p['u_max']
    t9['wake_q_pde'] = met_p['wake_q']
    # analitik tutarlilik (s ~ 2): geri-besleme + tasinim farki sinirli olmali (kaba sinir)
    j2 = int(np.argmin(np.abs(fld_p['s'] - 2.0)))
    l_an = np.asarray(ltilde_global(float(fld_p['s'][j2]), X9, h9, p), dtype=float)
    rel = np.abs(fld_p['L'][:, j2] - l_an) / np.maximum(np.abs(l_an), 1e-300)
    t9['early_s'] = float(fld_p['s'][j2])
    t9['analytic_sanity_maxrelerr'] = float(np.max(rel))
    # izgarasal yakinsama: Nx = 160 cozumu vs Nx = 80 (enterpole), s = 40
    Xf = pde_grid(Nx=160)
    fld_f = solve_l_field(Xf, h9, mu9, 40.0, p, p['bmax'], 121, transport=True)
    Lc = np.interp(Xf, X9, fld_p['L'][:, -1])
    relc = np.abs(fld_f['L'][:, -1] - Lc) / np.maximum(np.abs(Lc), 1e-300)
    t9['grid_conv_maxrelerr'] = float(np.max(relc))
    # uzatilmis alan iz teshisi (Xmax = 300): q <= 1 iraksak iz demektir
    Xe = np.concatenate(([0.0], np.logspace(-6, np.log10(300.0), 100)))
    fld_e = solve_l_field(Xe, h9, mu9, 40.0, p, p['bmax'], 41, transport=True)
    F2e = np.asarray(F_prof(Xe, 0.0, h9, p['F0'], p['X0'], p['alpha_eta']), dtype=float) ** 2
    Ae, qe = wake_exponent(Xe, fld_e['Pi'][:, -1] - F2e, npts=12)
    t9['wake_q_extended'] = qe
    t9['Pi_300_40'] = float(fld_e['Pi'][-1, -1])
    t9['pass'] = bool(t9['ode_success'] and t9['pde_success']
                      and t9['fene_max_pde'] < 1.0 and np.isfinite(met_p['Pi_axis_40'])
                      and t9['analytic_sanity_maxrelerr'] < 0.25
                      and t9['grid_conv_maxrelerr'] < 0.10)
    rep['T9_full_pde'] = t9
    # T10 (ist. bag.): K_curv egrilik carpani - PDE izi yakinsar (h = 0.005, mu_s = 0.1)
    h10, mu10 = 0.005, 0.1
    t10 = {}
    K0 = float(np.asarray(K_curv(0.0, p)))
    Ks = float(np.asarray(K_curv(p['X_star'], p)))
    t10['K0'] = K0
    t10['K_star'] = Ks
    t10['K0_expected'] = (p['X0'] + p['X_star']) / (2.0 * p['X_star'] * p['X0'])
    t10['K_star_expected'] = 1.0 / (2.0 * p['X_star'])
    t10['K0_err'] = float(abs(K0 - t10['K0_expected']))
    t10['K_star_err'] = float(abs(Ks - t10['K_star_expected']))
    Xk = 1e6
    Kasy = float(np.asarray(K_curv(Xk, p)))
    t10['K_asymptotic_relerr'] = float(abs(Xk * Kasy / ((p['X0'] + p['X_star']) / (2.0 * p['X_star'])) - 1.0))
    X10 = pde_grid(Nx=80)
    fld_c = solve_l_field(X10, h10, mu10, 40.0, p, p['bmax'], 121, transport=True, curv=True)
    met_c = field_metrics(X10, fld_c['s'], fld_c['L'], h10, mu10, p['bmax'], p, curv=True)
    t10['pde_success'] = bool(fld_c['sol'].success)
    t10['fene_max'] = met_c['fene_max']
    t10['Pi_star_40'] = met_c['Pi_star_40']
    t10['Pi_axis_40'] = met_c['Pi_axis_40']
    t10['u_max'] = met_c['u_max']
    t10['I50_40'] = met_c['I50_40']
    t10['wake_q_grid'] = met_c['wake_q']
    Xe2 = np.concatenate(([0.0], np.logspace(-6, np.log10(300.0), 100)))
    fld_e2 = solve_l_field(Xe2, h10, mu10, 40.0, p, p['bmax'], 41, transport=True, curv=True)
    F2e2 = np.asarray(F_prof(Xe2, 0.0, h10, p['F0'], p['X0'], p['alpha_eta']), dtype=float) ** 2
    Ae2, qe2 = wake_exponent(Xe2, fld_e2['Pi'][:, -1] - F2e2, npts=12)
    t10['wake_A_extended'] = Ae2
    t10['wake_q_extended'] = qe2
    t10['Pi_300_40'] = float(fld_e2['Pi'][-1, -1])
    t10['tail_est'] = float(-Ae2 * 300.0 ** (1.0 - qe2) / (qe2 - 1.0)) if qe2 > 1 else None
    t10['I50_300'] = float(np.trapezoid(fld_e2['Pi'][:, -1], Xe2))
    t10['Iinf_est'] = float(t10['I50_300'] + tail_F_analytic(300.0, h10, p) + t10['tail_est']) if qe2 > 1 else None
    t10['pass'] = bool(t10['K0_err'] < 1e-12 and t10['K_star_err'] < 1e-12
                       and t10['K_asymptotic_relerr'] < 1e-6
                       and t10['pde_success'] and t10['fene_max'] < 1.0
                       and np.isfinite(met_c['Pi_axis_40']) and qe2 > 1.3)
    rep['T10_curvature'] = t10
    return rep

if __name__ == '__main__':
    import json
    print(json.dumps(run_verification(), indent=2))
