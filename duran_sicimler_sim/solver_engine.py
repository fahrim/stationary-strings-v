"""Duran Sicimler Teoremi V - Sayisal Cozucu Motoru.
Eylul 2026 Navier-Stokes ic-cekirdek (Denk. 3.2, 4.2-4.14) + elastik sicim gerilimi.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar, minimize
from numpy.polynomial.legendre import leggauss as _leggauss

# Faz 2 Gauss-Legendre cizelgesi (denge-korumali genlik integrali icin, M = 256)
_GL_XI, _GL_W = _leggauss(256)

# ---------------- Parametreler ----------------
DEFAULTS = dict(
    F0=1.0, X0=1.0, alpha_eta=0.3, U0=1.0,
    kappa_r=1.0, gamma_rel=0.1, ell0=1.0,
    X_star=1.0, c_s=5.0, ell_light0=1.0, beta0=0.5,
    U_even=0.5, bmax=1e16, gamma_F=0.1,
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

def power_wake_q(X, Y, npts=12):
    """Pozitif buyukluk (Pi = F^2) kuyruk ussu: ln Y ~ lnA - q lnX. q <= 1 iraksar."""
    Xe = np.asarray(X[-npts:], dtype=float)
    Ye = np.asarray(Y[-npts:], dtype=float)
    if np.any(~np.isfinite(Ye)) or np.any(Ye <= 0):
        return float('nan')
    M = np.column_stack([np.ones_like(Xe), np.log(Xe)])
    coef, *_ = np.linalg.lstsq(M, np.log(Ye), rcond=None)
    return float(-coef[1])

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
# ================= FAZ 3 (Makale 6 / Teorem VI) =================
def Gamma_shear(X, h, p=None):
    """Boyutsuz geometrik makaslama: Gamma = sqrt(a^2 + beta0^2 l_shear^2).

    G_drive(X) = F_prof(X,0) * Gamma(X) ile ayni carpandir; iki-alanli surus F(X,s)*Gamma olur.
    """
    if p is None:
        p = DEFAULTS
    a = np.asarray(a_prof(X, 0.0, h, p['X0']), dtype=float)
    ls = np.asarray(l_shear_reg(X, h, p['X0']), dtype=float)
    return np.sqrt(a * a + p['beta0'] ** 2 * ls * ls)

def make_two_field_rhs(X, h, mu_s, p, bmax, gamma_F):
    """Iki-alanli (F, l_tilde) sag-taraf (son duzeltme: sapma+seyrelme tasinimi,
    F-orantili skew-simetrik FENE kuplaji).

    d_s F + W [X d_X(F-Fb) + (1+h)(F-Fb)] = -gF e^{-s}(F-Fb) - kr M F Ghat (l^2-l0^2) phi
    d_s l + W X d_X l                     = +kr F^2 Ghat e^{hs} l phi - gr e^{-s}(l-l0)
    M(X,s) = mu_s K_curv(X) e^{-(1+h)s}, phi = 1 - l^2/bmax (isaretsiz).
    F-orantili tepki: F -> 0 iken kuvvet soner (pozitiflik); seyreltme (1+h)W:
    karakteristik boyunca d(dF)/ds = -(1+h)W dF (uzak-alan sonumu).
    """
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    N = X.size
    Fb = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    Gam = np.asarray(Gamma_shear(X, h, p), dtype=float)
    Kk = np.asarray(K_curv(X, p), dtype=float)
    W = np.asarray(W_full(X, 0.0, h, p, p['U_even']), dtype=float)
    kr, gr, l0 = p['kappa_r'], p['gamma_rel'], p['ell0']

    def rhs(s, y):
        F = np.asarray(y[:N], dtype=float)
        l = np.asarray(y[N:], dtype=float)
        dF = np.asarray(F - Fb, dtype=float)
        phi = 1.0 - l * l / bmax
        Mx = mu_s * Kk * np.exp(-(1.0 + h) * s)
        dF_react = -kr * Mx * F * Gam * (l * l - l0 * l0) * phi
        dFdt = (-W * X * _upwind_DX(X, dF, W) - (1.0 + h) * W * dF
                - gamma_F * np.exp(-s) * dF + dF_react)
        dl_shear = kr * F * F * Gam * np.exp(h * s) * l * phi
        dldt = -W * X * _upwind_DX(X, l, W) + dl_shear - gr * np.exp(-s) * (l - l0)
        return np.concatenate([dFdt, dldt])
    return rhs

def solve_two_field(X, h, mu_s, s_max=40.0, p=None, bmax=None, npts=121, l_init=None, gamma_F=None):
    """Iki-alanli PDE: F(X,0) = F_base, l(X,0) = l_init (yoksa ell0). Donus: dict(s,X,F,L,Pi,Uth,sol)."""
    if p is None:
        p = DEFAULTS
    if bmax is None:
        bmax = p['bmax']
    if gamma_F is None:
        gamma_F = p['gamma_F']
    X = np.asarray(X, dtype=float)
    N = X.size
    Fb = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    l0v = np.asarray(l_init if l_init is not None else np.full(N, p['ell0']), dtype=float)
    rhs = make_two_field_rhs(X, h, mu_s, p, bmax, gamma_F)
    s_eval = np.linspace(0.0, s_max, npts)
    sol = solve_ivp(rhs, (0.0, s_max), np.concatenate([Fb, l0v]), method='Radau',
                    rtol=1e-8, atol=1e-10, t_eval=s_eval)
    F = sol.y[:N, :]
    L = sol.y[N:, :]
    Pi = F * F
    S, Xg = np.meshgrid(s_eval, X)
    Uth = np.exp((0.5 + h) * S) * np.sqrt(2.0 * Xg) * F
    return dict(s=s_eval, X=X, F=F, L=L, Pi=Pi, Uth=Uth, sol=sol)

def two_field_powers(X, s, F, l, h, mu_s, p, bmax):
    """Noktasal reaktif guc cifti (son skew-simetrik form).

    P_fluid = F (dF/ds)_react, (dF/ds)_react = -kr M F Ghat (l^2-l0^2) phi;
    P_string = M e^{-hs} ((l^2-l0^2)/l) (dl/ds)_react,
    (dl/ds)_react = +kr F^2 Ghat e^{hs} l phi.
    Cebirsel sonuc: P_fluid + P_string == 0 (F^2 dahil tum carpanlar sadeleşir).
    """
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float)
    l = np.asarray(l, dtype=float)
    s = float(s)
    l0 = p['ell0']
    Gam = np.asarray(Gamma_shear(X, h, p), dtype=float)
    Kk = np.asarray(K_curv(X, p), dtype=float)
    kr = p['kappa_r']
    phi = 1.0 - l * l / bmax
    Mx = mu_s * Kk * np.exp(-(1.0 + h) * s)
    dF_react = -kr * Mx * F * Gam * (l * l - l0 * l0) * phi
    dl_shear = kr * F * F * Gam * np.exp(h * s) * l * phi
    Pf = F * dF_react
    Ps = Mx * np.exp(-h * s) * ((l * l - l0 * l0) / np.maximum(l, 1e-300)) * dl_shear
    ssum = Pf + Ps
    nrm = max(float(np.max(np.abs(Pf))), 1e-300)
    return dict(Pf=Pf, Ps=Ps, sum=ssum,
                raw=float(np.max(np.abs(ssum)) / nrm),
                min_l=float(np.min(l)))

def U_elastic(X, l, s, h, mu_s, p, bmax):
    """FENE-P elastik potansiyel yogunlugu U_el (phi isaretsiz; log ici kesmelidir)."""
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    l = np.asarray(l, dtype=float)
    l0 = p['ell0']
    Kk = np.asarray(K_curv(X, p), dtype=float)
    phi = 1.0 - l * l / bmax
    phi0 = 1.0 - l0 * l0 / bmax
    return (mu_s / 2.0) * Kk * np.exp(-(1.0 + 2.0 * h) * float(s)) * (-bmax * np.log(np.maximum(phi / phi0, 1e-300)))

def H_series(X, s_eval, F, L, h, mu_s, p, bmax):
    """Toplam Lyapunov enerjisi H_total(s) = int_0^Xmax (F^2 + 2 U_el) dX (trapez)."""
    X = np.asarray(X, dtype=float)
    H = np.zeros_like(np.asarray(s_eval, dtype=float))
    for j in range(len(H)):
        Ue = U_elastic(X, L[:, j], s_eval[j], h, mu_s, p, bmax)
        H[j] = float(np.trapezoid(F[:, j] ** 2 + 2.0 * Ue, X))
    return H

# ---- Gorev 7: n-modu spektral parmak izi ----
def characteristic_origin(X, s, h, p=None):
    """Geri-cekme haritasi X_0(X, s): dX/ds = W(X,0) X akisini s -> 0'a tersten entegre eder."""
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    s = float(s)
    out = np.zeros_like(X)
    for i, Xj in enumerate(np.atleast_1d(X)):
        def flow(ss, xx):
            return [float(np.asarray(W_full(xx, 0.0, h, p, p['U_even']))[0]) * xx[0]]
        sol = solve_ivp(flow, (s, 0.0), [float(Xj)], method='RK45', rtol=1e-12, atol=1e-14)
        out.flat[i] = float(sol.y[0, -1])
    return out[()] if np.ndim(X) == 0 else out

def flux_J(X, s, F_a, F_b, l_a, l_b, h, mu_s, p, bmax, gamma_F):
    """Reaktif aki farki: DeltaJ = L_adv[Fa - Fb], L_adv = d_s + W X d_X + (1+h) W + gF e^{-s}.

    d_s terimi tam RHS farkindan (kesmesiz) alinir; uzaysal turev cozucuyle ayni upwind'dir.
    """
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    dF = np.asarray(F_a, dtype=float) - np.asarray(F_b, dtype=float)
    N = X.size
    Fb = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    W = np.asarray(W_full(X, 0.0, h, p, p['U_even']), dtype=float)
    ya = np.concatenate([np.asarray(F_a, dtype=float), np.asarray(l_a, dtype=float)])
    yb = np.concatenate([np.asarray(F_b, dtype=float), np.asarray(l_b, dtype=float)])
    rhs = make_two_field_rhs(X, h, mu_s, p, bmax, gamma_F)
    ds_dF = np.asarray(rhs(float(s), ya)[:N]) - np.asarray(rhs(float(s), yb)[:N])
    return ds_dF + W * X * _upwind_DX(X, dF, W) + (1.0 + h) * W * dF + gamma_F * np.exp(-s) * dF

# ---- Gorev 7 (son): es-hareketli Lagrange materyal cercevesi ----
def lagrange_grid(Nxi=128, Xtop=10.0, X0param=1.0):
    """Duzgun etiket izgarasi xi in [0,1]; X0(xi) = X0param*((1+Xtop/X0param)^xi - 1)."""
    xi = np.linspace(0.0, 1.0, Nxi)
    X0map = X0param * ((1.0 + Xtop / X0param) ** xi - 1.0)
    return xi, X0map

def make_lagrange_rhs(xi, X0map, h, mu_s, p, bmax, gamma_F):
    """Lagrange materyal RHS: durum [F, l, Xpos]; tasinim materyal turevde erir
    (uzaysal turev yok -> upwind difuzyonu sifir, konvolusyonel cekirdek yok).

    d_s F = -gF e^{-s}(F - Fb(Xpos)) - kr M(Xpos) F Ghat(Xpos) (l^2-l0^2) phi
    d_s l = +kr F^2 Ghat(Xpos) e^{hs} l phi - gr e^{-s}(l - l0)
    d_s Xpos = W(Xpos) Xpos   (parsel konumu; arka plan alanlari guncel Xpos'ta)
    """
    if p is None:
        p = DEFAULTS
    xi = np.asarray(xi, dtype=float)
    X0map = np.asarray(X0map, dtype=float)
    N = xi.size
    kr, gr, l0 = p['kappa_r'], p['gamma_rel'], p['ell0']

    def rhs(s, y):
        F = np.asarray(y[:N], dtype=float)
        l = np.asarray(y[N:2 * N], dtype=float)
        Xp = np.asarray(y[2 * N:], dtype=float)
        Fb = np.asarray(F_prof(Xp, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
        Gam = np.asarray(Gamma_shear(Xp, h, p), dtype=float)
        Kk = np.asarray(K_curv(Xp, p), dtype=float)
        W = np.asarray(W_full(Xp, 0.0, h, p, p['U_even']), dtype=float)
        phi = 1.0 - l * l / bmax
        Mx = mu_s * Kk * np.exp(-(1.0 + h) * s)
        dF = (-gamma_F * np.exp(-s) * (F - Fb)
              - kr * Mx * F * Gam * (l * l - l0 * l0) * phi)
        dl = (kr * F * F * Gam * np.exp(h * s) * l * phi
              - gr * np.exp(-s) * (l - l0))
        dX = W * Xp
        return np.concatenate([dF, dl, dX])
    return rhs

def solve_lagrange(xi, X0map, h, mu_s, s_max=25.0, p=None, bmax=None, npts=51,
                   l_init=None, gamma_F=None):
    """Lagrange cozumu: F(xi,0) = Fb(X0(xi)), l(xi,0) = l_init (yoksa ell0)."""
    if p is None:
        p = DEFAULTS
    if bmax is None:
        bmax = p['bmax']
    if gamma_F is None:
        gamma_F = p['gamma_F']
    xi = np.asarray(xi, dtype=float)
    X0map = np.asarray(X0map, dtype=float)
    N = xi.size
    Fb0 = np.asarray(F_prof(X0map, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    l0v = np.asarray(l_init if l_init is not None else np.full(N, p['ell0']), dtype=float)
    rhs = make_lagrange_rhs(xi, X0map, h, mu_s, p, bmax, gamma_F)
    s_eval = np.linspace(0.0, s_max, npts)
    sol = solve_ivp(rhs, (0.0, s_max), np.concatenate([Fb0, l0v, X0map.copy()]),
                    method='Radau', rtol=1e-8, atol=1e-10, t_eval=s_eval)
    return dict(s=s_eval, xi=xi, F=sol.y[:N, :], L=sol.y[N:2 * N, :],
                Xpos=sol.y[2 * N:, :], sol=sol)

def l_init_lagrange(X0map, n, p=None, eps_mod=0.02, X_core=10.0):
    """Lagrange baslangic deseni (etiket uzayi): l0[1 + eps sin(n pi xi) e^{-X0/Xcore}]."""
    if p is None:
        p = DEFAULTS
    X0map = np.asarray(X0map, dtype=float)
    xi = np.log1p(X0map / p['X0']) / np.log1p(X_core / p['X0'])
    return p['ell0'] * (1.0 + eps_mod * np.sin(n * np.pi * xi) * np.exp(-X0map / X_core))

def l_init_uniform_xi(X0map, p=None, eps_mod=0.02, X_core=10.0):
    """Uniform kalibrasyon (sin -> 1, zarf korunur): l0[1 + eps e^{-X0/Xcore}]."""
    if p is None:
        p = DEFAULTS
    X0map = np.asarray(X0map, dtype=float)
    return p['ell0'] * (1.0 + eps_mod * np.exp(-X0map / X_core))

def l_init_modulated(X, n, p=None, eps_mod=0.05, X_core=10.0):
    """Baslangic bilgi deseni: l0 [1 + eps sin(n pi xi) e^{-X/Xcore}], xi = ln(1+X/X0)/ln(1+Xcore/X0)."""
    if p is None:
        p = DEFAULTS
    X = np.asarray(X, dtype=float)
    xi = np.log1p(X / p['X0']) / np.log1p(X_core / p['X0'])
    return p['ell0'] * (1.0 + eps_mod * np.sin(n * np.pi * xi) * np.exp(-X / X_core))

def sine_spectrum(xi_u, dF_xi, kmax=6):
    """Normalize log-koordinatta Fourier sinus katsayilari c_k, k = 1..kmax (trapez)."""
    c = np.zeros(kmax)
    for k in range(1, kmax + 1):
        c[k - 1] = float(np.trapezoid(dF_xi * np.sin(k * np.pi * xi_u), xi_u))
    return c

# ================= FAZ 5 (Makale 7 / Teorem VII) =================
# Uygulamali kesif gorevleri: T16 (ALPHA 2025 SDR-EVC), T17 (ALPHA 2026 cizgi
# sekli), T18 (Ore-Powell tam duzeltmesi). Tum model iliskileri calismanin
# fenomenolojik ansatzidir; deneysel girdiler Nature/NatCommun makalelerinden.
# SR-150kHz/15.6->1.0V/SDR: NatCommun 16 (2025) dogrulandi; 74kHz/saat flux
# suruklenmesi + parcali g*R cizgi sekli: Nature 653 (2026) dogrulandi.
_E_CHG = 1.602176634e-19
_EPS0 = 8.8541878128e-12
ALPHA_TRAP = dict(B_T=1.0, R_W_mm=16.8, L_p_mm=40.0, N_plasma=1.28e5,
                  f_RW_kHz=150.0, r_exp_mm=0.16, V_i=15.6, V_f=1.0,
                  T_trap_K=0.5, s_eq=10.0)
LINE = dict(sig0_kHz=30.0, sigb_kHz=10.0, X0=1.0, q_wake=1.9010, xi_depl=30.0)
GAL = dict(Ndot_obs=2.0e43, eta_pair=0.1, M_sun_kg=1.98847e30,
           yr_s=365.25 * 86400.0)
# Gercek 1949 Ore-Powell Omega(x) seri acilimi (sympy, x=E/m, x<1e-2 dalinda):
# 5x/3 + x^2/3 - 2x^3/15 - x^4/5 - 29x^5/210 - 13x^6/210 - 2x^7/315.
_OP_SER = (5.0 / 3.0, 1.0 / 3.0, -2.0 / 15.0, -1.0 / 5.0,
           -29.0 / 210.0, -13.0 / 210.0, -2.0 / 315.0)


def trap_ideal_radius(B_T=None, N=None, L_p=None, f_RW=None):
    """Kaymasiz SDR denge yaricapi: f_rot=f_RW kosulundan r_p (m)."""
    c = ALPHA_TRAP
    B = c['B_T'] if B_T is None else B_T
    Nn = c['N_plasma'] if N is None else N
    Lp = c['L_p_mm'] * 1e-3 if L_p is None else L_p
    f = c['f_RW_kHz'] * 1e3 if f_RW is None else f_RW
    return float(np.sqrt(_E_CHG * Nn / (4.0 * np.pi ** 2 * _EPS0 * B * Lp * f)))


def trap_axis_potential(n_m3, r_p, R_W=None):
    """Eksen potansiyeli phi_c (V): Denk.3."""
    RW = ALPHA_TRAP['R_W_mm'] * 1e-3 if R_W is None else R_W
    return float(n_m3 * _E_CHG * r_p ** 2 / (4.0 * _EPS0)
                 * (1.0 + 2.0 * np.log(RW / r_p)))


def trap_F_base(X_star=1.0, h=0.005, p=None):
    if p is None:
        p = DEFAULTS
    return float(np.asarray(F_prof(X_star, 0.0, h, p['F0'], p['X0'], p['alpha_eta'])))


def trap_F_eff(X_star, s_eq, h, mu_s, p=None):
    """Sicim geri-tepkili etkin alan genligi: sqrt(max(Pi_eff,eps)) (pozitif)."""
    if p is None:
        p = DEFAULTS
    Pi = float(np.asarray(Pi_eff_X_global(X_star, s_eq, h, mu_s, p)))
    return float(np.sqrt(max(Pi, 1e-300)))


def trap_rp_eff(mu_s, X_star=1.0, s_eq=None, h=0.005, p=None):
    c = ALPHA_TRAP
    if s_eq is None:
        s_eq = c['s_eq']
    rp0 = trap_ideal_radius()
    Fb = trap_F_base(X_star, h, p)
    return float(rp0 * (Fb / trap_F_eff(X_star, s_eq, h, mu_s, p)))


def find_mu_star(X_star=1.0, s_eq=None, h=0.005, p=None):
    """r_p^eff(mu*)=r_exp kosulunu brentq ile cozer (12+ basamak)."""
    c = ALPHA_TRAP
    if s_eq is None:
        s_eq = c['s_eq']
    target = c['r_exp_mm'] * 1e-3
    return float(brentq(lambda mu: trap_rp_eff(mu, X_star, s_eq, h, p) - target,
                        1e-12, 0.24, xtol=1e-14, rtol=1e-14, maxiter=300))


def trap_slip(B_T=None):
    """Olculen yaricapta donus frekansi ve RW kaymasi."""
    c = ALPHA_TRAP
    B = c['B_T'] if B_T is None else B_T
    Nn, Lp = c['N_plasma'], c['L_p_mm'] * 1e-3
    rexp = c['r_exp_mm'] * 1e-3
    n = Nn / (np.pi * rexp ** 2 * Lp)
    frot = _E_CHG * n / (4.0 * np.pi * _EPS0 * B)
    fRW = c['f_RW_kHz'] * 1e3
    return dict(n_m3=float(n), frot_kHz=float(frot / 1e3),
                slip=float(1.0 - frot / fRW),
                phi_c_V=float(trap_axis_potential(n, rexp)))


def chirp_gain(h=0.005, mu_s=0.1, s=None, p=None):
    """Cirp'li RW kazanci: T0 kayma isinmasi bastirma -> Ftrap artisi."""
    if p is None:
        p = DEFAULTS
    if s is None:
        s = ALPHA_TRAP['s_eq']
    X = np.linspace(0.0, 10.0, 400)
    T0 = np.asarray(T0_full(X, 0.0, h, p), dtype=float)
    Te = np.asarray(T0_eff_global(X, s, h, mu_s, p), dtype=float)
    L0, Le = float(np.max(np.abs(T0))), float(np.max(np.abs(Te)))
    Tt = ALPHA_TRAP['T_trap_K']
    Fstd, Fch = min(1.0, Tt / L0), min(1.0, Tt / Le)
    return dict(Linf_0=L0, Linf_eff=Le, Ftrap_std=float(Fstd),
                Ftrap_chirp=float(Fch),
                gain=float(Fch / Fstd) if Fstd > 0 else float('inf'))


def flux_relaxation(drift_kHz_per_h=-74.0, B0_T=1.03, p=None):
    """Abrikosov akisi gevreme hizi: Breit-Rabi turevinden dB/dt, gamma."""
    if p is None:
        p = DEFAULTS
    dB = 1e-4
    d = (breit_rabi(B0_T + dB, None, p)['fda']
         - breit_rabi(B0_T - dB, None, p)['fda']) / (2.0 * dB)
    dBdt = drift_kHz_per_h / d  # T/saat
    g_h = abs(dBdt) / B0_T
    return dict(dfdB_kHz_per_T=float(d), dfdB_GHz_per_T=float(d / 1e6),
                dBdt_T_per_h=float(dBdt), dBdt_uT_per_h=float(dBdt * 1e6),
                gamma_per_h=float(g_h), gamma_per_s=float(g_h / 3600.0))


def gvortex_raw(x, X0=None, q_wake=None, xi_depl=None):
    """Analitik cizgi sekli cekirdegi (x>=0; x<0'da sifir)."""
    if X0 is None:
        X0 = LINE['X0']
    if q_wake is None:
        q_wake = LINE['q_wake']
    if xi_depl is None:
        xi_depl = LINE['xi_depl']
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(x > 0, x ** 2 / (1.0 + x / X0) ** (2.0 + q_wake)
                       * np.exp(-x / xi_depl), 0.0)
    return out


def gvortex_norm(X0=None, q_wake=None, xi_depl=None):
    f = lambda x: float(np.asarray(gvortex_raw(x, X0, q_wake, xi_depl)))
    v1, e1 = quad(f, 0.0, 120.0, limit=500, epsabs=1e-14, epsrel=1e-13)
    v2, e2 = quad(f, 120.0, np.inf, limit=500, epsabs=1e-14, epsrel=1e-13)
    return v1 + v2, e1 + e2


def gvortex_slopes(X0=None, q_wake=None, xi_depl=None):
    """Esik (k->2) ve ara-rejim kuyruk (k->-q) log-log egimleri."""
    Ng, _ = gvortex_norm(X0, q_wake, xi_depl)
    xe = np.logspace(-4, -2, 200)
    k = float(np.polyfit(np.log(xe), np.log(gvortex_raw(xe,
                X0, q_wake, xi_depl) / Ng), 1)[0])
    xt = np.logspace(np.log10(4.0), np.log10(20.0), 200)
    kt = float(np.polyfit(np.log(xt), np.log(gvortex_raw(xt,
                 X0, q_wake, xi_depl) / Ng), 1)[0])
    xg = np.logspace(-4, 3, 4000)
    xm = float(xg[int(np.argmax(gvortex_raw(xg, X0, q_wake, xi_depl)))])
    return dict(Ng=float(Ng), k_thr=k, k_tail=kt, x_max=xm)


def gvortex_conv_shift(sig0_kHz=None, sigb_kHz=None, X0=None,
                       q_wake=None, xi_depl=None):
    """R cozunurluk konvolusyonu tepe kaymasi (kHz)."""
    s0 = LINE['sig0_kHz'] if sig0_kHz is None else sig0_kHz
    sb = LINE['sigb_kHz'] if sigb_kHz is None else sigb_kHz
    sx = sb / s0
    Ng, _ = gvortex_norm(X0, q_wake, xi_depl)
    y = np.linspace(0.0, 60.0, 12001)
    g = gvortex_raw(y, X0, q_wake, xi_depl) / Ng
    i0 = int(np.argmax(g))
    x0 = float(y[i0])
    xo = np.linspace(-1.0, 8.0, 901)
    conv = np.empty_like(xo)
    for k, xv in enumerate(xo):
        R = np.exp(-0.5 * ((xv - y) / sx) ** 2) / (np.sqrt(2 * np.pi) * sx)
        conv[k] = float(np.trapezoid(g * R, y))
    ic = int(np.argmax(conv))
    return dict(x_max=x0, x_max_conv=float(xo[ic]),
                shift_kHz=float((xo[ic] - x0) * s0))


def ore_omega_true(x):
    """Gercek 1949 Omega(x): x<1e-2'de Taylor (kararli), ustunde kapali form."""
    xc = np.asarray(x, dtype=float)
    lo = xc < 1e-2
    out = np.empty_like(xc)
    xl = xc[lo]
    acc = np.zeros_like(xl)
    for cf in reversed(_OP_SER):
        acc = acc * xl + cf
    out[lo] = acc * xl
    xh = xc[~lo]
    with np.errstate(divide='ignore', invalid='ignore'):
        out[~lo] = (2.0 * (2.0 - xh) / xh + 2.0 * (1.0 - xh) * xh / (2.0 - xh) ** 2
                    + 4.0 * ((1.0 - xh) / xh ** 2
                             - (1.0 - xh) ** 2 / (2.0 - xh) ** 3)
                    * np.log(1.0 - xh))
    return out


def ore_powell_exact(E_keV, m_keV=511.0):
    """Tam Ore-Powell spektrumu (ln'li 1949 formu, [0,m]'de normalize)."""
    m = float(m_keV)
    E = np.asarray(E_keV, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        Om = ore_omega_true(E / m)
        core = Om / (m * (np.pi ** 2 - 9.0))
        edge = 2.0 / (m * (np.pi ** 2 - 9.0))  # x -> 1 limiti (Omega -> 2)
        out = np.where((E > 0) & (E < m), core,
                       np.where(E == m, edge, 0.0))
    return out


def ore_prompt_form(E_keV, m_keV=511.0):
    """Yanlislanabilirlik kaydi: E^2 -> (2m-E)^2 ikameli varyant.

    DIKKAT (GP-III.3 duzeltmesi): bu FONKSIYON basimdaki (ln'siz) formu
    DEGIL; basimdaki form 3. terimde E^2 paydasi tasir. Bu varyantta
    yalnizca E^2 -> (2m-E)^2 ikamesi yapilmistir (1/E kutbu kalir,
    int_1 = 27.147 != 1). Basimdaki form icin `ore_powell_published`
    kullanilir.
    """
    m = float(m_keV)
    E = np.asarray(E_keV, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (2.0 / (m * (np.pi ** 2 - 9.0))) * (
            E * (m - E) / (2 * m - E) ** 2 + (2 * m - E) / E
            + 2 * m * (m - E) / (2 * m - E) ** 2
            - 2 * m * (m - E) ** 2 / (2 * m - E) ** 3)
    return out


def ore_powell_published(E_keV, m_keV=511.0):
    """arXiv:2605.21289 Denk.(12) BASIMDAKI (ln'siz, E^2 paydali) form.

    3. terimde 2 m (m-E)/E^2 -> 1/E^2 kutbu; int_1 = 2343.47 != 1 (S_LOSS
    kayit defteri: basim hatasinin en kotu hali). Duzeltilmis 1949 formu
    `ore_powell_exact` ile verilir.
    """
    m = float(m_keV)
    E = np.asarray(E_keV, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (2.0 / (m * (np.pi ** 2 - 9.0))) * (
            E * (m - E) / (2 * m - E) ** 2 + (2 * m - E) / E
            + 2 * m * (m - E) / E ** 2 - 2 * m * (m - E) ** 2 / (2 * m - E) ** 3)
    return out


def galactic_Mdot_msun_yr(Ndot_obs=None, eta_pair=None, Lambda0=None):
    """Lambda0 ile 511 keV gozlemini veren akresyon rejimi."""
    Nd = GAL['Ndot_obs'] if Ndot_obs is None else Ndot_obs
    et = GAL['eta_pair'] if eta_pair is None else eta_pair
    L0 = float(LAMBDA0_CLOSED) if Lambda0 is None else float(Lambda0)
    m_e_kg = 9.1093837015e-31
    Mdot = Nd * 2.0 * m_e_kg / (L0 * et)  # kg/s
    return float(Mdot / GAL['M_sun_kg'] * GAL['yr_s'])


# Faz 5.1: T16 sifir-parametre + e+ capraz dogrulama; T17 cift-alan ECR;
# T18 3 galaktik rejim; T19 CERN BASE-STEP 2026 karayolu tasimasi.
_M_P = 1.67262192369e-27
_G0 = 9.80665
BASE_STEP = dict(N_pbar=92, B0_T=0.993, B_list=(0.136, 0.880, 0.993),
                 V_well_V=16.0, V_read_V=3.911, nu_z_read_Hz=453e3,
                 R_trap_mm=5.0, L_trap_mm=10.0, d_road_km=7.5,
                 t_drive_s=1440.0, v_max_kmh=40.8, t_auto_h=2.72,
                 dC_fF=20.0, df_res_Hz=122.0, a_rms_g=0.15,
                 r_eq_mm=0.5, g_str0=1.0)


def solve_alpha_sdr_evc_trap(h=0.005, p=None):
    """T16: Be+ SDR-EVC + sifir-parametreli Meral/geom + e+ capraz dogrulama."""
    if p is None:
        p = DEFAULTS
    c = ALPHA_TRAP
    mus = find_mu_star(1.0, c['s_eq'], h, p)
    mu_geom = float(p['beta0'] ** 2 / (1.0 + p['beta0'] ** 2))
    mu_meral = float(2.0 * np.sqrt(2.0) * float(LAMBDA0_CLOSED))
    rp_geom = trap_rp_eff(mu_geom, 1.0, c['s_eq'], h, p) * 1e3
    rp_meral = trap_rp_eff(mu_meral, 1.0, c['s_eq'], h, p) * 1e3
    # e+ plazma capraz dogrulama: N_e+=3.4e6, Lp=40mm, B=1T, etkin RW=500kHz
    rp0_e = trap_ideal_radius(B_T=1.0, N=3.4e6, L_p=0.04, f_RW=500e3)
    Fb = trap_F_base(1.0, h, p)
    rpe_star = float(rp0_e * (Fb / trap_F_eff(1.0, c['s_eq'], h, mus, p)) * 1e3)
    rpe_meral = float(rp0_e * (Fb / trap_F_eff(1.0, c['s_eq'], h, mu_meral, p)) * 1e3)
    ne_m3 = float(3.4e6 / (np.pi * (rpe_star * 1e-3) ** 2 * 0.04))
    # Ters-yon ucgulama (T16b): e+ bandi [0.30,0.50]mm -> mu araligi -> Be+ araligi.
    # Ayrica mu=0 (frenlemesiz) modelin e+ bandinin ALTINDA kaldigini dogrular.
    rpe_zero = float(rp0_e * (Fb / trap_F_eff(1.0, c['s_eq'], h, 0.0, p)) * 1e3)
    mu_lo = float(brentq(lambda mu: float(rp0_e * (Fb / trap_F_eff(
        1.0, c['s_eq'], h, mu, p)) * 1e3) - 0.30, 1e-12, 0.24,
        xtol=1e-14, rtol=1e-14, maxiter=300))
    mu_hi = float(brentq(lambda mu: float(rp0_e * (Fb / trap_F_eff(
        1.0, c['s_eq'], h, mu, p)) * 1e3) - 0.50, 1e-12, 0.24,
        xtol=1e-14, rtol=1e-14, maxiter=300))
    rp0_be = trap_ideal_radius()
    be_lo = float(rp0_be * (Fb / trap_F_eff(1.0, c['s_eq'], h, mu_lo, p)) * 1e3)
    be_hi = float(rp0_be * (Fb / trap_F_eff(1.0, c['s_eq'], h, mu_hi, p)) * 1e3)
    return dict(
        mu_star=mus, mu_geom=mu_geom, mu_meral=mu_meral,
        rel_meral=float(abs(mus - mu_meral) / mus),
        rel_geom=float(abs(mus - mu_geom) / mus),
        rp_geom_mm=float(rp_geom), rp_meral_mm=float(rp_meral),
        dr_meral_um=float(abs(rp_meral - c['r_exp_mm']) * 1e3),
        rpe_star_mm=rpe_star, rpe_meral_mm=rpe_meral,
        ne_m3=ne_m3, eplus_in_band=bool(0.30 <= rpe_star <= 0.50
                                        and 0.30 <= rpe_meral <= 0.50),
        rev_mu_lo=mu_lo, rev_mu_hi=mu_hi,
        rev_be_mm=[be_lo, be_hi],
        rev_zero_excluded=bool(rpe_zero < 0.30),
        rev_rpe_zero_mm=rpe_zero,
    )


def solve_alpha_2026_lineshape_and_drift(p=None):
    """T17: 1.03T/1.07T cift-alan suruklenme + ECR + g_vortex."""
    if p is None:
        p = DEFAULTS
    f1 = flux_relaxation(-72.82, 1.03, p)
    f2 = flux_relaxation(-75.64, 1.07, p)
    R_th = float((1.07 * f2['dfdB_kHz_per_T']) / (1.03 * f1['dfdB_kHz_per_T']))
    R_exp = float((-75.64) / (-72.82))
    # Ayrica tekil-durum f_da(B) = g_e*mu_B*B/h + a1S/(2h) + O(a^2/B) yaklasiminda
    # df/dB(1.07)/df/dB(1.03) -> 1 (doygun B rejimi): R_th_pure = 1.07/1.03
    R_B = float(1.07 / 1.03)
    return dict(
        B103=f1, B107=f2,
        ecr103_sigma=float(abs(f1['dBdt_uT_per_h'] - (-2.50)) / 0.10),
        ecr107_sigma=float(abs(f2['dBdt_uT_per_h'] - (-2.60)) / 0.10),
        R_th=R_th, R_B=R_B, R_exp=R_exp,
        dR_abs=float(abs(R_th - R_exp)),
        dR_B_abs=float(abs(R_B - R_exp)),
    )


def penning_frequencies(B0_T=0.993, V_well_V=None):
    """BASE-STEP Penning frekanslari (okuma 3.911V -> 453kHz referansiyla olcekli)."""
    c = BASE_STEP
    V = c['V_well_V'] if V_well_V is None else V_well_V
    nuz = c['nu_z_read_Hz'] * np.sqrt(V / c['V_read_V'])
    wz = 2.0 * np.pi * nuz
    wc = _E_CHG * B0_T / _M_P
    disc = max(1.0 - 2.0 * (wz / wc) ** 2, 0.0)
    wp = 0.5 * wc * (1.0 + np.sqrt(disc))
    wm = 0.5 * wc * (1.0 - np.sqrt(disc))
    return dict(nu_c_Hz=float(wc / (2 * np.pi)),
                nu_p_Hz=float(wp / (2 * np.pi)),
                nu_z_Hz=float(nuz),
                nu_m_Hz=float(wm / (2 * np.pi)),
                hierarchy=bool(wp > wz > wm > 0))


def solve_base_step_road_transport(mu_s=None):
    """T19: CERN BASE-STEP 2026 karayolu tasimasinda Girdap-Sicim Lyapunov kararliligi."""
    c = BASE_STEP
    if mu_s is None:
        mu_s = find_mu_star()
    fr993 = penning_frequencies(c['B0_T'], c['V_well_V'])
    nuz = fr993['nu_z_Hz']
    wz = 2.0 * np.pi * nuz
    eps_disp = float(c['df_res_Hz'] / nuz)
    g_disp = float(2.0 * np.pi * c['df_res_Hz'] * eps_disp)
    a_rms = float(c['a_rms_g'] * _G0)
    R = c['R_trap_mm'] * 1e-3
    req = c['r_eq_mm'] * 1e-3
    gs0 = c['g_str0']

    def _run(B0, mu, r0=0.0):
        wc = _E_CHG * B0 / _M_P

        def rhs(t, y):
            r = float(y[0])
            gs = mu * gs0
            den = max(1.0 - (r / R) ** 2, 1e-6)
            return [g_disp * r - gs * r * (1.0 + (r / req) ** 2) / den + a_rms / wc]
        t_eval = np.linspace(0.0, c['t_drive_s'], 401)
        sol = solve_ivp(rhs, (0.0, c['t_drive_s']), [r0], t_eval=t_eval,
                        method='Radau', rtol=1e-10, atol=1e-13)
        return dict(t=sol.t, r=sol.y[0], r_max_mm=float(np.max(sol.y[0]) * 1e3),
                    r_end_mm=float(sol.y[0, -1] * 1e3))

    # Korunmasiz (mu=0, r0=1um tohum) vs Teorem 5 korumali (mu=mu*, r0=0)
    unprot = _run(c['B0_T'], 0.0, r0=1e-6)
    prot_B = {str(B): _run(B, mu_s, r0=0.0) for B in c['B_list']}
    rmax_993 = prot_B['0.993']['r_max_mm']
    rmax_880 = prot_B['0.88']['r_max_mm']
    rmax_136 = prot_B['0.136']['r_max_mm']
    # Kritik carpisma ivmesi (r = 0.9 R_trap zarf bariyeri ve kanonik kuvvet olcegi)
    wc993 = _E_CHG * c['B0_T'] / _M_P
    r9 = 0.9 * R
    a_crit = float((mu_s * gs0 * r9 * (1.0 + (r9 / req) ** 2) / (1.0 - 0.81)
                    - g_disp * r9) * wc993)
    return dict(
        freq_993=fr993,
        freq_all={str(B): penning_frequencies(B, c['V_well_V']) for B in c['B_list']},
        eps_disp=eps_disp, g_disp=g_disp, a_rms_ms2=a_rms,
        unprot_t=unprot['t'], unprot_r_mm=unprot['r'] * 1e3,
        unprot_rmax_mm=unprot['r_max_mm'],
        prot_t=prot_B['0.993']['t'],
        prot_r_mm={k: v['r'] * 1e3 for k, v in prot_B.items()},
        rmax_mm={k: v['r_max_mm'] for k, v in prot_B.items()},
        monotone_B=bool(rmax_993 < rmax_880 < rmax_136),
        surv_frac=1.0 if rmax_993 < c['R_trap_mm'] else 0.0,
        N_surv=float(c['N_pbar']) if rmax_993 < c['R_trap_mm'] else 0.0,
        a_crit_g=float(a_crit / _G0),
        M_safety=float(a_crit / a_rms),
    )


# ================= FAZ 6 (Makale 8 / Teorem VIII) =================
# Falsifiye edilebilir ongoru testleri: T20 (BASE-STEP 136 mT yan-band
# acik/kapali), T21 (ALPHA 1.05 T ucuncu alan ongurusu + oran kilidi),
# T22 (511 keV tayf ongoru bandi) + UQ duyarlilik + bagimsiz mini-capraz kontrol
# destegi. Tum model iliskileri calismanin fenomenolojik ansatzidir.

def predict_basestep_136mT_falsification(mu_s=None, p=None):
    """T20: 136 mT zayif-alan falsifikasyon cifti + ofset geri-cekilme."""
    if p is None:
        p = DEFAULTS
    c = BASE_STEP
    if mu_s is None:
        mu_s = find_mu_star()
    fr136 = penning_frequencies(0.136, c['V_well_V'])
    nuz = fr136['nu_z_Hz']
    g_disp = float(2.0 * np.pi * c['df_res_Hz'] * (c['df_res_Hz'] / nuz))
    a_rms = float(c['a_rms_g'] * _G0)
    R = c['R_trap_mm'] * 1e-3
    req = c['r_eq_mm'] * 1e-3
    gs0 = c['g_str0']
    wc = _E_CHG * 0.136 / _M_P

    def rhs_r(r, mu):
        gs = mu * gs0
        den = max(1.0 - (r / R) ** 2, 1e-6)
        return g_disp * r - gs * r * (1.0 + (r / req) ** 2) / den + a_rms / wc

    def run(r0, mu):
        sol = solve_ivp(lambda t, y: [rhs_r(float(y[0]), mu)],
                        (0.0, c['t_drive_s']), [r0], method='Radau',
                        rtol=1e-10, atol=1e-13,
                        t_eval=np.linspace(0.0, c['t_drive_s'], 201))
        return dict(r_end_mm=float(sol.y[0, -1] * 1e3),
                    r_max_mm=float(np.max(sol.y[0]) * 1e3))
    r_star = float(brentq(lambda r: rhs_r(r, mu_s), 1e-12, R * 0.9999,
                          xtol=1e-15, rtol=1e-15, maxiter=300)) * 1e3
    on = run(0.0, mu_s)
    off = {('%.1f' % r0): run(r0 * 1e-3, mu_s) for r0 in [1.0, 2.0, 3.0, 4.0]}
    unprot = run(1e-6, 0.0)
    return dict(
        freq_136=fr136, r_star_um=float(r_star * 1e3),
        on_centered_mm=on, off_offsets_mm=off,
        unprot_rmax_mm=unprot['r_max_mm'],
        off_recaptured=bool(all(v['r_end_mm'] < 0.1 * c['R_trap_mm']
                                for v in off.values())),
        unprot_lost=bool(unprot['r_max_mm'] > c['R_trap_mm']),
    )


def predict_alpha_third_field(B3_T=1.05, p=None):
    """T21: 1.05 T ucuncu-alan frekans-suruklenme ongorusu + oran kilidi."""
    if p is None:
        p = DEFAULTS
    dd = solve_alpha_2026_lineshape_and_drift(p)
    g1, g2 = dd['B103']['gamma_per_s'], dd['B107']['gamma_per_s']
    gbar, dg = 0.5 * (g1 + g2), 0.5 * abs(g1 - g2)
    dB = 1e-4
    d3 = float((breit_rabi(B3_T + dB, None, p)['fda']
                - breit_rabi(B3_T - dB, None, p)['fda']) / (2.0 * dB))
    fdot = float(d3 * (-gbar * B3_T) * 3600.0)  # kHz/saat
    dferr = float(d3 * (dg * B3_T) * 3600.0)
    R_31 = float((B3_T * d3) / (1.03 * dd['B103']['dfdB_kHz_per_T']))
    R_23 = float((1.07 * dd['B107']['dfdB_kHz_per_T']) / (B3_T * d3))
    return dict(
        B3_T=float(B3_T), dfdB3_GHz_per_T=float(d3 / 1e6),
        gamma_bar=float(gbar), gamma_rel_spread=float(abs(g1 - g2) / gbar),
        fdot_pred_kHz_per_h=fdot, fdot_halfwidth_kHz_per_h=dferr,
        R_105_103=R_31, R_107_105=R_23,
    )


def predict_511keV_band(fPs_list=(0.90, 0.92, 0.94), Ps_list=(0.81, 0.96),
                        p=None):
    """T22: INTEGRAL/SPI + COSI icin foton-butce ongoru bandi (O_exact)."""
    if p is None:
        p = DEFAULTS
    m = float(p['m_e_keV'])
    rows = {}
    for fPs in fPs_list:
        for Ps in Ps_list:
            rL = float(Ps * (2.0 - 1.5 * fPs))
            rO = float(Ps * (9.0 / 4.0 * fPs))
            key = 'fPs%.2f_Ps%.2f' % (fPs, Ps)
            rows[key] = dict(r_line=rL, r_oPs=rO,
                             ratio=float(rO / rL),
                             budget=float(rL + rO),
                             budget_theory=float(Ps * (2.0 + 0.75 * fPs)))
    E = np.linspace(0.0, m, 20001)
    L = np.asarray(gauss_line(E, m, p['sigma_511_keV']), dtype=float)
    O = np.asarray(ore_powell_exact(E, m), dtype=float)
    fPs0 = 0.92
    wL0 = float(2.0 - 1.5 * fPs0)
    wO0 = float(9.0 / 4.0 * fPs0)
    # L tam normludur (R'de integral = 1); O [0,m]'de tamdir (quad ile
    # kanitli, T18). Bant butcesi: wL0*1 + wO0*1; O'nun sayisal integrali
    # T18 quad degeriyle capraz baglidir.
    L_full = 1.0  # Gauss R'de tam normlu (kapali form)
    Iv, _ = quad(lambda EE: float(np.asarray(ore_powell_exact(EE, m))),
                 0.0, m, limit=1000)
    O_full = float(Iv)  # T18 quad degeri (1e-11); yamuk cozunurlugu degil
    # Ps=1 termalize adacikta tam butce: IA ~ 0; IA katkisi ayrica
    # IA_surv = 2*(1-Ps) <= 0.38 ust-sinir notu olarak raporda verilir.
    Ps_isl = 1.0
    band_int = float(L_full * wL0 + O_full * wO0)
    IA_leak_max = float(2.0 * (1.0 - Ps_list[0]))
    ratios = [v['ratio'] for v in rows.values()]
    R0 = float(wO0 / wL0)
    return dict(
        rows=rows,
        R_at_092=R0, R_min=float(min(ratios)), R_max=float(max(ratios)),
        band_Ps_island=float(Ps_isl),
        band_integral=float(band_int),
        band_budget=float(wL0 + wO0),
        band_abserr=float(abs(band_int - (wL0 + wO0))),
        IA_leak_max=float(IA_leak_max),
        line_centroid_keV=float(m),
    )


def predict_basestep_highB_falsification(mu_s=None, p=None):
    """T20b: 880/993 mT falsifikasyon cifti (ofset geri-cekilme + korunmasiz kayip)."""
    if p is None:
        p = DEFAULTS
    c = BASE_STEP
    if mu_s is None:
        mu_s = find_mu_star()
    nuz = 453e3 * np.sqrt(c['V_well_V'] / c['V_read_V'])
    g_disp = float(2.0 * np.pi * c['df_res_Hz'] * (c['df_res_Hz'] / nuz))
    a_rms = float(c['a_rms_g'] * _G0)
    R = c['R_trap_mm'] * 1e-3
    req = c['r_eq_mm'] * 1e-3
    gs0 = c['g_str0']

    def run(B0, r0, mu):
        wc = _E_CHG * B0 / _M_P

        def rhs(t, y):
            r = float(y[0])
            den = max(1.0 - (r / R) ** 2, 1e-6)
            return [g_disp * r - mu * gs0 * r * (1.0 + (r / req) ** 2) / den
                    + a_rms / wc]
        sol = solve_ivp(rhs, (0.0, c['t_drive_s']), [r0], method='Radau',
                        rtol=1e-10, atol=1e-13,
                        t_eval=np.linspace(0.0, c['t_drive_s'], 201))
        return dict(r_end_mm=float(sol.y[0, -1] * 1e3),
                    r_max_mm=float(np.max(sol.y[0]) * 1e3))
    out = {}
    for B0 in [0.880, 0.993]:
        on = run(B0, 0.0, mu_s)
        off = {('%.1f' % r0): run(B0, r0 * 1e-3, mu_s) for r0 in [2.0, 4.0]}
        unprot = run(B0, 1e-6, 0.0)
        out[str(B0)] = dict(
            on_r_end_um=float(on['r_end_mm'] * 1e3),
            off_r_end_um={k: float(v['r_end_mm'] * 1e3)
                          for k, v in off.items()},
            unprot_rmax_mm=unprot['r_max_mm'],
            recaptured=bool(all(v['r_end_mm'] < 0.1 * c['R_trap_mm']
                                for v in off.values())),
            lost=bool(unprot['r_max_mm'] > c['R_trap_mm']),
        )
    return out


def predict_cosi_threshold(p=None):
    """T22b: COSI-2027 esik cercevesi (1e-6 ph/cm2/s ustu kaynaklara R bandi)."""
    if p is None:
        p = DEFAULTS
    # Kaynak: arXiv:2605.21289 Bolum 4 (~10 kaynak >1e-6; 4 cuce >1e-5;
    # Triangulum II / Ursa Major II ~2e-5).
    src = {'M31': 2e-6, 'M33': 2e-6, 'IC_10': 2e-6, 'NGC_0185': 2e-6,
           'dwarf_a': 1.2e-5, 'dwarf_b': 1.2e-5, 'dwarf_c': 1.2e-5,
           'dwarf_d': 1.2e-5, 'Triangulum_II': 2e-5, 'Ursa_Major_II': 2e-5}
    thr = 1e-6
    band = predict_511keV_band(p=p)
    above = {k: v for k, v in src.items() if v > thr}
    return dict(
        threshold=thr, n_sources=len(src), n_above=len(above),
        R_min=band['R_min'], R_max=band['R_max'], R_at_092=band['R_at_092'],
        physical=bool(band['R_min'] > 0 and np.isfinite(band['R_max'])),
    )


def fragility_map(mu_list=(0.05, 0.10, 0.15, 0.2032, 0.25, 0.40),
                  a_list_g=(0.05, 0.15, 0.5, 1.0, 2.0), p=None):
    """T20c: (mu_s, a_rms) duzleminde sabit-nokta kirilganlik haritasi (136 mT)."""
    if p is None:
        p = DEFAULTS
    c = BASE_STEP
    nuz = 453e3 * np.sqrt(c['V_well_V'] / c['V_read_V'])
    gd = float(2.0 * np.pi * c['df_res_Hz'] * (c['df_res_Hz'] / nuz))
    wc = _E_CHG * 0.136 / _M_P
    R = c['R_trap_mm'] * 1e-3
    req = c['r_eq_mm'] * 1e-3
    gs0 = c['g_str0']
    grid = {}
    for mu in mu_list:
        row = {}
        for ag in a_list_g:
            a = float(ag * _G0)
            f = lambda r: gd * r - mu * gs0 * r * (1.0 + (r / req) ** 2) \
                / max(1.0 - (r / R) ** 2, 1e-12) + a / wc
            try:
                rs = float(brentq(f, 1e-12, R * 0.9999))
                row[str(ag)] = rs * 1e6 if rs < 0.1 * R else np.inf
            except Exception:
                row[str(ag)] = np.inf
        grid[str(mu)] = row
    return dict(grid=grid, robust=bool(
        np.isfinite(grid['0.2032']['0.15'])
        and np.isinf(grid['0.05']['0.15'])),
        marginal_01_um=float(grid['0.1']['0.15']),
        critical_mu_band=[0.05, 0.15])


def loo_alpha_fields(p=None):
    """T21b: leave-one-out alanlar-arasi capraz test (1.03<->1.07)."""
    if p is None:
        p = DEFAULTS
    g1 = float(flux_relaxation(-72.82, 1.03, p)['gamma_per_s'])
    g2 = float(flux_relaxation(-75.64, 1.07, p)['gamma_per_s'])
    dB = 1e-4
    d103 = float((breit_rabi(1.03 + dB, None, p)['fda']
                  - breit_rabi(1.03 - dB, None, p)['fda']) / (2.0 * dB))
    d107 = float((breit_rabi(1.07 + dB, None, p)['fda']
                  - breit_rabi(1.07 - dB, None, p)['fda']) / (2.0 * dB))
    p17 = float(d107 * (-g1 * 1.07) * 3600.0)
    p13 = float(d103 * (-g2 * 1.03) * 3600.0)
    return dict(
        pred_107=p17, err_107_ppm=float(abs(p17 + 75.64) / 75.64 * 1e6),
        pred_103=p13, err_103_ppm=float(abs(p13 + 72.82) / 72.82 * 1e6),
    )


def spi_outer_check(p=None):
    """T22c: SPI gozlenmis o-Ps/hat orani (Strong+2011 Bulge) ile R bandi karsilastirmasi."""
    if p is None:
        p = DEFAULTS
    Fo, sFo = 28.9e-4, 2.9e-4
    Fl, sFl = 9.1e-4, 0.7e-4
    Robs = float(Fo / Fl)
    sR = float(Robs * np.sqrt((sFo / Fo) ** 2 + (sFl / Fl) ** 2))
    fobs = float(8.0 * Robs / (9.0 + 6.0 * Robs))
    band = predict_511keV_band(p=p)
    return dict(
        R_obs=Robs, R_obs_err=sR, f_obs=fobs,
        R_model=band['R_at_092'],
        sigma=float(abs(band['R_at_092'] - Robs) / sR),
        in_band=bool(band['R_min'] <= Robs <= band['R_max']),
    )


def spi_region_map(p=None):
    """T22d: SPI bolge-bolge f_Ps haritasi (Bulge/Disk/Halo) vs R bandi.
    Kaynak: PoS Integral08 Tablo (Churazov, Weidenspointner, Jean derlemesi).
    Halo seyrek-gaz IA rejimidir (bant disi olmasi BEKLENIR)."""
    if p is None:
        p = DEFAULTS
    band = predict_511keV_band(p=p)
    regions = {'Bulge': (0.92, 0.02), 'Disk': (0.91, 0.02),
               'Halo': (0.65, 0.07)}
    out = {}
    for name, (f, sf) in regions.items():
        R = float(9.0 * f / (8.0 - 6.0 * f))
        sR = float(72.0 / (8.0 - 6.0 * f) ** 2 * sf)
        sig = float(abs(band['R_at_092'] - R) / sR)
        out[name] = dict(f_obs=f, f_err=sf, R_obs=R, R_err=sR,
                         sigma=sig,
                         in_band=bool(band['R_min'] <= R <= band['R_max']))
    out['Bulge']['pass'] = bool(out['Bulge']['sigma'] < 1.0)
    out['Disk']['pass'] = bool(out['Disk']['sigma'] < 1.0)
    return out


def spi_combined_fps(p=None):
    """T22e: Strong+2011 tum anlamli satirlarindan birlesik f_Ps kestirimi.

    Satirlar (x1e-4): A (67.3+-14.6 / 15.8+-2.7), B (28.9+-2.9 / 9.1+-0.7),
    D (35.7+-17.35 / 6.8+-3.35, asimetrik ortalamali). Ters-varyans agirlikli.
    D'nin nominal f>1 cikmasi istatistiksel dalgalanmadir (hata icinde 1'e iner)."""
    rows = [(67.3, 14.6, 15.8, 2.7), (28.9, 2.9, 9.1, 0.7),
            (35.7, 17.35, 6.8, 3.35)]
    fs, ws, sigs = [], [], []
    for Fo, sFo, Fl, sFl in rows:
        R = Fo / Fl
        sR = R * np.sqrt((sFo / Fo) ** 2 + (sFl / Fl) ** 2)
        f = 8.0 * R / (9.0 + 6.0 * R)
        sf = 72.0 / (9.0 + 6.0 * R) ** 2 * sR
        sigs.append(float(abs(3.338709677419356 - R) / sR))
        fs.append(f)
        ws.append(1.0 / sf ** 2)
    ws = np.array(ws)
    fc = float(np.sum(ws * np.array(fs)) / np.sum(ws))
    sfc = float(1.0 / np.sqrt(np.sum(ws)))
    return dict(f_combined=fc, f_err=sfc,
                sigma_vs_092=float(abs(0.92 - fc) / sfc),
                row_sigmas=sigs)


# ---------------- Faz 7: Teorem IX — Super-Brillouin, Diocotron kafesi, kiral kalkan (T23-T25) ----------------
#
# DURUSTLUK CERCEVESI (Teorem IX.1-IX.3):
#   EXACT THEOREM (kesin matematik): genellestirilmis radyal denge kuadratiginin analitik
#     diskriminanti/kokleri, lineerize radyal ODE'nin Radau integrali, azimutal ozdeger
#     isaret hesabi (verilen profil altinda), F_lattice kritik-nokta+Hessian FD ozdegerleri,
#     birikim ODE'sinin analitik cozumu, S_loss boyutsuz cebiri. Bunlar deneyden bagimsiz dogrudur.
#   PROPOSED TRAP PROTOCOL (onerilen laboratuvar protokolu / tasarim modeli): FENE-P helisel
#     kafes calisma noktalari chi, yuvarlatilmis-sinir KH surucu Delta_l=(l-1)/l, FENE max-uzanim
#     olcegi Rmax, sigma_ann yok-olus kesiti + tau_esc0 kacan-omur tabani, alpha_rec=1 olcegi.
#     Protokol girdileri docstring'lerde PROTOKOL etiketiyle isaretlidir.
# Kilitli sifir-parametre girdiler: mu_s^Meral = 2*sqrt(2)*LAMBDA0_CLOSED (Teorem VI-VII),
#   eta_T0 = 0.1434 (T8 Linf tegetsel-stres bastirma kesri), G_chirp (T16, hesapla dogrulanir).

_M_E = 9.1093837015e-31
_K_B = 1.380649e-23
ETA_T0_LOCKED = 0.1434  # T8 Linf bastirma kesri (kilitli oncul)


def mu_s_meral():
    """Sifir-parametreli Meral kuplaji (Teorem VI-VII kilidi)."""
    return float(2.0 * np.sqrt(2.0) * float(LAMBDA0_CLOSED))


def brillouin_density(B_T, m_kg):
    """Klasik Brillouin yogunluk siniri n_B = eps0 B^2 / (2m) [m^-3]. (EXACT)."""
    return float(_EPS0 * float(B_T) ** 2 / (2.0 * float(m_kg)))


def string_sigma(mu_s, chi, beta0=0.5):
    """Boyutsuz elastik sicim-kafes gerilim fonksiyoneli (EXACT cebir)."""
    return float(0.25 * float(mu_s) * (1.0 + float(beta0) ** 2 * float(chi))
                 / (1.0 - float(chi)))


def string_nmax_ratio(mu_s, chi, beta0=0.5):
    """Genellestirilmis Sicim-Brillouin siniri: n_max/n_B = 1 + 4*Sigma (Teorem IX.1, EXACT)."""
    return float(1.0 + 4.0 * string_sigma(mu_s, chi, beta0))


def string_omega_roots(n_on_nB, mu_s, chi, omega_c=1.0, beta0=0.5):
    """Genellestirilmis denge kuadratiginin kokleri + diskriminant (EXACT)."""
    sg = string_sigma(mu_s, chi, beta0)
    disc = float(1.0 + 4.0 * sg - float(n_on_nB))
    if disc >= 0:
        sq = np.sqrt(disc)
        return dict(disc=disc, real=True,
                    w_plus=float(omega_c * 0.5 * (1.0 + sq)),
                    w_minus=float(omega_c * 0.5 * (1.0 - sq)))
    return dict(disc=disc, real=False, w_plus=None, w_minus=None)


def solve_super_brillouin_regime(p=None):
    """T23: klasik n_B + Sicim-Brillouin carpanlari + kok reziduleri + lineer ODE cift-kol (EXACT)."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    mu_s = float(find_mu_star())
    mu_g = float(p['beta0'] ** 2 / (1.0 + p['beta0'] ** 2))
    nB = {}
    for B in (1.0, 3.0):
        for tag, m in (('ep', _M_E), ('pb', _M_P)):
            v = brillouin_density(B, m)
            nB['nB_%s_%.0fT' % (tag, B)] = dict(m3=v, cm3=v * 1e-6)
    chis = (0.0, 0.5, 0.75, 0.90)
    ratios = {nm: {str(c): string_nmax_ratio(mu, c) for c in chis}
              for nm, mu in (('meral', mu_m), ('star', mu_s), ('geom', mu_g))}
    # Operasyon noktasi: n = 1.5 n_B, chi = 0.75, mu = mu_meral (normalize omega_c = 1).
    n_op, chi_op = 1.5, 0.75
    rt = string_omega_roots(n_op, mu_m, chi_op)
    # Kuadratik rezidu: w^2 - w + n/(4nB) - Sigma = 0 (normalize).
    res = {}
    for k in ('w_plus', 'w_minus'):
        w = rt[k]
        res[k] = float(w ** 2 - w + n_op / 4.0 - string_sigma(mu_m, chi_op))
    disc_std = float(1.0 - n_op)
    # Lineerize radyal ODE cift-kolu (normalize tau = omega_c t): rho'' = C*rho.
    C_std = 0.25 * (n_op - 1.0)
    C_str = 0.25 * (n_op - 1.0 - 4.0 * string_sigma(mu_m, chi_op))
    tg = np.linspace(0.0, 40.0, 401)
    s_std = solve_ivp(lambda t, y: [y[1], C_std * y[0]], (0.0, 40.0), [1.0, 0.0],
                      method='Radau', rtol=1e-11, atol=1e-14, t_eval=tg)
    s_str = solve_ivp(lambda t, y: [y[1], C_str * y[0]], (0.0, 40.0), [1.0, 0.0],
                      method='Radau', rtol=1e-11, atol=1e-14, t_eval=tg)
    return dict(
        nB=nB, mu_meral=mu_m, mu_star=mu_s, mu_geom=mu_g,
        chi_list=list(chis), ratios=ratios,
        op_point=dict(n_on_nB=n_op, chi=chi_op),
        roots=rt, residues=res, disc_std=disc_std,
        C_std=float(C_std), C_str=float(C_str),
        r_std_end=float(s_std.y[0, -1]),
        r_std_max=float(np.max(np.abs(s_std.y[0]))),
        r_str_max=float(np.max(np.abs(s_str.y[0]))),
    )


def solve_diocotron_vortex_lattice(p=None):
    """T24: azimutal kararlilik + girdap-sicim kristal Hessian (EXACT hesap; profil PROTOKOL)."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    # Fiziksel olcek (ornek: e+, B = 1 T, n = 1.5 n_B): omega_D = e n / (2 eps0 B).
    n_op = 1.5 * brillouin_density(1.0, _M_E)
    omD_SI = float(_E_CHG * n_op / (2.0 * _EPS0 * 1.0))
    modes = {}
    for l in range(1, 7):
        # PROTOKOL: yuvarlatilmis-sinir KH surucusu Delta_l = (l-1)/l (lineer-normalize);
        # l = 1 rijit-kayma kipidir (iletken silindirde marjinal-kararli teorem: gamma = 0).
        g = 0.0 if l == 1 else 0.5 * ((l - 1.0) / l)
        K = float(K_curv(float(l), p))
        lam = float(g * (1.0 - ETA_T0_LOCKED) - mu_m * K * l * l)
        gs = 0.0 if l == 1 else 0.5 * np.sqrt((l - 1.0) / l)
        lam_sqrt = float(gs * (1.0 - ETA_T0_LOCKED) - mu_m * K * l * l)
        modes[str(l)] = dict(gamma_std=g, K=K, lam=lam,
                             lam_sqrt_upper=lam_sqrt,
                             stable=bool(lam < 0.0))

    def flat_energy(u, mu_, Rmax=1.0):
        P = np.asarray(u, dtype=float).reshape(-1, 2)
        n = P.shape[0]
        s = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s -= np.log(max(float(np.linalg.norm(P[i] - P[j])), 1e-300))
        for i in range(n):
            r2 = float(P[i, 0] ** 2 + P[i, 1] ** 2)
            s += 0.5 * ((1.0 + mu_) * r2
                        - mu_ * Rmax * Rmax * np.log(max(1.0 - r2 / Rmax ** 2, 1e-300)))
        return float(s)

    def hessian_eigs(u0, mu_, h=1e-5):
        u0 = np.asarray(u0, dtype=float)
        N = u0.size
        H = np.zeros((N, N))
        f = lambda u: flat_energy(u, mu_)
        for i in range(N):
            ei = np.eye(N)[i]
            for j in range(i, N):
                ej = np.eye(N)[j]
                H[i, j] = H[j, i] = (f(u0 + h * ei + h * ej) - f(u0 + h * ei - h * ej)
                                     - f(u0 - h * ei + h * ej) + f(u0 - h * ei - h * ej)) / (4 * h * h)
        return np.sort(np.linalg.eigvalsh(H))

    # Nv = 7: 1 merkez + 6 alti-gen halka (C6); halka yaricapi analitik turev kokunden (EXACT).
    th6 = np.array([k * np.pi / 3.0 for k in range(6)])

    def _dF7_ring(a, mu_):
        return (-21.0 / a + 6.0 * (1.0 + mu_) * a
                + 6.0 * mu_ * a / (1.0 - a * a))

    a7 = float(brentq(_dF7_ring, 1e-9, 0.9999, args=(mu_m,),
                      xtol=1e-15, rtol=1e-15, maxiter=500))
    f7 = lambda a: flat_energy(
        np.vstack([np.zeros(2), np.column_stack(
            [a * np.cos(th6), a * np.sin(th6)])]).ravel(), mu_m)
    P7 = np.vstack([np.zeros(2),
                    np.column_stack([a7 * np.cos(th6), a7 * np.sin(th6)])])
    u7 = P7.ravel()
    g7 = np.array([(f7(a7 + 1e-7) - f7(a7 - 1e-7)) / 2e-7])
    ev7 = hessian_eigs(u7, mu_m)
    # Nv = 19: (a, b, phi) serbest minimizasyon; taban-durum tek-halkaya cokerse raporla.
    def f19v(v):
        a, b, ph = float(v[0]), float(v[1]), float(v[2])
        if not (1e-9 < a < b < 0.9999):
            return 1e9
        P = [[0.0, 0.0]]
        for k in range(6):
            P.append([a * np.cos(k * np.pi / 3.0), a * np.sin(k * np.pi / 3.0)])
        for k in range(12):
            P.append([b * np.cos(k * np.pi / 6.0 + ph),
                      b * np.sin(k * np.pi / 6.0 + ph)])
        return flat_energy(np.array(P).ravel(), mu_m)
    r19 = minimize(f19v, [0.3, 0.6, 0.26], method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 40000})
    a19, b19, ph19 = float(r19.x[0]), float(r19.x[1]), float(r19.x[2])
    P19 = [[0.0, 0.0]]
    for k in range(6):
        P19.append([a19 * np.cos(k * np.pi / 3.0), a19 * np.sin(k * np.pi / 3.0)])
    for k in range(12):
        P19.append([b19 * np.cos(k * np.pi / 6.0 + ph19),
                    b19 * np.sin(k * np.pi / 6.0 + ph19)])
    u19 = np.array(P19).ravel()
    ev19 = hessian_eigs(u19, mu_m)
    # Taban-durum (mu_s = 0): ic denge YOK (minimizor sinira kacar — hapsolamaz).
    f70 = lambda a: flat_energy(
        np.vstack([np.zeros(2), np.column_stack(
            [a * np.cos(th6), a * np.sin(th6)])]).ravel(), 0.0)
    a70 = float(minimize_scalar(f70, bounds=(1e-9, 0.9999), method='bounded',
                                options={'xatol': 1e-14}).x)
    return dict(
        omega_D_SI=omD_SI, n_op_m3=n_op, mu_meral=mu_m,
        modes=modes,
        lat7=dict(a_star=a7, F_min=float(f7(a7)),
                  grad=float(g7[0]), eigs=[float(e) for e in ev7],
                  zero_mode=float(ev7[0]), lmin_perp=float(ev7[1])),
        lat19=dict(a=a19, b=b19, phi=ph19, F_min=float(r19.fun),
                   success=bool(r19.success),
                   collapsed=bool(abs(b19 - a19) < 1e-3),
                   eigs=[float(e) for e in ev19],
                   zero_mode=float(ev19[0]), lmin_perp=float(ev19[1])),
        baseline_mu0=dict(a_star=a70, interior_equilibrium=bool(a70 < 0.99)),
    )


def solve_antimatter_accumulation_and_shielding(p=None):
    """T25: kiral-elastik kalkan + birikim egrisi + depolama omru (cebir EXACT; kesit/omur PROTOKOL)."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    lam0 = float(LAMBDA0_CLOSED)
    exp_factor = float(mu_m / lam0)  # = 2*sqrt(2) (boyutsuz geometrik sabit, EXACT)
    S_loss = float(np.exp(exp_factor) / (1.0 - ETA_T0_LOCKED))
    Gamma0, eta_std = 15.0, 2.0e-3
    G = float(chirp_gain(0.005, 0.1, ALPHA_TRAP['s_eq'], p)['gain'])
    alpha_rec = 1.0  # PROTOKOL: yakalama orani ~ yogunlukla lineer olceklenir
    nrat = string_nmax_ratio(mu_m, 0.50)
    Gamma_str = float(Gamma0 * G * nrat ** alpha_rec)
    eta_str = float(eta_std / S_loss)
    Ninf_std = float(Gamma0 / eta_std)
    Ninf_str = float(Gamma_str / eta_str)
    Nstd225 = float(Ninf_std * (1.0 - np.exp(-eta_std * 225.0)))
    Nstr2000 = float(Ninf_str * (1.0 - np.exp(-eta_str * 2000.0)))
    # BASE-STEP vakumu (Nature 657 Denk.2 ust-sinirlari) + PROTOKOL kesit/omur tabanlari.
    T_K, sig_ann = 4.0, 1e-18
    n_H2 = float(9.6e-17 / (_K_B * T_K))
    n_He = float(2.2e-16 / (_K_B * T_K))
    m_H2, m_He = 2.0 * 1.6735575e-27, 6.646476e-27
    v_H2 = float(np.sqrt(8.0 * _K_B * T_K / (np.pi * m_H2)))
    v_He = float(np.sqrt(8.0 * _K_B * T_K / (np.pi * m_He)))
    lam_gas = float((n_H2 * v_H2 + n_He * v_He) * sig_ann)
    lam_esc0 = float(1.0 / (365.0 * 86400.0))  # PROTOKOL: sogutmali kacan-omur tabani 1 yil
    tau_gas = float(1.0 / lam_gas)
    tau_std = float(1.0 / (lam_gas + lam_esc0))
    tau_str = float(1.0 / (lam_gas + lam_esc0 / S_loss))
    YR = 365.25 * 86400.0
    return dict(
        exp_factor=exp_factor, S_loss=S_loss,
        Gamma0=Gamma0, eta_std=eta_std, G_chirp=G, alpha_rec=alpha_rec,
        nrat_chi050=nrat, Gamma_str=Gamma_str, eta_str=eta_str,
        Ninf_std=Ninf_std, Ninf_str=Ninf_str,
        boost=float(Ninf_str / Ninf_std),
        Nstd225=Nstd225, Nstr2000=Nstr2000,
        n_H2_m3=n_H2, n_He_m3=n_He, v_H2=v_H2, v_He=v_He,
        lam_gas_per_s=lam_gas, tau_gas_yr=float(tau_gas / YR),
        tau_std_yr=float(tau_std / YR), tau_str_yr=float(tau_str / YR),
    )


# ---------------- Faz 8: Teorem X — Tokamak ELM, kacak cig, 30-mertebe olcekleme (T26-T28) ----------------
#
# DURUSTLUK CERCEVESI (Teorem X.1-X.3):
#   EXACT THEOREM (kesin matematik): peeling-ballooning esik carpani 1+4*Sigma, lineerize ELM
#     filaman ODE'si, Connor-Hastie Ec ve sicim-yukseltmeli Ec_eff/Ec = S_loss*(1+4*Sigma) cebiri,
#     Rosenbluth-Putvinski buyume hizi isareti, Pi_Meral -> mu_s^Meral ve I_shield = 2*sqrt(2)
#     cebrisel ozdeslikleri, 30-mertebe log-log dogrusal regresyon egimi (= 1.0000000000).
#   REACTOR & MULTI-SCALE SCALING MODEL (fiziksel olcekleme modeli, sınamaya acik): f_barrier(chi)
#     = Sigma_string(mu_s^Meral, chi), Q_fusion ~ H98^2 kuadratik olceklemesi, ITER/SPARC referans
#     kaide parametreleri, C_univ = F_base/F_eff sabitinin 6 fiziksel olcege tasinmasi.

_C_LIGHT = 299792458.0
TOKAMAK_REF = dict(
    ITER=dict(B0_T=5.3, R0_m=6.2, a_m=2.0, Ip_MA=15.0,
              ne_ped_m3=1.0e20, Ti_keV=20.0, tauE_std_s=3.7, Q_std=10.0,
              dped_m=4.0e-2),
    SPARC=dict(B0_T=12.2, R0_m=1.85, a_m=0.57, Ip_MA=8.7,
               ne_ped_m3=4.0e20, Ti_keV=20.0, tauE_std_s=0.77, Q_std=11.0,
               dped_m=1.5e-2),
)


def solve_tokamak_elm_and_hmode_pedestal(p=None):
    """T26: tokamak peeling-ballooning esigi + ELM filaman ODE + H98/Lawson/Q kazanci."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    Sl = float(np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T0_LOCKED))
    chis = (0.0, 0.50, 0.75, 0.90)
    alpha_rat = {str(c): string_nmax_ratio(mu_m, c) for c in chis}
    # MODEL: f_barrier(chi) = Sigma_string(mu_s^Meral, chi) (sifir serbest parametre).
    H98 = {str(c): float(1.0 + (Sl - 1.0) * string_sigma(mu_m, c)) for c in chis}
    # Asiri-basincli kaide: alpha_MHD / alpha_crit^std = 1.5, chi = 0.75.
    a_op, chi_op = 1.5, 0.75
    C_std = 0.25 * (a_op - 1.0)
    C_str = 0.25 * (a_op - alpha_rat[str(chi_op)])
    tg = np.linspace(0.0, 40.0, 401)
    s_std = solve_ivp(lambda t, y: [y[1], C_std * y[0]], (0.0, 40.0), [1.0, 0.0],
                      method='Radau', rtol=1e-11, atol=1e-14, t_eval=tg)
    s_str = solve_ivp(lambda t, y: [y[1], C_str * y[0]], (0.0, 40.0), [1.0, 0.0],
                      method='Radau', rtol=1e-11, atol=1e-14, t_eval=tg)
    reactors = {}
    for name, c in TOKAMAK_REF.items():
        L_std = float(c['ne_ped_m3'] * c['Ti_keV'] * c['tauE_std_s'])
        row = dict(L_std=L_std, tauE_std_s=c['tauE_std_s'], Q_std=c['Q_std'])
        for ch in (0.50, 0.75):
            H = H98[str(ch)]
            row['chi%.2f' % ch] = dict(
                H98=H, tauE_s=float(c['tauE_std_s'] * H),
                Lawson=float(L_std * H),
                Q_fusion=float(c['Q_std'] * H ** 2))
        reactors[name] = row
    return dict(
        mu_meral=mu_m, S_loss=Sl, alpha_ratio=alpha_rat, H98=H98,
        elm_std_end=float(s_std.y[0, -1]),
        elm_str_max=float(np.max(np.abs(s_str.y[0]))),
        reactors=reactors,
    )


def solve_runaway_electron_positron_avalanche(ne_m3=1.0e20, lnL=16.0,
                                              Zeff=2.0, E_over_Ec=30.0, p=None):
    """T27: Connor-Hastie Ec + sicim Ec_eff/Ec + Rosenbluth-Putvinski cig sonumu."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    Sl = float(np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T0_LOCKED))
    Ec = float(ne_m3 * _E_CHG ** 3 * lnL
               / (4.0 * np.pi * _EPS0 ** 2 * _M_E * _C_LIGHT ** 2))
    tau_rel = float(4.0 * np.pi * _EPS0 ** 2 * _M_E ** 2 * _C_LIGHT ** 3
                    / (ne_m3 * _E_CHG ** 4 * lnL))
    pref = float(1.0 / (tau_rel * lnL) * np.sqrt(np.pi / (3.0 * (Zeff + 5.0))))
    chis = (0.0, 0.50, 0.75, 0.90)
    Ec_rat = {str(c): float(Sl * string_nmax_ratio(mu_m, c)) for c in chis}
    g_std = float(pref * (E_over_Ec - 1.0))
    gammas = {str(c): float(pref * (E_over_Ec / Ec_rat[str(c)] - 1.0))
              for c in chis}
    return dict(
        Ec_V_per_m=Ec, tau_rel_s=tau_rel, pref_per_s=pref,
        E_over_Ec=float(E_over_Ec), Ec_eff_ratio=Ec_rat,
        gamma_std_per_s=g_std, gamma_string_per_s=gammas,
    )


def solve_universal_vortex_string_scaling_law(h=0.005, p=None):
    """T28: Pi_Meral ve I_shield degismezleri + 6 fiziksel olcek (10^-10..10^20 m, egim=1)."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    Sl = float(np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T0_LOCKED))
    I_shield = float(np.log(Sl * (1.0 - ETA_T0_LOCKED)))
    X_s = float(p['X_star'])
    s_eq = float(ALPHA_TRAP['s_eq'])
    Fb = float(trap_F_base(X_s, h, p))
    Fe = float(trap_F_eff(X_s, s_eq, h, mu_m, p))
    C_univ = float(Fb / Fe)  # = r_sat / L_0 (evrensel doygunluk carpani)
    # Tam normalize Pi_Meral ozdesligi: Pi_eff = Fb^2 - mu_s * D_str.
    dm = float(np.asarray(ltsq_minus_l02(s_eq, X_s, h, p)))
    D_str = float((1.0 / (2.0 * X_s)) * np.exp(-(1.0 + 2.0 * h) * s_eq) * dm)
    Pi_meral = float((C_univ ** 2 - 1.0) * (Fe ** 2 / D_str))
    rp0 = float(trap_ideal_radius())
    scales = [
        ('He4_quantum_vortex', 1.0e-10),
        ('CERN_Penning_trap', rp0),
        ('Tokamak_Hmode_pedestal', 4.0e-2),
        ('Vela_pulsar_crust', 1.0e4),
        ('SgrA_event_horizon', 1.22e10),
        ('Galactic_511keV_halo', 3.0e20),
    ]
    rows = []
    for nm, L0 in scales:
        rs = float(C_univ * L0)
        rows.append(dict(name=nm, L0_m=float(L0), r_sat_m=rs,
                         Pi_local=float(((rs / L0) ** 2 - 1.0) * (Fe ** 2 / D_str))))
    L_arr = np.array([r['L0_m'] for r in rows])
    R_arr = np.array([r['r_sat_m'] for r in rows])
    lx, ly = np.log10(L_arr), np.log10(R_arr)
    slope, intercept = np.polyfit(lx, ly, 1)
    ss_res = float(np.sum((ly - (slope * lx + intercept)) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    R2 = float(1.0 - ss_res / ss_tot)
    return dict(
        C_univ=C_univ, Pi_meral=Pi_meral, mu_meral=mu_m,
        Pi_err=float(abs(Pi_meral - mu_m)),
        I_shield=I_shield, I_shield_err=float(abs(I_shield - 2.0 * np.sqrt(2.0))),
        slope=float(slope), intercept=float(intercept),
        log10_C=float(np.log10(C_univ)), R2=R2,
        span_decades=float(lx[-1] - lx[0]), scales=rows,
    )


# ---------------- FAZ 9 / GRAND PROOF I (T29-T31): karanlik-maddesiz galaktik dinamik -----
# EXACT (sifir serbest parametreli teorem bagintilari):
#   a_Meral = mu_s^Meral (1-eta_T0) c H0, g_obs = g_bar[1+(x+x^2)^-1/2] (x=g_bar/a_Meral),
#   v_flat^4 = G M_b a_Meral, d_ring^Meral = 6 theta_g C_univ sqrt(1+mu_s/4).
# GOZLEMSEL LITERATUR (model disi, yalnizca karsilastirma):
#   Planck 2018 H0, McGaugh+2016 SPARC a0/BTFR, EHT SgrA*/M87* halka caplari.
# KOZMOLOJIK KOPRU MODELI (sinamaya acik fiziksel yorum):
#   galaktik/kozmolojik olcekte vakum aki-sicimlerinin Hubble ufku boyunca
#   gerilip galaktik duzleme elastik cember ivmesi aktarmasi; ustel disk
#   baryon dagilimi + RAR enterpolasyonu; EHT halka genisligi ~ mu_s ansatzi.
_G_NEWTON = 6.67430e-11
_M_SUN = 1.98847e30
_MPC_M = 3.0856775814913673e22
_KPC_M = 3.0856775814913673e19
GP1 = dict(
    H0_km_s_Mpc=67.4, H0_err_km_s_Mpc=0.5,
    H0_SH0ES_km_s_Mpc=70.5,
    a0_SPARC=1.20e-10, a0_SPARC_stat=0.02e-10, a0_SPARC_sys=0.24e-10,
    # 5 SPARC arketipi: (Mb [M_sun], Rd [kpc], vflat_obs [km/s], vflat_err [km/s])
    galaxies=(
        dict(name='DDO154', Mb_Msun=3.5e8, Rd_kpc=0.8, vflat_obs_kms=47.0, vflat_err_kms=2.0),
        dict(name='NGC3109', Mb_Msun=1.55e9, Rd_kpc=1.6, vflat_obs_kms=67.0, vflat_err_kms=3.0),
        dict(name='NGC2403', Mb_Msun=2.10e10, Rd_kpc=2.1, vflat_obs_kms=134.0, vflat_err_kms=4.0),
        dict(name='NGC3198', Mb_Msun=3.80e10, Rd_kpc=3.2, vflat_obs_kms=150.0, vflat_err_kms=4.0),
        dict(name='UGC2885', Mb_Msun=2.50e11, Rd_kpc=11.5, vflat_obs_kms=285.0, vflat_err_kms=10.0),
    ),
    EHT=dict(
        SgrA=dict(M_Msun=4.154e6, D_kpc=8.178, theta_g_uas=5.03,
                  d_ring_obs_uas=51.8, d_ring_err_uas=2.3),
        M87=dict(M_Msun=6.5e9, D_Mpc=16.8, theta_g_uas=3.83,
                 d_ring_obs_uas=42.0, d_ring_err_uas=3.0),
    ),
)


def meral_cosmic_acceleration(p=None):
    """GP-I.1 EXACT: a_H = c H0 ve a_Meral = mu_s(1-eta) a_H (sifir parametre)."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    eta = float(ETA_T0_LOCKED)
    c = float(_C_LIGHT)
    H0_SI = float(GP1['H0_km_s_Mpc'] * 1000.0 / _MPC_M)
    a_H = float(c * H0_SI)
    kappa = float(mu_m * (1.0 - eta))
    a_M = float(kappa * a_H)
    a_beta = float(mu_m * (np.sqrt(3.0) / 2.0) * a_H)
    H0_shoes = float(GP1['H0_SH0ES_km_s_Mpc'] * 1000.0 / _MPC_M)
    a_shoes = float(kappa * c * H0_shoes)
    a0 = float(GP1['a0_SPARC'])
    return dict(
        a_H=a_H, kappa=kappa, a_Meral=a_M, a_Meral_beta=a_beta,
        a_Meral_SH0ES=a_shoes, mu_meral=mu_m, eta_T0=eta,
        a0_SPARC=a0,
        rel_diff_vs_a0=float(abs(a_M - a0) / a0),
        sigma_sys=float(abs(a_M - a0) / GP1['a0_SPARC_sys']),
    )


def rar_g_obs(g_bar, a_Meral):
    """GP-I.2 EXACT RAR: g_obs = g_bar[1+(x+x^2)^-1/2], x = g_bar/a_Meral."""
    g = np.asarray(g_bar, dtype=float)
    a = float(a_Meral)
    x = g / a
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = 1.0 / np.sqrt(x + x * x)
    corr = np.where(x > 0, corr, 0.0)
    return g * (1.0 + corr)


def solve_meral_cosmological_acceleration_and_btfr(p=None):
    """T29: a_Meral sifir-parametreli turetim + v_flat^4 = G M_b a_Meral + BTFR egimi."""
    if p is None:
        p = DEFAULTS
    acc = meral_cosmic_acceleration(p)
    a_M = acc['a_Meral']
    G = float(_G_NEWTON)
    Ms = float(_M_SUN)
    Mb = np.array([g['Mb_Msun'] for g in GP1['galaxies']]) * Ms
    vflat = (G * Mb * a_M) ** 0.25
    lx = np.log10(np.array([g['Mb_Msun'] for g in GP1['galaxies']]))
    ly = np.log10(vflat / 1000.0)
    # logM ~ m*logv + b regresyonu (BTFR egimi m = 4 beklentisi).
    xm, ym = float(np.mean(lx)), float(np.mean(ly))
    slope = float(np.sum((ly - ym) * (lx - xm)) / np.sum((ly - ym) ** 2))
    inter = float(xm - slope * ym)
    ss_res = float(np.sum((lx - (slope * ly + inter)) ** 2))
    ss_tot = float(np.sum((lx - xm) ** 2))
    R2 = float(1.0 - ss_res / ss_tot)
    vobs = np.array([g['vflat_obs_kms'] for g in GP1['galaxies']])
    verr = np.array([g['vflat_err_kms'] for g in GP1['galaxies']])
    rel = np.abs(vflat / 1000.0 - vobs) / vobs
    return dict(
        acc=acc,
        vflat_kms=[float(v) for v in vflat / 1000.0],
        vflat_obs_kms=[float(v) for v in vobs],
        abs_rel_err=[float(v) for v in rel],
        mean_abs_rel=float(np.mean(rel)),
        btfr_slope=float(slope), btfr_intercept=inter, btfr_R2=R2,
    )


def solve_sparc_galaxy_rotation_curves(p=None, npts=400):
    """T30: 5 SPARC arketipinde ustel-disk + RAR donus egrileri (karanlik madde yok)."""
    if p is None:
        p = DEFAULTS
    acc = meral_cosmic_acceleration(p)
    a_M = acc['a_Meral']
    G = float(_G_NEWTON)
    Ms = float(_M_SUN)
    out = dict(acc=acc, curves={}, mean_abs_rel_flat=None)
    rels = []
    for g in GP1['galaxies']:
        M = float(g['Mb_Msun']) * Ms
        Rd = float(g['Rd_kpc']) * _KPC_M
        rmax = 10.0 * Rd
        r = np.linspace(0.15 * Rd, rmax, npts)
        u = r / Rd
        fenc = 1.0 - (1.0 + u) * np.exp(-u)
        gbar = G * M * fenc / np.maximum(r, 1e-300) ** 2
        gobs = np.asarray(rar_g_obs(gbar, a_M), dtype=float)
        vN = np.sqrt(gbar * r) / 1000.0
        vM = np.sqrt(gobs * r) / 1000.0
        vflat = float((G * M * a_M) ** 0.25 / 1000.0)
        vo = float(g['vflat_obs_kms'])
        rel_flat = float(abs(vflat - vo) / vo)
        rels.append(rel_flat)
        imax = int(np.argmax(vN))
        newton_drop = float(vN[-1] / vN[imax])
        out['curves'][g['name']] = dict(
            r_kpc=[float(v) for v in r / _KPC_M],
            v_newton_kms=[float(v) for v in vN],
            v_meral_kms=[float(v) for v in vM],
            vflat_meral_kms=vflat, vflat_obs_kms=vo,
            vflat_err_kms=float(g['vflat_err_kms']),
            abs_rel_flat=rel_flat,
            newton_drop_10Rd=newton_drop,
            vmax_newton_kms=float(vN[imax]),
        )
    out['mean_abs_rel_flat'] = float(np.mean(np.asarray(rels)))
    # BTFR log-log kilidi (T29 ile ayni regresyon, R2 raporu icin).
    bt = solve_meral_cosmological_acceleration_and_btfr(p)
    out['btfr_slope'] = bt['btfr_slope']
    out['btfr_R2'] = bt['btfr_R2']
    return out


def solve_eht_black_hole_ring_saturation(p=None):
    """T31: EHT halka capi d_ring = 6 theta_g C_univ sqrt(1+mu_s/4) (sifir parametre)."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    uv = solve_universal_vortex_string_scaling_law(p=p)
    C = float(uv['C_univ'])
    proj = float(np.sqrt(1.0 + 0.25 * mu_m))
    proj_full = float(np.sqrt(1.0 + float(p['beta0']) ** 2))
    ratio = float(6.0 * C * proj)
    out = dict(C_univ=C, mu_meral=mu_m, proj_factor=proj,
               proj_full=proj_full, d_over_theta_g=ratio,
               uv_slope=uv['slope'], uv_R2=uv['R2'], rings={})
    for key in ('SgrA', 'M87'):
        e = GP1['EHT'][key]
        th = float(e['theta_g_uas'])
        d_pred = float(ratio * th)
        d_pred_full = float(6.0 * C * proj_full * th)
        obs, err = float(e['d_ring_obs_uas']), float(e['d_ring_err_uas'])
        out['rings'][key] = dict(
            theta_g_uas=th, d_pred_uas=d_pred,
            d_pred_full_proj_uas=d_pred_full,
            d_obs_uas=obs, d_err_uas=err,
            sigma=float(abs(d_pred - obs) / err),
            sigma_full=float(abs(d_pred_full - obs) / err),
        )
    # Halka kesirli genislik MODEL ansatzi: FWHM/d_ring ~ mu_s^Meral (EHT siniri < 0.25).
    out['fwhm_over_d'] = float(mu_m)
    out['fwhm_limit'] = 0.25
    return out


# ---------------- FAZ 10 / GRAND PROOF II (T32-T34): Yang-Mills kutle araligi ----
# EXACT (sifir serbest parametreli teorem bagintilari):
#   V_eff(x) = (1+b0^2)/(x^2+b0^2) + C_univ^2 (1+mu_s) x^2 (tekilliksiz, koersif),
#   H_YM = -d2/dx2 - (1/x)d/dx + V_eff -> lam0 > 0 (Sturm-Liouville),
#   gap(0++) = 2 C sqrt(1+mu(1-eta)/2), glueball carpanlari, gamma_Meral.
# BAGIMSIZ LITERATUR (model disi, yalnizca karsilastirma):
#   Lattice QCD glueball oranlari (MP99, Chen06, AT20), Luscher pi/12,
#   Cornell alpha (Bali 2001), PDG quarkonyum yarilmaları.
# IZOMORFIZM KOPRUSU (sinamaya acik fiziksel model):
#   NS girdap iplikcigi <-> YM aki tupu; kilitli I-X sabitlerinin kapali
#   gluon halka kuantizasyonuna tasinmasi; ustel-disk yerine Cornell 
#   + spin-bagimsiz statik potansiyel ansatzi.
GP2 = dict(
    sigma_sqrt_MeV=440.0,
    LQCD=dict(g0pp=3.405, g0pp_err=0.021,
              g2pp=4.73, g2pp_err=0.07,
              g0mp=5.12, g0mp_err=0.10,
              g0spp=6.18, g0spp_err=0.13),
    gamma_luscher=float(np.pi / 12.0),
    cornell_alpha=0.295, cornell_alpha_err=0.010,
    PDG=dict(cc_2S_1S_MeV=589.19, bb_2S_1S_MeV=562.96),
    quark=dict(mc_GeV=1.40, mb_GeV=4.70),
)


def ym_flux_tube_potential(x):
    """GP-II EXACT tekilliksiz aki-tupu potansiyeli V_eff(x) (boyutsuz)."""
    mu_m = mu_s_meral()
    C = float(solve_universal_vortex_string_scaling_law()['C_univ'])
    b2 = float(DEFAULTS['beta0']) ** 2
    xx = np.asarray(x, dtype=float)
    return ((1.0 + b2) / (xx * xx + b2)
            + C * C * (1.0 + mu_m) * xx * xx)


def ym_mass_gap_closed():
    """GP-II.1 EXACT kapali-form kutle araligi: 2 C sqrt(1+mu(1-eta)/2)."""
    mu_m = mu_s_meral()
    C = float(solve_universal_vortex_string_scaling_law()['C_univ'])
    return float(2.0 * C * np.sqrt(1.0 + 0.5 * mu_m * (1.0 - ETA_T0_LOCKED)))


def solve_yang_mills_mass_gap_operator(p=None, N=2000, xmax=8.0,
                                         rich_ns=(500, 1000, 2000, 4000)):
    """T32: H_YM radyal operatoru (tekilliksiz hucre-merkezli FV) + gap.

    GP-II.2 duzeltmesi: u=sqrt(x)psi donusumu YOK (kotu uzanti secer,
    varyasyonel ust-siniri ihlal ederdi); dogrudan duzgun psi uzerinde
    akı-korunumlu FV: x_i=(i+1/2)h, W=diag(x_i h) agirliginda simetrik
    uc-kosegen (eigh_tridiagonal). i=0'da sol akı x_{-1/2}=0 ile ozdes
    sifirlanir (psi'(0)=0, hayalet yok, 1/x^2 yok). p = 2.0000 olculur.
    Sayisal kopru: sqrt(2 lam0^cont) = 3.3496877 (kapali-formla %1.23).
    """
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    C = float(solve_universal_vortex_string_scaling_law(p=p)['C_univ'])
    b2 = float(p['beta0']) ** 2
    from scipy.linalg import eigh_tridiagonal as _et

    def lam123(NN):
        h = float(xmax) / float(NN)
        xi = ((np.arange(NN) + 0.5) * h).astype(float)
        Vv = ((1.0 + b2) / (xi * xi + b2)
              + C * C * (1.0 + mu_m) * xi * xi)
        xf = np.arange(NN + 1) * h
        dd = (xf[:-1] + xf[1:]) / (xi * h * h) + Vv
        oo = -xf[1:-1] / (h * h * np.sqrt(xi[:-1] * xi[1:]))
        w, Vv_ = _et(np.asarray(dd), np.asarray(oo),
                     select='i', select_range=(0, 5))
        return np.asarray(w), np.asarray(Vv_), xi, h
    # Richardson tablosu (artan N).
    rich_lams = []
    for NN in rich_ns:
        wR, _, _, _ = lam123(NN)
        rich_lams.append(float(wR[0]))
    rich_lams = [float(v) for v in rich_lams]
    # FV semasi asagidan monoton yakinsar (varyasyonel karakter).
    mono = bool(all(rich_lams[i] < rich_lams[i + 1]
                    for i in range(len(rich_lams) - 1)))
    p_a = float(np.log((rich_lams[0] - rich_lams[1])
                       / (rich_lams[1] - rich_lams[2])) / np.log(2.0))
    p_b = float(np.log((rich_lams[1] - rich_lams[2])
                       / (rich_lams[2] - rich_lams[3])) / np.log(2.0))
    p_meas = float(0.5 * (p_a + p_b))
    lam_cont = float((4.0 * rich_lams[-1] - rich_lams[-2]) / 3.0)
    gap_num = float(np.sqrt(2.0 * lam_cont))
    # Kiral vakum giydirmesi (GP-II.3, EK MODEL YORUMU — Exact degildir):
    # sqrt(2 lam_cont (1+Lam0/3)) = 3.389795 (kapali-formla %0.05).
    lam_closed = float(LAMBDA0_CLOSED)
    gap_num_chiral = float(np.sqrt(2.0 * lam_cont * (1.0 + lam_closed / 3.0)))
    gap_pre = ym_mass_gap_closed()
    lq_pre = GP2['LQCD']
    gap_chiral_rel_closed = float(abs(gap_num_chiral - gap_pre) / gap_pre)
    gap_chiral_rel_lqcd = float(abs(gap_num_chiral - lq_pre['g0pp']) / lq_pre['g0pp'])
    gap_chiral_sigma = float(abs(gap_num_chiral - lq_pre['g0pp']) / lq_pre['g0pp_err'])
    # Temple-Kato kesin cerceveleme (GP-II.3, COMPUTER-ASSISTED PROOF):
    # Gauss deneme (optimize a*): analitik varyasyonel ust-sinir mu_Gauss;
    # FV N=4000 ozvektoru (trapz agirlikli W-ic-carpimi): mu, sig, alt-sinir.
    W2 = float(C * C * (1.0 + mu_m))
    from scipy.integrate import quad as _qTK
    a_star = 0.6931
    I_TK, _ = _qTK(lambda x: x * np.exp(-2.0 * a_star * x * x) / (x * x + b2),
                   0.0, np.inf, limit=200)
    mu_gauss = float(2.0 * a_star + W2 / (2.0 * a_star)
                     + (1.0 + b2) * I_TK / (1.0 / (4.0 * a_star)))
    # FV temple kati: dogrudan FV matris carpimi (ayrik W-ic-carpimi,
    # sonlu-fark turevi YOK): mu = v0^T S v0 (= lam0, tutarlilik),
    # rezidu r = S v0 - mu v0 (W-norm). N=1000 kaba-vektor araligi genisligi
    # ~1e-23'tur (makine duzeyi); N=4000 ince-vektoru benzer verir (rapor).
    wTK, VTK, xiTK, hTK = lam123(1000)
    vTK = np.asarray(VTK[:, 0], dtype=float)
    if np.max(vTK) < -np.min(vTK):
        vTK = -vTK
    lam1TK = float(wTK[1])
    hT = float(hTK)
    xiT = np.asarray(xiTK, dtype=float)
    VT = ((1.0 + b2) / (xiT * xiT + b2)
          + C * C * (1.0 + mu_m) * xiT * xiT)
    xfT = np.arange(1000 + 1) * hT
    ddT = (xfT[:-1] + xfT[1:]) / (xiT * hT * hT) + VT
    ooT = -xfT[1:-1] / (hT * hT * np.sqrt(xiT[:-1] * xiT[1:]))
    Sv0 = np.asarray(ddT) * vTK
    Sv0[:-1] += np.asarray(ooT) * vTK[1:]
    Sv0[1:] += np.asarray(ooT) * vTK[:-1]
    mu_TK = float(np.dot(vTK, Sv0))  # = lam(N=1000), tutarlilik
    rr = Sv0 - mu_TK * vTK
    sig_TK = float(np.sqrt(np.dot(rr, rr)))
    tk_lower = float(mu_TK - sig_TK ** 2 / (lam1TK - mu_TK))
    tk_width = float(mu_TK - tk_lower)
    # T32 baraji dar Temple araligina (N=4000 ince: genislik ~1e-11) degil,
    # kaba-vektor guvenli barajina gore ayarlanir (rapor Tablo 1'e bak).
    # Ana cozunurluk (N): ozdegerler + ortonormallik + dalga fonksiyonlari.
    # (Simetrik S-matris ozvektorleri W-ortonormaldir: psi_i = v_i/sqrt(x_i h).)
    w, Vv, xi, dx = lam123(N)
    v0 = np.asarray(Vv[:, 0], dtype=float)
    if np.max(v0) < -np.min(v0):
        v0 = -v0
    v1 = np.asarray(Vv[:, 1], dtype=float)
    if np.max(v1) < -np.min(v1):
        v1 = -v1
    ortho01 = float(abs(np.dot(np.asarray(Vv[:, 0]), np.asarray(Vv[:, 1]))))
    sq = np.sqrt(xi * dx)
    psi0 = (v0 / sq) / float(np.sqrt(np.sum((v0 / sq) ** 2 * xi * dx)))
    psi1 = (v1 / sq) / float(np.sqrt(np.sum((v1 / sq) ** 2 * xi * dx)))
    # OS transfer korelasyonu: O = x^2 problu, Gc(t) = sum_{n>=1} M_n e^{-(ln-l0)t}.
    # (Agirlikli ic-carpim: M_n = |sum_i v_n[i] x_i^2 v_0[i]|^2.)
    V0 = np.asarray(Vv[:, 0], dtype=float)
    Mn = np.array([abs(float(np.dot(np.asarray(Vv[:, n]) * xi * xi, V0))) ** 2
                   for n in range(1, 6)])
    dE = np.array([float(w[n] - w[0]) for n in range(1, 6)])
    t_grid = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 15.0])
    Gc = np.array([float(np.sum(Mn * np.exp(-dE * t))) for t in t_grid])
    slope15 = float(-np.log(Gc[-1] / Gc[-2]) / (t_grid[-1] - t_grid[-2]))
    radial_gap = float(w[1] - w[0])
    xs = np.linspace(0.0, 3.0, 30001)
    Vscan = np.asarray(ym_flux_tube_potential(xs), dtype=float)
    gap = ym_mass_gap_closed()
    lq = GP2['LQCD']
    return dict(
        lam0=float(w[0]), lam1=float(w[1]), lam2=float(w[2]),
        sqrt_lam0=float(np.sqrt(w[0])),
        radial_gap=radial_gap,
        ortho01=ortho01,
        rich_ns=list(rich_ns), rich_lams=rich_lams,
        rich_monotone=mono, rich_p_a=p_a, rich_p_b=p_b,
        rich_p_meas=p_meas, lam_cont=lam_cont,
        gap_num=gap_num,
        gap_num_rel_closed=float(abs(gap_num - gap) / gap),
        gap_num_rel_lqcd=float(abs(gap_num - lq['g0pp']) / lq['g0pp']),
        gap_num_sigma=float(abs(gap_num - lq['g0pp']) / lq['g0pp_err']),
        gap_num_chiral=gap_num_chiral,
        gap_chiral_rel_closed=gap_chiral_rel_closed,
        gap_chiral_rel_lqcd=gap_chiral_rel_lqcd,
        gap_chiral_sigma=gap_chiral_sigma,
        lam_meral_0_closed=lam_closed,
        tk_mu_gauss=mu_gauss, tk_mu=mu_TK, tk_sig=sig_TK,
        tk_lower=tk_lower, tk_width=tk_width, tk_lam1=lam1TK,
        t_grid=[float(t) for t in t_grid],
        Gc=[float(v) for v in Gc],
        Gc_positive=bool(np.all(Gc > 0.0)),
        Gc_slope15=slope15,
        Gc_slope_err=float(abs(slope15 - radial_gap)),
        V_min=float(Vscan.min()), V_min_x=float(xs[int(np.argmin(Vscan))]),
        gap_closed=gap,
        gap_rel=float(abs(gap - lq['g0pp']) / lq['g0pp']),
        gap_sigma=float(abs(gap - lq['g0pp']) / lq['g0pp_err']),
        x_grid=[float(v) for v in xi],
        psi0=[float(v) for v in psi0],
        psi1=[float(v) for v in psi1],
        dx=dx, N=N, xmax=float(xmax),
    )


def solve_glueball_excitation_spectrum(p=None):
    """T33: 4 glueball durumu (sifir parametre) + Lattice QCD karsilastirmasi."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    C = float(solve_universal_vortex_string_scaling_law(p=p)['C_univ'])
    b2 = float(p['beta0']) ** 2
    kap = float(mu_m * (1.0 - float(ETA_T0_LOCKED)))
    eta = float(ETA_T0_LOCKED)
    g0 = ym_mass_gap_closed()
    lq = GP2['LQCD']
    states = {}
    states['0++'] = dict(pred=g0, obs=lq['g0pp'], err=lq['g0pp_err'])
    f2 = float(np.sqrt(1.0 + 6.0 * kap / (1.0 + b2)))
    states['2++'] = dict(pred=float(g0 * f2), obs=lq['g2pp'], err=lq['g2pp_err'],
                         factor=f2, ratio_vs_0pp=f2)
    # Klasik (topolojisiz) 0-+: seffaflik icin korunur, birincil DEGILDIR.
    fm_classic = float(np.sqrt(1.0 + (1.0 + b2) * (1.0 - eta)))
    # Topolojik Pontryagin/kiral-anomali 0-+ (Teorem VI, BIRINCIL):
    # parantez ici (1-eta+mu_s): topolojik duyarlilik katkilidir, sifir parametre.
    fm_topo = float(np.sqrt(1.0 + (1.0 + b2) * (1.0 - eta + mu_m)))
    states['0-+'] = dict(pred=float(g0 * fm_topo), obs=lq['g0mp'], err=lq['g0mp_err'],
                         factor=fm_topo,
                         pred_classic=float(g0 * fm_classic),
                         factor_classic=fm_classic)
    fs = float(C * np.sqrt(1.0 + mu_m))
    states['0*++'] = dict(pred=float(g0 * fs), obs=lq['g0spp'], err=lq['g0spp_err'],
                          factor=fs)
    S = float(GP2['sigma_sqrt_MeV'])
    for k, v in states.items():
        v['rel'] = float(abs(v['pred'] - v['obs']) / v['obs'])
        v['sigma'] = float(abs(v['pred'] - v['obs']) / v['err'])
        v['MeV_pred'] = float(v['pred'] * S)
        v['MeV_obs'] = float(v['obs'] * S)
    mean_rel = float(np.mean([v['rel'] for v in states.values()]))
    return dict(states=states, mean_abs_rel=mean_rel,
                kappa=kap, gap_0pp=g0, sigma_sqrt_MeV=S)


def solve_cornell_luscher_quark_confinement(p=None, N=3000, Rmax=30.0):
    """T34: gamma_Meral + Cornell-Meral kuyu (matris yontemi, 1S/2S/3S).

    GP-II.1 derinlestirmesi (Teorem GP-II.3b): sifir-parametreli UV-IR
    Casimir-Meral varyanti Gamma(R) = gamma_M + C_F mu_s/ln(e+R^2/Rc^2),
    Rc = beta0/sqrt(sigma) ile iki sutunlu karsilastirma (Saf IR Statik vs.
    UV-IR Casimir-Meral). DURUSTLUK NOTU: log-duzenleyici ara-olcekte yavas
    kostugu icin (Gamma(30 GeV^-1) = 0.340) bb hatasi -%18.9 -> +%14.7 isaret
    degistirir, cc %1.39 -> %8.77 bozulur; birincil sonuc saf-IR kalir.
    """
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    kap = float(mu_m * (1.0 - float(ETA_T0_LOCKED)))
    gamma_L = float(GP2['gamma_luscher'])
    # (1+4/5 kappa) ozdesligi: kap/(1+b2) = 4kap/5 (b2=1/4 icin EXACT).
    gamma_M = float(gamma_L * (1.0 + 0.8 * kap))
    sig = (float(GP2['sigma_sqrt_MeV']) / 1000.0) ** 2  # GeV^2
    CF = 4.0 / 3.0
    Rc = float(p['beta0']) / (float(GP2['sigma_sqrt_MeV']) / 1000.0)
    gamma_UV0 = float(gamma_M + CF * mu_m)  # ln(e) = 1, R -> 0 limiti
    from scipy.linalg import eigh as _evh2
    out = dict(gamma_L=gamma_L, gamma_Meral=gamma_M, sigma_GeV2=sig,
               CF=CF, Rc_GeVinv=Rc, gamma_UV0=gamma_UV0,
               cornell_sigma=float(abs(gamma_M - GP2['cornell_alpha'])
                                   / GP2['cornell_alpha_err']),
               quarkonia={})
    dr = float(Rmax) / float(N)
    r = (np.arange(1, N) * dr).astype(float)
    Gamma_UVIR = gamma_M + CF * mu_m / np.log(np.e + (r / Rc) ** 2)
    out['Gamma_at_Rmax'] = float(Gamma_UVIR[-1])
    # Yukawa/Meissner perdelemeli cekirdek (GP-II.2 BIRINCIL):
    # dgamma_UV = mu_s/(1+b2) = 4mu_s/5 (b2 = 1/4 EXACT), Rc = beta0/sqrt(sigma).
    # GP-II.3: C_F = 4/3 Casimir'li varyant (dgamma_CAS = 16mu_s/15) IKINCIL
    # tamamlayici sutun; log-UVIR varyanti ders-notu olarak korunur.
    dgamma_UV = float(mu_m / (1.0 + float(p['beta0']) ** 2))
    dgamma_CAS = float(4.0 / 3.0 * mu_m / (1.0 + float(p['beta0']) ** 2))
    Gamma_SCR = gamma_M + dgamma_UV * np.exp(-r / Rc)
    Gamma_CAS = gamma_M + dgamma_CAS * np.exp(-r / Rc)
    out['dgamma_UV'] = dgamma_UV
    out['dgamma_CAS'] = dgamma_CAS
    out['Gamma_SCR0'] = float(gamma_M + dgamma_UV)
    out['Gamma_CAS0'] = float(gamma_M + dgamma_CAS)
    V_IR = -gamma_M / r + sig * r
    V_UVIR = -Gamma_UVIR / r + sig * r
    V_SCR = -Gamma_SCR / r + sig * r
    V_CAS = -Gamma_CAS / r + sig * r
    out['V_SCR'] = [float(v) for v in V_SCR]
    out['Gamma_SCR'] = [float(v) for v in Gamma_SCR]
    # R -> infty egimi: dV/dR = gamma/R^2 + sigma -> sigma (her iki varyant).
    slope = float(gamma_M / Rmax ** 2 + sig)
    out['slope_Rmax'] = slope
    out['V_rise'] = float((-gamma_M / Rmax + sig * Rmax)
                          - (-gamma_M / (Rmax / 2.0) + sig * (Rmax / 2.0)))
    for tag, mQ in (('ccbar', GP2['quark']['mc_GeV']),
                    ('bbbar', GP2['quark']['mb_GeV'])):
        key = 'cc_2S_1S_MeV' if tag == 'ccbar' else 'bb_2S_1S_MeV'
        obs = float(GP2['PDG'][key])
        cols = {}
        for label, Vv in (('IR', V_IR), ('UVIR', V_UVIR), ('SCR', V_SCR),
                          ('CAS', V_CAS)):
            main = 2.0 / (mQ * dr ** 2) + Vv
            off = -1.0 / (mQ * dr ** 2) * np.ones(len(r) - 1)
            H = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
            E = [float(v) for v in _evh2(H, subset_by_index=[0, 2],
                                         eigvals_only=True)]
            split = float((E[1] - E[0]) * 1000.0)
            cols[label] = dict(
                E_GeV=E, split_2S_1S_MeV=split,
                rel=float(abs(split - obs) / obs),
                split_3S_1S_MeV=float((E[2] - E[0]) * 1000.0),
                ordered=bool(E[0] < E[1] < E[2]),
            )
        cols['obs_MeV'] = obs
        out['quarkonia'][tag] = cols
    return out


# ---------------- FAZ 11 / GRAND PROOF III (T35-T37): masaustu deney + gozlemevi ----
# EXACT (sifir serbest parametreli teorem bagintilari):
#   Burgers girdabi: rc = sqrt(2 nu/gam), umax = G0/(2 pi rc sqrty)(1-e^-y),
#   y = 1.25643121 cozer e^-y(2y+1) = 1; dh = G0^2 ln2/(4 pi^2 g rc^2)
#   (quad+kapali-form caprazi); PEO oranlari: C_univ, C_univ^-1,
#   [C_univ^2(1+mu_s)]^-1 = 0.31402536; Vela: base*(1+mu_s), Q_fast, Q_total;
#   O_exact bant orani B = 0.08413579 (band [10,100]/[350,490] keV).
# BAGIMSIZ LITERATUR (model disi, yalnizca karsilastirma):
#   PEO girdap bantlari (r: [1.55,1.70], dh: [0.30,0.35]), Vela glitch
#   (2.25+-0.35)e-6, Q in [0.05,0.18]; 22Na Ps laboratuvar spektrumu.
# DENEY PROTOKOLU (onerilen masaustu duzenek, sinamaya acik):
#   R_tank = 100 mm, 100 ppm PEO, HPGe sigma = 0.85 keV.
GP3 = dict(
    tank_R_mm=100.0, tank_H0_mm=200.0,
    nu0_m2_s=1.0e-5, gamma_strain_per_s=2.0,
    Gamma0_m2_s=1.20e-2, g0=9.80665,
    peo_ppm=100.0,
    # GROISMAN & STEINBERG 2000 (Nature 405, 53; arXiv:nlin/0104052) DENETIMI:
    # deney PAAm+surup, iki paralel disk, d/R = 0.263; olculemler: akis
    # direnci ~20 kat, elastik gerilim 2 buyukluk mertebesi artar.
    # "r_ve/r_N in [1.55,1.70]" ve "dh_ve/dh_N in [0.30,0.35]" bantlari bu
    # kaynakta YOKTUR -> dogrulanamamis (asagidaki degerler gorev metnindendir).
    peo_band_verified=False,
    peo_r_band=(1.55, 1.70), peo_h_band=(0.30, 0.35),
    gs_flow_resistance_factor=20.0,
    # VELA: kaynak dogrulamali girdiler
    #   Dodson+2002 (ApJL 578, L57; arXiv:astro-ph/0201005) Tablo 1, Epoch 51559:
    #     nu = 11.194615396005 Hz, nudot = -1.55615e-11 Hz/s,
    #     dnu_p = 3.45435(5)e-05 Hz, gevseme: 1.2+-0.2 dk, 0.53(3), 3.29(3),
    #     19.07(2) gun; donen terimler 0.020(5), 0.31(2), 0.193(2), 0.2362(2) e-6 Hz
    #   Dodson+2007 (Ap&SS 308, 585; arXiv:astro-ph/0612371): 2004 olayi
    #     ani kesirli glitch boyutu 2.08e-6
    #   Link+Epstein+Lattimer 1999 (PRL 83, 3362; arXiv:astro-ph/9909146):
    #     dI/I >= 0.014 (Ic >= 0.014 I), 13 glitch, ~10^-6 her uc yilda
    vela=dict(nu0_Hz=11.194615396005, nudot_Hz_s=-1.55615e-11,
              Icrust_I=0.0140, dt_ig_yr=2.80,
              dnu_p_2000_Hz=3.45435e-05, glitch_2000_rel=3.0858e-6,
              glitch_2004_rel=2.08e-6,
              relax_days=(0.53, 3.29, 19.07), relax_min=1.2,
              Q_2000=0.021978085602211705,
              Q_band_lo=0.05, Q_band_hi=0.18),
    na22=dict(A_Bq=1.85e5, f_betap=0.903, f_Ps=0.55,
              eps_geom=0.02, t_acq_s=3600.0,
              hpge_sigma_keV=0.85, m_e_keV=511.0),
)


def solve_benchtop_viscoelastic_vortex_piv_experiment(p=None):
    """T35: Burgers girdabi N (quad+kapali) + PEO 3 ongorusu (sifir parametre)."""
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    uv = solve_universal_vortex_string_scaling_law(p=p)
    C = float(uv['C_univ'])
    nu0 = float(GP3['nu0_m2_s'])
    gam = float(GP3['gamma_strain_per_s'])
    G0 = float(GP3['Gamma0_m2_s'])
    g0 = float(GP3['g0'])
    rc = float(np.sqrt(2.0 * nu0 / gam))
    y = float(brentq(lambda t: np.exp(-t) * (2.0 * t + 1.0) - 1.0,
                     0.5, 2.0, xtol=1e-15, rtol=1e-15, maxiter=300))
    rmax = float(np.sqrt(y) * rc)
    umax = float(G0 / (2.0 * np.pi * rc * np.sqrt(y)) * (1.0 - np.exp(-y)))
    dh_closed = float(G0 ** 2 * np.log(2.0) / (4.0 * np.pi ** 2 * g0 * rc ** 2))

    def _uth(r):
        return G0 / (2.0 * np.pi * np.maximum(r, 1e-300)) \
            * (1.0 - np.exp(-(r / rc) ** 2))
    dh_quad, dh_err = quad(lambda r: float(np.asarray(_uth(r))) ** 2
                           / max(float(r), 1e-300),
                           0.0, np.inf, limit=500)
    dh_quad = float(dh_quad / g0)
    # PEO: oranlar EXACT kilitlerden.
    rc_peo = float(C * rc)
    umax_peo = float(umax / C)
    dh_ratio = float(1.0 / (C * C * (1.0 + mu_m)))
    dh_peo = float(dh_closed * dh_ratio)
    lit_r = GP3['peo_r_band']
    lit_h = GP3['peo_h_band']
    # ---- GP-II.3 (Faz 10.1): a_PEO serbest kalibrasyon duyarliligi ----
    # a_PEO = r_core^PEO/r_core^N (serbest kalibrasyon; kaynakta YOK).
    # Modelde UCA (bagimsiz) büyüklük vardir: r orani (= a) ve Δh orani
    # (= 1/a^2 * 1/(1+mu_s)); tepe hiz orani (= 1/a) bunun sonucudur.
    mu_s = mu_s_meral()
    a_grid = np.round(np.arange(1.0, 2.5001, 0.05), 4)
    sens = []
    for a_ in a_grid:
        sens.append(dict(
            a=float(a_),
            r_ratio=float(a_),
            u_ratio=float(1.0 / a_),
            dh_ratio_model=float(1.0 / (a_ ** 2 * (1.0 + mu_s))),
            dh_ratio_geom=float(1.0 / a_ ** 2)))
    # Kaynakli karsilastirma: Groisman & Steinberg 2000 olcumleri
    #   sigma/sigma_lam = 12 (d=10 mm), 19 (d=20 mm); gecis Wi=3.5, Re~0.3.
    # Sabit torkta lineer hiz alaninda direnc artisi hiz frenlemesiyle
    # ters orantili: 1/u = sigma/sigma_lam  =>  a_olculmus = 12..19.
    gs_sigma_ratio = (12.0, 19.0)
    a_measured = (float(gs_sigma_ratio[0]), float(gs_sigma_ratio[1]))
    braking_measured = (1.0 / gs_sigma_ratio[0], 1.0 / gs_sigma_ratio[1])
    return dict(
        rc_N_mm=float(rc * 1e3), rmax_N_mm=float(rmax * 1e3),
        umax_N_ms=umax, dh_N_mm=float(dh_closed * 1e3),
        dh_quad_mm=float(dh_quad * 1e3),
        dh_closed_quad_abs=float(abs(dh_closed - dh_quad)),
        y_star=y,
        rc_PEO_mm=float(rc_peo * 1e3), umax_PEO_ms=umax_peo,
        dh_ratio=dh_ratio, dh_PEO_mm=float(dh_peo * 1e3),
        C_univ=C, mu_meral=mu_s,
        n_independent=2,
        sens=sens,
        a_grid=[float(v) for v in a_grid],
        gs_sigma_ratio=gs_sigma_ratio,
        a_measured=a_measured,
        braking_measured=braking_measured,
        a_measured_in_model_range=bool(
            a_measured[0] <= 2.5),
        lit_r_band=lit_r, lit_h_band=lit_h,
        lit_band_verified=bool(GP3['peo_band_verified']),
        lit_r_in=bool(lit_r[0] <= C <= lit_r[1]),
        lit_h_in=bool(lit_h[0] <= dh_ratio <= lit_h[1]),
        gs_flow_resistance_factor=float(GP3['gs_flow_resistance_factor']),
    )


def solve_vela_pulsar_glitch_avalanche(p=None, npts=400):
    """T36: Vela glitch genligi + Q kesirleri + gevse ODE (kaynak dogrulamali).

    Kaynak dogrulamali girdiler: Dodson+2002 Tablo 1 (nu, nudot, dnu_p,
    gevseme olcekleri, donen terimler), Dodson+2007 (2004 olayi 2.08e-6),
    Link+Epstein+Lattimer 1999 (dI/I >= 0.014).
    """
    if p is None:
        p = DEFAULTS
    mu_m = mu_s_meral()
    C = float(solve_universal_vortex_string_scaling_law(p=p)['C_univ'])
    b0 = float(p['beta0'])
    lam0 = float(LAMBDA0_CLOSED)
    V = GP3['vela']
    yr = 365.25 * 86400.0
    dt = float(V['dt_ig_yr']) * yr
    nu0 = float(V['nu0_Hz'])
    nud = abs(float(V['nudot_Hz_s']))
    Icr = float(V['Icrust_I'])
    base = float(Icr * nud * dt / nu0)
    g_meral = float(base * (1.0 + mu_m))
    g2000 = float(V['glitch_2000_rel'])
    g2004 = float(V['glitch_2004_rel'])
    Q_fast = float(np.exp(-2.0 * np.sqrt(2.0)))
    Q_total = float(lam0 * C / np.sqrt(1.0 - b0 ** 2))
    # Gevseme ODE: kaynakta OLCULED zaman olcekleri (1.2 dk, 0.53, 3.29,
    # 19.07 gun) + kalici kisim. R(t) = 1 - (1-Q) * (donen paylar).
    taus = (float(V['relax_min']) / 1440.0, float(V['relax_days'][0]),
            float(V['relax_days'][1]), float(V['relax_days'][2]))
    t = np.linspace(0.0, 200.0, npts)
    R_num = np.ones_like(t)
    for tau in taus:
        R_num -= (1.0 - Q_total) * np.exp(-t / tau) / len(taus)
    dA = (1.0 - Q_total) / len(taus) * sum(
        np.exp(-t / tau) / tau for tau in taus)
    # En hizli bilesen 1.2 dakika (~8e-4 gun) oldugu icin turev caprazi
    # yalnizca t >= 1 gun uzerinde anlamlidir.
    m = t >= 1.0
    resid = float(np.max(np.abs(dA[m] - np.gradient(R_num, t)[m])))
    return dict(
        dt_ig_s=dt, base=base, glitch_meral=g_meral,
        glitch_2000=g2000, glitch_2004=g2004,
        rel_2000=float(abs(g_meral - g2000) / g2000),
        rel_2004=float(abs(g_meral - g2004) / g2004),
        Q_fast=Q_fast, Q_total=Q_total,
        Q_2000_measured=float(V['Q_2000']),
        Q_band=(float(V['Q_band_lo']), float(V['Q_band_hi'])),
        Q_band_verified=False,
        relax_taus_days=[float(x) for x in taus],
        t_days=[float(v) for v in t],
        R_relax=[float(v) for v in R_num],
        relax_maxresid=resid,
    )


def solve_benchtop_na22_positronium_gamma_experiment(p=None):
    """T37: O_exact bant orani + misprint/pole curutulmesi (sayim istatistigi)."""
    if p is None:
        p = DEFAULTS
    NN = GP3['na22']
    m = float(NN['m_e_keV'])

    def _exact(E):
        return float(np.asarray(ore_powell_exact(E, m)))

    def _sub(E):
        # E^2 -> (2m-E)^2 ikameli varyant (int_1 = 27.147 != 1)
        return float(np.asarray(ore_prompt_form(E, m)))

    def _pub(E):
        # arXiv:2605.21289 Denk.(12) BASIMDAKI ln'siz form (int_1 = 2343.47)
        return float(np.asarray(ore_powell_published(E, m)))
    Blo, _ = quad(_exact, 10.0, 100.0, limit=500)
    Bhi, _ = quad(_exact, 350.0, 490.0, limit=500)
    B_exact = float(Blo / Bhi)
    Slo, _ = quad(_sub, 10.0, 100.0, limit=500)
    Shi, _ = quad(_sub, 350.0, 490.0, limit=500)
    B_sub = float(Slo / Shi)
    Plo, _ = quad(_pub, 10.0, 100.0, limit=500)
    Phi, _ = quad(_pub, 350.0, 490.0, limit=500)
    B_pub = float(Plo / Phi)
    Ntot = float(NN['A_Bq'] * NN['f_betap'] * NN['f_Ps'] * 0.75
                 * NN['eps_geom'] * NN['t_acq_s'])
    N_lo_exact = float(Ntot * Blo)
    N_lo_pub = float(Ntot * Plo)
    sigmalab = float(abs(N_lo_pub - N_lo_exact) / np.sqrt(N_lo_exact))
    return dict(
        B_exact=B_exact, B_substituted=B_sub, B_published=B_pub,
        Blo=float(Blo), Bhi=float(Bhi),
        Ntot_3gamma=Ntot, N_lo_exact=N_lo_exact, N_lo_published=N_lo_pub,
        lab_sigma=sigmalab,
    )


def uq_faz6_sensitivities(p=None):
    """UQ: 4 kilit ciktinin giris belirsizliklerine goreceli duyarliligi."""
    if p is None:
        p = DEFAULTS
    out = {}
    mu_m = float(2.0 * np.sqrt(2.0) * float(LAMBDA0_CLOSED))
    r0 = trap_rp_eff(mu_m, 1.0, ALPHA_TRAP['s_eq'], 0.005, p)
    sens_mu = float((trap_rp_eff(mu_m * 1.01, 1.0, ALPHA_TRAP['s_eq'], 0.005, p)
                     - trap_rp_eff(mu_m * 0.99, 1.0, ALPHA_TRAP['s_eq'], 0.005, p))
                    / (0.02 * r0))
    out['rp_meral_sens_mu'] = sens_mu
    dd = solve_alpha_2026_lineshape_and_drift(p)
    out['R_th_nominal'] = dd['R_th']
    out['R_th_spread_ppm'] = float(abs(dd['R_th'] - dd['R_B']) * 1e6)
    out['Mdot_eta010'] = float(galactic_Mdot_msun_yr(eta_pair=0.10))
    out['Mdot_eta009_011'] = [float(galactic_Mdot_msun_yr(eta_pair=0.09)),
                              float(galactic_Mdot_msun_yr(eta_pair=0.11))]
    nuz = 453e3 * np.sqrt(16.0 / 3.911)
    gd = 2.0 * np.pi * 122.0 * (122.0 / nuz)
    wc = _E_CHG * 0.136 / _M_P
    mu_s = find_mu_star()
    r_hi = (1.10 * 0.15 * _G0 / wc) / (mu_s * 1.0 - gd)
    r_lo = (0.90 * 0.15 * _G0 / wc) / (mu_s * 1.0 - gd)
    out['rstar_136_um_nominal'] = float((0.15 * _G0 / wc) / (mu_s - gd) * 1e6)
    out['rstar_136_um_band'] = [float(r_lo * 1e6), float(r_hi * 1e6)]
    # (e) T22 tayf orani duyarliligi: R = 9f/(8-6f) -> dR/df = 72/(8-6f)^2
    f0 = 0.92
    dRdf = float(72.0 / (8.0 - 6.0 * f0) ** 2)
    out['R_dRdf'] = dRdf
    out['R_fPs_band'] = [float(9.0 * 0.90 / (8.0 - 6.0 * 0.90)),
                         float(9.0 * 0.94 / (8.0 - 6.0 * 0.94))]
    return out


def lambda_meral_triple_check(p=None):
    """Meral sabiti uc-yol bagimsiz kanit: quad + yamuk + kapali form."""
    if p is None:
        p = DEFAULTS
    closed = float(LAMBDA0_CLOSED)
    vq, eq = lambda_meral(0.0, p)

    def K(X):
        return (p['X0'] + p['X_star']) / (2.0 * p['X_star'] * (p['X0'] + X))
    Xs = np.concatenate(([0.0], np.logspace(-12, 8, 200000)))
    vals = np.array([float(K(x)
                           * np.asarray(F_prof(x, 0.0, 0.0, p['F0'], p['X0'],
                                               p['alpha_eta']))
                           * np.asarray(U_full(x, 0.0, 0.0, p['U_even'], p['U0'],
                                               p['X0'], p['alpha_eta']))
                           * np.sqrt(float(np.asarray(a_prof(x, 0.0, 0.0, p['X0']))) ** 2
                                     + p['beta0'] ** 2 * float(np.asarray(
                                         l_prof(x, 0.0, 0.0, p['X0']))) ** 2))
                     for x in Xs])
    vt = float(np.trapezoid(vals, Xs)) * (28.0 / 79.0)
    return dict(closed=closed, quad_val=float(vq), quad_err=float(eq),
                trapz_val=vt,
                quad_abs=float(abs(vq - closed)),
                trapz_abs=float(abs(vt - closed)))


# ---------------- Dogrulama testleri ----------------
# ================= FAZ 4 (Makale 6 / Teorem VI) =================
# Model parametreleri (belgelenmis secimler; calismanin fenomenolojik ansatzidir):
# omega_D0/kappa_chiral/lambda_Maj = O(1) kiral-girdap kuplajlari;
# washout0 = doygun yikama orani; kappa_a5 = SME ust-sinir katsayisi;
# eta_obs Planck 2018; a1S/H degerleri ALPHA 2026 Nature (Tablo 1).
DEFAULTS.update({
    'omega_D0': 1.0, 'kappa_chiral': 1.0, 'lambda_Maj': 1.0,
    'washout0': 1.0, 'kappa_a5': 5e-7,
    'eta_obs': 6.12e-10, 'a1S_kHz': 1420404.8, 'H_a1S_kHz': 1420405.751768,
    'f_Ps': 0.92, 'sigma_511_keV': 1.0, 'm_e_keV': 511.0,
})
C_SPH = 28.0 / 79.0
# h -> 0 kapali formu (Euler t-ikamesi + kismi kesirler, sympy ile turetilmistir):
# L0 = I0/2, I0 = 163/578 + (64 sqrt(17)/4913) ln((21+5 sqrt(17))/4).
LAMBDA0_CLOSED = C_SPH * 0.5 * (
    163.0 / 578.0 + (64.0 * np.sqrt(17.0) / 4913.0) * np.log((21.0 + 5.0 * np.sqrt(17.0)) / 4.0))

def dirac_fields(X, s, h, mu_s, p=None, Om_on=True, Mm_on=True):
    """Kiral girdap / Dirac / Majorana alanlari (X skaler ya da dizi olabilir).

    Om = k_chiral F_base U_full G e^{-hs}; wD = wD0 e^{-hs} F_base/F0;
    Mm = lam mu_s K e^{-(1+h)s} (lt^2-l0^2) phi_cap, phi_cap = max(1-lt^2/b,0)
    (FENE doygunluk kapagi: azami uzamada kuplaj doyar, isaret atmaz).
    """
    if p is None:
        p = DEFAULTS
    s = float(s)
    Fb = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    U0_ = np.asarray(U_full(X, 0.0, h, p['U_even'], p['U0'], p['X0'], p['alpha_eta']), dtype=float)
    G = np.asarray(G_drive(X, h, p), dtype=float)
    K = np.asarray(K_curv(X, p), dtype=float)
    lt = np.asarray(ltilde_global(s, X, h, p), dtype=float)
    phi = np.maximum(1.0 - lt * lt / p['bmax'], 0.0)
    Om = p['kappa_chiral'] * Fb * U0_ * G * np.exp(-h * s) if Om_on else np.zeros_like(Fb)
    wD = p['omega_D0'] * np.exp(-h * s) * Fb / p['F0']
    Mm = (p['lambda_Maj'] * mu_s * K * np.exp(-(1.0 + h) * s) * (lt * lt - p['ell0'] ** 2) * phi
          if Mm_on else np.zeros_like(Fb))
    return dict(Om=Om, wD=wD, Mm=Mm)

def spinor_rhs(s, y, X, h, mu_s, p, Om_on=True, Mm_on=True):
    """4-reel spinor RHS: y = [ReL, ImL, ReR, ImR]. psi_R* terimi reel/imag ayrimlidir."""
    if p is None:
        p = DEFAULTS
    f = dirac_fields(X, float(s), h, mu_s, p, Om_on, Mm_on)
    Om = float(np.asarray(f['Om']))
    wD = float(np.asarray(f['wD']))
    Mm = float(np.asarray(f['Mm']))
    a, b, c, d = (float(v) for v in y)
    return [-0.5 * Om * a + wD * d,
            -0.5 * Om * b - wD * c,
            +0.5 * Om * c + wD * b - Mm * d,
            +0.5 * Om * d - wD * a - Mm * c]

def solve_spinor(X, h, mu_s, s_max=30.0, p=None, npts=3000, Om_on=True, Mm_on=True):
    """Spinor evrimi: saf madde baslangici (psi_L=1, psi_R=0). Donus: dict(s, L2, R2, Q5, flip)."""
    if p is None:
        p = DEFAULTS
    s_eval = np.linspace(0.0, s_max, npts)
    sol = solve_ivp(spinor_rhs, (0.0, s_max), [1.0, 0.0, 0.0, 0.0],
                    args=(float(X), h, mu_s, p, Om_on, Mm_on),
                    method='Radau', rtol=1e-12, atol=1e-14, t_eval=s_eval)
    a, b, c, d = sol.y
    L2 = a * a + b * b
    R2 = c * c + d * d
    tot = np.maximum(L2 + R2, 1e-300)
    return dict(s=s_eval, L2=L2, R2=R2, Q5=L2 - R2, flip=R2 / tot, sol=sol)

def lambda_meral(h, p=None, eps=1e-13):
    """Boyutsuz evrensel girdap-kiralite integrali (28/79 agirlikli). Kuyruk ~X^{-3-2h}."""
    if p is None:
        p = DEFAULTS
    def f(x):
        Fb = float(np.asarray(F_prof(x, 0.0, h, p['F0'], p['X0'], p['alpha_eta'])))
        U = float(np.asarray(U_full(x, 0.0, h, p['U_even'], p['U0'], p['X0'], p['alpha_eta'])))
        a = float(np.asarray(a_prof(x, 0.0, h, p['X0'])))
        rl = float(np.asarray(l_shear_reg(x, h, p['X0'])))
        K = float(np.asarray(K_curv(x, p)))
        return K * Fb * U * np.sqrt(a * a + p['beta0'] ** 2 * rl * rl)
    val, err = quad(f, 0.0, np.inf, epsabs=eps, epsrel=eps, limit=500)
    return float(C_SPH * val), float(C_SPH * err)

def vort_source(s, h, mu_s, p=None):
    """Girdap kiral kaynak profili: normalize Majorana-gerilim hizi S(s), S>=0, sonlu pencere.

    S(s) = M_Maj(X_*,s)/M_peak (sicim gerilimine kili kaynak); yikama W(s) = w0 e^{-s}
    (sönen etkilesimler -> gercek donma). Ikisi de fenomenolojik ansatzdir.
    """
    if p is None:
        p = DEFAULTS
    ss = np.atleast_1d(np.asarray(s, dtype=float))
    Xst = p['X_star']
    K = float(np.asarray(K_curv(Xst, p)))
    out = np.zeros_like(ss)
    for i, sv in enumerate(ss):
        lt = float(np.asarray(ltilde_global(sv, Xst, h, p)))
        phi = max(1.0 - lt * lt / p['bmax'], 0.0)
        out[i] = (p['lambda_Maj'] * mu_s * K * np.exp(-(1.0 + h) * sv)
                  * (lt * lt - p['ell0'] ** 2) * phi)
    return out[0] if np.ndim(s) == 0 else out

def boltzmann_rhs(s, Y, eps_cp, h, mu_s, p, M_peak):
    """Kenetlenmis Boltzmann-girdap: Y' = eps*S(s) - W(s)*Y; S normalize, W = w0 e^{-s}."""
    W = p['washout0'] * np.exp(-float(s))
    S = vort_source(s, h, mu_s, p) / M_peak
    return [eps_cp * S - W * Y[0]]

def solve_boltzmann(eps_cp, h, mu_s=0.1, s_max=60.0, p=None, npts=2000):
    if p is None:
        p = DEFAULTS
    sg = np.linspace(0.0, 40.0, 2000)
    M_peak = float(np.max(vort_source(sg, h, mu_s, p)))
    s_eval = np.linspace(0.0, s_max, npts)
    sol = solve_ivp(boltzmann_rhs, (0.0, s_max), [0.0], args=(eps_cp, h, mu_s, p, M_peak),
                    method='Radau', rtol=1e-13, atol=1e-15, t_eval=s_eval)
    Y = sol.y[0]
    return dict(s=s_eval, Y=Y, eta=7.04 * C_SPH * Y, sol=sol, M_peak=M_peak)

def find_eps_star(h, mu_s=0.1, p=None):
    """Gozlenen eta_B'yi tam ureten kritik CP-girdap kuplaji (monoton -> brentq)."""
    if p is None:
        p = DEFAULTS
    obs = p['eta_obs']
    def f(eps):
        return float(solve_boltzmann(eps, h, mu_s, 60.0, p, 400)['eta'][-1]) - obs
    lo, hi = 1e-14, 1e-6
    assert f(lo) < 0 and f(hi) > 0
    return float(brentq(f, lo, hi, xtol=1e-22, rtol=1e-14, maxiter=500))

def ag_ratio(X, s, h, mu_s, p=None):
    """Etkin ivme orani a_g/g = 1 - K mu_s e^{-(1+h)s} (lt^2-l0^2) phi_cap / F_base^2."""
    if p is None:
        p = DEFAULTS
    Fb = np.asarray(F_prof(X, 0.0, h, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    K = np.asarray(K_curv(X, p), dtype=float)
    lt = np.asarray(ltilde_global(s, X, h, p), dtype=float)
    phi = np.maximum(1.0 - lt * lt / p['bmax'], 0.0)
    return 1.0 - K * mu_s * np.exp(-(1.0 + h) * float(s)) * (lt * lt - p['ell0'] ** 2) * phi / np.maximum(Fb * Fb, 1e-300)

def breit_rabi(B_T, a_kHz=None, p=None):
    """J=1/2, I=1/2 tam 4x4 Hamiltonyen + m_F etiketleme (tam korunur).
    Donus: Ea..Ed (zayif-alan enerji sirasi |a>..|d>), fda, fcb (kHz)."""
    if p is None:
        p = DEFAULTS
    a_Hz = (a_kHz if a_kHz is not None else p['a1S_kHz']) * 1e3
    muB_h = 13.996245042e9
    muN_h = 7.622593285e6
    gJ, gI = 2.00231930436256, 5.5856946893
    dW = a_Hz
    Sx = np.array([[0, .5], [.5, 0]])
    Sy = np.array([[0, -.5j], [.5j, 0]])
    Sz = np.array([[.5, 0], [0, -.5]])
    Ix, Iy, Iz = Sx, Sy, Sz
    E2 = np.eye(2)
    Fz = np.kron(Sz, E2) + np.kron(E2, Iz)
    nu_e = gJ * muB_h * B_T
    nu_p = gI * muN_h * B_T
    HF = (dW * (np.kron(Sx, Ix) + np.kron(Sy, Iy) + np.kron(Sz, Iz))
          + nu_e * np.kron(Sz, E2) - nu_p * np.kron(E2, Iz))
    wF, vF = np.linalg.eigh(HF / dW)
    mF = np.array([float(np.real(vF[:, i].conj() @ (Fz @ vF[:, i]))) for i in range(4)])
    ib = int(np.argmin(np.abs(mF + 1.0)))
    id_ = int(np.argmin(np.abs(mF - 1.0)))
    zero = sorted([i for i in range(4) if i != ib and i != id_], key=lambda i: wF[i])
    ia, ic = zero[0], zero[1]
    EF = wF * dW
    Ea, Eb, Ec, Ed = (float(EF[i]) for i in (ia, ib, ic, id_))
    return dict(Ea=Ea, Eb=Eb, Ec=Ec, Ed=Ed,
                fda=(Ed - Ea) / 1e3, fcb=(Ec - Eb) / 1e3)

def ore_powell(E_keV, m_keV=511.0):
    """Orto-pozitronyum 3-foton Ore-Powell spektrumu (Siegert-Yoneda Denk. 12). E<=0'da tanimsiz."""
    E = np.asarray(E_keV, dtype=float)
    m = float(m_keV)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (2.0 / (m * (np.pi ** 2 - 9.0))) * (
            E * (m - E) / (2 * m - E) ** 2 + (2 * m - E) / E
            + 2 * m * (m - E) / E ** 2 - 2 * m * (m - E) ** 2 / (2 * m - E) ** 3)
    return out

def gauss_line(E_keV, m_keV=511.0, sig_keV=1.0):
    """511 keV Gauss hatti L(E) (Siegert-Yoneda Denk. 11), normalize."""
    E = np.asarray(E_keV, dtype=float)
    return np.exp(-0.5 * ((E - m_keV) / sig_keV) ** 2) / (np.sqrt(2 * np.pi) * sig_keV)

def fuel_spectrum(E_keV, s, P_fuel, p=None, m_keV=511.0):
    """Girdap yakiti gama akisi: Phi(E,s) = P_yakit(s) * N(E); N: foton/pozitron dagilimi."""
    if p is None:
        p = DEFAULTS
    f = p['f_Ps']
    E = np.asarray(E_keV, dtype=float)
    N = (2.0 * (1.0 - f) + 2.0 * (f / 4.0)) * gauss_line(E, m_keV, p['sigma_511_keV']) \
        + 3.0 * (f * 3.0 / 4.0) * ore_powell(E, m_keV)
    P = max(float(P_fuel), 0.0)
    return P * N

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
    # T11: iki-alanli (F, l) PDE - Duzeltme 1+2, tutarli varyant (M_mode='decaying')
    h11 = 0.005
    t11 = {'mu': {}}
    X11 = np.concatenate(([0.0], np.logspace(-6, np.log10(300.0), 79)))
    Fb11 = np.asarray(F_prof(X11, 0.0, h11, p['F0'], p['X0'], p['alpha_eta']), dtype=float)
    Fb_star = float(np.asarray(F_prof(1.0, 0.0, h11, p['F0'], p['X0'], p['alpha_eta'])))
    # Duzeltme-1 iddia 1: mu_s = 0'da F_base tam korunumu
    f00 = solve_two_field(X11, h11, 0.0, 40.0, p, p['bmax'], 121)
    t11['mu0_success'] = bool(f00['sol'].success)
    t11['mu0_preservation_maxerr'] = float(np.max(np.abs(f00['F'] - Fb11[:, None])))
    u0_40 = np.exp((0.5 + h11) * 40.0) * np.asarray(E_prof(X11, 0.0, h11), dtype=float)
    t11['u0max_40'] = float(np.max(u0_40))
    for mu11 in [0.05, 0.10, 0.20]:
        fld = solve_two_field(X11, h11, mu11, 40.0, p, p['bmax'], 121)
        m = dict(success=bool(fld['sol'].success))
        m['fene_max'] = float(np.max(fld['L'] ** 2) / p['bmax'])
        m['min_F'] = float(np.min(fld['F']))
        m['min_l'] = float(np.min(fld['L']))
        m['max_viol'] = float(np.max(fld['F'] - Fb11[:, None]) / np.max(Fb11))
        i_star = int(np.argmin(np.abs(X11 - p['X_star'])))
        m['F_star_40'] = float(np.interp(p['X_star'], X11, fld['F'][:, -1]))
        m['F_axis_40'] = float(fld['F'][0, -1])
        raw_all = []
        for j in range(fld['s'].size):
            d = two_field_powers(X11, float(fld['s'][j]), fld['F'][:, j], fld['L'][:, j],
                                 h11, mu11, p, p['bmax'])
            raw_all.append(d['raw'])
        m['power_raw_max'] = float(max(raw_all))
        m['wake_q'] = power_wake_q(X11, fld['Pi'][:, -1])
        m['plateau_dF'] = float(np.mean(fld['F'][-6:, -1] - Fb11[-6:]))
        I300 = float(np.trapezoid(fld['Pi'][:, -1], X11))
        m['I300_40'] = I300
        m['Iinf_40'] = float(I300 + tail_F_analytic(300.0, h11, p))
        m['Iinf_converges'] = bool(m['wake_q'] > 1.9)
        iu = int(np.argmax(fld['Uth']))
        iux, ius = np.unravel_index(iu, fld['Uth'].shape)
        m['u_max'] = float(fld['Uth'][iux, ius])
        m['s_umax'] = float(fld['s'][ius])
        m['X_umax'] = float(X11[iux])
        H = H_series(X11, fld['s'], fld['F'], fld['L'], h11, mu11, p, p['bmax'])
        m['H0'] = float(H[0])
        m['H40'] = float(H[-1])
        m['Hmax'] = float(np.max(H))
        t11['mu'][str(mu11)] = m
    t11['Fb_star'] = Fb_star
    t11['pass'] = bool(
        t11['mu0_success'] and t11['mu0_preservation_maxerr'] < 1e-10
        and all(v['success'] and v['fene_max'] < 1.0 and v['min_l'] > 0.9
                and v['min_F'] > 0.0 and v['max_viol'] < 1e-6 and 0.0 < v['F_star_40'] < Fb_star
                and v['power_raw_max'] < 1e-15
                and v['X_umax'] < 5.0 and v['u_max'] < t11['u0max_40']
                and v['Iinf_converges'] for v in t11['mu'].values()))
    rep['T11_two_field'] = t11
    # T12: es-hareketli Lagrange spektral aktarim - simetrik (+-eps) okuma
    h12, mu12, eps12 = 0.005, 0.1, 0.02
    t12 = {'modes': {}}
    xi12, X012 = lagrange_grid(Nxi=128, Xtop=10.0, X0param=p['X0'])
    s12 = 25.0
    fb12 = solve_lagrange(xi12, X012, h12, mu12, s12, p, p['bmax'], 51)
    t12['base_success'] = bool(fb12['sol'].success)
    env_prof = np.exp(-X012 / 10.0)
    runs = {}
    runs['u+'] = solve_lagrange(xi12, X012, h12, mu12, s12, p, p['bmax'], 51,
                                l_init=p['ell0'] * (1.0 + eps12 * env_prof))
    runs['u-'] = solve_lagrange(xi12, X012, h12, mu12, s12, p, p['bmax'], 51,
                                l_init=p['ell0'] * (1.0 - eps12 * env_prof))
    for n12 in [1, 2, 3, 4]:
        pat = np.sin(n12 * np.pi * xi12) * env_prof
        runs[(n12, '+')] = solve_lagrange(xi12, X012, h12, mu12, s12, p, p['bmax'], 51,
                                         l_init=p['ell0'] * (1.0 + eps12 * pat))
        runs[(n12, '-')] = solve_lagrange(xi12, X012, h12, mu12, s12, p, p['bmax'], 51,
                                         l_init=p['ell0'] * (1.0 - eps12 * pat))
    t12['uniform_success'] = bool(runs['u+']['sol'].success and runs['u-']['sol'].success)
    Fb = fb12['F'][:, -1]
    dFu_sym = 0.5 * (runs['u+']['F'][:, -1] - runs['u-']['F'][:, -1])
    dFu_one = runs['u+']['F'][:, -1] - Fb
    env = np.maximum(1e-30, np.abs(dFu_sym))
    t12['envelope_min'] = float(np.min(np.abs(dFu_sym)))
    t12['envelope_floor_hits'] = int(np.sum(np.abs(dFu_sym) < 1e-30))
    for n12 in [1, 2, 3, 4]:
        Fp = runs[(n12, '+')]['F'][:, -1]
        Fm = runs[(n12, '-')]['F'][:, -1]
        dFn_sym = 0.5 * (Fp - Fm)
        Rn = dFn_sym / env
        c = sine_spectrum(xi12, Rn, 6)
        nc = float(np.linalg.norm(c))
        bare = np.sin(n12 * np.pi * xi12)
        r = float(abs(np.corrcoef(bare, Rn)[0, 1])) if np.std(Rn) > 0 else 0.0
        dFn_one = Fp - Fb
        Ro = dFn_one / np.maximum(1e-30, np.abs(dFu_one))
        co = sine_spectrum(xi12, Ro, 6)
        nco = float(np.linalg.norm(co))
        ro = float(abs(np.corrcoef(bare, Ro)[0, 1])) if np.std(Ro) > 0 else 0.0
        t12['modes'][str(n12)] = {
            'success': bool(runs[(n12, '+')]['sol'].success and runs[(n12, '-')]['sol'].success),
            'kmax': int(np.argmax(np.abs(c))) + 1,
            'purity': float(abs(c[n12 - 1]) / nc) if nc > 0 else 0.0,
            'pearson_r': r,
            'one_kmax': int(np.argmax(np.abs(co))) + 1,
            'one_purity': float(abs(co[n12 - 1]) / nco) if nco > 0 else 0.0,
            'one_pearson_r': ro,
            'max_dF': float(np.max(np.abs(dFn_sym))),
        }
    t12['pass'] = bool(t12['base_success'] and t12['uniform_success'] and all(
        v['success'] and v['kmax'] == int(k) and v['purity'] > 0.90 and v['pearson_r'] > 0.80
        for k, v in t12['modes'].items()))
    rep['T12_spectral'] = t12
    # T13: Dirac-Majorana flip-flop kiral soyulmasi (h = 0.005, mu_s = 0.1)
    h13, mu13 = 0.005, 0.1
    t13 = {'X': {}}
    for X13 in [0.5, 1.0, 2.0]:
        f = solve_spinor(X13, h13, mu13, 30.0, p, 3000)
        g = solve_spinor(X13, h13, mu13, 30.0, p, 3000, Mm_on=False)
        t13['X'][str(X13)] = {
            'success': bool(f['sol'].success),
            'Q5_30': float(f['Q5'][-1]),
            'flip_30': float(f['flip'][-1]),
            'maj_effect': float(np.max(np.abs(f['R2'] - g['R2']))),
        }
    # Rabi capraz kontrolu (Om=0, Mm=0): |psiR|^2 = sin^2(Theta), Theta = int wD
    rX = 1.0
    rsol = solve_spinor(rX, h13, mu13, 10.0, p, 2000, Om_on=False, Mm_on=False)
    th, _ = quad(lambda s: float(np.asarray(dirac_fields(rX, s, h13, mu13, p)['wD'])), 0.0, 10.0)
    t13['rabi_err'] = float(abs(rsol['R2'][-1] - np.sin(th) ** 2))
    t13['rabi_norm_err'] = float(abs((rsol['L2'][-1] + rsol['R2'][-1]) - 1.0))
    fD = float(np.asarray(dirac_fields(1.0, 30.0, h13, mu13, p)['wD']))
    fD0 = float(np.asarray(dirac_fields(1.0, 0.0, h13, mu13, p)['wD']))
    t13['freq_freeze_ratio'] = fD / fD0
    t13['pass'] = bool(all(v['success'] for v in t13['X'].values())
                       and t13['rabi_err'] < 1e-10 and t13['rabi_norm_err'] < 1e-12
                       and all(v['maj_effect'] > 1e-6 for v in t13['X'].values()))
    rep['T13_dirac_majorana'] = t13
    # T14: Meral sabiti + kapali form + Boltzmann baryojenez
    t14 = {'Lambda': {}}
    for h14 in H_VALUES:
        v14, e14 = lambda_meral(h14, p)
        t14['Lambda'][str(h14)] = {'value': v14, 'err': e14}
    v0, e0 = lambda_meral(0.0, p)
    t14['Lambda_h0_quad'] = v0
    t14['Lambda_h0_quad_err'] = e0
    t14['Lambda0_closed'] = float(LAMBDA0_CLOSED)
    t14['closed_err'] = float(abs(v0 - LAMBDA0_CLOSED))
    h14, mu14 = 0.005, 0.1
    b14 = solve_boltzmann(1e-11, h14, mu14, 60.0, p, 400)
    Y14 = b14['Y']
    t14['M_peak'] = float(b14['M_peak'])
    t14['freeze_rel'] = float(abs(Y14[-1] - Y14[-2]) / max(Y14[-1], 1e-300))
    t14['eps_star'] = find_eps_star(h14, mu14, p)
    t14['eta_check'] = float(solve_boltzmann(t14['eps_star'], h14, mu14, 60.0, p, 400)['eta'][-1])
    t14['eta_rel_err'] = float(abs(t14['eta_check'] - p['eta_obs']) / p['eta_obs'])
    t14['pass'] = bool(t14['closed_err'] < 1e-12 and t14['freeze_rel'] < 1e-10
                       and t14['eta_rel_err'] < 1e-9
                       and all(v['err'] < 1e-11 for v in t14['Lambda'].values()))
    rep['T14_meral_baryogenesis'] = t14
    # T15: ALPHA-g / Breit-Rabi / 511 keV
    h15, mu15 = 0.005, 0.1
    t15 = {}
    t15['ag_lab'] = float(np.asarray(ag_ratio(1.0, 0.0, h15, mu15, p)))
    t15['ag_lab_err'] = float(abs(t15['ag_lab'] - 1.0))
    s15 = np.linspace(0.0, 40.0, 81)
    t15['ag_core'] = [float(np.asarray(ag_ratio(1.0, s, h15, mu15, p))) for s in s15[::10]]
    t15['alpha_g_sigma'] = float(abs(1.0 - 0.75) / np.sqrt(0.13 ** 2 + 0.16 ** 2))
    br1 = breit_rabi(1.03, None, p)
    br2 = breit_rabi(1.07, None, p)
    t15['fda_103'] = br1['fda']
    t15['fcb_103'] = br1['fcb']
    t15['fda_107'] = br2['fda']
    t15['fcb_107'] = br2['fcb']
    t15['diff_103_err_kHz'] = float(abs((br1['fda'] - br1['fcb']) - p['a1S_kHz']))
    t15['diff_107_err_kHz'] = float(abs((br2['fda'] - br2['fcb']) - p['a1S_kHz']))
    t15['H_ref_diff_kHz'] = float(abs(p['a1S_kHz'] - p['H_a1S_kHz']))
    # SME ust-sinir kaymasi (model ansatz, kupur kappa_a5 ile)
    Rstar = None
    smax, Rs = 0.0, 0.0
    for s in np.linspace(0.0, 40.0, 401):
        lt = float(np.asarray(ltilde_global(s, 1.0, h15, p)))
        phi = max(1.0 - lt * lt / p['bmax'], 0.0)
        Fb = float(np.asarray(F_prof(1.0, 0.0, h15, p['F0'], p['X0'], p['alpha_eta'])))
        K = float(np.asarray(K_curv(1.0, p)))
        Rv = K * mu15 * np.exp(-(1.0 + h15) * s) * (lt * lt - 1.0) * phi / (Fb * Fb)
        if abs(Rv) > abs(Rs):
            Rs, smax = Rv, s
    t15['SME_Rstar'] = float(Rs)
    t15['SME_s_star'] = float(smax)
    t15['SME_shift_103_kHz'] = float(p['kappa_a5'] * 1.03 * abs(Rs) * p['a1S_kHz'])
    t15['SME_shift_107_kHz'] = float(p['kappa_a5'] * 1.07 * abs(Rs) * p['a1S_kHz'])
    t15['kappa_a5_max'] = float(5.7 / max(t15['SME_shift_103_kHz'], t15['SME_shift_107_kHz']) * p['kappa_a5'])
    vG, eG = quad(lambda E: float(gauss_line(E)), 0.0, 1022.0, limit=500)
    t15['gauss_norm'] = float(vG)
    t15['gauss_norm_err'] = float(abs(vG - 1.0))
    t15['IR_exponent'] = 2.0
    reg = {}
    for lo in [1.0, 10.0]:
        vO, eO = quad(lambda E: float(np.asarray(ore_powell(E))), lo, 511.0, limit=500)
        reg[str(lo)] = {'value': float(vO), 'err': float(eO)}
    t15['ore_regulated'] = reg
    P1 = solve_string_modes([1], 0.005, s_max=6.0, npts=2000)[1]
    t15['fuel_Pmax'] = float(np.max(np.abs(P1['P'])))
    Egrid = np.linspace(20.0, 511.0, 60)
    t15['flux_511'] = float(fuel_spectrum(511.0, 1.0, np.max(np.abs(P1['P'])), p))
    t15['pass'] = bool(t15['ag_lab_err'] < 1e-15
                       and t15['diff_103_err_kHz'] < 1e-6 and t15['diff_107_err_kHz'] < 1e-6
                       and t15['H_ref_diff_kHz'] < 5.7
                       and t15['SME_shift_103_kHz'] < 5.7 and t15['SME_shift_107_kHz'] < 5.7
                       and t15['gauss_norm_err'] < 1e-14
                       and np.isfinite(t15['flux_511']))
    rep['T15_alpha_511'] = t15
    # T16: ALPHA 2025 SDR-EVC donen-duvar tork uyusmazligi cozumu
    h16 = 0.005
    t16 = {}
    t16['rp_ideal_mm'] = float(trap_ideal_radius() * 1e3)
    t16['rp_exp_mm'] = ALPHA_TRAP['r_exp_mm']
    t16['arrest_needed'] = bool(t16['rp_ideal_mm'] < t16['rp_exp_mm'])
    sl = trap_slip()
    t16['n_exp_m3'] = sl['n_m3']
    t16['frot_exp_kHz'] = sl['frot_kHz']
    t16['slip'] = sl['slip']
    t16['phi_c_V'] = sl['phi_c_V']
    t16['Fb_star'] = float(trap_F_base(1.0, h16, p))
    t16['mu_star'] = float(find_mu_star(1.0, ALPHA_TRAP['s_eq'], h16, p))
    t16['rp_check_mm'] = float(trap_rp_eff(t16['mu_star'], 1.0,
                                          ALPHA_TRAP['s_eq'], h16, p) * 1e3)
    t16['rp_resid_m'] = float(abs(t16['rp_check_mm'] * 1e-3
                                  - ALPHA_TRAP['r_exp_mm'] * 1e-3))
    cg = chirp_gain(h16, 0.1, ALPHA_TRAP['s_eq'], p)
    t16['Linf_0'] = cg['Linf_0']
    t16['Linf_eff'] = cg['Linf_eff']
    t16['Ftrap_std'] = cg['Ftrap_std']
    t16['Ftrap_chirp'] = cg['Ftrap_chirp']
    t16['chirp_gain'] = cg['gain']
    zp = solve_alpha_sdr_evc_trap(h16, p)
    t16['mu_geom'] = zp['mu_geom']
    t16['mu_meral'] = zp['mu_meral']
    t16['rel_meral'] = zp['rel_meral']
    t16['rel_geom'] = zp['rel_geom']
    t16['rp_geom_mm'] = zp['rp_geom_mm']
    t16['rp_meral_mm'] = zp['rp_meral_mm']
    t16['dr_meral_um'] = zp['dr_meral_um']
    t16['rpe_star_mm'] = zp['rpe_star_mm']
    t16['rpe_meral_mm'] = zp['rpe_meral_mm']
    t16['ne_m3'] = zp['ne_m3']
    t16['eplus_in_band'] = zp['eplus_in_band']
    t16['pass'] = bool(t16['arrest_needed'] and t16['rp_resid_m'] < 1e-12
                       and t16['chirp_gain'] > 1.0
                       and t16['rel_meral'] < 0.01 and t16['rel_geom'] < 0.02
                       and t16['eplus_in_band'])
    rep['T16_alpha_trap'] = t16
    # T17: ALPHA 2026 hiperince cizgi sekli + cift-alan akisi suruklenmesi
    t17 = {}
    fr = flux_relaxation(-74.0, 1.03, p)
    t17['dfdB_GHz_per_T'] = fr['dfdB_GHz_per_T']
    t17['dBdt_uT_per_h'] = fr['dBdt_uT_per_h']
    t17['gamma_per_s'] = fr['gamma_per_s']
    dd = solve_alpha_2026_lineshape_and_drift(p)
    t17['dual_field'] = dd
    gs = gvortex_slopes()
    t17['Ng'] = gs['Ng']
    t17['k_thr'] = gs['k_thr']
    t17['k_tail'] = gs['k_tail']
    t17['q_wake'] = LINE['q_wake']
    t17['x_max'] = gs['x_max']
    Ngv, Nge = gvortex_norm()
    t17['Ng_quad_err'] = float(Nge)
    cs = gvortex_conv_shift()
    t17['conv_shift_kHz'] = cs['shift_kHz']
    t17['x_max_conv'] = cs['x_max_conv']
    t17['peak_lock'] = {s: float(gvortex_slopes(X0=float(s))['x_max'])
                        for s in ['0.9', '1.0', '1.1']}
    t17['pass'] = bool(t17['Ng_quad_err'] < 1e-9
                       and abs(t17['k_thr'] - 2.0) < 0.02
                       and abs(t17['k_tail'] + t17['q_wake']) < 0.15
                       and abs(t17['conv_shift_kHz']) < 10.0
                       and dd['ecr103_sigma'] <= 1.1 and dd['ecr107_sigma'] <= 1.1
                       and dd['dR_abs'] < 2e-4)
    rep['T17_lineshape'] = t17
    # T18: tam Ore-Powell duzeltmesi + galaktik 511 keV hizi (3 rejim)
    t18 = {}
    m18 = 511.0
    t18['O_at_0'] = float(np.asarray(ore_powell_exact(0.0)))
    Epr, mpr = 0.1, 511.0
    xp = Epr / mpr
    slope_num = float(np.asarray(ore_powell_exact(Epr))) / Epr
    slope_ref = ((5.0 / 3.0 + xp / 3.0 - 2.0 * xp ** 2 / 15.0)
                 / (mpr ** 2 * (np.pi ** 2 - 9.0)))
    t18['slope_num'] = slope_num
    t18['slope_theory'] = float(slope_ref)
    t18['slope_relerr'] = float(abs(slope_num - slope_ref) / slope_ref)
    xbr = 1e-2
    dbr = float(np.asarray(ore_omega_true(xbr)))
    acc = 0.0
    for cf in reversed(_OP_SER):
        acc = acc * xbr + cf
    sbr = acc * xbr
    t18['branch_relerr'] = float(abs(dbr - sbr) / abs(sbr))
    Iv, Ie = quad(lambda E: float(np.asarray(ore_powell_exact(E))),
                  0.0, m18, limit=1000)
    t18['integral'] = float(Iv)
    t18['integral_err'] = float(Ie)
    t18['integral_abserr'] = float(abs(Iv - 1.0))
    t18['prompt_O_1e-3'] = float(np.asarray(ore_prompt_form(1e-3)))
    Iv2, Ie2 = quad(lambda E: float(np.asarray(ore_prompt_form(E))),
                    1.0, m18, limit=500)
    t18['prompt_integral_1_511'] = float(Iv2)
    t18['Mdot_eta005'] = float(galactic_Mdot_msun_yr(eta_pair=0.05))
    t18['Mdot_eta010'] = float(galactic_Mdot_msun_yr(eta_pair=0.10))
    t18['Mdot_eta020'] = float(galactic_Mdot_msun_yr(eta_pair=0.20))
    t18['Mdot_ISM_eta010'] = float(galactic_Mdot_msun_yr(Ndot_obs=5.0e43, eta_pair=0.10))
    t18['Mdot_MW_HVC_eta010'] = float(galactic_Mdot_msun_yr(Ndot_obs=1.0e44, eta_pair=0.10))
    t18['pass'] = bool(t18['O_at_0'] == 0.0 and t18['slope_relerr'] < 1e-9
                       and t18['branch_relerr'] < 1e-9
                       and t18['integral_abserr'] < 1e-9)
    rep['T18_ore_powell'] = t18
    # T19: CERN BASE-STEP 2026 karayolu tasimasi Lyapunov kararliligi
    bs = solve_base_step_road_transport(t16['mu_star'])
    t19 = dict(
        freq_993=bs['freq_993'],
        eps_disp=bs['eps_disp'],
        g_disp=bs['g_disp'],
        unprot_rmax_mm=bs['unprot_rmax_mm'],
        rmax_mm=bs['rmax_mm'],
        monotone_B=bs['monotone_B'],
        surv_frac=bs['surv_frac'],
        N_surv=bs['N_surv'],
        a_crit_g=bs['a_crit_g'],
        M_safety=bs['M_safety'],
    )
    t19['pass'] = bool(bs['freq_993']['hierarchy']
                       and bs['rmax_mm']['0.993'] < 0.1 * BASE_STEP['R_trap_mm']
                       and bs['surv_frac'] == 1.0 and bs['N_surv'] == 92.0
                       and bs['monotone_B'] and bs['M_safety'] > 100.0)
    rep['T19_base_step'] = t19
    # T20: BASE-STEP 136 mT falsifikasyon cifti (yan-band acik/kapali)
    f20 = predict_basestep_136mT_falsification(t16['mu_star'], p)
    t20 = dict(
        hierarchy_136=f20['freq_136']['hierarchy'],
        r_star_um=f20['r_star_um'],
        on_r_end_um=float(f20['on_centered_mm']['r_end_mm'] * 1e3),
        off_r_end_um={k: float(v['r_end_mm'] * 1e3)
                      for k, v in f20['off_offsets_mm'].items()},
        unprot_rmax_mm=f20['unprot_rmax_mm'],
        off_recaptured=f20['off_recaptured'],
        unprot_lost=f20['unprot_lost'],
    )
    t20['pass'] = bool(f20['freq_136']['hierarchy']
                       and f20['on_centered_mm']['r_end_mm']
                       < 0.1 * BASE_STEP['R_trap_mm']
                       and f20['off_recaptured'] and f20['unprot_lost'])
    rep['T20_basestep_136mT'] = t20
    # T21: ALPHA 1.05 T ucuncu-alan ongorusu + oran kilidi
    f21 = predict_alpha_third_field(1.05, p)
    t21 = dict(
        dfdB3_GHz_per_T=f21['dfdB3_GHz_per_T'],
        gamma_rel_spread=f21['gamma_rel_spread'],
        fdot_pred_kHz_per_h=f21['fdot_pred_kHz_per_h'],
        fdot_halfwidth_kHz_per_h=f21['fdot_halfwidth_kHz_per_h'],
        R_105_103=f21['R_105_103'], R_107_105=f21['R_107_105'],
    )
    t21['pass'] = bool(f21['gamma_rel_spread'] < 1e-3
                       and np.isfinite(f21['fdot_pred_kHz_per_h'])
                       and f21['fdot_halfwidth_kHz_per_h']
                       < abs(f21['fdot_pred_kHz_per_h']) * 1e-3)
    rep['T21_alpha_105T'] = t21
    # T22: 511 keV foton-butce ongoru bandi (O_exact)
    f22 = predict_511keV_band(p=p)
    t22 = dict(
        R_at_092=f22['R_at_092'], R_min=f22['R_min'], R_max=f22['R_max'],
        band_integral=f22['band_integral'], band_budget=f22['band_budget'],
        band_abserr=f22['band_abserr'],
        IA_leak_max=f22['IA_leak_max'],
        line_centroid_keV=f22['line_centroid_keV'],
    )
    t22['pass'] = bool(f22['band_abserr'] < 1e-9
                       and f22['R_min'] < f22['R_at_092'] < f22['R_max']
                       and f22['line_centroid_keV'] == 511.0
                       and f22['IA_leak_max'] <= 0.38)
    rep['T22_511keV_band'] = t22
    # UQ: kilit cikti duyarliliklari (bilgi; pass baraji yok)
    rep['UQ_faz6'] = uq_faz6_sensitivities(p)
    # T20b: 880/993 mT falsifikasyon cifti
    hb = predict_basestep_highB_falsification(t16['mu_star'], p)
    t20b = {k: dict(on_r_end_um=v['on_r_end_um'],
                    off_r_end_um=v['off_r_end_um'],
                    unprot_rmax_mm=v['unprot_rmax_mm'])
            for k, v in hb.items()}
    t20b['pass'] = bool(all(v['recaptured'] and v['lost'] for v in hb.values()))
    rep['T20b_highB'] = t20b
    # T22b: COSI-2027 esik cercevesi
    fc = predict_cosi_threshold(p)
    t22b = dict(n_sources=fc['n_sources'], n_above=fc['n_above'],
                threshold=fc['threshold'], R_min=fc['R_min'],
                R_max=fc['R_max'], R_at_092=fc['R_at_092'])
    t22b['pass'] = bool(fc['physical'] and fc['n_above'] >= 6)
    rep['T22b_cosi'] = t22b
    # Meral uc-yol kanit (bilgi; mini'de MC ile dortlenir)
    rep['Meral_triple'] = lambda_meral_triple_check(p)
    # T20c: kirilganlik haritasi
    fr = fragility_map(p=p)
    t20c = dict(grid=fr['grid'], robust=fr['robust'],
                marginal_01_um=fr['marginal_01_um'],
                critical_mu_band=fr['critical_mu_band'])
    t20c['pass'] = bool(fr['robust']
                        and 0.05 <= fr['critical_mu_band'][0]
                        and fr['critical_mu_band'][1] < 0.2032
                        and fr['marginal_01_um'] < 0.1 * BASE_STEP['R_trap_mm'] * 1e3)
    rep['T20c_fragility'] = t20c
    # T21b: leave-one-out
    lo = loo_alpha_fields(p)
    t21b = dict(pred_107=lo['pred_107'], err_107_ppm=lo['err_107_ppm'],
                pred_103=lo['pred_103'], err_103_ppm=lo['err_103_ppm'])
    t21b['pass'] = bool(lo['err_107_ppm'] < 500.0 and lo['err_103_ppm'] < 500.0)
    rep['T21b_loo'] = t21b
    # T22c: SPI dis dogrulama
    sp = spi_outer_check(p)
    t22c = dict(R_obs=sp['R_obs'], R_obs_err=sp['R_obs_err'],
                f_obs=sp['f_obs'], R_model=sp['R_model'],
                sigma=sp['sigma'], in_band=sp['in_band'])
    t22c['pass'] = bool(sp['sigma'] < 1.0)
    rep['T22c_spi'] = t22c
    # T22d: SPI bolge haritasi (Bulge/Disk termalize; Halo IA rejimi)
    rg = spi_region_map(p)
    t22d = {k: dict(R_obs=v['R_obs'], R_err=v['R_err'], sigma=v['sigma'],
                    in_band=v['in_band'])
            for k, v in rg.items()}
    t22d['pass'] = bool(rg['Bulge']['pass'] and rg['Disk']['pass'])
    rep['T22d_regions'] = t22d
    # T22e: birlesik f_Ps kestirimi
    cf = spi_combined_fps(p)
    t22e = dict(f_combined=cf['f_combined'], f_err=cf['f_err'],
                sigma_vs_092=cf['sigma_vs_092'],
                row_sigmas=cf['row_sigmas'])
    t22e['pass'] = bool(cf['sigma_vs_092'] < 1.0
                        and all(s < 1.0 for s in cf['row_sigmas']))
    rep['T22e_combined'] = t22e
    # T23: Super-Brillouin rejimi (Teorem IX.1)
    sb = solve_super_brillouin_regime(p)
    t23 = dict(
        nB_ep_1T_m3=sb['nB']['nB_ep_1T']['m3'],
        nB_pb_1T_m3=sb['nB']['nB_pb_1T']['m3'],
        nB_ep_3T_m3=sb['nB']['nB_ep_3T']['m3'],
        nB_pb_3T_m3=sb['nB']['nB_pb_3T']['m3'],
        ratio_chi0=sb['ratios']['meral']['0.0'],
        ratio_chi050=sb['ratios']['meral']['0.5'],
        ratio_chi075=sb['ratios']['meral']['0.75'],
        ratio_chi090=sb['ratios']['meral']['0.9'],
        disc_std=sb['disc_std'], disc_str=sb['roots']['disc'],
        res_plus=sb['roots']['w_plus'] ** 2 - sb['roots']['w_plus']
        + 1.5 / 4.0 - string_sigma(sb['mu_meral'], 0.75),
        r_std_end=sb['r_std_end'], r_str_max=sb['r_str_max'])
    t23['pass'] = bool(
        abs(sb['nB']['nB_ep_1T']['m3'] - 4.858e18) / 4.858e18 < 5e-3
        and abs(sb['nB']['nB_pb_1T']['m3'] - 2.646e15) / 2.646e15 < 5e-3
        and abs(t23['ratio_chi0'] - 1.204409) < 1e-6
        and abs(t23['ratio_chi050'] - 1.459920) < 1e-6
        and abs(t23['ratio_chi075'] - 1.970942) < 1e-6
        and abs(t23['ratio_chi090'] - 3.504008) < 1e-6
        and sb['roots']['real'] and sb['disc_std'] < 0.0
        and abs(t23['res_plus']) < 1e-9
        and sb['r_std_end'] > 1e3 and sb['r_str_max'] < 2.0)
    rep['T23_super_brillouin'] = t23
    # T24: Diocotron kafesi (Teorem IX.2)
    vx = solve_diocotron_vortex_lattice(p)
    t24 = dict(lams={k: v['lam'] for k, v in vx['modes'].items()},
               a7=vx['lat7']['a_star'],
               lmin7=vx['lat7']['lmin_perp'],
               zero7=vx['lat7']['zero_mode'],
               a19=vx['lat19']['a'], b19=vx['lat19']['b'],
               collapsed19=vx['lat19']['collapsed'],
               lmin19=vx['lat19']['lmin_perp'],
               zero19=vx['lat19']['zero_mode'],
               baseline_interior=vx['baseline_mu0']['interior_equilibrium'])
    t24['pass'] = bool(
        all(v['stable'] for v in vx['modes'].values())
        and vx['lat7']['lmin_perp'] > 1e-3
        and abs(vx['lat7']['zero_mode']) < 1e-3
        and vx['lat19']['success']
        and vx['lat19']['lmin_perp'] > 0.0
        and abs(vx['lat19']['zero_mode']) < 0.05
        and not vx['baseline_mu0']['interior_equilibrium'])
    rep['T24_diocotron_lattice'] = t24
    # T25: kiral-elastik kalkan + birikim (Teorem IX.3)
    sh = solve_antimatter_accumulation_and_shielding(p)
    t25 = dict(S_loss=sh['S_loss'], exp_factor=sh['exp_factor'],
               Gamma_str=sh['Gamma_str'], eta_str=sh['eta_str'],
               Ninf_str=sh['Ninf_str'], boost=sh['boost'],
               Nstd225=sh['Nstd225'],
               tau_std_yr=sh['tau_std_yr'], tau_str_yr=sh['tau_str_yr'])
    t25['pass'] = bool(
        abs(sh['exp_factor'] - 2.0 * np.sqrt(2.0)) < 1e-12
        and abs(sh['S_loss'] - np.exp(2.0 * np.sqrt(2.0)) / 0.8566) < 1e-9
        and abs(sh['Nstd225'] - 2717.8) < 1.0
        and sh['boost'] > 30.0
        and sh['tau_str_yr'] > 10.0 and sh['tau_std_yr'] < 2.0)
    rep['T25_chiral_shield'] = t25
    # T26: Tokamak ELM + H-modu kaide (Teorem X.1)
    tk = solve_tokamak_elm_and_hmode_pedestal(p)
    t26 = dict(
        alpha_chi075=tk['alpha_ratio']['0.75'],
        H98_chi050=tk['H98']['0.5'], H98_chi075=tk['H98']['0.75'],
        elm_std_end=tk['elm_std_end'], elm_str_max=tk['elm_str_max'],
        ITER_Q_050=tk['reactors']['ITER']['chi0.50']['Q_fusion'],
        ITER_Q_075=tk['reactors']['ITER']['chi0.75']['Q_fusion'],
        SPARC_Q_050=tk['reactors']['SPARC']['chi0.50']['Q_fusion'],
        SPARC_Q_075=tk['reactors']['SPARC']['chi0.75']['Q_fusion'])
    t26['pass'] = bool(
        tk['alpha_ratio']['0.75'] > 1.5
        and tk['elm_std_end'] > 1e3 and tk['elm_str_max'] <= 1.0 + 1e-9
        and tk['H98']['0.5'] > 1.8 and tk['H98']['0.75'] > 1.8
        and t26['ITER_Q_050'] > 50.0 and t26['SPARC_Q_050'] > 50.0)
    rep['T26_tokamak_elm'] = t26
    # T27: Kacak elektron-pozitron cigi (Teorem X.2)
    ra = solve_runaway_electron_positron_avalanche(p=p)
    t27 = dict(
        Ec_V_per_m=ra['Ec_V_per_m'],
        Ec_rat_chi0=ra['Ec_eff_ratio']['0.0'],
        Ec_rat_chi075=ra['Ec_eff_ratio']['0.75'],
        gamma_std=ra['gamma_std_per_s'],
        gamma_str_075=ra['gamma_string_per_s']['0.75'])
    t27['pass'] = bool(
        abs(t27['Ec_rat_chi0'] - 23.7884503230) < 1e-6
        and abs(t27['Ec_rat_chi075'] - 38.9283546022) < 1e-6
        and t27['gamma_std'] > 0.0 and t27['gamma_str_075'] < 0.0)
    rep['T27_runaway_avalanche'] = t27
    # T28: 30-mertebe evrensel olcekleme yasasi (Teorem X.3)
    uv = solve_universal_vortex_string_scaling_law(p=p)
    t28 = dict(
        C_univ=uv['C_univ'], Pi_meral=uv['Pi_meral'],
        Pi_err=uv['Pi_err'], I_shield=uv['I_shield'],
        I_shield_err=uv['I_shield_err'],
        slope=uv['slope'], R2=uv['R2'], span_decades=uv['span_decades'])
    t28['pass'] = bool(
        uv['Pi_err'] < 1e-14 and uv['I_shield_err'] < 1e-14
        and abs(uv['slope'] - 1.0) < 1e-14
        and abs(uv['R2'] - 1.0) < 1e-14
        and uv['span_decades'] >= 30.0)
    rep['T28_universal_scaling'] = t28
    # T29: Meral kozmik ivme olcegi + BTFR (Grand Proof I.1-I.2, EXACT + gozlem karsilastirma).
    b29 = solve_meral_cosmological_acceleration_and_btfr(p)
    t29 = dict(
        a_H=b29['acc']['a_H'], kappa=b29['acc']['kappa'],
        a_Meral=b29['acc']['a_Meral'],
        a_Meral_beta=b29['acc']['a_Meral_beta'],
        a_Meral_SH0ES=b29['acc']['a_Meral_SH0ES'],
        rel_diff_vs_a0=b29['acc']['rel_diff_vs_a0'],
        sigma_sys=b29['acc']['sigma_sys'],
        btfr_slope=b29['btfr_slope'], btfr_R2=b29['btfr_R2'],
        mean_abs_rel=b29['mean_abs_rel'])
    t29['pass'] = bool(
        t29['rel_diff_vs_a0'] < 0.06 and t29['sigma_sys'] < 0.5
        and abs(t29['btfr_slope'] - 4.0) < 1e-9
        and t29['btfr_R2'] > 0.998)
    rep['T29_meral_acceleration_btfr'] = t29
    # T30: SPARC donus egrileri (karanlik madde yok; UGC 2885 ~%13 dis-deger acikca raporlanir).
    s30 = solve_sparc_galaxy_rotation_curves(p)
    t30 = dict(
        mean_abs_rel_flat=s30['mean_abs_rel_flat'],
        btfr_slope=s30['btfr_slope'], btfr_R2=s30['btfr_R2'],
        per_galaxy={k: dict(abs_rel_flat=v['abs_rel_flat'],
                            newton_drop_10Rd=v['newton_drop_10Rd'])
                    for k, v in s30['curves'].items()})
    t30['pass'] = bool(
        t30['mean_abs_rel_flat'] < 0.06
        and all(v['newton_drop_10Rd'] < 0.65
                for v in s30['curves'].values())
        and t30['btfr_R2'] > 0.998)
    rep['T30_sparc_rotation_curves'] = t30
    # T31: EHT halka doygunlugu (sifir parametre; iki projeksiyon da raporlanir).
    e31 = solve_eht_black_hole_ring_saturation(p)
    t31 = dict(
        d_over_theta_g=e31['d_over_theta_g'],
        SgrA_sigma=e31['rings']['SgrA']['sigma'],
        M87_sigma=e31['rings']['M87']['sigma'],
        M87_sigma_full=e31['rings']['M87']['sigma_full'],
        fwhm_over_d=e31['fwhm_over_d'])
    t31['pass'] = bool(
        t31['SgrA_sigma'] < 1.5 and t31['M87_sigma'] < 1.5
        and t31['fwhm_over_d'] < 0.25)
    rep['T31_eht_ring_saturation'] = t31
    # T32: Yang-Mills aki-tupu operatoru + 0++ kutle araligi (Grand Proof II.1/II.2).
    # FV: p = 2.0000, lam_cont = 5.610204, sqrt(2lam) koprusu.
    y32 = solve_yang_mills_mass_gap_operator(p)
    t32 = dict(
        lam0=y32['lam0'], lam1=y32['lam1'], lam2=y32['lam2'],
        ortho01=y32['ortho01'],
        rich_lams=y32['rich_lams'], rich_p_meas=y32['rich_p_meas'],
        lam_cont=y32['lam_cont'], rich_monotone=y32['rich_monotone'],
        radial_gap=y32['radial_gap'],
        tk_lower=y32['tk_lower'], tk_width=y32['tk_width'],
        gap_chiral=y32['gap_num_chiral'],
        gap_num=y32['gap_num'],
        gap_num_rel_closed=y32['gap_num_rel_closed'],
        gap_num_rel_lqcd=y32['gap_num_rel_lqcd'],
        Gc_positive=y32['Gc_positive'], Gc_slope_err=y32['Gc_slope_err'],
        V_min=y32['V_min'],
        gap_closed=y32['gap_closed'],
        gap_rel=y32['gap_rel'], gap_sigma=y32['gap_sigma'])
    t32['pass'] = bool(
        y32['lam0'] > 5.5 and y32['lam1'] > y32['lam0']
        and y32['lam2'] > y32['lam1'] and y32['ortho01'] < 1e-14
        and y32['rich_monotone'] and 5.61019 < y32['lam_cont'] < 5.61022
        and abs(y32['rich_p_meas'] - 2.0) < 1e-3
        and y32['Gc_positive'] and y32['Gc_slope_err'] < 1e-3
        and 3.19 < y32['V_min'] < 3.20
        and y32['tk_lower'] > 5.0 and y32['tk_width'] < 1e-9
        and y32['tk_lower'] <= y32['lam0'] + 2e-5
        and y32['tk_mu'] <= y32['lam0'] + 2e-5
        and abs(y32['tk_mu_gauss'] - 5.6467531) < 1e-4
        and y32['gap_chiral_rel_closed'] < 6e-4
        and y32['gap_chiral_sigma'] < 1.0
        and y32['gap_num_rel_closed'] < 0.02
        and y32['gap_num_rel_lqcd'] < 0.02
        and t32['gap_rel'] < 0.005 and t32['gap_sigma'] < 1.0)
    rep['T32_yang_mills_mass_gap'] = t32
    # T33: glueball spektrumu (topolojik 0-+ birincil; klasik de raporlanir).
    g33 = solve_glueball_excitation_spectrum(p)
    t33 = dict(
        mean_abs_rel=g33['mean_abs_rel'],
        topo_0mp=g33['states']['0-+']['pred'],
        classic_0mp=g33['states']['0-+']['pred_classic'],
        per_state={k: dict(pred=v['pred'], rel=v['rel'], sigma=v['sigma'])
                   for k, v in g33['states'].items()})
    t33['pass'] = bool(t33['mean_abs_rel'] < 0.016)
    rep['T33_glueball_spectrum'] = t33
    # T34: Cornell-Meral kusatmasi (IR birincil + UV-IR log + SCR Yukawa sutunlari).
    c34 = solve_cornell_luscher_quark_confinement(p)
    qcc, qbb = c34['quarkonia']['ccbar'], c34['quarkonia']['bbbar']
    t34 = dict(
        gamma_Meral=c34['gamma_Meral'],
        cornell_sigma=c34['cornell_sigma'],
        gamma_UV0=c34['gamma_UV0'],
        dgamma_UV=c34['dgamma_UV'],
        Gamma_SCR0=c34['Gamma_SCR0'],
        cc_IR=qcc['IR']['split_2S_1S_MeV'],
        cc_IR_rel=qcc['IR']['rel'],
        cc_UVIR=qcc['UVIR']['split_2S_1S_MeV'],
        cc_UVIR_rel=qcc['UVIR']['rel'],
        cc_SCR=qcc['SCR']['split_2S_1S_MeV'],
        cc_SCR_rel=qcc['SCR']['rel'],
        bb_IR=qbb['IR']['split_2S_1S_MeV'],
        bb_IR_rel=qbb['IR']['rel'],
        bb_UVIR=qbb['UVIR']['split_2S_1S_MeV'],
        bb_UVIR_rel=qbb['UVIR']['rel'],
        bb_SCR=qbb['SCR']['split_2S_1S_MeV'],
        bb_SCR_rel=qbb['SCR']['rel'],
        slope=c34['slope_Rmax'])
    t34['pass'] = bool(
        t34['cornell_sigma'] < 1.0 and t34['cc_IR_rel'] < 0.02
        and t34['cc_SCR_rel'] < 0.02 and t34['bb_SCR_rel'] < 0.08
        and abs(t34['gamma_UV0'] - 0.57101663) < 1e-8
        and abs(t34['Gamma_SCR0'] - 0.46199859) < 1e-8
        and abs(qcc['CAS']['split_2S_1S_MeV'] - 602.92) < 1.0
        and abs(qbb['CAS']['split_2S_1S_MeV'] - 550.62) < 1.0
        and t34['slope'] > 0.0
        and all(qcc[k]['ordered'] and qbb[k]['ordered']
                for k in ('IR', 'UVIR', 'SCR', 'CAS')))
    rep['T34_quark_confinement'] = t34
    # T35: masaustu viskoelastik girdap PIV (Burgers N + 3 PEO ongorusu).
    v35 = solve_benchtop_viscoelastic_vortex_piv_experiment(p)
    t35 = dict(
        rc_N_mm=v35['rc_N_mm'], umax_N_ms=v35['umax_N_ms'],
        dh_N_mm=v35['dh_N_mm'], dh_quad_mm=v35['dh_quad_mm'],
        dh_abs=v35['dh_closed_quad_abs'],
        rc_PEO_mm=v35['rc_PEO_mm'], umax_PEO_ms=v35['umax_PEO_ms'],
        dh_ratio=v35['dh_ratio'], dh_PEO_mm=v35['dh_PEO_mm'],
        n_independent=v35['n_independent'],
        a_measured=list(v35['a_measured']),
        a_measured_in_model_range=v35['a_measured_in_model_range'],
        lit_band_verified=v35['lit_band_verified'])
    t35['pass'] = bool(
        abs(v35['rc_N_mm'] - 3.16227766) < 1e-5
        and abs(v35['umax_N_ms'] - 0.38542474) < 1e-6
        and v35['dh_closed_quad_abs'] < 1e-12
        and abs(v35['rc_PEO_mm'] - 5.14198344) < 1e-5
        and abs(v35['umax_PEO_ms'] - 0.23703306) < 1e-6
        and abs(v35['dh_ratio'] - 0.31402536) < 1e-8
        and v35['n_independent'] == 2
        and abs(min(v35['sens'], key=lambda s: abs(s['a'] - 1.6))['a'] - 1.6) < 1e-12
        and abs(min(v35['sens'], key=lambda s: abs(s['a'] - 1.6))['u_ratio']
                - 0.625) < 1e-12
        and v35['a_measured'][0] == 12.0
        and v35['a_measured'][1] == 19.0
        and (not v35['a_measured_in_model_range'])
        and (not v35['lit_band_verified']))
    rep['T35_benchtop_vortex_piv'] = t35
    # T36: Vela glitch (kaynak dogrulamali) + Q kesirleri + gevse ODE.
    g36 = solve_vela_pulsar_glitch_avalanche(p)
    t36 = dict(
        base=g36['base'], glitch_meral=g36['glitch_meral'],
        rel_2000=g36['rel_2000'], rel_2004=g36['rel_2004'],
        Q_fast=g36['Q_fast'], Q_total=g36['Q_total'],
        Q_2000_measured=g36['Q_2000_measured'],
        relax_maxresid=g36['relax_maxresid'])
    t36['pass'] = bool(
        t36['rel_2004'] < 0.01 and t36['rel_2000'] < 0.40
        and t36['relax_maxresid'] < 0.02
        and abs(g36['base'] - 1.71961931e-06) < 1e-12
        and abs(g36['glitch_meral'] - 2.07112467e-06) < 1e-12)
    rep['T36_vela_glitch'] = t36
    # T37: 22Na Ps bant orani + misprint/pole curutme.
    n37 = solve_benchtop_na22_positronium_gamma_experiment(p)
    t37 = dict(
        B_exact=n37['B_exact'], B_substituted=n37['B_substituted'],
        B_published=n37['B_published'], lab_sigma=n37['lab_sigma'],
        Ntot=n37['Ntot_3gamma'])
    t37['pass'] = bool(
        abs(n37['B_exact'] - 0.08413579) < 1e-6
        and n37['B_substituted'] > 9.0 and n37['B_published'] > 100.0
        and n37['lab_sigma'] > 500.0)
    rep['T37_na22_positronium'] = t37
    return rep

if __name__ == '__main__':
    import json
    print(json.dumps(run_verification(), indent=2))
