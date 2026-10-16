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
    muN_h = 7.622593285e9
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
    return rep

if __name__ == '__main__':
    import json
    print(json.dumps(run_verification(), indent=2))
