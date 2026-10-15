"""Duran Sicimler Teoremi V - Sayisal Cozucu Motoru.
Eylul 2026 Navier-Stokes ic-cekirdek (Denk. 3.2, 4.2-4.14) + elastik sicim gerilimi.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar

# ---------------- Parametreler ----------------
DEFAULTS = dict(
    F0=1.0, X0=1.0, alpha_eta=0.3, U0=1.0,
    kappa_r=1.0, gamma_rel=0.1, ell0=1.0,
    X_star=1.0, c_s=5.0, ell_light0=1.0, beta0=0.5,
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
    """int_0^X (1+x/X0)^{-(1+h)} dx = X0/h * (1-(1+X/X0)^{-h}). h->0 limiti log."""
    X = np.asarray(X, dtype=float)
    if abs(h) < 1e-12:
        return X0 * np.log(1.0 + X / X0)
    return X0 / h * (1.0 - (1.0 + X / X0) ** (-h))

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
    return rep

if __name__ == '__main__':
    import json
    print(json.dumps(run_verification(), indent=2))
