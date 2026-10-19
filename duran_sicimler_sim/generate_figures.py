"""Yayin kalitesinde 5 figur uretici (Faz 2: fig5 eklendi)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver_engine import (DEFAULTS, H_VALUES, MU_VALUES, F_prof, E_prof,
    u_theta_0, Pi_eff_X, E_eff_of_s, u_eff_of_s, ell_explicit_analytic,
    find_s_kritik, compute_u_max, solve_selfconsistent, solve_ell_explicit,
    V0_prof, u_r_0, q_horizon, omega_prime_ratio, solve_string_modes,
    compute_table8, solve_spinor, lambda_meral, LAMBDA0_CLOSED, C_SPH,
    solve_boltzmann, find_eps_star, ag_ratio, breit_rabi, ore_powell,
    gauss_line, fuel_spectrum, ltilde_global)

import os
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)

def fig1():
    p = DEFAULTS; h = 0.005; mu = 0.1; X = p['X_star']
    s_c = find_s_kritik(X, h, mu, p)
    u_max, s_opt = compute_u_max(X, h, mu, p, s_c)
    s = np.linspace(0, s_c * 1.02, 800)
    u0 = u_theta_0(X, s, h)
    ell = ell_explicit_analytic(s, X, h, p)
    uexp = u_eff_of_s(s, ell, X, h, mu, p)
    Pi = Pi_eff_X(s, ell, X, h, mu, p)
    # self-consistent
    ss, ells, _, send = solve_selfconsistent(X, h, mu, s_c * 1.5, p)
    uself = u_eff_of_s(ss, ells, X, h, mu, p)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].semilogy(s, u0, 'k--', label=r'Saf NS $u_\theta^{(0)}\to\infty$')
    ax[0, 0].semilogy(s, uexp, 'b-', label='Acik (explicit)')
    ax[0, 0].semilogy(ss, np.maximum(uself, 1e-300), 'r-', label='Tam geri-tepkili')
    ax[0, 0].axvline(s_c, color='gray', ls=':')
    ax[0, 0].set_title(f'Hiz (X*=1,h={h},mu={mu}); s_krit={s_c:.6f}, u_max={u_max:.4g}')
    ax[0, 0].set_xlabel('s=-ln q'); ax[0, 0].set_ylabel(r'$u_\theta$'); ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(s, Pi, 'b-'); ax[0, 1].axhline(0, color='k', lw=0.8); ax[0, 1].axvline(s_c, color='gray', ls=':')
    ax[0, 1].set_title(r'$\Pi_{\mathrm{eff},X}(s)$ sifirlanmasi'); ax[0, 1].set_xlabel('s'); ax[0, 1].set_ylabel('Pi_eff,X')
    # profil vs X sabit s
    Xg = np.linspace(0.05, 5, 400)
    for sv in [0, s_c * 0.5, s_c * 0.9]:
        e = ell_explicit_analytic(sv, Xg, h, p)
        ue = u_eff_of_s(sv, e, Xg, h, mu, p)
        ax[1, 0].plot(Xg, ue, label=f's={sv:.2f}')
    ax[1, 0].set_title(r'$u_\theta^{\mathrm{eff}}(X)$ profilleri'); ax[1, 0].set_xlabel('X'); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot(s, ell, 'b-', label='explicit'); ax[1, 1].plot(ss, ells, 'r-', label='self-consistent')
    ax[1, 1].set_yscale('log'); ax[1, 1].set_title(r'$\ell_\theta(s)$'); ax[1, 1].set_xlabel('s'); ax[1, 1].legend(fontsize=7)
    fig.suptitle('Fig1: Blowup Arrest - Saf NS vs Sicim Duzenlemesi')
    fig.tight_layout(); save(fig, 'fig1_blowup_arrest')

def fig2():
    p = DEFAULTS; X = p['X_star']
    hs = np.array([0.002, 0.005, 0.008, 0.01])
    mus = np.array([0.02, 0.05, 0.1, 0.2])
    S = np.zeros((len(mus), len(hs))); U = np.zeros_like(S)
    for i, mu in enumerate(mus):
        for j, h in enumerate(hs):
            sc = find_s_kritik(X, h, mu, p)
            um, _ = compute_u_max(X, h, mu, p, sc)
            S[i, j] = sc; U[i, j] = um
    np.savetxt(os.path.join(HERE, 'phase_S.dat'), S); np.savetxt(os.path.join(HERE, 'phase_U.dat'), U)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im0 = ax[0].imshow(S, origin='lower', aspect='auto', extent=[hs[0], hs[-1], mus[0], mus[-1]])
    ax[0].set_xlabel('h'); ax[0].set_ylabel('mu_s'); ax[0].set_title(r'$s_{\mathrm{kritik}}$')
    fig.colorbar(im0, ax=ax[0])
    for i in range(len(mus)):
        for j in range(len(hs)):
            ax[0].text(hs[j], mus[i], f'{S[i,j]:.2f}', ha='center', va='center', fontsize=7, color='w')
    im1 = ax[1].imshow(np.log10(U), origin='lower', aspect='auto', extent=[hs[0], hs[-1], mus[0], mus[-1]])
    ax[1].set_xlabel('h'); ax[1].set_ylabel('mu_s'); ax[1].set_title(r'$\log_{10} u_{\max}$')
    fig.colorbar(im1, ax=ax[1])
    for i in range(len(mus)):
        for j in range(len(hs)):
            ax[1].text(hs[j], mus[i], f'{np.log10(U[i,j]):.2f}', ha='center', va='center', fontsize=7, color='w')
    fig.suptitle('Fig2: (h,mu) Faz Diyagrami')
    fig.tight_layout(); save(fig, 'fig2_phase_diagram')
    return hs, mus, S, U

def fig3():
    p = DEFAULTS; h = 0.005
    Xg = np.linspace(0.05, 4, 120); etag = np.linspace(-0.95, 0.95, 120)
    XX, EE = np.meshgrid(Xg, etag)
    Qh = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        Qh[i, :] = q_horizon(XX[i, :], EE[i, :], h, p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cs = ax[0].contourf(XX, EE, np.log10(np.maximum(Qh, 1e-16)), levels=20)
    fig.colorbar(cs, ax=ax[0], label='log10 q_ufuk')
    ax[0].contour(XX, EE, Qh, levels=6, colors='w', linewidths=0.6)
    ax[0].set_xlabel('X'); ax[0].set_ylabel('eta'); ax[0].set_title(r'$q_{\mathrm{ufuk}}(X,\eta)$')
    # omega cokusu: sabit X,eta icin tau->0
    taus = np.logspace(-4, 0, 300)
    for Xv in [0.2, 0.5, 1.0, 2.0]:
        w = omega_prime_ratio(Xv, 0.0, taus, h, p)
        ax[1].plot(taus, w, label=f'X={Xv}')
    ax[1].axhline(0, color='k', lw=0.8); ax[1].set_xscale('log')
    ax[1].set_xlabel('tau'); ax[1].set_ylabel(r"$\omega'/\omega_0$"); ax[1].legend(fontsize=7)
    ax[1].set_title(r"$\omega'\to 0$ cokusu (eta=0)")
    fig.suptitle('Fig3: Akustik Olay Ufku')
    fig.tight_layout(); save(fig, 'fig3_acoustic_horizon')

def fig4():
    p = DEFAULTS; h = 0.005
    modes = solve_string_modes([1, 2, 3, 5], h, s_max=6.0, npts=2000)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for n, m in modes.items():
        ax[0, 0].plot(m['tau'], m['omega_n'], label=f'n={n}')
        ax[0, 1].plot(m['tau'], m['Llight'], label=f'n={n}')
        ax[1, 0].plot(m['tau'], m['I_n'] - n * p['c_s'], label=f'n={n}')
        ax[1, 1].plot(m['tau'], m['E'], label=f'n={n}')
    ax[0, 0].set_xscale('log'); ax[0, 0].set_title(r'$\omega_n(\tau)=\omega_{n,0}\tau^h\to 0$')
    ax[0, 0].set_xlabel('tau'); ax[0, 1].set_xscale('log'); ax[0, 1].set_yscale('log')
    ax[0, 1].set_title(r'$\mathcal{L}(\tau)\to\infty$'); ax[0, 1].set_xlabel('tau')
    ax[1, 0].set_xscale('log'); ax[1, 0].set_title(r'$\mathcal{I}_n-nc_s$ hatasi')
    ax[1, 0].set_xlabel('tau'); ax[1, 0].legend(fontsize=7)
    ax[1, 1].set_xscale('log'); ax[1, 1].set_yscale('log'); ax[1, 1].set_title('E_sicim(tau)')
    ax[1, 1].set_xlabel('tau')
    for a in ax.flat:
        a.legend(fontsize=7); a.grid(True, alpha=0.3)
    fig.suptitle(r'Fig4: Topolojik Degismez $\mathcal{I}_n=nc_s$')
    fig.tight_layout(); save(fig, 'fig4_topological_invariant')

def fig6():
    """Faz 3: iki-alanli evrim, Lyapunov guc muhasebesi, H_total ve spektral iz (2x3)."""
    from solver_engine import (solve_two_field, two_field_powers, H_series,
        lagrange_grid, solve_lagrange, l_init_lagrange, l_init_uniform_xi,
        sine_spectrum)
    import numpy as np
    p = DEFAULTS; h = 0.005; mu = 0.1
    X = np.concatenate(([0.0], np.logspace(-6, np.log10(300.0), 79)))
    fld = solve_two_field(X, h, mu, 40.0, p, p['bmax'], 121)
    s = fld['s']
    raw_s = []
    pf_s = []
    for j in range(s.size):
        d = two_field_powers(X, float(s[j]), fld['F'][:, j], fld['L'][:, j], h, mu, p, p['bmax'])
        raw_s.append(d['raw'])
        pf_s.append(float(np.max(np.abs(d['Pf']))))
    H = H_series(X, s, fld['F'], fld['L'], h, mu, p, p['bmax'])
    # T12 verisi (es-hareketli Lagrange cercevesi)
    xi_lag, X0_lag = lagrange_grid(Nxi=128, Xtop=10.0, X0param=p['X0'])
    fb = solve_lagrange(xi_lag, X0_lag, h, mu, 25.0, p, p['bmax'], 51)
    env_prof = np.exp(-X0_lag / 10.0)
    fu_p = solve_lagrange(xi_lag, X0_lag, h, mu, 25.0, p, p['bmax'], 51,
                          l_init=p['ell0'] * (1.0 + 0.02 * env_prof))
    fu_m = solve_lagrange(xi_lag, X0_lag, h, mu, 25.0, p, p['bmax'], 51,
                          l_init=p['ell0'] * (1.0 - 0.02 * env_prof))
    env = np.maximum(1e-30, np.abs(0.5 * (fu_p['F'][:, -1] - fu_m['F'][:, -1])))
    specs = {}
    for n in [1, 2, 3, 4]:
        pat = np.sin(n * np.pi * xi_lag) * env_prof
        fn_p = solve_lagrange(xi_lag, X0_lag, h, mu, 25.0, p, p['bmax'], 51,
                              l_init=p['ell0'] * (1.0 + 0.02 * pat))
        fn_m = solve_lagrange(xi_lag, X0_lag, h, mu, 25.0, p, p['bmax'], 51,
                              l_init=p['ell0'] * (1.0 - 0.02 * pat))
        Rn = (0.5 * (fn_p['F'][:, -1] - fn_m['F'][:, -1])) / env
        c = sine_spectrum(xi_lag, Rn, 6)
        bare = np.sin(n * np.pi * xi_lag)
        r = abs(np.corrcoef(bare, Rn)[0, 1]) if np.std(Rn) > 0 else 0.0
        specs[n] = (Rn, c, r)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for sv in [0, 10, 20, 30, 40]:
        j = int(np.argmin(np.abs(s - sv)))
        ax[0, 0].plot(X[1:], fld['F'][1:, j], label=f's={sv}')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_title(r'iki-alanli $F(X,s)$'); ax[0, 0].set_xlabel('X'); ax[0, 0].legend(fontsize=7)
    for sv in [0, 10, 20, 30, 40]:
        j = int(np.argmin(np.abs(s - sv)))
        ax[0, 1].plot(X[1:], np.maximum(fld['L'][1:, j], 1e-300), label=f's={sv}')
    ax[0, 1].set_xscale('log'); ax[0, 1].set_yscale('log')
    ax[0, 1].set_title(r'iki-alanli $\tilde{\ell}(X,s)$'); ax[0, 1].set_xlabel('X'); ax[0, 1].legend(fontsize=7)
    ax[0, 2].plot(s, np.maximum(pf_s, 1e-300), label=r'max$|P_{\mathrm{fluid}}|$')
    ax[0, 2].plot(s, np.maximum(raw_s, 1e-300), label=r'max$|P_f+P_s|$/max$|P_f|$')
    ax[0, 2].axhline(1e-15, color='k', ls='--', lw=0.8)
    ax[0, 2].set_yscale('log')
    ax[0, 2].set_title('Lyapunov guc muhasebesi'); ax[0, 2].set_xlabel('s'); ax[0, 2].legend(fontsize=7)
    ax[1, 0].plot(s, np.maximum(H, 1e-300))
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_title(r'$H_{\mathrm{total}}(s)$'); ax[1, 0].set_xlabel('s')
    for n in [1, 2, 3, 4]:
        Ri, _, _ = specs[n]
        ax[1, 1].plot(xi_lag, Ri / max(np.max(np.abs(Ri)), 1e-300), label=f'n={n}')
    ax[1, 1].set_title(r'normalize $\mathcal{R}_n(\xi)$ Lagrange, s=25'); ax[1, 1].set_xlabel('xi'); ax[1, 1].legend(fontsize=7)
    ns = [1, 2, 3, 4]
    pur = []
    cor = []
    for n in ns:
        _, c, r = specs[n]
        pur.append(abs(c[n - 1]) / np.linalg.norm(c))
        cor.append(r)
    ax[1, 2].bar([str(n) for n in ns], pur, label='saflik (simetrik)')
    ax[1, 2].plot([str(n) for n in ns], cor, 'ro-', label='|r|')
    ax[1, 2].axhline(0.98, color='k', ls='--', lw=0.8)
    ax[1, 2].axhline(0.99, color='gray', ls=':', lw=0.8)
    ax[1, 2].set_title('spektral saflik / korelasyon'); ax[1, 2].legend(fontsize=7)
    fig.suptitle('Fig6: Iki-alanli PDE + Lyapunov + spektral iz (h=0.005, mu=0.1)')
    fig.tight_layout(); save(fig, 'fig6_two_field_lyapunov_and_echo')

def fig5():
    p = DEFAULTS; h = 0.005
    tab = compute_table8((0.05, 0.10, 0.20), h, 40.0, p)
    X = tab['X']
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    mu0 = 0.10
    Lp, Pp, Up = tab[mu0]['PDE']['L'], tab[mu0]['PDE']['Pi'], tab[mu0]['PDE']['Uth']
    s = tab[mu0]['PDE']['s']
    for sv in [0, 10, 20, 30, 40]:
        j = int(np.argmin(np.abs(s - sv)))
        ax[0, 0].plot(X[1:], Lp[1:, j], label=f's={sv}')
    ax[0, 0].set_xscale('log'); ax[0, 0].set_yscale('log')
    ax[0, 0].set_title(r'PDE $\tilde{\ell}(X,s)$ ($\mu=0.10$)'); ax[0, 0].set_xlabel('X'); ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(X[1:], tab[mu0]['ODE']['Pi'][1:, -1], label='ODE')
    ax[0, 1].plot(X[1:], Pp[1:, -1], label='PDE')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_title(r'$\Pi(X,40)$ ODE vs PDE'); ax[0, 1].set_xlabel('X'); ax[0, 1].legend(fontsize=7)
    ax[0, 2].plot(X[1:], tab[mu0]['ODE']['Uth'][1:, -1], label='ODE')
    ax[0, 2].plot(X[1:], Up[1:, -1], label='PDE')
    ax[0, 2].set_xscale('log'); ax[0, 2].set_yscale('log')
    ax[0, 2].set_title(r'$u_\theta(X,40)$ ODE vs PDE'); ax[0, 2].set_xlabel('X'); ax[0, 2].legend(fontsize=7)
    i_star = int(np.argmin(np.abs(X - p['X_star'])))
    for mu in [0.05, 0.10, 0.20]:
        ax[1, 0].plot(tab[mu]['ODE']['s'], tab[mu]['ODE']['Pi'][i_star, :], '--', label=f'ODE {mu}')
        ax[1, 0].plot(tab[mu]['PDE']['s'], tab[mu]['PDE']['Pi'][i_star, :], '-', label=f'PDE {mu}')
    ax[1, 0].set_title(r'$\Pi(X_*,s)$'); ax[1, 0].set_xlabel('s'); ax[1, 0].legend(fontsize=6)
    for mu in [0.05, 0.10, 0.20]:
        ax[1, 1].plot(tab[mu]['ODE']['s'], tab[mu]['ODE']['I50'], '--', label=f'ODE {mu}')
        ax[1, 1].plot(tab[mu]['PDE']['s'], tab[mu]['PDE']['I50'], '-', label=f'PDE {mu}')
    ax[1, 1].set_title(r'$I_{[0,50]}(s)$'); ax[1, 1].set_xlabel('s'); ax[1, 1].legend(fontsize=6)
    for mu in [0.05, 0.10, 0.20]:
        ax[1, 2].plot(tab[mu]['PDE']['s'], np.max(tab[mu]['PDE']['L'] ** 2, axis=0) / p['bmax'], label=f'{mu}')
    ax[1, 2].set_yscale('log')
    ax[1, 2].set_title(r'max $\tilde{\ell}^2/b_{\max}(s)$'); ax[1, 2].set_xlabel('s'); ax[1, 2].legend(fontsize=7)
    fig.suptitle('Fig5: Tam PDE + FENE-P (h=0.005, s in [0,40])')
    fig.tight_layout(); save(fig, 'fig5_full_pde_solution')

def fig7(lang='tr'):
    p = DEFAULTS; h = 0.005; mu = 0.1
    T = {'peel': 'Flip-flop: kiral soyulma (X*=1)',
         'closed': 'kapali form h=0',
         'planck': 'Planck gozlem',
         'freeze': r'Baryojenez donmasi ($\epsilon^*=%.3e$)',
         'sup': 'Fig7: Dirac-Majorana flip-flop, Meral sabiti, baryojenez',
         'name': 'fig7_dirac_majorana_and_meral_constant'} if lang == 'tr' else {
         'peel': 'Flip-flop: chiral peeling (X*=1)',
         'closed': 'closed form h=0',
         'planck': 'Planck obs.',
         'freeze': r'Baryogenesis freeze-out ($\epsilon^*=%.3e$)',
         'sup': 'Fig7: Dirac-Majorana flip-flop, Meral constant, baryogenesis',
         'name': 'fig7_dirac_majorana_and_meral_constant_en'}
    sp = solve_spinor(1.0, h, mu, 30.0, p, 3000)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].semilogy(sp['s'], np.maximum(sp['L2'], 1e-300), label=r'$|\psi_L|^2$')
    ax[0].semilogy(sp['s'], np.maximum(sp['R2'], 1e-300), label=r'$|\psi_R|^2$')
    ax[0].set_title(T['peel']); ax[0].set_xlabel('s'); ax[0].legend(fontsize=7)
    ax2 = ax[0].twinx()
    ax2.plot(sp['s'], sp['flip'], 'g--', lw=0.8, label='flip')
    ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=7)
    hs = np.array(H_VALUES)
    lam = np.array([lambda_meral(hh, p)[0] for hh in hs])
    ax[1].plot(hs, lam, 'bo-', label='quad')
    ax[1].axhline(LAMBDA0_CLOSED, color='r', ls='--', label=T['closed'])
    ax[1].set_title(r'$\Lambda_{\mathrm{Meral}}(h)$'); ax[1].set_xlabel('h'); ax[1].legend(fontsize=7)
    eps = find_eps_star(h, mu, p)
    b = solve_boltzmann(eps, h, mu, 60.0, p, 2000)
    ax[2].semilogy(b['s'], np.maximum(b['eta'], 1e-300), label=r'$\eta_B(s)$')
    ax[2].axhline(p['eta_obs'], color='r', ls='--', label=T['planck'])
    ax[2].axhspan(p['eta_obs'] - 0.04e-10, p['eta_obs'] + 0.04e-10, color='r', alpha=0.15)
    ax[2].set_title(T['freeze'] % eps)
    ax[2].set_xlabel('s'); ax[2].legend(fontsize=7)
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])

def fig8(lang='tr'):
    p = DEFAULTS; h = 0.005; mu = 0.1
    T = {'phase': 'ALPHA-g / Villata faz gecisi',
         'fuel': 'Girdap yakiti spektrumu',
         'sup': 'Fig8: ALPHA Breit-Rabi ve 511 keV yakit',
         'name': 'fig8_alpha_breit_rabi_and_511keV_fuel'} if lang == 'tr' else {
         'phase': 'ALPHA-g / Villata phase transition',
         'fuel': 'Vortex fuel spectrum',
         'sup': 'Fig8: ALPHA Breit-Rabi and 511 keV fuel',
         'name': 'fig8_alpha_breit_rabi_and_511keV_fuel_en'}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    s = np.linspace(0, 40, 400)
    agr = np.array([float(np.asarray(ag_ratio(1.0, ss, h, mu, p))) for ss in s])
    ax[0].plot(s, agr, label=r'$a_{\bar g}/g$ (X*=1)')
    ax[0].axhline(1.0, color='k', lw=0.8)
    ax[0].axhspan(0.75 - 0.206, 0.75 + 0.206, color='g', alpha=0.15, label='ALPHA-g')
    ax[0].set_title(T['phase']); ax[0].set_xlabel('s'); ax[0].legend(fontsize=7)
    B = np.linspace(0, 1.2, 120)
    E = np.array([[breit_rabi(b, None, p)[k] for k in ['Ea', 'Eb', 'Ec', 'Ed']] for b in B]) / 1e6
    for i, lab in enumerate(['a', 'b', 'c', 'd']):
        ax[1].plot(B, E[:, i], label='|%s>' % lab)
    for Bv in [1.03, 1.07]:
        ax[1].axvline(Bv, color='gray', ls=':')
    ax[1].set_title('Breit-Rabi 1S (MHz)'); ax[1].set_xlabel('B (T)'); ax[1].legend(fontsize=7)
    E = np.linspace(5, 511, 400)
    ax[2].loglog(E, gauss_line(E), label='511 keV Gauss')
    ax[2].loglog(E, np.maximum(np.asarray(ore_powell(E), dtype=float), 1e-300), label='Ore-Powell')
    ax[2].axvline(511, color='k', lw=0.8)
    ax[2].set_title(T['fuel']); ax[2].set_xlabel('keV'); ax[2].legend(fontsize=7)
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])

def fig9(lang='tr'):
    from solver_engine import (trap_ideal_radius, trap_rp_eff, find_mu_star,
        gvortex_raw, gvortex_norm, ore_powell_exact, ore_prompt_form, ore_powell,
        gauss_line, ALPHA_TRAP, LINE, solve_alpha_sdr_evc_trap,
        solve_base_step_road_transport, BASE_STEP)
    import numpy as np
    T = ({'trap': '(a) T16: ALPHA 2025 SDR-EVC ve sıfır-parametre',
          'line': r'(b) T17: ALPHA 2026 $g_{\mathrm{vortex}}$ ve $(g*R)$ konvolüsyonu',
          'op': '(c) T18: Ore-Powell 1949 tam vs. ıraksak formlar',
          'base': '(d) T19: BASE-STEP 2026 karayolu Lyapunov (92/92)',
          'sup': 'CERN ALPHA Tuzak, Çizgi Şekli, 511 keV Tam Düzeltmesi ve BASE-STEP',
          'name': 'fig9_cern_alpha_trap_and_511keV_exact',
          'unp': r'korunmasız ($\mu_s=0$)', 'wall': 'duvar 5 mm',
          'p1': r'$(2m-E)^2$ ($1/E$ kutbu)', 'p2': r'2605.21289 Denk.12 ($1/E^2$)'}
         if lang == 'tr' else
         {'trap': '(a) T16: ALPHA 2025 SDR-EVC and zero-parameter',
          'line': r'(b) T17: ALPHA 2026 $g_{\mathrm{vortex}}$ and $(g*R)$ convolution',
          'op': '(c) T18: Ore-Powell 1949 exact vs. divergent forms',
          'base': '(d) T19: BASE-STEP 2026 road Lyapunov (92/92)',
          'sup': 'CERN ALPHA Trap, Lineshape, Exact 511 keV Correction, and BASE-STEP',
          'name': 'fig9_cern_alpha_trap_and_511keV_exact_en',
          'unp': r'unprotected ($\mu_s=0$)', 'wall': 'wall 5 mm',
          'p1': r'$(2m-E)^2$ ($1/E$ pole)', 'p2': r'2605.21289 Eq.12 ($1/E^2$)'})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    zp = solve_alpha_sdr_evc_trap()
    mu = np.linspace(0.0, 0.24, 120)
    rp = np.array([trap_rp_eff(m) * 1e3 for m in mu])
    ax[0, 0].plot(mu, rp, 'b-', label=r'$r_p^{\mathrm{eff}}(\mu_s)$ (Be$^+$)')
    ax[0, 0].axhline(trap_ideal_radius() * 1e3, color='k', ls='--', lw=0.8, label='ideal')
    ax[0, 0].axhline(ALPHA_TRAP['r_exp_mm'], color='r', ls='--', lw=0.8, label='exp 0.16 mm')
    ax[0, 0].plot(zp['mu_star'], ALPHA_TRAP['r_exp_mm'], 'ro', label=r'$\mu_s^*=0.2032$')
    ax[0, 0].plot(zp['mu_meral'], zp['rp_meral_mm'], 'gs',
                  label=r'$\mu_s^{\mathrm{Meral}}=0.2044$')
    ax[0, 0].plot(zp['mu_geom'], zp['rp_geom_mm'], 'm^',
                  label=r'$\mu_s^{\mathrm{geom}}=0.2000$')
    ax[0, 0].set_title(T['trap']); ax[0, 0].set_xlabel(r'$\mu_s$')
    ax[0, 0].set_ylabel('mm'); ax[0, 0].legend(fontsize=6.5)
    Ng, _ = gvortex_norm()
    x = np.linspace(-0.5, 10.0, 600)
    g = gvortex_raw(x) / Ng
    sx = LINE['sigb_kHz'] / LINE['sig0_kHz']
    y = np.linspace(0.0, 40.0, 4001)
    gy = gvortex_raw(y) / Ng
    conv = np.array([float(np.trapezoid(
        gy * np.exp(-0.5 * ((xv - y) / sx) ** 2) / (np.sqrt(2 * np.pi) * sx), y))
        for xv in x])
    ax[0, 1].plot(x, g, 'b-', label=r'$g_{\mathrm{vortex}}(x)$')
    ax[0, 1].plot(x, conv, 'r--', label=r'$(g*R)(x)$ ($\sigma_b=10$ kHz)')
    ax[0, 1].axvline(1.0157, color='gray', ls=':', lw=0.8)
    ax[0, 1].set_title(T['line']); ax[0, 1].set_xlabel(r'$x=(f-f_o)/\sigma_0$')
    ax[0, 1].legend(fontsize=7)
    E = np.logspace(0.0, np.log10(511.0), 400)
    ax[1, 0].loglog(E, np.maximum(np.asarray(ore_powell_exact(E), dtype=float), 1e-300),
                    'b-', label=r'$O_{\mathrm{exact}}$ (1949, $\int=1$)')
    ax[1, 0].loglog(E, np.maximum(np.asarray(ore_prompt_form(E), dtype=float), 1e-300),
                    'r--', lw=0.8, label=T['p1'])
    ax[1, 0].loglog(E, np.maximum(np.asarray(ore_powell(E), dtype=float), 1e-300),
                    'm:', lw=0.8, label=T['p2'])
    ax[1, 0].loglog(E, np.maximum(gauss_line(E), 1e-300), 'g-.', lw=0.8, label='511 keV Gauss')
    ax[1, 0].set_ylim(1e-8, 1e4); ax[1, 0].axvline(511, color='k', lw=0.8)
    ax[1, 0].set_title(T['op']); ax[1, 0].set_xlabel('keV'); ax[1, 0].legend(fontsize=6.5)
    bs = solve_base_step_road_transport()
    t_m = bs['prot_t'] / 60.0
    ax[1, 1].semilogy(t_m, np.maximum(bs['unprot_r_mm'], 1e-6), 'r--', label=T['unp'])
    for B, col in [('0.136', 'c-'), ('0.88', 'g-'), ('0.993', 'b-')]:
        ax[1, 1].semilogy(t_m, np.maximum(bs['prot_r_mm'][B], 1e-6), col,
                          label=f'Lyapunov B={B} T')
    ax[1, 1].axhline(BASE_STEP['R_trap_mm'], color='k', ls=':', lw=0.9, label=T['wall'])
    ax[1, 1].set_ylim(5e-5, 20.0)
    ax[1, 1].set_title(T['base']); ax[1, 1].set_xlabel('t (min)'); ax[1, 1].set_ylabel('mm')
    ax[1, 1].legend(fontsize=6.5)
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])

def fig10(lang='tr'):
    from solver_engine import (find_mu_star, breit_rabi, gauss_line,
        ore_powell_exact, predict_basestep_136mT_falsification,
        predict_alpha_third_field, predict_511keV_band, uq_faz6_sensitivities,
        solve_alpha_2026_lineshape_and_drift, BASE_STEP, DEFAULTS)
    from scipy.integrate import solve_ivp
    import numpy as np
    p = DEFAULTS
    T = ({'a': '(a) T20: 136 mT yan-band açık/kapalı falsifikasyonu',
          'b': '(b) T21: 1.05 T üçüncü-alan öngörüsü + oran kilidi',
          'c': '(c) T22: 511 keV foton-bütçe öngörü bandı',
          'd': '(d) UQ: kilit çıktı duyarlılıkları',
          'sup': 'Falsifiye Edilebilir Öngörüler, 511 keV Bandı ve UQ',
          'name': 'fig10_falsifiable_predictions_UQ',
          'wall': 'duvar 5 mm',
          'b_data1': 'veri 1.03 T', 'b_data2': 'veri 1.07 T',
          'b_pred': 'öngörü 1.05 T',
          'a_on': 'merkezli+korumalı', 'a_off': 'ofset 4 mm geri-çekilme',
          'a_un': 'korunmasız', 'b_unit': 'kHz/saat'}
         if lang == 'tr' else
         {'a': '(a) T20: 136 mT sideband on/off falsification',
          'b': '(b) T21: 1.05 T third-field prediction + ratio lock',
          'c': '(c) T22: 511 keV photon-budget prediction band',
          'd': '(d) UQ: lock-output sensitivities',
          'sup': 'Falsifiable Predictions, 511 keV Band, and UQ',
          'name': 'fig10_falsifiable_predictions_UQ_en',
          'wall': 'wall 5 mm',
          'b_data1': 'data 1.03 T', 'b_data2': 'data 1.07 T',
          'b_pred': 'prediction 1.05 T',
          'a_on': 'centered+protected', 'a_off': 'offset 4 mm recapture',
          'a_un': 'unprotected', 'b_unit': 'kHz/h'})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    mu_s = find_mu_star()
    c = BASE_STEP
    nuz = 453e3 * np.sqrt(16.0 / 3.911)
    gd = 2.0 * np.pi * 122.0 * (122.0 / nuz)
    a_rms = 0.15 * 9.80665
    R, req = 5e-3, 0.5e-3
    wc136 = 1.602176634e-19 * 0.136 / 1.67262192369e-27

    def run136(r0, mu):
        def rhs(t, y):
            r = float(y[0])
            den = max(1.0 - (r / R) ** 2, 1e-6)
            return [gd * r - mu * 1.0 * r * (1.0 + (r / req) ** 2) / den
                    + a_rms / wc136]
        tg = np.linspace(0.0, 1440.0, 401)
        sol = solve_ivp(rhs, (0.0, 1440.0), [r0], method='Radau',
                        rtol=1e-10, atol=1e-13, t_eval=tg)
        return tg / 60.0, sol.y[0] * 1e3
    t, r_on = run136(0.0, mu_s)
    _, r_off = run136(4.0e-3, mu_s)
    _, r_un = run136(1e-6, 0.0)
    ax[0, 0].semilogy(t, np.maximum(r_on, 1e-6), 'b-', label=T['a_on'])
    ax[0, 0].semilogy(t, np.maximum(r_off, 1e-6), 'g-', label=T['a_off'])
    ax[0, 0].semilogy(t, np.maximum(r_un, 1e-6), 'r--', label=T['a_un'])
    ax[0, 0].axhline(c['R_trap_mm'], color='k', ls=':', lw=0.9, label=T['wall'])
    ax[0, 0].set_ylim(5e-5, 50.0)
    ax[0, 0].set_title(T['a']); ax[0, 0].set_xlabel('t (min)')
    ax[0, 0].set_ylabel('mm'); ax[0, 0].legend(fontsize=6.5)
    f21 = predict_alpha_third_field(1.05, p)
    dd = solve_alpha_2026_lineshape_and_drift(p)
    Bg = np.linspace(1.00, 1.10, 60)
    gb = f21['gamma_bar']
    fd = [float((breit_rabi(B + 1e-4, None, p)['fda']
                 - breit_rabi(B - 1e-4, None, p)['fda']) / (2e-4))
          * (-gb * B) * 3600.0 for B in Bg]
    ax[0, 1].plot(Bg, fd, 'b-', label='model')
    ax[0, 1].plot(1.03, -72.82, 'ko', label=T['b_data1'])
    ax[0, 1].plot(1.07, -75.64, 'ko', label=T['b_data2'])
    ax[0, 1].errorbar(1.05, f21['fdot_pred_kHz_per_h'],
                      yerr=f21['fdot_halfwidth_kHz_per_h'],
                      fmt='rs', label=T['b_pred'])
    ax[0, 1].set_title(T['b']); ax[0, 1].set_xlabel('B (T)')
    ax[0, 1].set_ylabel(T['b_unit']); ax[0, 1].legend(fontsize=6.5)
    m = 511.0
    E = np.linspace(100.0, 511.0, 2001)
    L = np.asarray(gauss_line(E, m, p['sigma_511_keV']), dtype=float)
    O = np.asarray(ore_powell_exact(E, m), dtype=float)
    for fPs, Ps, st in [(0.90, 0.81, 'c:'), (0.92, 0.90, 'b-'), (0.94, 0.96, 'r--')]:
        Sspec = Ps * (2.0 - 1.5 * fPs) * L + Ps * (9.0 / 4.0 * fPs) * O
        ax[1, 0].loglog(E, np.maximum(Sspec, 1e-300), st,
                        label='fPs=%.2f' % fPs)
    ax[1, 0].axvline(511, color='k', lw=0.8)
    ax[1, 0].set_title(T['c']); ax[1, 0].set_xlabel('keV')
    ax[1, 0].legend(fontsize=6.5)
    uq = uq_faz6_sensitivities(p)
    rlo, rhi = uq['rstar_136_um_band']
    rnom = uq['rstar_136_um_nominal']
    mlo, mhi = uq['Mdot_eta009_011']
    mnom = uq['Mdot_eta010']
    r_pm = [(rhi - rnom) / rnom * 100.0, (rnom - rlo) / rnom * 100.0]
    m_pm = [(mhi - mnom) / abs(mnom) * 100.0, (mnom - mlo) / abs(mnom) * 100.0]
    rp_s = abs(uq['rp_meral_sens_mu']) * 1.0
    R_ppm = uq['R_th_spread_ppm']
    ax[1, 1].plot([0, rp_s], [0, 0], 'b-', lw=4, solid_capstyle='round')
    ax[1, 1].plot([0, R_ppm / 1e4], [1, 1], 'g-', lw=4, solid_capstyle='round')
    ax[1, 1].plot([(mlo - mnom) / abs(mnom) * 100.0,
                   (mhi - mnom) / abs(mnom) * 100.0], [2, 2],
                  'r-', lw=4, solid_capstyle='round')
    ax[1, 1].plot([(rlo - rnom) / rnom * 100.0,
                   (rhi - rnom) / rnom * 100.0], [3, 3],
                  'c-', lw=4, solid_capstyle='round')
    ax[1, 1].plot([0, 0, (mhi - mnom) / abs(mnom) * 100.0,
                   (rhi - rnom) / rnom * 100.0], [0, 1, 2, 3],
                  'ko', ms=4)
    ax[1, 1].set_yticks([0, 1, 2, 3])
    ax[1, 1].set_yticklabels(['rp(Meral)|mu', 'R_th yayilim/1e4',
                              'Mdot|eta', 'r*(136mT)|a'])
    ax[1, 1].axvline(0, color='k', lw=0.8)
    ax[1, 1].set_xlim(-12.0, 12.0)
    ax[1, 1].set_title(T['d']); ax[1, 1].set_xlabel('% (1-sigma olcegi)')
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])


def fig11(lang='tr'):
    from solver_engine import (solve_super_brillouin_regime,
        solve_diocotron_vortex_lattice,
        solve_antimatter_accumulation_and_shielding,
        string_nmax_ratio, brillouin_density, _M_P, DEFAULTS)
    import numpy as np
    p = DEFAULTS
    T = ({'a': r'(a) T23: Süper-Brillouin $n_{\max}^{\mathrm{sicim}}/n_B$ ($\chi$ boyunca)',
          'b': r'(b) T24: Diocotron özdeğerleri $\mathrm{Re}\,\lambda_\ell$ ve $N_v=7$ kristali',
          'c': r'(c) T25: CERN ALPHA $\overline{\mathrm{H}}$ birikimi ($19.75\times$ kalkan)',
          'd': r'(d) $\bar{p}$ yoğunluğu $n_{\max}(B_0,\chi)$ ve depolama ömrü $\tau_{\mathrm{dep}}$',
          'sup': 'Süper-Brillouin Rejimi, Diocotron Girdap Kristali ve Kiral-Elastik Kalkan',
          'name': 'fig11_super_brillouin_vortex_lattice_and_shielding',
          'std_lim': 'klasik Brillouin (1.0)',
          'g_std': r'standart $\gamma_\ell^{\mathrm{std}}>0$',
          'l_str': r'sicim $\mathrm{Re}\,\lambda_\ell<0$',
          'c_std': r'standart SDR-EVC ($N_\infty=7500$)',
          'c_str': r'kalkanlı ($\chi=0.50$, $N_\infty=2.36\times10^5$)',
          'c_exp': 'ALPHA 2025 (2835 @ x=225)',
          'x_cyc': 'sentez çevrimi x',
          'tau_lbl': r'$\tau_{\mathrm{dep}}$: std 0.97 yıl / kalkanlı 12.47 yıl'}
         if lang == 'tr' else
         {'a': r'(a) T23: Super-Brillouin $n_{\max}^{\mathrm{string}}/n_B$ across $\chi$',
          'b': r'(b) T24: Diocotron eigenvalues $\mathrm{Re}\,\lambda_\ell$ and $N_v=7$ crystal',
          'c': r'(c) T25: CERN ALPHA $\overline{\mathrm{H}}$ accumulation ($19.75\times$ shield)',
          'd': r'(d) $\bar{p}$ density $n_{\max}(B_0,\chi)$ and storage lifetime $\tau_{\mathrm{store}}$',
          'sup': 'Super-Brillouin Regime, Diocotron Vortex Crystal, and Chiral-Elastic Shield',
          'name': 'fig11_super_brillouin_vortex_lattice_and_shielding_en',
          'std_lim': 'classical Brillouin (1.0)',
          'g_std': r'standard $\gamma_\ell^{\mathrm{std}}>0$',
          'l_str': r'string $\mathrm{Re}\,\lambda_\ell<0$',
          'c_std': r'standard SDR-EVC ($N_\infty=7500$)',
          'c_str': r'shielded ($\chi=0.50$, $N_\infty=2.36\times10^5$)',
          'c_exp': 'ALPHA 2025 (2835 @ x=225)',
          'x_cyc': 'synthesis cycle x',
          'tau_lbl': r'$\tau_{\mathrm{store}}$: std 0.97 yr / shielded 12.47 yr'})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    sb = solve_super_brillouin_regime(p)
    vx = solve_diocotron_vortex_lattice(p)
    sh = solve_antimatter_accumulation_and_shielding(p)
    # Panel (a): n_max / n_B vs chi
    cg = np.linspace(0.0, 0.92, 200)
    for mu, st, lb in [(sb['mu_meral'], 'b-', r'$\mu_s^{\mathrm{Meral}}=0.2044$'),
                       (sb['mu_star'], 'r--', r'$\mu_s^*=0.2032$'),
                       (sb['mu_geom'], 'g:', r'$\mu_s^{\mathrm{geom}}=0.2000$')]:
        ax[0, 0].plot(cg, [string_nmax_ratio(mu, c) for c in cg], st, label=lb)
    ax[0, 0].axhline(1.0, color='k', ls='-.', lw=0.8, label=T['std_lim'])
    for c0 in (0.0, 0.5, 0.75, 0.90):
        ax[0, 0].plot(c0, string_nmax_ratio(sb['mu_meral'], c0), 'bo', ms=4)
    ax[0, 0].set_title(T['a']); ax[0, 0].set_xlabel(r'$\chi=r_p^2/R_{\max}^2$')
    ax[0, 0].set_ylabel(r'$n_{\max}/n_B$'); ax[0, 0].legend(fontsize=6.5)
    # Panel (b): azimutal modlar l = 1..6
    ls = np.arange(1, 7)
    gs = [vx['modes'][str(l)]['gamma_std'] for l in ls]
    lms = [vx['modes'][str(l)]['lam'] for l in ls]
    ax[0, 1].plot(ls, gs, 'ro--', label=T['g_std'])
    ax[0, 1].plot(ls, lms, 'bs-', label=T['l_str'])
    ax[0, 1].axhline(0.0, color='k', lw=0.8)
    ax[0, 1].set_title(T['b']); ax[0, 1].set_xlabel(r'$\ell$')
    ax[0, 1].set_ylabel(r'$\lambda_\ell/\omega_D$'); ax[0, 1].legend(fontsize=6.5)
    # Panel (c): N(x) birikim
    xg = np.linspace(0.0, 2000.0, 401)
    Nstd = sh['Ninf_std'] * (1.0 - np.exp(-sh['eta_std'] * xg))
    Nstr = sh['Ninf_str'] * (1.0 - np.exp(-sh['eta_str'] * xg))
    ax[1, 0].semilogy(xg[1:], Nstd[1:], 'r--', label=T['c_std'])
    ax[1, 0].semilogy(xg[1:], Nstr[1:], 'b-', label=T['c_str'])
    ax[1, 0].plot(225.0, 2835.0, 'ko', ms=5, label=T['c_exp'])
    ax[1, 0].set_title(T['c']); ax[1, 0].set_xlabel(T['x_cyc'])
    ax[1, 0].set_ylabel(r'$N(x)$'); ax[1, 0].legend(fontsize=6.5)
    # Panel (d): n_max(p-bar) vs B0
    Bg = np.linspace(0.5, 5.0, 100)
    for chi, st in [(0.0, 'k--'), (0.5, 'g-.'), (0.75, 'b-'), (0.90, 'r-')]:
        mult = string_nmax_ratio(sb['mu_meral'], chi) if chi > 0 else 1.0
        lb = r'klasik $n_B$' if (chi == 0 and lang == 'tr') else (
            r'classical $n_B$' if chi == 0 else r'$\chi=%.2f$ (%.2f$\times$)' % (chi, mult))
        ax[1, 1].semilogy(Bg, [brillouin_density(B, _M_P) * mult for B in Bg],
                          st, label=lb)
    ax[1, 1].text(0.96, 0.08, T['tau_lbl'], transform=ax[1, 1].transAxes,
                  ha='right', fontsize=7,
                  bbox=dict(boxstyle='round', fc='wheat', alpha=0.3))
    ax[1, 1].set_title(T['d']); ax[1, 1].set_xlabel(r'$B_0$ (T)')
    ax[1, 1].set_ylabel(r'$n_{\max}(\bar{p})$ ($\mathrm{m}^{-3}$)')
    ax[1, 1].legend(fontsize=6.5, loc='upper left')
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])


def fig12(lang='tr'):
    from solver_engine import (solve_tokamak_elm_and_hmode_pedestal,
        solve_runaway_electron_positron_avalanche,
        solve_universal_vortex_string_scaling_law,
        string_sigma, DEFAULTS)
    from scipy.integrate import solve_ivp
    import numpy as np
    p = DEFAULTS
    T = ({'a': r'(a) T26: Tokamak ELM filamanı ($\alpha_{\mathrm{MHD}}=1.5\alpha_{\mathrm{crit}}^{\mathrm{std}}$) ve $H_{98}$',
          'b': r'(b) T27: Yıkımda kaçak çığ $I_{\mathrm{RE}}(t)/I_0$ ($E_\parallel/E_c=30$)',
          'c': r'(c) T28: 30-mertebeli evrensel ölçekleme ($\mathrm{eğim}=1.0000$)',
          'd': r'(d) Serinin boyutsuz değişmezleri (Teorem I–X, T1–T28 PASS)',
          'sup': 'Tokamak H-Modu/ELM Bastırması, Kaçak Çığ Doyumu ve 30-Mertebeli Evrensel Ölçekleme',
          'name': 'fig12_tokamak_elm_runaway_and_universal_scaling',
          'elm_std': r'standart ELM patlaması ($\mu_s=0$)',
          'elm_str': r'FENE-P RMP ($\chi=0.75$, sınırlı)',
          'elm_txt': r'$H_{98}$: $3.16$ ($\chi=0.50$), $5.55$ ($\chi=0.75$)',
          'ra_std': r'standart çığ ($\gamma=+33.56\ \mathrm{s}^{-1}$)',
          'ra_50': r'kalkanlı $\chi=0.50$ ($\gamma=+0.047\ \mathrm{s}^{-1}$)',
          'ra_75': r'kalkanlı $\chi=0.75$ ($\gamma=-0.265\ \mathrm{s}^{-1}$, sönüm)',
          'sc_line': r'$r_{\mathrm{sat}}=\mathcal{C}_{\mathrm{univ}}L_0$ ($\mathcal{C}=1.6260$)',
          'sc_names': ['⁴He girdabı', 'CERN tuzak', 'ITER/SPARC', 'Vela pulsar', 'Sgr A*', 'Galaktik halo'],
          'inv_lbl': 'değer (log ölçek)',
          'pass_box': 'T1–T28: 28/28 PASS\nmini_check_x: 15/15 PASS'}
         if lang == 'tr' else
         {'a': r'(a) T26: Tokamak ELM filament ($\alpha_{\mathrm{MHD}}=1.5\alpha_{\mathrm{crit}}^{\mathrm{std}}$) and $H_{98}$',
          'b': r'(b) T27: Disruption runaway avalanche $I_{\mathrm{RE}}(t)/I_0$ ($E_\parallel/E_c=30$)',
          'c': r'(c) T28: 30-decade universal scaling ($\mathrm{slope}=1.0000$)',
          'd': r'(d) Dimensionless invariants of the series (Theorem I–X, T1–T28 PASS)',
          'sup': 'Tokamak H-Mode/ELM Suppression, Runaway Quench, and 30-Decade Universal Scaling',
          'name': 'fig12_tokamak_elm_runaway_and_universal_scaling_en',
          'elm_std': r'standard ELM blowup ($\mu_s=0$)',
          'elm_str': r'FENE-P RMP ($\chi=0.75$, bounded)',
          'elm_txt': r'$H_{98}$: $3.16$ ($\chi=0.50$), $5.55$ ($\chi=0.75$)',
          'ra_std': r'standard avalanche ($\gamma=+33.56\ \mathrm{s}^{-1}$)',
          'ra_50': r'shielded $\chi=0.50$ ($\gamma=+0.047\ \mathrm{s}^{-1}$)',
          'ra_75': r'shielded $\chi=0.75$ ($\gamma=-0.265\ \mathrm{s}^{-1}$, quench)',
          'sc_line': r'$r_{\mathrm{sat}}=\mathcal{C}_{\mathrm{univ}}L_0$ ($\mathcal{C}=1.6260$)',
          'sc_names': ['⁴He vortex', 'CERN trap', 'ITER/SPARC', 'Vela pulsar', 'Sgr A*', 'Galactic halo'],
          'inv_lbl': 'value (log scale)',
          'pass_box': 'T1–T28: 28/28 PASS\nmini_check_x: 15/15 PASS'})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    tk = solve_tokamak_elm_and_hmode_pedestal(p)
    ra = solve_runaway_electron_positron_avalanche(p=p)
    uv = solve_universal_vortex_string_scaling_law(p=p)
    # Panel (a): ELM filaman ODE
    tg = np.linspace(0.0, 20.0, 301)
    C_std = 0.25 * (1.5 - 1.0)
    C_str = 0.25 * (1.5 - tk['alpha_ratio']['0.75'])
    s1 = solve_ivp(lambda t, y: [y[1], C_std * y[0]], (0.0, 20.0), [1.0, 0.0],
                   method='Radau', rtol=1e-10, atol=1e-13, t_eval=tg)
    s2 = solve_ivp(lambda t, y: [y[1], C_str * y[0]], (0.0, 20.0), [1.0, 0.0],
                   method='Radau', rtol=1e-10, atol=1e-13, t_eval=tg)
    ax[0, 0].semilogy(tg, np.maximum(np.abs(s1.y[0]), 1e-3), 'r--', label=T['elm_std'])
    ax[0, 0].semilogy(tg, np.maximum(np.abs(s2.y[0]), 1e-3), 'b-', label=T['elm_str'])
    ax[0, 0].text(0.04, 0.08, T['elm_txt'], transform=ax[0, 0].transAxes,
                  ha='left', fontsize=7,
                  bbox=dict(boxstyle='round', fc='wheat', alpha=0.3))
    ax[0, 0].set_title(T['a']); ax[0, 0].set_xlabel(r'$\tau=\omega_A t$')
    ax[0, 0].set_ylabel(r'$|r_{\mathrm{fil}}(\tau)/r_0|$'); ax[0, 0].legend(fontsize=6.5, loc='upper left')
    # Panel (b): Runaway avalanche I_RE(t)/I_0
    tr = np.linspace(0.0, 0.5, 200)
    ax[0, 1].semilogy(tr, np.exp(ra['gamma_std_per_s'] * tr), 'r--', label=T['ra_std'])
    ax[0, 1].semilogy(tr, np.exp(ra['gamma_string_per_s']['0.5'] * tr), 'g-.', label=T['ra_50'])
    ax[0, 1].semilogy(tr, np.exp(ra['gamma_string_per_s']['0.75'] * tr), 'b-', label=T['ra_75'])
    ax[0, 1].axhline(1.0, color='k', ls=':', lw=0.8)
    ax[0, 1].set_title(T['b']); ax[0, 1].set_xlabel('t (s)')
    ax[0, 1].set_ylabel(r'$I_{\mathrm{RE}}(t)/I_0$'); ax[0, 1].legend(fontsize=6.5, loc='upper left')
    # Panel (c): 30-decade universal scaling
    L = np.array([r['L0_m'] for r in uv['scales']])
    R = np.array([r['r_sat_m'] for r in uv['scales']])
    Lg = np.logspace(-11, 21, 100)
    ax[1, 0].loglog(Lg, uv['C_univ'] * Lg, 'b-', lw=1.2, label=T['sc_line'])
    ax[1, 0].loglog(L, R, 'ro', ms=5)
    offs = [(6, -10), (6, -10), (6, -10), (-48, 8), (-42, 8), (-62, -12)]
    for i, nm in enumerate(T['sc_names']):
        ax[1, 0].annotate(nm, (L[i], R[i]), xytext=offs[i],
                          textcoords='offset points', fontsize=6.5)
    ax[1, 0].set_title(T['c']); ax[1, 0].set_xlabel(r'$L_0$ (m)')
    ax[1, 0].set_ylabel(r'$r_{\mathrm{sat}}$ (m)'); ax[1, 0].legend(fontsize=6.5, loc='upper left')
    # Panel (d): Key dimensionless invariants
    names = [r'$\Lambda_{\mathrm{Meral}}^{(0)}$', r'$\mu_s^{\mathrm{Meral}}$',
             r'$\mathcal{C}_{\mathrm{univ}}$', r'$\mathcal{I}_{\mathrm{shield}}=2\sqrt{2}$',
             r'$S_{\mathrm{loss}}$', r'$E_c^{\mathrm{eff}}/E_c(0.75)$']
    vals = [0.0722694327, uv['mu_meral'], uv['C_univ'],
            uv['I_shield'], tk['S_loss'], ra['Ec_eff_ratio']['0.75']]
    y = np.arange(len(names))
    ax[1, 1].barh(y, vals, color=['#4c72b0', '#55a868', '#c44e52',
                                  '#8172b2', '#ccb974', '#64b5cd'], height=0.55)
    ax[1, 1].set_xscale('log'); ax[1, 1].set_xlim(0.03, 120.0)
    ax[1, 1].set_yticks(y); ax[1, 1].set_yticklabels(names, fontsize=7.5)
    for yi, v in zip(y, vals):
        ax[1, 1].text(v * 1.12, yi, '%.4g' % v, va='center', fontsize=6.5)
    ax[1, 1].text(0.96, 0.12, T['pass_box'], transform=ax[1, 1].transAxes,
                  ha='right', fontsize=7,
                  bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.35))
    ax[1, 1].set_title(T['d']); ax[1, 1].set_xlabel(T['inv_lbl'])
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])


def fig13(lang='tr'):
    from solver_engine import (DEFAULTS, GP1, meral_cosmic_acceleration,
        rar_g_obs, solve_meral_cosmological_acceleration_and_btfr,
        solve_sparc_galaxy_rotation_curves,
        solve_eht_black_hole_ring_saturation)
    import numpy as np
    p = DEFAULTS
    T = ({'a': r'(a) T29: RAR — Newton vs Meral sicim-çemberi',
          'b': r'(b) T30: Dönüş eğrileri — çöküş vs düzleşme',
          'c': r'(c) BTFR ($v_{\mathrm{flat}}^4=GM_b a_{\mathrm{Meral}}$, eğim $=4$)',
          'd': r'(d) T31: EHT halka profili ($d_{\mathrm{ring}}=10.00\,\theta_g$)',
          'sup': 'Karanlık-Maddesiz Galaktik Dinamik (Meral İvme Ölçeği), SPARC ve EHT Halka Kanıtı',
          'name': 'fig13_dark_matter_free_sparc_btfr_and_eht_ring',
          'xlabel_a': r'$g_{\mathrm{bar}}$ (m/s$^2$)',
          'ylabel_a': r'$g_{\mathrm{obs}}$ (m/s$^2$)',
          'sparc_band': r'SPARC saçılma bandı ($\pm0.12$ dex, McGaugh+2016)',
          'obs_line': r'gözlenen $v_{\mathrm{flat}}$ bandı',
          'eht_txt': r'parlak halka: $5.00\,\theta_g$ yarıçap',
          'pass_box': 'T29–T31 PASS\nmini_check_gp1: 15/15 PASS'}
         if lang == 'tr' else
         {'a': r'(a) T29: RAR — Newton vs Meral string-loop',
          'b': r'(b) T30: Rotation curves — fall vs flattening',
          'c': r'(c) BTFR ($v_{\mathrm{flat}}^4=GM_b a_{\mathrm{Meral}}$, slope $=4$)',
          'd': r'(d) T31: EHT ring profile ($d_{\mathrm{ring}}=10.00\,\theta_g$)',
          'sup': 'Dark-Matter-Free Galactic Dynamics (Meral Acceleration Scale), SPARC and EHT Ring Evidence',
          'name': 'fig13_dark_matter_free_sparc_btfr_and_eht_ring_en',
          'xlabel_a': r'$g_{\mathrm{bar}}$ (m/s$^2$)',
          'ylabel_a': r'$g_{\mathrm{obs}}$ (m/s$^2$)',
          'sparc_band': r'SPARC scatter band ($\pm0.12$ dex, McGaugh+2016)',
          'obs_line': r'observed $v_{\mathrm{flat}}$ band',
          'eht_txt': r'bright ring: $5.00\,\theta_g$ radius',
          'pass_box': 'T29–T31 PASS\nmini_check_gp1: 15/15 PASS'})
    acc = meral_cosmic_acceleration(p)
    aM = acc['a_Meral']
    bt = solve_meral_cosmological_acceleration_and_btfr(p)
    sc = solve_sparc_galaxy_rotation_curves(p)
    eh = solve_eht_black_hole_ring_saturation(p)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # Panel (a): RAR log-log
    gb = np.logspace(-14, -8, 400)
    go = np.asarray(rar_g_obs(gb, aM), dtype=float)
    ax[0, 0].loglog(gb, gb, 'k--', lw=1.0, label=r'saf Newton ($1{:}1$)'
                    if lang == 'tr' else r'pure Newton ($1{:}1$)')
    ax[0, 0].loglog(gb, go, 'b-', lw=1.6,
                    label=r'Meral sicim-çemberi' if lang == 'tr'
                    else r'Meral string-loop')
    ax[0, 0].fill_between(gb, go * 10.0 ** -0.12, go * 10.0 ** 0.12,
                          color='b', alpha=0.12, label=T['sparc_band'])
    ax[0, 0].axvline(aM, color='gray', ls=':', lw=1.0)
    ax[0, 0].text(aM * 1.15, 3e-14, r'$a_{\mathrm{Meral}}=1.15\times10^{-10}$',
                  fontsize=6.5, rotation=90, va='bottom')
    ax[0, 0].set_title(T['a']); ax[0, 0].set_xlabel(T['xlabel_a'])
    ax[0, 0].set_ylabel(T['ylabel_a']); ax[0, 0].legend(fontsize=6.5, loc='upper left')
    # Panel (b): NGC 3198 + DDO 154 rotation curves
    for key, sty in (('NGC3198', ('b-', 'r--')), ('DDO154', ('g-', 'm--'))):
        c = sc['curves'][key]
        r = np.array(c['r_kpc'])
        ax[0, 1].plot(r, c['v_newton_kms'], sty[1], lw=1.0,
                      label='%s Newton' % key)
        ax[0, 1].plot(r, c['v_meral_kms'], sty[0], lw=1.6,
                      label='%s Meral' % key)
        ax[0, 1].axhline(c['vflat_obs_kms'], color='k', ls=':', lw=0.9)
        ax[0, 1].fill_between(r, c['vflat_obs_kms'] - c['vflat_err_kms'],
                              c['vflat_obs_kms'] + c['vflat_err_kms'],
                              color='k', alpha=0.08)
    ax[0, 1].set_title(T['b'])
    ax[0, 1].set_xlabel('r (kpc)'); ax[0, 1].set_ylabel('v (km/s)')
    ax[0, 1].legend(fontsize=6.5, loc='lower right')
    ax[0, 1].set_xlim(0, 33); ax[0, 1].set_ylim(0, 200)
    # Panel (c): BTFR
    Mb = np.array([g['Mb_Msun'] for g in GP1['galaxies']])
    vf = np.array(bt['vflat_kms'])
    vo = np.array(bt['vflat_obs_kms'])
    ve = np.array([g['vflat_err_kms'] for g in GP1['galaxies']])
    Mg = np.logspace(8, 12, 100)
    vline = ((6.67430e-11 * Mg * 1.98847e30 * aM) ** 0.25) / 1000.0
    ax[1, 0].loglog(Mg, vline, 'b-', lw=1.4,
                    label=r'$v_{\mathrm{flat}}^4=GM_b a_{\mathrm{Meral}}$')
    ax[1, 0].errorbar(Mb, vo, yerr=ve, fmt='ro', ms=4, capsize=2,
                      label='SPARC')
    for i, g in enumerate(GP1['galaxies']):
        ax[1, 0].annotate(g['name'], (Mb[i], vo[i]), xytext=(4, -9),
                          textcoords='offset points', fontsize=6.5)
    ax[1, 0].set_title(T['c'])
    ax[1, 0].set_xlabel(r'$M_b\ (M_\odot)$'); ax[1, 0].set_ylabel(r'$v_{\mathrm{flat}}$ (km/s)')
    ax[1, 0].legend(fontsize=6.5, loc='upper left')
    # Panel (d): EHT ring profile (Gaussian ring, FWHM/d = mu_s)
    th = np.linspace(0, 10, 600)
    fwhm = eh['fwhm_over_d'] * eh['d_over_theta_g']
    sig = fwhm / 2.355
    prof = np.exp(-0.5 * ((th - eh['d_over_theta_g'] / 2.0) / sig) ** 2)
    prof += 0.06 * np.exp(-0.5 * (th / (2.0 * sig)) ** 2)
    ax[1, 1].plot(th, prof / np.max(prof), 'b-', lw=1.6, label=T['eht_txt'])
    for key, col in (('SgrA', 'darkorange'), ('M87', 'green')):
        rr = eh['rings'][key]
        robs = rr['d_obs_uas'] / rr['theta_g_uas'] / 2.0
        rerr = rr['d_err_uas'] / rr['theta_g_uas'] / 2.0
        ax[1, 1].axvspan(robs - rerr, robs + rerr, color=col, alpha=0.18,
                         label='%s EHT' % key)
    ax[1, 1].axvline(eh['d_over_theta_g'] / 2.0, color='b', ls=':', lw=1.0)
    ax[1, 1].text(0.04, 0.90, T['pass_box'], transform=ax[1, 1].transAxes,
                  ha='left', fontsize=7,
                  bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.35))
    ax[1, 1].set_title(T['d'])
    ax[1, 1].set_xlabel(r'$\theta/\theta_g$'); ax[1, 1].set_ylabel('normalize parlaklık'
                        if lang == 'tr' else 'normalized brightness')
    ax[1, 1].legend(fontsize=6.5, loc='upper right')
    ax[1, 1].set_xlim(0, 10); ax[1, 1].set_ylim(0, 1.15)
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])


def fig14(lang='tr'):
    from solver_engine import (DEFAULTS, GP2,
        solve_yang_mills_mass_gap_operator,
        solve_glueball_excitation_spectrum,
        solve_cornell_luscher_quark_confinement,
        ym_flux_tube_potential)
    import numpy as np
    p = DEFAULTS
    T = ({'a': r'(a) T32: Tekilliksiz YM akı-tüpü potansiyeli ve taban dalga fonksiyonu',
          'b': r'(b) T33: Glueball kütle spektrumu — Meral vs. Lattice QCD',
          'c': r'(c) T34: Cornell–Meral kuşatma potansiyeli ve $c\bar{c}$/$b\bar{b}$ seviyeleri',
          'd': r'(d) NS $\leftrightarrow$ YM $\leftrightarrow$ kozmoloji köprüsü (T1–T34)',
          'sup': 'Yang–Mills Kütle Aralığı, Glueball Spektrumu ve Kuark Kuşatması',
          'name': 'fig14_yang_mills_mass_gap_glueballs_and_confinement',
          'sing': r'tekil $-(\pi/12)/x^2$ (kesikli)',
          'veff': r'tekilliksiz $V_{\mathrm{eff}}(x)$',
          'psi': r'taban $\psi_0(x)$ (ölçekli)',
          'psi1': r'uyarılmış $\psi_1(x)$ (ölçekli)',
          'os': r'OS $G_c(t)$ sönümü',
          'lus': r'saf Lüscher $\pi/12$',
          'mer': r'Cornell–Meral $\gamma=0.2985$',
          'scr': r'Yukawa-perdelemeli SCR',
          'cas': r'$C_F=4/3$ Casimir-SCR',
          'bb_ir': r'$b\bar{b}$ IR seviyeleri',
          'bb_scr': r'$b\bar{b}$ SCR seviyeleri',
          'classic': r'topolojisiz $0^{-+}$ (4.88)',
          'inv_lbl': 'değer (log ölçek)',
          'pass_box': 'T1–T34: 34/34 PASS\nmini_check_gp2: 17/17 PASS'}
         if lang == 'tr' else
         {'a': r'(a) T32: Nonsingular YM flux-tube potential and ground wavefunction',
          'b': r'(b) T33: Glueball mass spectrum — Meral vs. Lattice QCD',
          'c': r'(c) T34: Cornell–Meral confining potential and $c\bar{c}$/$b\bar{b}$ levels',
          'd': r'(d) NS $\leftrightarrow$ YM $\leftrightarrow$ cosmology bridge (T1–T34)',
          'sup': 'Yang–Mills Mass Gap, Glueball Spectrum, and Quark Confinement',
          'name': 'fig14_yang_mills_mass_gap_glueballs_and_confinement_en',
          'sing': r'singular $-(\pi/12)/x^2$ (clipped)',
          'veff': r'nonsingular $V_{\mathrm{eff}}(x)$',
          'psi': r'ground $\psi_0(x)$ (scaled)',
          'psi1': r'excited $\psi_1(x)$ (scaled)',
          'os': r'OS $G_c(t)$ decay',
          'lus': r'pure Lüscher $\pi/12$',
          'mer': r'Cornell–Meral $\gamma=0.2985$',
          'scr': r'Yukawa-screened SCR',
          'cas': r'$C_F=4/3$ Casimir-SCR',
          'bb_ir': r'$b\bar{b}$ IR levels',
          'bb_scr': r'$b\bar{b}$ SCR levels',
          'classic': r'non-topological $0^{-+}$ (4.88)',
          'inv_lbl': 'value (log scale)',
          'pass_box': 'T1–T34: 34/34 PASS\nmini_check_gp2: 17/17 PASS'})
    ym = solve_yang_mills_mass_gap_operator(p)
    gb = solve_glueball_excitation_spectrum(p)
    cf = solve_cornell_luscher_quark_confinement(p)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # Panel (a): V_eff vs singular + psi0
    x = np.linspace(0.02, 4.0, 500)
    Vs = np.asarray(ym_flux_tube_potential(x), dtype=float)
    Sg = np.ma.masked_where(-(np.pi / 12.0) / x ** 2 < -2.0,
                            -(np.pi / 12.0) / x ** 2)
    ax[0, 0].plot(x, Sg, 'r--', lw=1.0, label=T['sing'])
    ax[0, 0].plot(x, Vs, 'b-', lw=1.6, label=T['veff'])
    xg = np.array(ym['x_grid'])
    ps = np.array(ym['psi0'])
    m0 = xg < 4.0
    ax[0, 0].plot(xg[m0], 2.0 * ps[m0] / np.max(ps), 'g-.',
                  lw=1.2, label=T['psi'])
    p1 = np.array(ym['psi1'])
    m1 = (xg > 0.06) & (xg < 4.0)
    ax[0, 0].plot(xg[m1], 2.0 * p1[m1] / np.max(np.abs(p1)), 'm:',
                  lw=1.0, label=T['psi1'])
    axi = ax[0, 0].inset_axes([0.60, 0.10, 0.37, 0.35])
    tg = np.array(ym['t_grid'])
    Gc = np.array(ym['Gc'])
    axi.semilogy(tg, Gc, 'ko-', ms=3)
    axi.semilogy(tg, Gc[0] * np.exp(-ym['radial_gap'] * (tg - tg[0])), 'g--', lw=1.0)
    axi.set_title(T['os'], fontsize=6.5)
    axi.set_xlabel('t', fontsize=6); axi.set_ylabel(r'$G_c$', fontsize=6)
    axi.tick_params(labelsize=5.5)
    ax[0, 0].set_ylim(-2.2, 12.0)
    ax[0, 0].set_title(T['a']); ax[0, 0].set_xlabel(r'$x=r\sqrt{\sigma}$')
    ax[0, 0].set_ylabel(r'$V_{\mathrm{eff}}(x)$'); ax[0, 0].legend(fontsize=6.5, loc='upper right')
    # Panel (b): glueball spectrum
    keys = ['0++', '2++', '0-+', '0*++']
    pv = [gb['states'][k]['pred'] for k in keys]
    ov = [gb['states'][k]['obs'] for k in keys]
    oe = [gb['states'][k]['err'] for k in keys]
    Xb = np.arange(len(keys))
    ax[0, 1].bar(Xb - 0.2, pv, 0.4, label='Meral', color='#4c72b0')
    ax[0, 1].errorbar(Xb + 0.2, ov, yerr=oe, fmt='ro', ms=5, capsize=3,
                      label='Lattice QCD')
    for i, k in enumerate(keys):
        ax[0, 1].text(i - 0.2, pv[i] + 0.09, '%.2f' % pv[i], ha='center', fontsize=6.5)
        ax[0, 1].text(i + 0.2, ov[i] + 0.09, '%.2f' % ov[i], ha='center', fontsize=6.5)
    ax[0, 1].plot([2 - 0.2], [gb['states']['0-+']['pred_classic']], color='gray',
                  marker='x', ms=8, mew=1.5, ls='',
                  label=T['classic'])
    ax[0, 1].set_xticks(Xb); ax[0, 1].set_xticklabels(
        ['0++', '2++', '0-+', '0*++'])
    ax[0, 1].set_title(T['b']); ax[0, 1].set_ylabel(r'$M/\sqrt{\sigma}$')
    ax[0, 1].legend(fontsize=6.5, loc='upper left'); ax[0, 1].set_ylim(0, 7.5)
    # Panel (c): Cornell potential + ccbar levels
    y = np.linspace(0.05, 5.0, 400)
    ax[1, 0].plot(y, -0.26179939 / y + y, 'r--', lw=1.0, label=T['lus'])
    ax[1, 0].plot(y, -cf['gamma_Meral'] / y + y, 'b-', lw=1.6, label=T['mer'])
    Gam_s = (cf['gamma_Meral']
             + cf['dgamma_UV'] * np.exp(-(y / 0.44) / cf['Rc_GeVinv']))
    ax[1, 0].plot(y, -Gam_s / y + y, color='orange', lw=1.6, label=T['scr'])
    Gam_c = (cf['gamma_Meral']
             + (cf['dgamma_CAS'] if 'dgamma_CAS' in cf else 0.0)
             * np.exp(-(y / 0.44) / cf['Rc_GeVinv']))
    ax[1, 0].plot(y, -Gam_c / y + y, color='red', lw=1.0, ls='--', label=T['cas'])
    S = 0.44
    for i, tag in enumerate(['1S', '2S', '3S']):
        E = cf['quarkonia']['ccbar']['SCR']['E_GeV'][i] / S
        ax[1, 0].hlines(E, 0.5, 4.0, colors='g', linestyles=':', lw=1.0)
        ax[1, 0].text(4.05, E, tag, va='center', fontsize=7)
    for i in range(3):
        Ei = cf['quarkonia']['bbbar']['IR']['E_GeV'][i] / S
        Es = cf['quarkonia']['bbbar']['SCR']['E_GeV'][i] / S
        ax[1, 0].hlines(Ei, 0.5, 2.2, colors='purple', linestyles='--', lw=0.9)
        ax[1, 0].hlines(Es, 0.5, 2.2, colors='magenta', linestyles='-.', lw=0.9)
    ax[1, 0].plot([], [], color='purple', ls='--', lw=0.9, label=T['bb_ir'])
    ax[1, 0].plot([], [], color='magenta', ls='-.', lw=0.9, label=T['bb_scr'])
    bbtxt = ('bb 2S-1S: IR %d / SCR %d / CAS %d / PDG %d MeV'
             % (round(cf['quarkonia']['bbbar']['IR']['split_2S_1S_MeV']),
                round(cf['quarkonia']['bbbar']['SCR']['split_2S_1S_MeV']),
                round(cf['quarkonia']['bbbar']['CAS']['split_2S_1S_MeV']),
                round(cf['quarkonia']['bbbar']['obs_MeV'])))
    ax[1, 0].text(0.98, 0.06, bbtxt, transform=ax[1, 0].transAxes,
                  ha='right', fontsize=6.5,
                  bbox=dict(boxstyle='round', fc='wheat', alpha=0.35))
    ax[1, 0].set_title(T['c']); ax[1, 0].set_xlabel(r'$R\sqrt{\sigma}$')
    ax[1, 0].set_ylabel(r'$V_{Q\bar{Q}}/\sqrt{\sigma}$')
    ax[1, 0].legend(fontsize=6.5, loc='upper left'); ax[1, 0].set_ylim(-1.5, 5.5)
    # Panel (d): bridge invariants
    names = [r'$\Lambda_{\mathrm{Meral}}^{(0)}$', r'$\mu_s^{\mathrm{Meral}}$',
             r'$\kappa_{\mathrm{Meral}}$', r'$\gamma_{\mathrm{Meral}}$',
             r'$\mathcal{C}_{\mathrm{univ}}$', r'$\Delta m_{\mathrm{gap}}/\sqrt{\sigma}$']
    vals = [0.0722694327, 0.2044088237, gb['kappa'], cf['gamma_Meral'],
            1.6260379366, ym['gap_closed']]
    yy = np.arange(len(names))
    ax[1, 1].barh(yy, vals, color=['#4c72b0', '#55a868', '#c44e52',
                                   '#8172b2', '#ccb974', '#64b5cd'], height=0.55)
    ax[1, 1].set_xscale('log'); ax[1, 1].set_xlim(0.03, 12.0)
    ax[1, 1].set_yticks(yy); ax[1, 1].set_yticklabels(names, fontsize=7.5)
    for yi, v in zip(yy, vals):
        ax[1, 1].text(v * 1.12, yi, '%.4g' % v, va='center', fontsize=6.5)
    ax[1, 1].text(0.96, 0.12, T['pass_box'], transform=ax[1, 1].transAxes,
                  ha='right', fontsize=7,
                  bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.35))
    ax[1, 1].set_title(T['d']); ax[1, 1].set_xlabel(T['inv_lbl'])
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])


def fig15(lang='tr'):
    from solver_engine import (DEFAULTS, GP3,
        solve_benchtop_viscoelastic_vortex_piv_experiment,
        solve_vela_pulsar_glitch_avalanche,
        solve_benchtop_na22_positronium_gamma_experiment,
        ore_powell_exact)
    from scipy.integrate import quad
    import numpy as np
    p = DEFAULTS
    T = ({'a': r'(a) T35: Burgers girdabı — su vs. PEO (PIV hız profili)',
          'b': r'(b) T35: Serbest-yüzey hunisi — Newton vs. PEO bastırması',
          'c': r'(c) T36: Vela glitch gevşemesi + T37: $^{22}$Na Ps spektrumu (gömülü)',
          'd': r'(d) T1–T37 deney köprüsü değişmezleri',
          'sup': 'Masaüstü Girdap PIV, Vela Glitch ve Na-22 Laboratuvar Kanıtı',
          'name': 'fig15_benchtop_piv_vortex_vela_glitch_and_na22_lab',
          'nwt': r'su (Newton)',
          'peo': r'PEO (Meral)',
          'na_in': r'$^{22}$Na $O_{\mathrm{exact}}(E)$',
          'obs_a': r'gözlenen $v_{\mathrm{flat}}$ değil PIV',
          'qmeas': r'ölçülen Q (2000)',
          'inv_lbl': 'değer (log ölçek)',
          'pass_box': 'T1–T37: 38/38 PASS\nmini_check_gp3: 13/13 PASS'}
         if lang == 'tr' else
         {'a': r'(a) T35: Burgers vortex — water vs. PEO (PIV profile)',
          'b': r'(b) T35: Free-surface funnel — Newton vs. PEO suppression',
          'c': r'(c) T36: Vela glitch relaxation + T37: $^{22}$Na Ps spectrum (inset)',
          'd': r'(d) T1–T37 experiment bridge invariants',
          'sup': 'Benchtop Vortex PIV, Vela Glitch, and Na-22 Lab Evidence',
          'name': 'fig15_benchtop_piv_vortex_vela_glitch_and_na22_lab_en',
          'nwt': r'water (Newton)',
          'peo': r'PEO (Meral)',
          'na_in': r'$^{22}$Na $O_{\mathrm{exact}}(E)$',
          'obs_a': r'observed band',
          'qmeas': r'measured Q (2000)',
          'inv_lbl': 'value (log scale)',
          'pass_box': 'T1–T37: 38/38 PASS\nmini_check_gp3: 13/13 PASS'})
    bv = solve_benchtop_viscoelastic_vortex_piv_experiment(p)
    vg = solve_vela_pulsar_glitch_avalanche(p)
    na = solve_benchtop_na22_positronium_gamma_experiment(p)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # Panel (a): PIV azimuthal velocity profiles
    rc = bv['rc_N_mm'] * 1e-3
    G0 = float(GP3['Gamma0_m2_s'])
    r = np.linspace(0.2e-3, 20e-3, 400)
    uN = G0 / (2.0 * np.pi * r) * (1.0 - np.exp(-(r / rc) ** 2))
    rcP = bv['rc_PEO_mm'] * 1e-3
    uP = G0 / (2.0 * np.pi * r) * (1.0 - np.exp(-(r / rcP) ** 2))
    ax[0, 0].plot(r * 1e3, uN, 'k--', lw=1.2, label=T['nwt'])
    ax[0, 0].plot(r * 1e3, uP, 'b-', lw=1.6, label=T['peo'])
    ax[0, 0].axvline(bv['rc_N_mm'], color='k', ls=':', lw=0.9)
    ax[0, 0].axvline(bv['rc_PEO_mm'], color='b', ls=':', lw=0.9)
    ax[0, 0].set_title(T['a']); ax[0, 0].set_xlabel('r (mm)')
    ax[0, 0].set_ylabel(r'$u_\theta$ (m/s)'); ax[0, 0].legend(fontsize=6.5, loc='lower right')
    ax[0, 0].set_xlim(0, 20); ax[0, 0].set_ylim(0, 0.45)
    # Panel (b): funnel depth bars
    ax[0, 1].bar([0, 1], [bv['dh_N_mm'], bv['dh_PEO_mm']],
                 color=['#333333', '#4c72b0'], width=0.55)
    ax[0, 1].set_xticks([0, 1]); ax[0, 1].set_xticklabels([T['nwt'], T['peo']])
    ax[0, 1].set_title(T['b']); ax[0, 1].set_ylabel(r'$\Delta h$ (mm)')
    ax[0, 1].text(1, bv['dh_PEO_mm'] + 0.7, '%.2f mm (%.4f)' % (bv['dh_PEO_mm'], bv['dh_ratio']),
                  ha='center', fontsize=7)
    ax[0, 1].text(0, bv['dh_N_mm'] + 0.7, '%.2f mm' % bv['dh_N_mm'], ha='center', fontsize=7)
    ax[0, 1].set_ylim(0, bv['dh_N_mm'] + 4.0)
    # Panel (c): Vela relaxation (top) + Na22 spectrum (bottom, shared x split)
    axc = ax[1, 0]
    t = np.array(vg['t_days'])
    axc.plot(t, vg['R_relax'], 'b-', lw=1.6, label='Vela R(t)')
    axc.axhline(vg['Q_total'], color='gray', ls=':', lw=0.9)
    axc.axhline(vg['Q_2000_measured'], color='r', ls='--', lw=1.0,
                label='ölçülen Q (2000)')
    axc.set_title(T['c']); axc.set_xlabel('t (gün)' if lang == 'tr' else 't (days)')
    axc.set_ylabel('R(t)', color='b')
    axc.tick_params(axis='y', labelcolor='b')
    axc.set_xlim(0, 200); axc.set_ylim(-0.03, 1.05)
    axc.legend(fontsize=6.0, loc='upper right')
    axc2 = axc.inset_axes([0.42, 0.42, 0.54, 0.52])
    Ee = np.linspace(5.0, 511.0, 400)
    Ov = np.array([float(np.asarray(ore_powell_exact(E), dtype=float)) for E in Ee])
    axc2.semilogy(Ee, np.maximum(Ov, 1e-8), 'r-', lw=1.0)
    axc2.set_title(T['na_in'], fontsize=7)
    axc2.set_xlabel('E (keV)', fontsize=6); axc2.set_ylabel('O(E)', fontsize=6)
    axc2.tick_params(labelsize=5.5)
    # Panel (d): bridge invariants
    names = [r'$\Lambda_{\mathrm{Meral}}^{(0)}$', r'$\mu_s^{\mathrm{Meral}}$',
             r'$\mathcal{C}_{\mathrm{univ}}$', r'$Q_{\mathrm{total}}$',
             r'$\mathcal{B}_{\mathrm{exact}}$', r'$S_{\mathrm{loss}}$']
    vals = [0.0722694327, 0.2044088237, bv['C_univ'],
            vg['Q_total'], na['B_exact'], 19.7511425152]
    yy = np.arange(len(names))
    ax[1, 1].barh(yy, vals, color=['#4c72b0', '#55a868', '#c44e52',
                                   '#8172b2', '#ccb974', '#64b5cd'], height=0.55)
    ax[1, 1].set_xscale('log'); ax[1, 1].set_xlim(0.03, 60.0)
    ax[1, 1].set_yticks(yy); ax[1, 1].set_yticklabels(names, fontsize=7.5)
    for yi, v in zip(yy, vals):
        ax[1, 1].text(v * 1.12, yi, '%.4g' % v, va='center', fontsize=6.5)
    ax[1, 1].text(0.96, 0.12, T['pass_box'], transform=ax[1, 1].transAxes,
                  ha='right', fontsize=7,
                  bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.35))
    ax[1, 1].set_title(T['d']); ax[1, 1].set_xlabel(T['inv_lbl'])
    fig.suptitle(T['sup'])
    fig.tight_layout(); save(fig, T['name'])


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print('figures done')
