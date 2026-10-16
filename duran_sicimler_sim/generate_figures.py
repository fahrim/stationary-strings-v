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

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print('figures done')
