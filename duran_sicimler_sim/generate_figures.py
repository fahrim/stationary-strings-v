"""Yayin kalitesinde 5 figur uretici (Faz 2: fig5 eklendi)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver_engine import (DEFAULTS, H_VALUES, MU_VALUES, F_prof, E_prof,
    u_theta_0, Pi_eff_X, E_eff_of_s, u_eff_of_s, ell_explicit_analytic,
    find_s_kritik, compute_u_max, solve_selfconsistent, solve_ell_explicit,
    V0_prof, u_r_0, q_horizon, omega_prime_ratio, solve_string_modes,
    compute_table8)

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

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5()
    print('figures done')
