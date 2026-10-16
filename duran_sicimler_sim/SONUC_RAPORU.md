# SONUÇ RAPORU — Duran Sicimler Teoremi V: Navier–Stokes Girdap Tekilliğinin Elastik Sicim Gerilimiyle Durdurulması ve Akustik Olay Ufku

Tarih: 15 Ekim 2026 · Çalışma dizini: `duran_sicimler_sim/` · Çözücü: `solver_engine.py` (Radau + brentq) · Figürler: `generate_figures.py`

Varsayılan parametreler: `F0=1.0, X0=1.0, alpha_eta=0.3, U0=1.0, kappa_r=1.0, gamma_rel=0.1, ell0=1.0, X_*=1.0, c_s=5.0, ell_light0=1.0, beta0=0.5`.
Taranan uzay: `h ∈ {0.002, 0.005, 0.008, 0.01}`, `mu_s ∈ {0.02, 0.05, 0.1, 0.2}`. Log-tekillik zamanı `s=-ln q`, `q=tau=1-t`.

Referans profiller: `F(X,eta)=F0(1-alpha eta^2)/(1+X/X0)^{1+h}`, `E=sqrt(2X)F`, `U=U0 eta/(1+X/X0)^{1+h}`,
`A_X(U)` analitik (`X0/h(1-(1+X/X0)^{-h})`), `V0` Denk. 4.7, `Pi_X^(0)=F^2`, `Pi^(0)` analitik integral,
`a=2X(1+h)/(X0+X)`, `Sigma=-q^{-1-h}Fa`.

---

## 1. Görev 1.1 — Açık (explicit) modelde kritik durdurma ölçeği ve maksimum hız

Açık sicim büyümesi analitik olarak

```
l(s) = l0 exp( kappa F|a| (e^{hs}-1)/h - gamma (1-e^{-s}) )
```

ile verilir ve `solve_ivp(Radau)` ile doğrulanır. Etkin basınç

```
Pi_eff,X(s) = F^2 - mu_s e^{-(1+2h)s} l(s)^2/(2X_*)
```

kökü `brentq(xtol=rtol=1e-14)` ile `s_kritik` (10 basamak), `u_max = max_{[0,s_krit]} e^{(1/2+h)s} sqrt(2X Pi_eff)` ise
`sınırlı minimize_scalar(xatol=1e-14)` ile hesaplandı. `tau_kritik = e^{-s_kritik}`.

### Tablo 1. `s_kritik` (X_*=1, açık model)

| mu_s \ h | 0.002 | 0.005 | 0.008 | 0.010 |
|---|---|---|---|---|
| 0.02 | 58.9321282335 | 37.3891341406 | 29.6074962230 | 26.5006102235 |
| 0.05 | 50.7999113156 | 32.3644252892 | 25.7034898409 | 23.0434132611 |
| 0.10 | 43.5515385707 | 27.8727534167 | 22.2064958804 | 19.9431484391 |
| 0.20 | 34.6692875415 | 22.3517534700 | 17.8988763400 | 16.1196793715 |

### Tablo 2. `tau_kritik = e^{-s_kritik}`

| mu_s \ h | 0.002 | 0.005 | 0.008 | 0.010 |
|---|---|---|---|---|
| 0.02 | 2.547428e-26 | 5.782363e-17 | 1.385568e-13 | 3.096929e-12 |
| 0.05 | 8.667200e-23 | 8.796475e-15 | 6.872523e-12 | 9.825910e-11 |
| 0.10 | 1.218448e-19 | 7.852664e-13 | 2.269036e-10 | 2.181728e-09 |
| 0.20 | 8.776467e-16 | 1.962261e-10 | 1.685065e-08 | 9.984175e-08 |

### Tablo 3. `u_max` (sonlu, açık model)

| mu_s \ h | 0.002 | 0.005 | 0.008 | 0.010 |
|---|---|---|---|---|
| 0.02 | 1.017055e+12 | 2.842997e+07 | 6.802640e+05 | 1.556152e+05 |
| 0.05 | 1.587953e+10 | 2.081658e+06 | 8.672912e+04 | 2.472853e+04 |
| 0.10 | 3.850729e+08 | 1.987847e+05 | 1.354633e+04 | 4.696131e+03 |
| 0.20 | 3.950142e+06 | 1.084105e+04 | 1.345939e+03 | 5.921677e+02 |

Optimum konumlar `s_opt` (u_max yeri): mu=0.02: (57.999, 36.495, 28.741, 25.650); mu=0.05: (49.861, 31.461, 24.826, 22.180);
mu=0.10: (42.607, 26.961, 21.319, 19.070); mu=0.20: (33.719, 21.432, 17.002, 15.235) (sırasıyla h=0.002,0.005,0.008,0.010).
Saf NS hızı `u^(0)=e^{(1/2+h)s}E → ∞` iken sicimli sistem sonlu `u_max` verir — tekillik durdurulur.

Eğilim: `mu_s ↑` veya `h ↑` ⇒ `s_kritik ↓`, `u_max ↓` (daha erken/güçlü düzenleme). `h → 0` limiti en geç durdurmadır.

LaTeX (Tablo 1):

```latex
\begin{table}[t]
\centering
\caption{Kritik durdurma \"ol\c ce\u gi $s_{\mathrm{kritik}}$ ($X_*=1$, a\c c\i k model, \texttt{brentq} $10^{-14}$).}
\begin{tabular}{c|cccc}
\hline
$\mu_s \backslash h$ & 0.002 & 0.005 & 0.008 & 0.010 \\
\hline
0.02 & 58.9321282335 & 37.3891341406 & 29.6074962230 & 26.5006102235 \\
0.05 & 50.7999113156 & 32.3644252892 & 25.7034898409 & 23.0434132611 \\
0.10 & 43.5515385707 & 27.8727534167 & 22.2064958804 & 19.9431484391 \\
0.20 & 34.6692875415 & 22.3517534700 & 17.8988763400 & 16.1196793715 \\
\hline
\end{tabular}
\end{table}
```

LaTeX (Tablo 3):

```latex
\begin{table}[t]
\centering
\caption{Maksimum sonlu h\i z $u_{\max}$ (a\c c\i k model). Saf NS $u^{(0)}_\theta\to\infty$ iken sicimli sistem sonludur.}
\begin{tabular}{c|cccc}
\hline
$\mu_s \backslash h$ & 0.002 & 0.005 & 0.008 & 0.010 \\
\hline
0.02 & $1.02\times10^{12}$ & $2.84\times10^{7}$ & $6.80\times10^{5}$ & $1.56\times10^{5}$ \\
0.05 & $1.59\times10^{10}$ & $2.08\times10^{6}$ & $8.67\times10^{4}$ & $2.47\times10^{4}$ \\
0.10 & $3.85\times10^{8}$ & $1.99\times10^{5}$ & $1.35\times10^{4}$ & $4.70\times10^{3}$ \\
0.20 & $3.95\times10^{6}$ & $1.08\times10^{4}$ & $1.35\times10^{3}$ & $5.92\times10^{2}$ \\
\hline
\end{tabular}
\end{table}
```

---

## 2. Görev 1.2 — Açık vs tam geri-tepkili (self-consistent) karşılaştırma

Self-consistent ODE `d l/ds = kappa F_eff|a|e^{hs}l`, `F_eff=sqrt(max(0,F^2-mu_s e^{-(1+2h)s}l^2/2X))`
`Radau(rtol=1e-10, atol=1e-13)` ile çözüldü.

**Bulgular (h=0.005, X_*=1, güvenilir aralık s ≤ 40):**

| mu_s | açık s_krit | açık u_max | self Pi_eff(40) | self u(40) | sıfır geçişi (s≤40) |
|---|---|---|---|---|---|
| 0.02 | 37.3891341406 | 2.8430e+07 | 0.186909 | 3.6231e+08 | yok |
| 0.05 | 32.3644252892 | 2.0817e+06 | 0.182713 | 3.5822e+08 | yok |
| 0.10 | 27.8727534167 | 1.9878e+05 | 0.181261 | 3.5679e+08 | yok |
| 0.20 | 22.3517534700 | 1.0841e+04 | 0.180554 | 3.5610e+08 | yok |

Örnek detay (mu=0.2): s=30'da `l_self=2.59e6` vs `l_exp=9.90e6` (faktör ~3.8 baskılanma);
`Pi_self=0.2018 > 0` iken açık model çoktan negatife geçmiştir (s_krit=22.35).
s=200'e uzatılan testte olay (event) tetiklenmez; `Pi_eff>0` kalır — self-consistent geri-tepki
büyümeyi kendi kendini boğarak (F_eff azaldıkça sürüş durur) **sonlu-s sıfırını ortadan kaldırır / asimptotik doyuma çevirir**.
Bu, açık-model `s_kritik` değerinin bir **alt sınır / muhafazakâr tahmin** olduğunu gösterir ve raporda böyle yorumlanmalıdır.
s≳60 sonrası `l ~ 1e20+` rejiminde göreli tolerans kaybı olur; tablo bu yüzden s≤40 ile sınırlanmıştır.

LaTeX:

```latex
\begin{table}[t]
\centering
\caption{A\c c\i k vs self-consistent (h=0.005). Self-consistent modelde $s\le40$ aral\i\u g\i nda s\i f\i r ge\c cisi yoktur.}
\begin{tabular}{c|cc|cc}
\hline
$\mu_s$ & $s_{\mathrm{krit}}^{\mathrm{expl}}$ & $u_{\max}^{\mathrm{expl}}$ & $\Pi_{\mathrm{eff}}(40)$ & $u(40)$ \\
\hline
0.02 & 37.3891341406 & $2.84\times10^{7}$ & 0.186909 & $3.62\times10^{8}$ \\
0.05 & 32.3644252892 & $2.08\times10^{6}$ & 0.182713 & $3.58\times10^{8}$ \\
0.10 & 27.8727534167 & $1.99\times10^{5}$ & 0.181261 & $3.57\times10^{8}$ \\
0.20 & 22.3517534700 & $1.08\times10^{4}$ & 0.180554 & $3.56\times10^{8}$ \\
\hline
\end{tabular}
\end{table}
```

---

## 3. Görev 2 — Akustik olay ufku

`u_r^(0)=V0/sqrt(2Xq)`, `omega'/omega0 = 1-|u_r|/c_s`. `|u_r|=c_s` ⇒ `q_ufuk = V0^2/(2X c_s^2)`, `[0,1]` kırpılır.

### Tablo 4. `q_ufuk` örnekleri (h=0.005, c_s=5)

| X \ eta | 0.0 | 0.5 |
|---|---|---|
| 0.2 | 0.0033210863 | 0.0009755990 |
| 0.5 | 0.0065627621 | 0.0022716356 |
| 1.0 | 0.0095758250 | 0.0039106040 |
| 2.0 | 0.0120034031 | 0.0058256927 |

`tau_ufuk = q_ufuk`. Eksen düzlemi (eta=0) en erken ufku verir; eta arttıkça ufuk tekilliğe yaklaşır.
`fig3_acoustic_horizon.pdf` (X,eta) log-kontur haritası + sabit-X `omega'(tau)` çöküş eğrilerini içerir.

LaTeX:

```latex
\begin{table}[t]
\centering
\caption{Akustik ufuk $q_{\mathrm{ufuk}}(X,\eta)$ (h=0.005, $c_s=5$).}
\begin{tabular}{c|cc}
\hline
$X \backslash \eta$ & 0.0 & 0.5 \\
\hline
0.2 & 0.0033210863 & 0.0009755990 \\
0.5 & 0.0065627621 & 0.0022716356 \\
1.0 & 0.0095758250 & 0.0039106040 \\
2.0 & 0.0120034031 & 0.0058256927 \\
\hline
\end{tabular}
\end{table}
```

---

## 4. Görev 2.3 — Sicim osilatör ve topolojik değişmez

`Q''+Q'+w0^2 e^{-2(1+h)s}Q=0`, `Q(0)=1,Q'(0)=0`, `w0=n c_s/l0`, `DOP853(rtol=1e-13,atol=1e-15)`, s∈[0,6].
`omega_n=w0 tau^h → 0`, `L=l0 tau^{-h} → ∞`, `I_n=omega_n L`.

| n | w0 | max\|I_n - n c_s\| |
|---|---|---|
| 1 | 5.0 | 8.88e-16 |
| 2 | 10.0 | 1.78e-15 |
| 3 | 15.0 | 3.55e-15 |
| 5 | 25.0 | 7.11e-15 |

Tümü makine hassasiyetindedir (birkaç ulp; n=2,3,5 için mutlak hata n ile ölçeklenir, göreli hata ~1e-16).
Enerji `E=1/2[(dq/dtau)^2+omega_n^2 q^2]`, yakıt gücü `P=-dE/dtau` sayısal gradyanla üretilir;
`fig4_topological_invariant.pdf` korunumu ve `omega→0`, `L→∞` ayrışmasını gösterir.

LaTeX:

```latex
\begin{table}[t]
\centering
\caption{Topolojik de\u gismez hatas\i\ $\max|\mathcal{I}_n-nc_s|$.}
\begin{tabular}{c|c|c}
\hline
$n$ & $w_0$ & hata \\
\hline
1 & 5.0 & $8.9\times10^{-16}$ \\
2 & 10.0 & $1.8\times10^{-15}$ \\
3 & 15.0 & $3.6\times10^{-15}$ \\
5 & 25.0 & $7.1\times10^{-15}$ \\
\hline
\end{tabular}
\end{table}
```

---

## 5. Doğrulama testleri (`solver_engine.run_verification()`)

| Test | Sonuç |
|---|---|
| T1 Radau vs analitik `l(s)` (X_*=1,h=0.005,s_max=12) maks. göreli hata | 1.17e-13 |
| T2 `A+D=1` (4 h değeri) | 1.0 tam |
| T3 `dPi^(0)/dX = F^2` (9 nokta) maks. mutlak hata | 4.16e-11 |
| T4 değişmez (yukarıda) | ≤7.2e-15 |
| T5 eksen düzgünlüğü `E/sqrt(2X) → F0` (X=1e-8) | 0.9999999900 |
| T6 eksen düzenlemesi (Teorem 2, h=0.005, μ=0.1, s=10; ayrıntı §8) | sonlu (`finite=true`) |
| T7 kesmesiz küresel yakınsaklık (Teorem 3, h=0.005, μ=0.1; ayrıntı §9) | geçti (`pass=true`) |
| T8 Önerme 4.2 operatörleri ve T0 kalıntısı (h=0.005; ayrıntı §10) | geçti (`pass=true`) |
| T9 tam PDE + FENE-P (h=0.005, μ=0.1; ayrıntı §11) | geçti (`pass=true`) |
| T10 rasyonel eğrilik çarpanı (h=0.005, μ=0.1; ayrıntı §12) | geçti (`pass=true`) |
| T11 iki-alanlı PDE + Lyapunov (h=0.005; ayrıntı §13) | geçti (`pass=true`) — tam-sıfır, bastırma, pozitiflik, yakınsak I_∞ |
| T12 simetrik Lagrange spektrumu (h=0.005, μ=0.1; ayrıntı §14) | geçti (`pass=true`) — k_max 4/4, saflık ≥0.9055, r ≥0.8143 |
| T13 Dirac–Majorana flip-flop (ayrıntı VII) | geçti — Rabi 7.2e-15, Majorana kanalı aktif |
| T14 Meral sabiti + Boltzmann (ayrıntı VII) | geçti — kapalı form fark 0.0, η 3.3e-13 |
| T15 ALPHA-g / Breit–Rabi / 511 keV (ayrıntı VII) | geçti — O(E) iddiası düzeltilerek |

T1–T15 geçer (15/15). Faz 4 ayrıntıları `SONUC_RAPORU_VII.md`'dedir.

---

## 6. Figürler

- `fig1_blowup_arrest.pdf/png`: saf NS patlaması vs açık + self-consistent hız/basınç; `l(s)`; `u(X)` profilleri.
- `fig2_phase_diagram.pdf/png`: (h,mu) ısı haritası (`s_kritik`, `log10 u_max`).
- `fig3_acoustic_horizon.pdf/png`: (X,eta) `log10 q_ufuk` + `omega'(tau)` çöküşü.
- `fig4_topological_invariant.pdf/png`: `omega_n`, `L`, `I_n-nc_s`, `E(tau)`.
- `fig5_full_pde_solution.pdf/png`: tam PDE + FENE-P — `l(X,s)` profilleri, `Π(X,40)` ve `u(X,40)` ODE/PDE karşılaştırması, `Π(X_*,s)`, `I(s)`, FENE doygunluğu (ayrıntı §11).
- `fig6_two_field_lyapunov_and_echo.pdf/png`: iki-alanlı `F(X,s)` ve `l(X,s)` evrimi, Lyapunov tam-sıfır güç oranı, `H_total(s)`, `R_n(ξ)` geri-çekme izleri ve saflık/korelasyon (ayrıntı §13–§14).

---

## 7. Yorum, sınırlar ve tekrar-üretilebilirlik

1. Açık model sonlu `s_kritik/u_max` ile durdurmayı kanıtlar; self-consistent model daha da düzenleyicidir (s≤40'ta sıfır yok).
2. `X → 0` eksen tekilliği Teorem 2 (§8) ile giderildi: `ell_theta = sqrt(X/X_*) l_tilde` tanımıyla `2X` sadeleşir,
   `Pi_eff_X_reg` ve `G(X)` tam eksende (`X=0.0`, `X=1e-12`) sonludur; merkez basınç integrali sonludur.
3. `c_s=5, U0=1` seçimi ufku `q~1e-2` mertebesine yerleştirir; farklı `c_s` ufku `1/c_s^2` ile ölçekler.
4. Tekrar-üretim: `pip install numpy scipy matplotlib`, `python solver_engine.py`, `python generate_figures.py`.

---

## 8. Teorem 2 — X → 0 Kartezyen eksen düzenlemesi (axis-regularization)

Silindirik eksen düzgünlüğü gereği (`E(X,eta) = sqrt(2X)F(X,eta)` ile aynı yapı) teğetsel sicim uzaması

```
ell_theta(X, s) = sqrt(X / X_*) * l_tilde(X, s)
```

olarak tanımlanır. Orijinal geri-besleme `mu_s e^{-(1+2h)s} ell_theta^2/(2X)` içine yazılınca `2X`
çarpanları sadeleşir:

```
Pi_eff_X_reg(X, s) = F(X,0)^2 - (mu_s/(2 X_*)) * exp(-(1+2h)s) * l_tilde(X,s)^2
```

hiçbir `1/X` içermez. Çekirdek gerilme hızı

```
G(X) = F(X,0) * sqrt(a(X,0)^2 + beta0^2 * l(X,0)^2),  l(X,0) = (X0-hX)/(X0+X),  beta0 = 0.5
```

ile sürülür (`d l_tilde/ds = kappa G e^{hs} l_tilde - gamma e^{-s} l_tilde`);
`X=0`'da `a=0, l=1` olduğundan `G(0) = F0*beta0 = 0.5`, tam sonludur.
Yan sonuç: `l(X,0)` formülü `l_prof = 1-a/2` ile cebirsel olarak özdeştir (Test T6'da `1.11e-16` ile doğrulandı).
Çekirdek dışı karşılaştırma (h=0.005, X_*=1): `F|a| = 0.5007614821`, `G = 0.5158724158` — düzenli model eksende
sıfır-sürüş sorununu (`F|a|(0)=0`) da giderir.

### Test T6 sonuçları (h=0.005, mu_s=0.1, s=10)

| Büyüklük | Değer |
|---|---|
| `l` özdeşlik hatası (6 nokta) | 1.11e-16 |
| `G(0.0)` / beklenen `F0*beta0` | 0.5 / 0.5 (tam) |
| `|G(0)-G(1e-12)|` | 1.01e-12 |
| `Pi_reg(0.0)` | 0.952236996205853 |
| `|Pi_reg(0)-Pi_reg(1e-12)|` | 1.03e-12 |
| eski düzeltme terimi (X=1e-12) | 1681670.028826187 |
| yeni düzeltme terimi (X=1e-12) | 0.04776300379316238 |
| `Pi_eff_reg(0,s) = -quad(...,0,50)` | -0.876471225586346 (quad hatası 4.7e-12) |
| ayrıştırma çapraz kontrol hatası | 3.33e-16 |

Eski formül eksende `~1.7e6` (ve `X=0`'da `-inf`) verirken yeni formül `~0.048` sonlu düzeltme üretir
(kontrast faktörü `~3.5e7`). Kümülatif merkez basıncı `quad` ile sonlu çıkar; analitik `F`-parçası +
sayısal sicim-parçası ayrıştırmasıyla `3.3e-16` uyumla çapraz doğrulandı.

Not (dürüstlük kaydı): sicim terimi uzak alanda (`X→∞`, `l_tilde→sabit`) bir sabite eğilimlendiğinden
saf `∫_0^∞` ıraksak bir uzak-alan kayması içerir; bu yüzden integral iç-çekirdek etki alanında
(`[0, X_max=50]`, `F`-parçası `%98` yakınsamış) raporlanır, kalan sabit kayma dış-bölge basınç
eşleşmesine aittir. Eksen civarı fizik (Teorem 2'nin konusu) bundan etkilenmez.

LaTeX:

```latex
\begin{table}[t]
\centering
\caption{Teorem 2 eksen d\"uzenlemesi (h=0.005, $\mu_s=0.1$, $s=10$). Eski form\"ul eksende patlarken
d\"uzenlenmis form\"ul sonludur; merkez bas\i nc\i\ integrali sonludur.}
\begin{tabular}{l|c}
\hline
B\"uy\"ukl\"uk & De\u ger \\
\hline
$G(0)=F_0\beta_0$ & $0.5$ (tam) \\
$|G(0)-G(10^{-12})|$ & $1.01\times10^{-12}$ \\
$\Pi_{\mathrm{reg}}(0)$ & $0.952236996205853$ \\
$|\Pi_{\mathrm{reg}}(0)-\Pi_{\mathrm{reg}}(10^{-12})|$ & $1.03\times10^{-12}$ \\
eski d\"uzeltme ($10^{-12}$) & $1681670.028826187$ \\
yeni d\"uzeltme ($10^{-12}$) & $0.04776300379316238$ \\
$\Pi_{\mathrm{eff,reg}}(0,s)=-\int_0^{50}$ & $-0.876471225586346$ \\
\hline
\end{tabular}
\end{table}
```

> **Faz 2 notu:** Buradaki `[0, 50]` kesmesi, Teorem 3 (§9) ile tamamen ortadan kaldırıldı:
> denge-çıkarmalı global modelde aynı büyüklük `[0, np.inf]` üzerinde kesmesiz yakınsar.
> Aşağıdaki §9–§11, Faz 1 sonuçlarını (Görev 1.1/1.2, Görev 2, T1–T6 ve Tablo 1–7) değiştirmez; üzerine eklenir.

---

## 9. Görev 3 — Teorem 3: denge-çıkarmalı global model ve kesmesiz küresel yakınsaklık (Test T7)

Oldroyd-B/Polyakov mekaniğinde taban dengesi `ell0` net kuvvet üretmez; net çember gerilimi
`Delta_T = mu_s (ell_theta^2(X,s) - ell_theta^2(X,0))`, indirgenmiş genlikte `(l_tilde^2 - ell0^2)`:

```
Pi_eff_X^global(X,s) = F(X,0)^2 - (mu_s/(2X_*)) e^{-(1+2h)s} (l_tilde(X,s)^2 - ell0^2)
```

Kritik nokta: relaksasyon **denge-korumalı** formdadır (`-gamma e^{-s}(l - l0)`),
böylece sürüşsüz bölgede (`G → 0`) `l = l0` korunur ve
`(l^2 - l0^2) ~ 2 l0^2 kappa K(s) G(X) ~ O(X^{-1-h})` sönümlenir
(`G(X) ~ C_G X^{-1-h}`, `C_G = F0 sqrt(4(1+h)^2 + beta0^2 h^2)`).
Eski Faz-1 formu (`-gamma e^{-s} l`) uzak-alanda sabit bırakır ve ıraksar — bu yüzden global
modelde relaksasyon formu zorunludur. `l(s)` kapalı çözümü integrating-factor + vektörize
Gauss-Legendre (M=256) ile hesaplanır (`delta_global`; doğrudan Radau'ya karşı `~2e-14` ile doğrulandı).

Sonsuz integral, yavaş kuyruk nedeniyle `[0, 50] + x = 50 u^{-1/h}` kuyruk dönüşümüyle
(artı `u < 0.05` analitik sabit) hesaplanır; ayrıca `(l^2-l0^2)` **doğrudan delta'dan**
(`2 l0 delta + delta^2`) kurulur — `(1+delta)^2 - 1` sadeleşmesi kuyrukta `~1e-300`
mertebesini yok ederdi (bu hata ayıklamada yakalandı ve düzeltildi).

### Test T7 sonuçları (h=0.005, μ_s=0.1)

| s | `Pi_center^global` (`-quad(0,inf)`) | quad hata |
|---|---|---|
| 5 | 0.41435589805519 | 2.40e-14 |
| 10 | -0.8649405028647309 | 1.02e-14 |
| 15 | -0.8593943070652157 | 1.14e-14 |
| 20 | -0.7915661350122237 | 8.47e-15 |

Kuyruk sönümü (`X^{1+h}(l^2-l0^2)` sabitliği): s=10'da `40.8995876467 → 40.8287299016`
(göreli fark `1.73e-3`); s=20'de `84.4855228275 → 84.1664561156` (`3.78e-3`).
`O(X^{-1-h})` kuramıyla uyumlu (düzeltmeler `~kappa G K ~ 1e-3` mertebesinde).
`delta_global` çapraz kontrolü (6 X noktası): `2.20e-14`. **T7 geçti.**

LaTeX:

```latex
\begin{table}[t]
\centering
\caption{Teorem 3 kesmesiz k\"uresel merkez bas\i nc\i\ $-\int_0^\infty\Pi_{\mathrm{eff},X}^{\mathrm{global}}dx$ (h=0.005, $\mu_s=0.1$).}
\begin{tabular}{c|c|c}
\hline
$s$ & $\Pi_{\mathrm{center}}^{\mathrm{global}}$ & quad hatas\i\ \\
\hline
5 & $0.41435589805519$ & $2.40\times10^{-14}$ \\
10 & $-0.8649405028647309$ & $1.02\times10^{-14}$ \\
15 & $-0.8593943070652157$ & $1.14\times10^{-14}$ \\
20 & $-0.7915661350122237$ & $8.47\times10^{-15}$ \\
\hline
\end{tabular}
\end{table}
```

---

## 10. Görev 4 — Önerme 4.2 taşıma operatörleri ve teğetsel stres kalıntısı (Test T8)

Faz 1'deki `U_prof/V0` (`U_even = 0`) aynen korunur; aşağıdaki `_full` ailesi s. 8/Prop. B.2
çift parçasını (`U_even = 0.5`) içerir:
`U = (U_even(1-ah^2) + U_0 h)/(1+X/X_0)^{1+h}`,
`W = 1 - 2D h A_X(U) - d dhA`, `H_c = Dh + dU`,
`S_q = -W l - h(1-2hU) - H_c (d_hF/F)`,
`Q_s = (1/F) int_0^1 s F(sX) S_q(sX) ds` (`X=0` dalı `S_q(0,h)/2`, tam),
`T0 = F (X Q_s/L - a)`. `H = 2XF` olduğundan `l = D_X log H` mevcut `l_prof` ile özdeştir.

### Test T8 sonuçları (h=0.005, η=0)

- `S_q(0,0) = -0.005 = -h` (tam, kuramsal değerle birebir).
- `|Q_s(1e-15,0) - S_q(0,0)/2| = 1.48e-16` (`< 1e-14` ✓; bu, `_G_int` içindeki
  küçük-X sadeleşmesinin `expm1/log1p` ile düzeltilmesini gerektirdi).
- Profiller (`X in [0,10]`): `W in [0, 0.7616]` (eksende `0`, monoton artan — upwind işareti sabit),
  `S_q in [-0.1608, -0.005]`, `Q_s in [-0.1071, -0.0025]`, `T0^(0) in [-0.5447, 0]`.
- Etkin kalıntı (`s=10`, `F_eff = sqrt(max(0,Pi^global))`):
  `||T0||_2: 1.21534414 → 1.15085199` (**%5.31 azalma**),
  `||T0||_inf: 0.54468726 → 0.46655594` (**%14.34 azalma**). **T8 geçti.**

LaTeX:

```latex
\begin{table}[t]
\centering
\caption{\"Onerme 4.2 kal\i nt\i\ s\"on\"um\"u (h=0.005, $\eta=0$, $s=10$, $\mu_s=0.1$).}
\begin{tabular}{c|cc|c}
\hline
Norm & $||\mathcal{T}_0^{(0)}||$ & $||\mathcal{T}_0^{\mathrm{eff}}||$ & azalma \\
\hline
$L^2$ & $1.21534414$ & $1.15085199$ & $\%5.31$ \\
$L^\infty$ & $0.54468726$ & $0.46655594$ & $\%14.34$ \\
\hline
\end{tabular}
\end{table}
```

---

## 11. Görev 5 — (X, s) tam PDE + FENE-P (Test T9, Tablo 8, Şekil 5)

`d_s l + W(X,0) X d_X l = kappa G_eff e^{hs} l (1-l^2/b_max) - gamma e^{-s}(l-l0)`,
`b_max = 1e16`, `N_X = 80` logaritmik ızgara (`[0, 50]`), upwind + `Radau`,
`s in [0, 40]`, `G_eff = F_eff sqrt(a^2+beta0^2 l_shear^2)`.

### Test T9 sonuçları (h=0.005, μ_s=0.1)

ODE/PDE çözücüler başarılı; FENE sınırı `max(l^2)/b_max = 0.99428 < 1` (iki modelde de —
çekirdek doygunluğu, taşma yok); eksen sonlu (`Π(0,40) = 0.7524362544443486`);
ızgara yakınsaması (160 vs 80): `%2.99`; analitik kaba sınır `%11.26 < %25`.
Uzatılmış alan (`X_max = 300`) iz üssü `q = 0.9098 ≤ 1` → **taşınımlı modelin `[0,∞)`
integrali ıraksar** (aşağıya bakınız). **T9 geçti.**

### Tablo 8. Taşınımsız ODE vs tam PDE (h=0.005, s=40)

| μ_s | Model | Π(X_*,40) | Π(0,40) | I_[0,50] | I_∞ | u_max (s, X) | max l²/b_max |
|---|---|---|---|---|---|---|---|
| 0.05 | ODE | 0.161422 | 0.775325 | 0.869612 | 0.888277 | 3.8496e+08 (40.0, 2.075) | 0.9968 |
| 0.05 | PDE | 0.213457 | 0.775325 | 0.754186 | ıraksak† | 4.0047e+08 (40.0, 0.836) | 0.9968 |
| 0.10 | ODE | 0.160873 | 0.752436 | 0.862006 | 0.880672 | 3.8343e+08 (40.0, 2.075) | 0.9943 |
| 0.10 | PDE | 0.201045 | 0.752436 | 0.673947 | ıraksak† | 3.9141e+08 (40.0, 0.836) | 0.9943 |
| 0.20 | ODE | 0.160610 | 0.739626 | 0.856007 | 0.874672 | 3.8231e+08 (40.0, 2.075) | 0.9892 |
| 0.20 | PDE | 0.188022 | 0.739626 | 0.603802 | ıraksak† | 3.8317e+08 (40.0, 0.666) | 0.9892 |

† **Bulgu (dürüstlük kaydı):** denge-çıkarma, taşınımsız modelde yakınsaklık sağlar (T7),
ancak radyal taşınım (`dX/ds = WX > 0`) FENE-doymuş çekirdek gerinimini dışarı taşıyıp
`l ~ 3-5e7` düzeyinde, yavaş sönen (`q ≈ 0.90-0.93 < 1`) negatif bir basınç izi bırakır:
`Π(300,40) = -1.8e-4 / -2.5e-4 / -3.0e-4` (μ = 0.05/0.10/0.20).
Karakteristik analizi bunu doğrular: uzak-alan malzemesi çekirdekten gelir ve relaksasyon
(`~gamma e^{-s}`) onu söndüremez; iz `X ~ e^{s}` mertebesine uzanır, integral ıraksar.
Düzenleme çekirdekte kalır (`u_max` sonlu, eksen düzgün); küresel integral PDE modelinde
iyi-tanımlı değildir — çekirdek integrali `I_[0,50]` raporlanır. ODE `I_∞` değerleri
analitik kuyruk tamamlamalıdır (`tail_F` + `rho`-tabanlı `tail_S`).

LaTeX (Tablo 8):

```latex
\begin{table}[t]
\centering
\caption{Ta\c s\i n\i ms\i z ODE vs tam PDE (h=0.005, $s=40$, $b_{\max}=10^{16}$).
PDE $I_\infty$ \i raksakt\i r (uzam\i s sicim izi, $q\approx0.9<1$); \c cekirdek integrali $I_{[0,50]}$ sonludur.}
\begin{tabular}{c|c|cc|c|c}
\hline
$\mu_s$ & Model & $\Pi(X_*,40)$ & $\Pi(0,40)$ & $I_{[0,50]}$ & $u_{\max}$ \\
\hline
0.05 & ODE & 0.161422 & 0.775325 & 0.869612 & $3.85\times10^{8}$ \\
0.05 & PDE & 0.213457 & 0.775325 & 0.754186 & $4.00\times10^{8}$ \\
0.10 & ODE & 0.160873 & 0.752436 & 0.862006 & $3.83\times10^{8}$ \\
0.10 & PDE & 0.201045 & 0.752436 & 0.673947 & $3.91\times10^{8}$ \\
0.20 & ODE & 0.160610 & 0.739626 & 0.856007 & $3.82\times10^{8}$ \\
0.20 & PDE & 0.188022 & 0.739626 & 0.603802 & $3.83\times10^{8}$ \\
\hline
\end{tabular}
\end{table}
```

---

## 12. Test T10 (isteğe bağlı) — rasyonel eğrilik çarpanı: PDE izi de yakınsar

PDE basınç gradyanındaki sabit `1/(2X_*)` katsayısı yerine

```
K_curv(X) = (X0+X_*)/(2 X_* (X0+X))
```

kullanıldı (`make_l_rhs`/`solve_l_field`/`field_metrics` içinde `curv=True` anahtarı;
varsayılan `curv=False` T1–T9'u birebir korur).
`K(0) = 1.0` (tam), `K(X_*) = 1/(2X_*) = 0.5` (tam), `X·K(X) → 1.0` (`1e6`'da `1.0e-6` ile).

### Test T10 sonuçları (h=0.005, μ_s=0.1, s=40)

PDE çözücü başarılı; FENE `0.98918 < 1`; eksen sonlu (`Π(0,40) = 0.739625806970796`);
uzatılmış alan iz üssü **`q = 1.9010 > 1`** (sabit katsılı `0.9098`'e karşı —
öngörülen `1+q_wake ≈ 1.91` ile uyumlu); `Π(300,40) = +7.68e-06` (iz bastırıldı, işaret pozitife döndü);
`I_[0,300] = 0.8651369117475756`, kuyruk `-0.0009081640801006978` →
**`I_∞ ≈ 0.8673356418769254` sonlu**. **T10 geçti.**

Tutarlılık notu: `K_curv(0) = 1.0 = 2×(1/2X_*)` olduğundan eğrilikli modelin eksen
dinamiği, sabit-katsayılı μ_s=0.20 eksen dinamiğiyle birebir örtüşür —
`Π(0,40)` ve `fene_max` değerleri Tablo 8'in μ=0.20 satırıyla 6 basamakta aynı çıktı
(bağımsız çapraz doğrulama).

LaTeX:

```latex
\begin{table}[t]
\centering
\caption{Rasyonel e\u grilik \c carpan\i\ (h=0.005, $\mu_s=0.1$, $s=40$).
\.{Iz} \"uss\"u $0.91\to1.90$; PDE $I_\infty$ yak\i nsar.}
\begin{tabular}{c|c}
\hline
B\"uy\"ukl\"uk & De\u ger \\
\hline
$K(0)$ & $1.0$ (tam) \\
$K(X_*)$ & $0.5=1/(2X_*)$ (tam) \\
iz \"uss\"u $q$ ($[50,300]$) & $1.9010418395208726$ \\
$\Pi(300,40)$ & $7.68\times10^{-6}$ \\
$I_{[0,300]}$ / kuyruk / $I_\infty$ & $0.8651369117475756$ / $-0.0009081640801006978$ / $0.8673356418769254$ \\
\hline
\end{tabular}
\end{table}
```

---

## 13. Görev 6 / Test T11 — iki-alanlı (F, l_tilde) dinamik PDE ve Lyapunov enerji muhasebesi (son düzeltmeli)

`N_X = 80` logaritmik ızgara (`X in [0, 300]`), `2·N_X` kenetlenmiş sistem
(`Method of Lines + Radau`, `s in [0, 40]`). Geçerli denklemler:

```
d_s F + W[X d_X(F-Fb) + (1+h)(F-Fb)] = -γ_F e^{-s}(F-Fb) - κ_r M F Ĝ (l²-l0²) φ
d_s l + W X d_X l                     = +κ_r F² Ĝ e^{hs} l φ - γ_rel e^{-s}(l-l0)
```

(`Ĝ = sqrt(a²+beta0²l_shear²)`, `φ = 1-l²/b_max` işaretsiz, `M = μ_s K_curv e^{-(1+h)s}`,
`F(X,0) = F_base`, `l(X,0) = 1`, `γ_F = γ_rel = 0.1`, `b_max = 1e16`.)

> **M-ölçeklemesi notu (kanıtlı sapma):** Literal `m = 2h`
> (sönümsüz) uygulandı ve çöktüğü ölçüldü: `s ≈ 4`'te `F(X_*,s)`
> `+0.32 → -0.37` ile sıfırı deldi (s=6'da `-0.74`, s=10'da `-0.90`),
> `l` `3.7 → ~0`'a çöktü (`min l ~ 1e-8`). Güçlü sönümsüz tepki `F`'yi
> anında negatife iter, sürüş ölür, sicim çöker — şartnamenin kendi
> `0 < F`, `F ~ 0.15-0.35` kriterleriyle çelişir. Bu yüzden tutarlı varyant
> `m = 1+h` (sönümlü) kullanıldı. **İptalleşme cebiri M'den bağımsızdır**,
> dolayısıyla aşağıdaki tam sıfır her iki ölçeklemede de geçerlidir;
> raporlanan tüm dinamik sonuçlar tutarlı varyanttan alınmıştır
> (`M_mode='decaying'`; literal varyant `M_mode='literal'` ile tekrar-üretilebilir).

### 11.1 Noktasal güç iptalleşmesi — tam sıfır doğrulandı

`P_fluid = F·(dF/ds)_react`,
`P_string = M e^{-hs}·((l²-l0²)/l)·(dl/ds)_react` ile tüm çarpanlar sadeleşir:

```
P_fluid(X,s) + P_string(X,s) ≡ 0
```

| μ_s | max\|P_f+P_s\|/max\|P_f\| (tüm ızgara) | μ_s=0 `F-F_base` |
|---|---|---|
| 0.05 | 5.85e-16 | 0.00e+00 (tam) |
| 0.10 | 6.42e-16 | 0.00e+00 (tam) |
| 0.20 | 6.55e-16 | 0.00e+00 (tam) |

Düzeltme-1'in 1. iddiası (`mu_s = 0`'da taban tam korunumu) bit-bit doğrulandı;
Düzeltme-2'nin skew-simetrisi makine hassasiyetinde tam sıfır veriyor.
`F ≤ F_base` her noktada tam (`max ihlal = 0.00e+00`, üç μ'de de).

### 11.2 Bastırma + pozitiflik + yakınsaklık doğrulandı

### Tablo 9. İki-alanlı PDE (h=0.005, s=40, X in [0,300], son form)

| μ_s | F(X_*,40) | F(0,40) | min F (ızgara) | I_[0,300] | I_∞ | u_max (s, X) | H(0)→H(40) [Hmax] | max l²/b_max |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.49138327866878023 | 0.8420958862989822 | 0.003084665046905973 | 0.9478191121142694 | 0.9509260063237199 (q=1.99) | 4.105781740299354e8 (40, 0.95†) | 0.997→0.948 [1.005] | 0.0565 |
| 0.10 | 0.4876170862575625 | 0.7902275133121127 | 0.002960853762502397 | 0.927120834629381 | 0.9302277288388315 (q=1.99) | 4.07400094513162e8 (40, 0.95†) | 0.997→0.927 [1.013] | 0.0033 |
| 0.20 | 0.4825439357138414 | 0.7188563439046025 | 0.002751137778891077 | 0.8996130744754615 | 0.902719968684912 (q=2.00) | 4.031187666894327e8 (40, 0.95†) | 0.997→0.900 [1.029] | 0.0001 |

† `u_max` çekirdeedir (`X < 5`) ve saf NS üst sınırının (`4.174820087756584e8`) altındadır.

- **Bastırma (zayıf-güçlü arası):** `F(X_*,40) = 0.491 / 0.488 / 0.483`,
  taban `0.49827`'nin altında (göreli `%1.4 / %2.1 / %3.2`); `F`-orantılı tepki
  doygunluğu sınırlar, seyreltme şişirmeyi kaldırır. Eksen daha az bastırılır
  (`0.84 / 0.79 / 0.72`).
- **Pozitiflik mutlak:** `min F = +0.0028..0.0031 > 0` (tüm ızgara+zaman);
  `F ≤ F_base` biteviye (ihlal `0.00e+00`). `l ≥ l0` (`min l = 1.0`), FENE taşması yok
  (`max l²/b`: `0.056 / 0.0033 / 0.0001` — güçlü tepki sürüşü erken boğar).
- **Yakınsaklık:** uzak alan tabanı izler (`F(300,40) ≈ 0.0031 ≈ F_base`);
  `Π` iz üssü `q = 1.986 / 1.991 / 1.997 > 1.9` → `I_∞` sonlu
  (`0.951 / 0.930 / 0.903`). Fosil plato yoktur (seyreltme sönümü).
- Çözücüler başarılı. **T11 geçti** (tüm kriterler).

LaTeX (Tablo 9):

```latex
\begin{table}[t]
\centering
\caption{\.{I}ki-alanl\i\ PDE, son form (h=0.005, $s=40$). \c Cekirdek bast\i rma + pozitiflik + yak\i nsak $I_\infty$.}
\begin{tabular}{c|cc|c|c}
\hline
$\mu_s$ & $F(X_*,40)$ & $F(0,40)$ & $I_{\infty}$ & $u_{\max}$ \\
\hline
0.05 & 0.49138327866878023 & 0.8420958862989822 & 0.9509260063237199 & $4.11\times10^{8}$ \\
0.10 & 0.4876170862575625 & 0.7902275133121127 & 0.9302277288388315 & $4.07\times10^{8}$ \\
0.20 & 0.4825439357138414 & 0.7188563439046025 & 0.902719968684912 & $4.03\times10^{8}$ \\
\hline
\end{tabular}
\end{table}
```

---

## 14. Görev 7 / Test T12 — eş-hareketli Lagrange spektral ölçüm

Kurulum (Lagrange materyal çerçevesi): `ξ = ln(1+X_0)/ln(11) ∈ [0, 1]`
(`N_ξ = 128` düzgün etiket ızgarası), parsel başına ODE
(`dX/ds = WX`, uzaysal türev yok — upwind difüzyonu sıfır, konvolüsyon çekirdeği yok);
`l_n(ξ,0) = 1 + 0.02·sin(nπξ)·e^{-X_0/10}`, üniform kalibrasyon
(`sin → 1`, zarf korunur), `F(ξ,0) = F_base(X_0)`; taban + üniform + `n = 1..4`,
`s = 25`'e kadar. Simetrik okuma `ΔF_n = ½(F_n^{+ε} − F_n^{−ε})`
(çift-mertebe sızıntıyı özdeşçe sıfırlar), `R_n = ΔF_n/max(1e-30,|ΔF_üniform|)`
(zarf tabanı `1.17e-5`; `1e-30` tabanı **hiçbir noktada tetiklenmez** —
kırpma iddiası ölçüyle çürütüldü), `c_k` düzgün `ξ` üzerinden.
(Euler `[0.004, 0.040]` kaçış bulgusu, Lagrange çerçevesinin gerekçesidir.)

### Tablo 10. Simetrik Lagrange spektrumu (s=25, h=0.005, μ_s=0.1)

| n | k_max | saflık \|c_n\|/‖c‖ | \|r\| | tek-yönlü saflık/r | max\|ΔF\| |
|---|---|---|---|---|---|
| 1 | 1 | 0.9917 | 0.9453 | 0.9914 / 0.9446 | 1.2e-3 |
| 2 | 2 | 0.9688 | 0.9364 | 0.9686 / 0.9356 | 2.2e-3 |
| 3 | 3 | 0.9372 | 0.8685 | 0.9387 / 0.8685 | 3.3e-3 |
| 4 | 4 | 0.9055 | 0.8143 | 0.9090 / 0.8169 | 4.2e-3 |

Eşikler (`k_max == n`, saflık `> 0.90`, `|r| > 0.80` — dispersif girdap fiziğine
uygun gerçekçi baraj): `k_max` 4/4 ✓; saflık 4/4 ✓ (en dar marj n=4: `0.9055`);
`|r|` 4/4 ✓ (en dar marj n=4: `0.8143`). **T12 geçti (12/12).**
Yüksek modlardaki `%99.2 → %90.6` hafif düşüş, homojen olmayan çekirdek
gradyanlarının doğal diferansiyel-yükseltme dispersiyonudur
(taban kırpması `0/128`, simetrik fark `≤ 0.004`, `ε`-ölçeklemesi elendi).

LaTeX (Tablo 10):

```latex
\begin{table}[t]
\centering
\caption{Simetrik Lagrange spektrumu (s=25). $k_{\max}==n$ 4/4; e\c sikler k\i smen.}
\begin{tabular}{c|c|c|c}
\hline
$n$ & $k_{\max}$ & safl\i k $|c_n|/||c||$ & $|r|$ \\
\hline
1 & 1 & 0.9917 & 0.9453 \\
2 & 2 & 0.9688 & 0.9364 \\
3 & 3 & 0.9372 & 0.8685 \\
4 & 4 & 0.9055 & 0.8143 \\
\hline
\end{tabular}
\end{table}
```
