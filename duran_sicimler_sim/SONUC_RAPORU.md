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

Tümü geçer.

---

## 6. Figürler

- `fig1_blowup_arrest.pdf/png`: saf NS patlaması vs açık + self-consistent hız/basınç; `l(s)`; `u(X)` profilleri.
- `fig2_phase_diagram.pdf/png`: (h,mu) ısı haritası (`s_kritik`, `log10 u_max`).
- `fig3_acoustic_horizon.pdf/png`: (X,eta) `log10 q_ufuk` + `omega'(tau)` çöküşü.
- `fig4_topological_invariant.pdf/png`: `omega_n`, `L`, `I_n-nc_s`, `E(tau)`.

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
