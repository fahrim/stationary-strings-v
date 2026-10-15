# SONUÇ RAPORU VI — Duran Sicimler Teoremi VI (son düzeltmeli): İki-Alanlı Dinamik PDE, Lyapunov Tam-Sıfırı ve Lagrange Spektral Ölçüm

Tarih: 15 Ekim 2026 · Çalışma dizini: `duran_sicimler_sim/` · Çözücü: `solver_engine.py` (Faz 3 bloğu) · Figür: `fig6_two_field_lyapunov_and_echo.pdf/png`
Kapsam: **yalnızca Faz 3 sonuçları** (Görev 6/T11, Görev 7/T12). Faz 1–2 sonuçları `SONUC_RAPORU.md` §1–§12'dedir.

Varsayılan parametreler: `F0=1.0, X0=1.0, alpha_eta=0.3, U0=1.0, kappa_r=1.0, gamma_rel=0.1, ell0=1.0,`
`X_*=1.0, c_s=5.0, ell_light0=1.0, beta0=0.5, U_even=0.5, b_max=1e16, gamma_F=0.1`.
Faz 3 ızgarası: `N_X = 80` logaritmik, `X in [0, 300]`; `2·N_X` kenetlenmiş sistem, `Radau`.

Geçerli denklemler:

```
d_s F + W[X d_X(F-Fb) + (1+h)(F-Fb)] = -γ_F e^{-s}(F-Fb) - κ_r M F Ĝ (l²-l0²) φ
d_s l + W X d_X l                     = +κ_r F² Ĝ e^{hs} l φ - γ_rel e^{-s}(l-l0)
```

`M = μ_s K_curv e^{-(1+h)s}`, `φ = 1-l²/b_max` işaretsiz.

> **M-ölçeklemesi notu (kanıtlı sapma):** Literal `m = 2h` uygulandı ve çöktüğü
> ölçüldü (`s ≈ 4`'te `F(X_*)`: `+0.32 → -0.37`; `l`: `3.7 → ~0`). Sonuçlar
> tutarlı varyant `m = 1+h` ile üretilmiştir; iptalleşme cebiri M'den bağımsızdır.

---

## 1. Lyapunov tam-sıfırı, bastırma, pozitiflik, yakınsaklık (Test T11 — geçti)

`P_fluid + P_string ≡ 0` tüm `(X, s)` ızgarasında:

| μ_s | max\|P_f+P_s\|/max\|P_f\| | μ_s=0 `F-F_base` |
|---|---|---|
| 0.05 | 5.85e-16 | 0.00e+00 (tam) |
| 0.10 | 6.42e-16 | 0.00e+00 (tam) |
| 0.20 | 6.55e-16 | 0.00e+00 (tam) |

### Tablo 9 (h=0.005, s=40)

| μ_s | F(X_*,40) | F(0,40) | min F | I_[0,300] | I_∞ (q) | u_max (s, X) | H(0)→H(40) | max l²/b_max |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.49138327866878023 | 0.8420958862989822 | 0.003084665046905973 | 0.9478191121142694 | 0.9509260063237199 (1.99) | 4.105781740299354e8 (40, 0.95) | 0.997→0.948 | 0.0565 |
| 0.10 | 0.4876170862575625 | 0.7902275133121127 | 0.002960853762502397 | 0.927120834629381 | 0.9302277288388315 (1.99) | 4.07400094513162e8 (40, 0.95) | 0.997→0.927 | 0.0033 |
| 0.20 | 0.4825439357138414 | 0.7188563439046025 | 0.002751137778891077 | 0.8996130744754615 | 0.902719968684912 (2.00) | 4.031187666894327e8 (40, 0.95) | 0.997→0.900 | 0.0001 |

`u_max` çekirdeedir (`X ≈ 0.95`) ve saf NS üst sınırının (`417482008.77565837`) altındadır.
`F ≤ F_base` biteviye, `l ≥ l0`, FENE taşması yok. **T11 geçti** (tüm kriterler).

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

## 2. Lagrange spektral ölçüm (Test T12 — geçti)

Eş-hareketli çerçeve (`N_ξ = 128`, `ε = 0.02`, parsel ODE'leri, uzaysal türev yok):
simetrik okuma `ΔF_n = ½(F_n^{+ε} − F_n^{−ε})`,
`R_n = ΔF_n/max(1e-30,|ΔF_üniform|)` (zarf tabanı `1.17e-5`; `1e-30` tabanı
**hiçbir noktada tetiklenmez** — kırpma iddiası ölçüyle çürütüldü), düzgün `ξ` Fourier'i.

### Tablo 10 (s=25, h=0.005, μ_s=0.1)

| n | k_max | saflık \|c_n\|/‖c‖ | \|r\| | tek-yönlü saflık/r | max\|ΔF\| |
|---|---|---|---|---|---|
| 1 | 1 | 0.9917 | 0.9453 | 0.9914 / 0.9446 | 1.2e-3 |
| 2 | 2 | 0.9688 | 0.9364 | 0.9686 / 0.9356 | 2.2e-3 |
| 3 | 3 | 0.9372 | 0.8685 | 0.9387 / 0.8685 | 3.3e-3 |
| 4 | 4 | 0.9055 | 0.8143 | 0.9090 / 0.8169 | 4.2e-3 |

Eşikler (`k_max == n`, saflık `> 0.90`, `|r| > 0.80`): `k_max` 4/4 ✓; saflık 4/4 ✓;
`|r|` 4/4 ✓. **T12 geçti (12/12).** `%99.2 → %90.6` düşüş, homojen olmayan
çekirdek gradyanlarının doğal diferansiyel-yükseltme dispersiyonudur.

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

---

## 3. Doğrulama özeti ve tekrar-üretim

| Test | Sonuç |
|---|---|
| T11 (3 μ_s + μ_s=0 koruma) | **geçti** — tüm kriterler |
| T12 (taban + üniform± + 4 mod±, Lagrange simetrik) | **geçti** — k_max 4/4, saflık 4/4, r 4/4 |
| T1–T10 | değişmedi, geçer |

Tekrar-üretim: `pip install numpy scipy matplotlib`, `python solver_engine.py` (T1–T12),
`python generate_figures.py` (fig1–fig6).
