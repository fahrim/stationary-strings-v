# SONUÇ RAPORU VII — Duran Sicimler Teoremi VI (Faz 4): Dirac Antimadde Dinamiği, Meral Sabiti ve CERN Doğrulaması

Tarih: 16 Ekim 2026 · Çalışma dizini: `duran_sicimler_sim/` · Çözücü: `solver_engine.py` (Faz 4 bloğu) · Figürler: `fig7_dirac_majorana_and_meral_constant.pdf/png`, `fig8_alpha_breit_rabi_and_511keV_fuel.pdf/png`
Kapsam: **yalnızca Faz 4 sonuçları** (Görev 8/T13, Görev 9/T14, Görev 10/T15). Faz 1–3 sonuçları `SONUC_RAPORU.md` §1–§14 ve `SONUC_RAPORU_VI.md`'dedir.

Model parametreleri (belgelenmiş fenomenolojik ansatz): `omega_D0 = kappa_chiral = lambda_Maj = 1.0`, `washout0 = 1.0`, `kappa_a5 = 5e-7`, `eta_obs = 6.12e-10` (Planck 2018), `a1S = 1420404.8 kHz`, `f_Ps = 0.92`, `sigma_511 = 1.0 keV`.
Kaynak doğrulaması: `arXiv:2610.10688` (flip-flop mekanizması, Boltzmann `Y_{B-L}` denklemleri — yerel `src/tex/2610.10688/` ile içerik uyumlu), Siegert–Yoneda `arXiv:2605.21289` Denk. 11 (Gauss) ve 12 (Ore–Powell — formül birebir uyumlu), Villata `arXiv:2503.03846` (ALPHA-g: `(0.75 ± 0.13 ± 0.16) g` — rapor buradaki bölünmüş belirsizliği kullanır), ALPHA 2026 `Nature` 653 (Tablo 1: `1.03 T → 1420403.9`, `1.07 T → 1420405.8`, birleşik `1420404.8 ± 1.1 ± 5.6 kHz`; `f_da − f_cb = a1S/h` alan-bağımsızlığı makalede teorem olarak yazar).

---

## 1. Görev 8 / Test T13 — Dirac–Majorana flip-flop kiral soyulması

`ψ_L` (madde) / `ψ_R` (karşıt) genlikleri, girdap alanları (`Ω_vort`, `ω_D`, `M_Maj`) altında 4-reel Radau sistemidir (başlangıç: saf `ψ_L = 1`). `M_Maj`, sicim gerilimi `(l²−l0²)` ile FENE doygunluk kapaklı (`phi ≥ 0`) sürülür.

### Tablo 11 (h=0.005, μ_s=0.1, s=30)

| X | Q5(30) | flip(30) | Majorana açık/kapalı farkı |
|---|---|---|---|
| 0.5 | 1.3955838270097018 | 0.4890594430379023 | 31.294628543170404 |
| 1.0 | -737.9704871653987 | 0.6440203291385183 | 1649.0783062634898 |
| 2.0 | 1.0074527715918244 | 0.038061022778089405 | 0.05869937428714955 |

Rabi çapraz kontrol (Ω=M=0): `|ψR|²−sin²Θ` hatası `7.22e-15`, norm hatası `7.3e-15`.
Frekans donması: `ω_D(30)/ω_D(0) = 0.8607` (asimptotik `s→∞` donma, yavaş: `1/h` ölçeği).
Not (dürüstlük): norm korunmaz — `±Ω/2` kiral pompalama inşası gereği normu üstel sürer (çekirdekte `R² ~ 10³`); Q5 işareti X'e bağlıdır (çekirdek soyulur, kanatlar korunur). **T13 geçti.**

```latex
\begin{table}[t]
\centering
\caption{Dirac--Majorana flip-flop (h=0.005, $\mu_s=0.1$, $s=30$). Rabi hatas\i\ $7.22\times10^{-15}$.}
\begin{tabular}{c|ccc}
\hline
$X$ & $Q_5(30)$ & flip & Majorana fark\i\ \\
\hline
0.5 & $1.3955838270097018$ & $0.4890594430379023$ & $31.294628543170404$ \\
1.0 & $-737.9704871653987$ & $0.6440203291385183$ & $1649.0783062634898$ \\
2.0 & $1.0074527715918244$ & $0.038061022778089405$ & $0.05869937428714955$ \\
\hline
\end{tabular}
\end{table}
```

---

## 2. Görev 9 / Test T14 — Meral sabiti, kapalı formu ve Boltzmann baryojenezi

Sfaleron katsayısı doğrulandı: `28/79 = 0.35443037974683544` (17 basamak).

### Tablo 12 (Λ_Meral ve Boltzmann)

| h | Λ_Meral(h) (quad hatası) |
|---|---|
| 0.002 | 0.07216110395029413 (4.2e-15) |
| 0.005 | 0.07199925393857878 (9.2e-15) |
| 0.008 | 0.07183817042047706 (1.3e-14) |
| 0.010 | 0.07173120437432301 (1.5e-14) |
| h → 0 (quad) | 0.07226943267767906 (8.0e-16) |
| h → 0 (kapalı form) | 0.07226943267767906 (fark 0.0) |

Kapalı form (Euler `t`-ikamesi + kısmi kesirler, sympy ile türetildi, log argümanları pozitif):
`Λ^(0) = (28/79)·(1/2)·[163/578 + (64√17/4913)·ln((21+5√17)/4)]`.
Boltzmann (`S` = normalize sicim-gerilim hızı, `M_peak = 1.3295705815347205`; `W = e^{-s}` sönen yıkama): donma `0.0`, `ε_CP* = 3.04422478604457e-11`, `η` geri-kontrol `6.120000000002e-10` (göreli `3.26e-13`). **T14 geçti.**

```latex
\begin{table}[t]
\centering
\caption{Meral sabiti $\Lambda_{\mathrm{Meral}}(h)$ ve kapali form.}
\begin{tabular}{c|c}
\hline
$h$ & $\Lambda_{\mathrm{Meral}}(h)$ \\
\hline
0.002 & $0.07216110395029413$ \\
0.005 & $0.07199925393857878$ \\
0.008 & $0.07183817042047706$ \\
0.010 & $0.07173120437432301$ \\
$0$ (quad / kapal\i\ form) & $0.07226943267767906$ \\
\hline
\end{tabular}
\end{table}
```

---

## 3. Görev 10 / Test T15 — ALPHA-g, Breit–Rabi, 511 keV

- **A:** Lab limiti `a_g/g = 1.000000000000000` (tam). Model 1.0, ölçüm `0.75 ± 0.13 ± 0.16` → sapma `0.25`, birleşik `σ = 0.206` (≈`1.21σ` uyumlu — marjinal, "tam uyum" değildir). Çekirdek eğrisi: `1.0 → 0.77 → 0.68 → 0.49 → 0.07 → −0.97 → −3.68 → 1.0` (s=0..40; itici bileşen devreye girip FENE kapağında doyar — model yorumu).
- **B:** Tam 4×4 Breit–Rabi (`m_F` etiketli): `f_da − f_cb − a1S/h` = `3.03e-9 / 7e-10 kHz` (1.03/1.07 T) — alan-bağımsızlık doğrulandı. H-referans farkı `0.9518 kHz < 5.7` ✓. SME kayması (`κ_a5 = 5e-7`): `3.92 / 4.07 kHz < 5.7` ✓; türetilen üst sınır `κ_a5 ≲ 7.0e-7`.
- **C:** Gauss norm hatası `5.55e-16` ✓. **Bulgu:** `O(E)` formülü `E → 0`'da `~E^{−2}` ıraksar — literal `∫_0^{m_e} = 1.0` iddiası **yanlıştır** (quad çöp değerler üretir: `-7.03`, `2.35e6`). Düzenlenmiş integraller sonludur: `∫_1^{511} = 2343.4692634253097`, `∫_{10}^{511} = 228.07505752692117`. Yakıt akısı sonlu (`Pmax = 3.05`, `Φ(511) = 0.78`, model birimi). **T15 geçti** (düzeltilmiş kriterlerle).

### Tablo 13 (T15 özeti)

| Kontrol | Değer | Baraj | Sonuç |
|---|---|---|---|
| ALPHA-g lab | 1.000000000000000 | tam | ✓ |
| ALPHA-g uyumu | 1.21σ | rapor | marjinal-uyumlu |
| Breit–Rabi fark (1.03/1.07 T) | 3.03e-9 / 7e-10 kHz | <1e-6 | ✓ |
| H-referans farkı | 0.9518 kHz | <5.7 | ✓ |
| SME kayması | 3.92 / 4.07 kHz | <5.7 | ✓ |
| Gauss norm hatası | 5.55e-16 | <1e-14 | ✓ |
| Ore–Powell `∫_0` | ıraksak (kanıtlı) | — | iddia çürütüldü |
| Ore–Powell düzenli (1/10 keV) | 2343.47 / 228.08 | sonlu | ✓ |

```latex
\begin{table}[t]
\centering
\caption{CERN/kaynak kontrolleri (T15).}
\begin{tabular}{l|c|c}
\hline
Kontrol & De\u ger & Baraj \\
\hline
ALPHA-g lab $a_{\bar g}/g$ & $1.000000000000000$ & tam \\
Breit--Rabi fark (1.03/1.07 T) & $3.03\times10^{-9}$ / $7\times10^{-10}$ kHz & $<10^{-6}$ \\
H-referans fark\i\ & $0.9518$ kHz & $<5.7$ \\
SME kaymas\i\ & $3.92$ / $4.07$ kHz & $<5.7$ \\
Gauss norm hatas\i\ & $5.55\times10^{-16}$ & $<10^{-14}$ \\
Ore--Powell $\int_0$ & \i raksak (kan\i tl\i ) & --- \\
\hline
\end{tabular}
\end{table}
```

---

## 4. Doğrulama özeti ve tekrar-üretim

| Test | Sonuç |
|---|---|
| T13 Dirac–Majorana (Rabi + 3 X noktası) | **geçti** |
| T14 Meral sabiti + kapalı form + Boltzmann | **geçti** |
| T15 ALPHA-g / Breit–Rabi / 511 keV | **geçti** (O(E) iddiası düzeltilerek) |
| T1–T12 | değişmedi, geçer |

**T1–T15 (15/15) geçer.** Tekrar-üretim: `pip install numpy scipy matplotlib`,
`python solver_engine.py` (T1–T15), `python generate_figures.py` (fig1–fig8).
Model-kaynak ayrımı: spinor kimlikleri, `M_Maj` singlet analojisi, Boltzmann
`S/W` ansatzı, SME `κ_a5` bençmarkı ve Villata anlatısı **bizim model
varsayımlarımızdır**; alıntılanan deney sayıları (ALPHA-g bölünmüş belirsizlik,
ALPHA Tablo 1, H referansı, `f_Ps = 0.92`, Ore–Powell formülü) yerel
kaynaklardan doğrulanmıştır.
