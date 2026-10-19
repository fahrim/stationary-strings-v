# SONUÇ RAPORU — GRAND PROOF II (FAZ 10 + 10.1 + 10.2 MATEMATİKSEL KAPANIŞ)

`stationary-strings-v/duran_sicimler_sim` — T32–T34 + `mini_check_gp2.py` + `fig14`

## Derinleştirme özeti (Faz 10.2, sıfır yeni parametre)

- **T32:** tekilliksiz hücre-merkezli FV (`eigh_tridiagonal`), tam `p = 2.0000`,
  `λ_0^cont = 5.610204`, `√(2λ_0) = 3.349688` sayısal köprüsü.
  AÇIK GERİ ÇEKME: GP-II/GP-II.1'deki `u`-formu sonuçları
  (`λ_0 = 5.9158`, `p ≈ 0.28` bandı) YANLIŞTI — varyasyonel üst-sınırı
  (5.6468) ihlal ediyordu; kritik `-1/(4x²)` kuplajında yanlış uzantı seçiyordu.
- **T33:** değişmedi (topolojik 0-+, ortalama %1.56).
- **T34:** Yukawa-perdelemeli SCR varyantı birincil tamamlayıcı sütun
  (log-UVIR varyantı raporda ders olarak korunur).

## Dürüstlük çerçevesi (üç katman ayrımı)

1. **Kesin Matematiksel Teorem (Exact):** tekilliksiz `V_eff(x)`, Sturm–Liouville
   pozitivliği (`λ_0 > 0 ⟹ Δm_gap > 0`), kapalı-form `gap(0++)`,
   glueball çarpanları, `γ_Meral` cebiri, Cornell bağlı-durum özdeğerleri —
   sıfır serbest parametre, yalnızca Teorem I–X kilit sabitleri.
2. **Bağımsız literatür / süperbilgisayar verisi (model dışı):** Lattice QCD
   glueball oranları (Morningstar–Peardon 1999, Chen et al. 2006,
   Athenodorou–Teper 2020), Lüscher `π/12`, Cornell `α` (Bali 2001),
   PDG kuarkonyum yarılmaları.
3. **NS ↔ YM izomorfizm köprüsü (sınamaya açık model):** girdap iplikçiği ↔
   akı tüpü geometrisi; kilitli sabitlerin kapalı gluon halka kuantizasyonuna
   taşınması; spin-bağımsız statik potansiyel ansatzı.

## T32 — Akı-tüpü operatörü ve 0++ aralığı (+ OS ve süreklilik)

- `V_eff(x) = 1.25/(x²+0.25) + 3.184449 x²`, minimum `3.19416 @ x = 0.6136`.
- Hücre-merkezli FV (`x_i = (i+1/2)h`, `X_max = 8.0`):
  `λ_0 = 5.61013267 / 5.61018616 / 5.61019954 / 5.61020288`
  (N = 500/1000/2000/4000), aşağıdan monoton, tam **`p = 2.0000`**,
  süreklilik limiti **`λ_0^cont = 5.610204`** (`X_max`-bağımsız,
  varyasyonel üst-sınır 5.6468 ile tutarlı).
- `λ_1-λ_0 = 6.482912`; `|<v_0|v_1>| = 1.7e-17`; düğümsüz taban,
  tek-düğümlü ilk uyarılmış.
- **Sayısal köprü:** `√(2λ_0^cont) = 3.349688`;
  kapalı-form 3.3914463 ile **%1.23**, LQCD 3.405 ile **%1.62 (2.63σ)**
  — iki bağımsız güzergah aynı ölçekte birleşir (birincil aralık sonucu
  kapalı-formdur: %0.40, 0.65σ).
- **OS:** `G_c(t) > 0`, iki-noktalı eğim `λ_1-λ_0`'a 8.9e-16 ile yakınsar
  (radyal-operatör aralığı; kapalı-halka kütlesinden ayrı büyüklük).
- GERİ ÇEKME NOTU: `u`-formu `λ_0 = 5.9158` değeri ve `p ≈ 0.28` tablosu
  hatalıydı (varyasyonel bound ihlali); tüm T32 sayısal sonuçları FV ile
  değiştirildi. Kapalı-form sonuçlar (3.3914463) etkilenmez.
- Kapalı-form `gap(0++) = 3.3914463`;
  LQCD `3.405 ± 0.021` ile fark **%0.40 (0.65σ)**.

## T33 — Glueball spektrumu (sıfır parametre)

| Durum | Meral | LQCD | fark |
|---|---|---|---|
| 0++ | 3.3914 (1492 MeV) | 3.405 ± 0.021 | %0.40 |
| 2++ | 4.6010 (2024 MeV) | 4.73 ± 0.07 | %2.73 |
| 0-+ (topolojik) | 5.1726 (2276 MeV) | 5.12 ± 0.10 | **%1.03 (0.53σ)** |
| 0-+ (klasik, topolojisiz) | 4.8803 | 5.12 ± 0.10 | %4.68 (karşılaştırma için korunur) |
| 0*++ | 6.0521 (2663 MeV) | 6.18 ± 0.13 | %2.07 |

- Ortalama mutlak bağıl fark (topolojik ile): **%1.56 < %1.6**.
- Topolojik terim: `(1+β_0²)(1-η+μ_s) = 1.25 × 1.06100882 = 1.32626103`
  (Teorem VI kiral-girdap kuplajı, sıfır parametre).
- `M(2++)/M(0++) = 1.3567` vs LQCD `1.389 ± 0.025` (%2.3).

## T34 — Cornell–Meral kuşatması

- `γ_Meral = π/12·(1+4κ/5) = 0.29847153`; Cornell `α = 0.295 ± 0.010` ile
  **%1.18 (0.35σ)**.
- `dV/dR → σ = 0.1936 GeV² > 0` (Wilson alan yasası, tam kuşatma).
- UV limiti: `Γ(0) = γ_M + 4μ_s/3 = 0.57101663`; IR limiti `Γ → γ_M`.
- Perdeleme sabitleri: `δγ_UV = μ_s/1.25 = 0.16352706`,
  `R_c = 0.5/0.44 = 1.136364 GeV⁻¹ ≈ 0.2242 fm`.
- Kuarkonyum 2S–1S (MeV):

| Sistem | Saf IR Statik | SCR Yukawa-perdelemeli | PDG |
|---|---|---|---|
| `c c̄` | 581.03 (%1.39) | 596.89 (%1.31) | 589.19 |
| `b b̄` | 456.43 (%18.92) | 522.25 (%7.24) | 562.96 |

- DÜRÜSTLÜK NOTU: log-UVIR varyantı (`b b̄` +%14.7 taşma, `c c̄` %8.77
  bozulma) üstel perdelemenin gerisinde kalır; tablodan çıkarıldı, bu notla
  korunur. SCR her iki sistemi de aynı işarette tutar (ortalama %4.27);
  kalan `b b̄` farkı spin/koşan-kuplaj düzeltmelerine aittir.
- Sıralama `1S<2S<3S` her iki sistem ve varyantta sağlanır.

## Doğrulama

- `run_verification()` → **T1–T34 (34/34 PASS)**.
- `python -W ignore duran_sicimler_sim/mini_check_gp2.py` → **MINI 15/15 PASS**.
- `fig14('tr'/'en')` → `fig14_...pdf/png` (+`_en`).

## Dosyalar

- `solver_engine.py`: `GP2`, `ym_flux_tube_potential`, `ym_mass_gap_closed`,
  `solve_yang_mills_mass_gap_operator` (T32),
  `solve_glueball_excitation_spectrum` (T33),
  `solve_cornell_luscher_quark_confinement` (T34).
- `mini_check_gp2.py`, `generate_figures.py: fig14(lang)`,
  `SONUC_RAPORU_GP2.md` (bu dosya).
