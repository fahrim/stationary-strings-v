# SONUÇ RAPORU — GRAND PROOF I (FAZ 9): Karanlık-Maddesiz Galaktik Dinamik

`stationary-strings-v/duran_sicimler_sim` — T29–T31 + `mini_check_gp1.py` + `fig13`

## Dürüstlük çerçevesi (üç katman ayrımı)

1. **Kesin Matematiksel Teorem (Exact):** `a_Meral`, RAR enterpolasyonu,
   `v_flat^4 = G M_b a_Meral`, `d_ring = 6 θ_g C_univ √(1+μ_s/4)` —
   sıfır serbest parametre, yalnızca Teorem I–X kilit sabitleri.
2. **Gözlemsel literatür (model dışı):** Planck 2018 `H_0`,
   McGaugh–Lelli–Schombert 2016 (SPARC RAR/BTFR), EHT M87* (2019) ve Sgr A* (2022).
3. **Kozmolojik köprü modeli (sınamaya açık yorum):** vakum akı-sicimlerinin
   Hubble ufku boyunca gerilip galaktik düzleme elastik çember ivmesi aktarması;
   üstel-disk baryon dağılımı; halka genişliği `~ μ_s` ansatzı.

## T29 — Meral kozmik ivme ölçeği ve BTFR

- `a_H = c H_0 = 6.548322413981453e-10 m/s²` (Planck 2018 `H_0 = 67.4 km/s/Mpc`).
- `κ_Meral = μ_s^Meral (1-η_T0) = 0.17509659836048996`.
- `a_Meral = 1.1465889796559045e-10 m/s²`.
- Eksenel `√3/2` projeksiyonu: `1.1592052113953912e-10 m/s²`;
  SH0ES/JWST `H_0 = 70.5` ile `1.199325268037704e-10 m/s²` (gözlemsel `a_0` ile örtüşür).
- Gözlemsel `a_0^SPARC = (1.20 ± 0.02 ± 0.24)e-10`: fark **%4.45 = 0.22σ_sys** (tam uyum).
- BTFR: `v_flat^4 = G M_b a_Meral`, log-log eğim **4.000000000000000**, `R² = 1.0`.

## T30 — SPARC dönüş eğrileri (karanlık madde yok)

| Galaksi | M_b (M☉) | R_d (kpc) | v_flat^Meral (km/s) | v_flat^obs (km/s) | fark |
|---|---|---|---|---|---|
| DDO 154 | 3.5e8 | 0.8 | 48.04 | 47.0 ± 2.0 | %2.2 |
| NGC 3109 | 1.55e9 | 1.6 | 69.69 | 67.0 ± 3.0 | %4.0 |
| NGC 2403 | 2.10e10 | 2.1 | 133.70 | 134.0 ± 4.0 | %0.2 |
| NGC 3198 | 3.80e10 | 3.2 | 155.07 | 150.0 ± 4.0 | %3.4 |
| UGC 2885 | 2.50e11 | 11.5 | 248.35 | 285.0 ± 10.0 | **%12.9 dış-değer** |

- Ortalama mutlak bağıl fark: **%4.54** (4/5 galakside <%4.1; dev sarmal UGC 2885
  modelin altında kalır — bulge/disk `M/L` sistematiği ya da ek fizik gerektirir; gizlenmedi).
- Saf Newton eğrisi `r = 10 R_d`'de tepe değerinin %57.9'una düşer (Kepler çöküşü, gözlemle çelişir);
  Meral eğrisi düzleşir (`|v_M(30)-v_M(20)|/v_flat = %3.4`, NGC 3198).
- Not: sonlu yarıçapta (`r ~ 1.4 r_M`) Newton kalıntısı ~%9 yukarı iter;
  `v_flat` asimptotik (`r → ∞`) BTFR büyüklüğüdür — Şekil 1(b) bunu açıkça gösterir.

## T31 — EHT halka doygunluğu

- `d_ring/θ_g = 6 × 1.62603794 × √(1+μ_s/4) = 10.0024041396` (≈ 10.00, GRMHD `α_1 ≈ 10.0–10.4` ile uyumlu).
- Sgr A*: **50.31 μas** vs `51.8 ± 2.3` → **0.65σ**.
- M87*: **38.31 μas** vs `42.0 ± 3.0` → **1.23σ**
  (tam `√(1+β_0²)` projeksiyonuyla 41.78 μas → 0.07σ; iki projeksiyon da raporlandı).
- Halka kesirli genişliği `FWHM/d ≈ μ_s = 0.2044 < 0.25` (EHT sınırı sağlanır).

## Doğrulama

- `python -W ignore -c "import sys; sys.path.insert(0,'duran_sicimler_sim'); from solver_engine import run_verification; run_verification()"` → **T1–T31 (31/31 PASS)**.
- `python -W ignore duran_sicimler_sim/mini_check_gp1.py` → **MINI 15/15 PASS** (solver import etmez).
- `python -W ignore -c "import sys; sys.path.insert(0,'duran_sicimler_sim'); from generate_figures import fig13; fig13('tr'); fig13('en')"` → `fig13_...pdf/png` (+`_en`).
- Üstel-disk + RAR profili ve Gauss-halka şeması fiziksel model ansatzıdır (kanıt değil, gösterim).

## Dosyalar

- `solver_engine.py`: `GP1`, `meral_cosmic_acceleration`, `rar_g_obs`,
  `solve_meral_cosmological_acceleration_and_btfr` (T29),
  `solve_sparc_galaxy_rotation_curves` (T30),
  `solve_eht_black_hole_ring_saturation` (T31); `run_verification()` T29–T31.
- `mini_check_gp1.py`: bağımsız 15 nokta (C1–C6).
- `generate_figures.py`: `fig13(lang)` 2×2 (RAR / dönüş eğrileri / BTFR / EHT halkası).
- `SONUC_RAPORU_GP1.md`: bu dosya.
