# SONUÇ RAPORU — GRAND PROOF II (FAZ 10 + 10.1 DERİNLEŞTİRME)

`stationary-strings-v/duran_sicimler_sim` — T32–T34 + `mini_check_gp2.py` + `fig14`

## Derinleştirme özeti (Faz 10.1, sıfır yeni parametre)

- **T32:** OS yansıma pozitifliği + transfer korelasyonu `G_c(t)`,
  ilk 3 özdeğer + ortonormallik, ölçülen-mertebeli Richardson süreklilik limiti.
- **T33:** topolojik Pontryagin/kiral-anomali `0-+` (klasik değer de korunur).
- **T34:** UV–IR Casimir–Meral varyant sütunu (saf-IR birincil kalır).

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

- `V_eff(x) = 1.25/(x²+0.25) + 3.184449 x²`, minimum `3.19416 @ x = 0.6136`
  (tekilliksiz; klasik `-(\pi/12)/x²` aşağıdan sınırsızken).
- Simetrik `u`-formu sonlu-fark: `λ_0 = 5.9158 < λ_1 = 12.5394 < λ_2 = 19.4291`
  (N=2000), `|<u_0|u_1>| = 3.4e-18`, düğümsüz taban, tek düğümlü ilk uyarılmış.
- **OS yansıma pozitifliği:** `G_c(t) > 0` (tüm t), iki-noktalı eğim
  `-(ln G(15)-ln G(12))/3 = λ_1-λ_0 = 6.6237` (1.8e-15 hassasiyet).
  DÜRÜSTLÜK NOTU: korelasyon eğimi **radyal-operatör aralığına**
  (`λ_1-λ_0`) yakınsar; kapalı-halka kütlesi (3.3914) ayrı spektral
  büyüklüktür — ikisi de `> 0`, ikisi de sıfır-parametrelidir.
- **Richardson:** `λ_0` = 5.986051 / 5.947392 / 5.915765 / 5.889499
  (N = 500/1000/2000/4000), monoton azalan, süreklilik bandı `~5.77`.
  DÜRÜSTLÜK NOTU: ölçülen mertebe **p ≈ 0.28, O(h²) DEĞİL**
  (orijin `√x` cusp'u); limit yine sonlu ve sıfırdan kesin uzaktır.
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
- Kuarkonyum 2S–1S (MeV):

| Sistem | Saf IR Statik | UV–IR Casimir–Meral | PDG |
|---|---|---|---|
| `c c̄` | 581.03 (%1.39) | 640.89 (%8.77) | 589.19 |
| `b b̄` | 456.43 (%18.92) | 645.80 (%14.71) | 562.96 |

- DÜRÜSTLÜK NOTU: log-düzenleyici ara-ölçekte yavaş koşar
  (`Γ(30 GeV⁻¹) = 0.340`, hâlâ `γ_M` üstünde); bu yüzden UV–IR varyantı
  `b b̄` hatasının işaretini çevirir (-%18.9 → +%14.7) ama `c c̄`'yi bozar.
  **Birincil sonuç saf-IR kalır**; hızlı sönen (kompakt-destekli) UV
  tamamlaması açık araştırmadır.
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
