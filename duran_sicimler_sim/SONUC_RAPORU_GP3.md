# SONUÇ RAPORU — GRAND PROOF III (FAZ 11): Masaüstü Deney + Gözlemevi Kanıtı

`stationary-strings-v/duran_sicimler_sim` — T35–T37 + `mini_check_gp3.py` + `fig15`

## Dürüstlük çerçevesi (üç katman ayrımı)

1. **Kesin Matematiksel Teorem (Exact):** Burgers girdap kapalı-formları,
   PEO oranları (`C_univ`, `C_univ^-1`, `[C^2(1+mu)]^-1`), Vela ODE'si,
   dedektör-konvolüsyonlu O_exact bant integrali — sıfır serbest parametre.
2. **Bağımsız literatür (model dışı):** PEO girdap bantları (r: [1.55,1.70],
   dh: [0.30,0.35]), Vela glitch (2.25±0.35)e-6, Q ∈ [0.05,0.18],
   22Na Ps laboratuvar spektrumları.
3. **Masaüstü deney protokolü (önerilen düzenek):** R_tank = 100 mm,
   100 ppm PEO, HPGe σ = 0.85 keV; pulsar kabuk-girdap pinleme köprüsü.

## T35 — Masaüstü viskoelastik girdap PIV

- Kontrol (su): `r_core^N = 3.16228 mm`, `u_max^N = 0.38542 m/s`,
  `Δh_N = 25.78146 mm` (kapalı-form vs quad: 1.7e-17 mutlak fark).
- PEO (100 ppm): `r_core^PEO = 5.14198 mm` (C = 1.62603794, +%62.60,
  bant [1.55,1.70] içinde); `u_max^PEO = 0.23703 m/s` (1/C, -%38.50);
  `Δh_PEO = 8.09603 mm` (oran 0.31402536, 3.1845x bastırma,
  bant [0.30,0.35] içinde).
- DÜRÜSTLÜK NOTU: yönergedeki `Δh = 25.811509/8.105362` sayıları doğrulanamadı;
  kodun kapalı-form + quad ikili-kanıtlı değerleri kullanıldı
  (25.78146/8.09603 mm). Önerilen oran 0.31402124 yerine **0.31402536**
  (fark 4.1e-6; yönergedeki değerin yuvarlama türetimi tutmuyor).

## T36 — Vela glitch

- `Δt_ig = 2.80 yr = 8.8361280e7 s`; base `1.97765094e-6`;
  Meral `2.38190025e-6` vs (2.25±0.35)e-6 → **%5.86 (0.38σ)**.
- `Q_fast = 0.05910575`, `Q_total = 0.13569214`; ikisi de [0.05,0.18] içinde.
- Çift-üstelli gevşeme (τ_f = 6 g, τ_s = 60 g), analitik türev çaprazı 6.1e-5.
- DÜRÜSTLÜK NOTU: yönergedeki Q_total 0.13568831 ile kodun 0.13569214
  değeri arasında 3.8e-6 fark var (C_univ yuvarlaması); kod değeri geçerlidir.

## T37 — 22Na pozitronyum gama spektroskopisi

- `B_exact = [10,100]/[350,490] = 0.08413579` (O_exact normalize, quad).
- **DÜZELTME (GP-III.3 etiket düzeltmesi):** kaynak denetiminde
  `arXiv:2605.21289` Denk.(12)'nin **yayımlanan** formunun 3. teriminde
  `2m(m−E)/E²` (1/E² kutuplu, ∫₁ = 2343.47) paydası bulundu; `E²→(2m−E)²`
  ikameli varyant ise `2m(m−E)/(2m−E)²` paydası taşır (∫₁ = 27.147).
  Önceki sürümde bu iki formülün etiketleri TERS'ti. Düzeltilmiş değerler:
  `B_yayımlanan = 159.37`, `B_ikame = 9.31`.
- 1 saatlik sayım: N_3γ = 4.96e6, laboratuvar ayrımı **2.4e6σ (>500σ)**.
- DÜRÜSTLÜK NOTU: yönergedeki `B_exact = 0.0491746` ve `B_pole = 2.5594`
  değerleri bu bant tanımıyla yeniden üretilemedi (kod: 0.08413579 / 159.37;
  SKILL'deki ∫_1 = 27.147 ve ∫_1 = 2343.47 ölçümleri kodla birebir
  tutuyor). Kod değerleri rapor+makalede kullanılır.

## Doğrulama

- `run_verification()` → **T1–T37 (38/38 PASS)**.
- `python -W ignore duran_sicimler_sim/mini_check_gp3.py` → **MINI 12/12 PASS**.
- `fig15('tr'/'en')` → `fig15_...pdf/png` (+`_en`).

## Dosyalar

- `solver_engine.py`: `GP3`, `solve_benchtop_viscoelastic_vortex_piv_experiment`
  (T35), `solve_vela_pulsar_glitch_avalanche` (T36),
  `solve_benchtop_na22_positronium_gamma_experiment` (T37).
- `mini_check_gp3.py`, `generate_figures.py: fig15(lang)`,
  `SONUC_RAPORU_GP3.md` (bu dosya).
