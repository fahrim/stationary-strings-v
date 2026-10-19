# SONUÇ RAPORU — GRAND PROOF III (FAZ 11): Masaüstü Deney + Gözlemevi Kanıtı

`stationary-strings-v/duran_sicimler_sim` — T35–T37 + `mini_check_gp3.py` + `fig15`

## Dürüstlük çerçevesi (üç katman ayrımı)

1. **Kesin Matematiksel Teorem (Exact):** Burgers girdap kapalı-formları,
   PEO oranları (`C_univ`, `C_univ^-1`, `[C^2(1+mu)]^-1`), Vela ODE'si,
   dedektör-konvolüsyonlu O_exact bant integrali — sıfır serbest parametre.
2. **Bağımsız literatür (model dışı):** PEO girdap deneyi (Groisman–Steinberg
   2000: akış direnci ~20×, elastik gerilim 2 büyüklük mertebesi artışı);
   Vela glitch (Dodson+2002/2007: 2000 olayı 3.0858e-6, 2004 olayı 2.08e-6,
   ölçülen $Q_{2000}=0.02198$, gevşeme 1.2 dk/0.53/3.29/19.07 gün);
   Link+1999: $\Delta I/I \ge 0.014$; $^{22}$Na Ps spektrumları.
3. **Masaüstü deney protokolü (önerilen düzenek):** R_tank = 100 mm,
   100 ppm PEO, HPGe σ = 0.85 keV; pulsar kabuk-girdap pinleme köprüsü.

## T35 — Masaüstü viskoelastik girdap PIV

- Kontrol (su): `r_core^N = 3.16228 mm`, `u_max^N = 0.38542 m/s`,
  `Δh_N = 25.78146 mm` (kapalı-form vs quad: 1.7e-17 mutlak fark).
- PEO (100 ppm): `r_core^PEO = 5.14198 mm` (C = 1.62603794, +%62.60);
  `u_max^PEO = 0.23703 m/s` (1/C, -%38.50);
  `Δh_PEO = 8.09603 mm` (oran 0.31402536, 3.1845x bastırma).
- **KAYNAK DENETİMİ (19 Ekim 2026):** Groisman & Steinberg (Nature 405, 53;
  arXiv:nlin/0104052) indirildi ve tam metin/LaTeX taranarak doğrulandı:
  - Deney düzeni: iki paralel disk arasında swirling akış, **PAAm + viskoz şurup**
    (100 ppm PEO sulu çözelti DEĞİL), $d/R = 0.263$.
  - Gerçekte ölçülenler: **akış direnci ~20 kat artar**, **elastik gerilimler
    iki büyüklük mertebesi artar**.
  - `$r_ve/r_N ∈ [1.55,1.70]`, `Δh_ve/Δh_N ∈ [0.30,0.35]`, "funnel", "dimple",
    "vortex core" kavramları **bu kaynakta hiç geçmiyor** → bantlar
    **doğrulanamaz**; "bandın merkezinde uyum" iddiası **geri çekildi**,
    T35 barajı bandı artık kullanmaz.
  - Kaynak deposu: `src/08_groisman_steinberg_elastic_turbulence_Nature2000.pdf`,
    `src/tex/nlin_0104052/`.

## T36 — Vela glitch (KAYNAK DOĞRULAMALI)

Kaynak denetimi (19 Ekim 2026): Dodson+2002 (ApJL 578, L57; astro-ph/0201005)
Tablo 1, Dodson+2007 (Ap&SS 308, 585; astro-ph/0612371), Link+Epstein+Lattimer
1999 (PRL 83, 3362; astro-ph/9909146) indirildi ve taranarak doğrulandı.

- Girdiler (kaynaktan): $\nu_0 = 11.194615396005$ Hz,
  $|\dot\nu_0| = 1.55615\times10^{-11}$ Hz/s, $\Delta I/I \ge \mathbf{0.0140}$,
  $\Delta t_{ig} = 2.80$ yıl.
- Model: base `1.71961931e-6`; Meral `2.07112467e-6`.
- Karşılaştırma: **2004 olayı $2.08\times10^{-6}$ → %0.43 uyum**;
  **2000 olayı (en büyük) $3.0858\times10^{-6}$ → %32.9 sapma**.
- İyileşme: model $Q_{\mathrm{fast}} = 0.05910575$,
  $Q_{\mathrm{total}} = 0.13569214$; **ölçülen** $Q_{2000} = 0.02198$
  (dönen terimler 0.020/0.31/0.193/0.2362 ×10⁻⁶ Hz / 3.45435×10⁻⁵ Hz).
- Gevşeme zaman ölçekleri **kaynaktan**: 1.2±0.2 dk, 0.53(3), 3.29(3),
  19.07(2) gün (keyfi 6/60 gün değerleri yerine).
- DÜRÜSTLÜK NOTLARI: (i) görev metnindeki $I_{\mathrm{crust}}/I = 0.0160$
  **düzeltildi** (kaynak ≥0.014); (ii) $(2.25\pm0.35)\times10^{-6}$ bandı
  **doğrulanamadı** → iki bireysel olay kullanıldı; (iii) $Q\in[0.05,0.18]$
  bandı **doğrulanamadı** → ölçülen 0.02198 raporlandı; modelin $Q$ değerleri
  bu tek ölçümle doğrudan karşılaştırılamaz.

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
- `python -W ignore duran_sicimler_sim/mini_check_gp3.py` → **MINI 13/13 PASS**.
- `fig15('tr'/'en')` → `fig15_...pdf/png` (+`_en`).

## Dosyalar

- `solver_engine.py`: `GP3`, `solve_benchtop_viscoelastic_vortex_piv_experiment`
  (T35), `solve_vela_pulsar_glitch_avalanche` (T36),
  `solve_benchtop_na22_positronium_gamma_experiment` (T37).
- `mini_check_gp3.py`, `generate_figures.py: fig15(lang)`,
  `SONUC_RAPORU_GP3.md` (bu dosya).
