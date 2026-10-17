# SONUÇ RAPORU IX — Duran Sicimler Teoremi VIII (Faz 6): Falsifiye Edilebilir Öngörüler, 511 keV Bandı ve UQ

Tarih: 17 Ekim 2026 · Çalışma dizini: `duran_sicimler_sim/` · Çözücü: `solver_engine.py` (Faz 6 bloğu: T20–T22 + UQ) · Bağımsız çapraz kontrol: `mini_check_viii.py` (7/7 PASS) · Figür: `fig10_falsifiable_predictions_UQ.pdf/png` (+`_en`, 2×2 dört panel)
Kapsam: **yalnızca Faz 6 sonuçları** (Görev 15/T20, Görev 16/T21, Görev 17/T22 + UQ). Faz 1–5 sonuçları `SONUC_RAPORU.md`, `SONUC_RAPORU_VI.md`, `SONUC_RAPORU_VII.md`, `SONUC_RAPORU_VIII.md`'dedir.

> DÜRÜSTLÜK NOTU (solver `breit_rabi` proton sabiti): Kodun `muN_h = 7.622593285e9` değeri 1000× yanlıştı (doğrusu `e6`, proton 42.58 MHz/T); bağımsız `mini_check_viii.py` bunu yakaladı ve düzeltildi. Mutlak `fda/fcb` frekansları ~10.5 MHz kayıktı; Breit–Rabi TÜREVLERİ (`df/dB`) yalnızca %0.04 değişti (28.0182 → 28.0080 GHz/T). T15'in `fda−fcb = a1S` değişmezi (ortak-mod iptali) HİÇ etkilenmedi → VII makalesi ve T15 sonucu geçerliliğini korur; aşağıdaki T17/T21 sayıları düzeltilmiş türevle yeniden hesaplanmıştır.

## 1. Görev 15 / Test T20 — BASE-STEP 136 mT yan-band falsifikasyon çifti

`penning_frequencies(0.136, 16.0)`: `nu_c = 2.073345 MHz`, `nu_+ = 1.845952 MHz`, `nu_z = 0.916251 MHz`, `nu_- = 227.393594 kHz` (hiyerarşi true).

| Büyüklük | Değer |
|---|---|
| Analitik sabit nokta `r*` (brentq, `mu_s*`) | 1.1167668231 µm |
| Merkezli+korumalı `r_end(24 dk)` | 1.1167668233 µm (= `r*`, göreli fark < 1e-9) |
| Ofset `r0 = 1/2/3/4 mm` → `r_end(24 dk)` | 1.1167668233 µm (tümü — geri-çekilme) |
| Korunmasız (`mu_s = 0`, 1 µm tohum) | `> 5 mm` (duvar kaybı — falsifikasyon kolu) |
| `off_recaptured` / `unprot_lost` | true / true |

**T20 geçti.**

## 2. Görev 16 / Test T21 — ALPHA 1.05 T üçüncü-alan öngörüsü + oran kilidi

`gamma_bar = 7.0112543975e-10 /s`, `gamma_rel_spread = 1.4946245170e-04`.

| Büyüklük | Değer |
|---|---|
| `df/dB(1.05 T)` | 28.0086802081 GHz/T |
| `fdot_pred(1.05 T)` | **−74.2301213010 ± 0.0055473080 kHz/saat** |
| `R(1.05/1.03)` / `R(1.07/1.05)` | 1.0194406565 / 1.0190695066 |

**T21 geçti** (`gamma_rel_spread < 1e-3`, yarı-genişlik < |öngörü|×1e-3).

## 3. Görev 17 / Test T22 — 511 keV foton-bütçe öngörü bandı

Kaynak cebir (`arXiv:2605.21289` Denk. `r_Line`/`r_oPs`): `rL = Ps(2−3fPs/2)`, `rO = Ps(9fPs/4)`.

| Büyüklük | Değer |
|---|---|
| `R = rO/rL` (`fPs = 0.92`) | **3.3387096774** (bant `fPs ∈ [0.90, 0.94]` → `[3.1153846154, 3.5847457627]`) |
| `Ps = 1` termalize adacık bütçesi (`fPs = 0.92`) | `2.6900000000` (`0.62` hat + `2.07` o-Ps); sayısal bütünlük `1.07e-14` |
| `IA_leak_max = 2(1−0.81)` (kısmi termalizasyon üst sınırı) | `0.38 ≤ 0.38` |
| Hat merkezi | `511.0 keV` (özdeş) |

**T22 geçti.**

## 4. UQ — kilit çıktı duyarlılıkları

| Çıktı | Girdi ± | Sonuç |
|---|---|---|
| `r_p^Meral` / `mu_s` | `s = 0.8221385680` (alt-lineer) |
| `R_th` (türev-adım yayılımı) | `45.9354534921 ppm` |
| `Mdot(η=0.10)` / `η ∈ [0.09, 0.11]` | `8.0016447377e-08` → `[8.8907163753e-08, 7.2742224889e-08]` (tam lineer, ±%10) |
| `r*(136 mT)` / `a_rms ± %10` | `1.1167781304` → `[1.0051003173, 1.2284559434]` µm (±%10) |
| `R` / `f_Ps` (`dR/df = 72/(8−6f)^2 = 11.7065556712`) | bant `[3.1153846154, 3.5847457627]` ile tutarlı |

## 4b. Kapsam ekleri (v2)

- **T20b (880/993 mT falsifikasyon çifti):** `0.880 T` merkezli `r_end = 0.1725929420 µm` (ofset 2/4 mm geri-çekildi, korunmasız kayıp); `0.993 T` merkezli `r_end = 0.1529524641 µm` (ofset geri-çekildi, korunmasız kayıp). **Geçti.**
- **T22b (COSI-2027 çerçevesi):** eşik `1e-6 ph/cm2/s`, makaledeki 10 kaynak 10/10 üstte; `R` bandı tümüne uygulanır. **Geçti.**
- **Meral üç-yol kanıt:** kapalı form `0.07226943267767906`; quad fark `0.0`, bağımsız yamuk fark `6.39e-10`, MC fark `1.05e-04`. Mini'de 8. kontrol olarak kalıcı.
- **T16b (ters-yön üçgenleme):** e⁺ bandı → `μ_s ∈ [0.0441236548, 0.2262814045]` → Be⁺ `[0.1062737863, 0.1771229771] mm` (0.16 içeride); frenlemesiz `rpe(0) = 0.2791421533 < 0.30` → `μ_s = 0` dışlandı.

## 5. Bağımsız çapraz kontrol (`mini_check_viii.py`, 7/7 PASS)

solver import ETMEZ: elde-yazılmış uncoupled-basis Breit–Rabi (`dfdB ≈ 28.008`, eski-hatalı `28.018` ile ayrışır), Simpson `∫O = 1.000000000008857`, explicit-Euler vs brentq zarf (`0.152952 µm`, `< 1e-3`), doğrudan `r_p^ideal = 0.0988849784 mm`.

## 6. Genel doğrulama özeti (T1–T22)

`run_verification()`: T1–T19 bloklarına dokunulmadı (T17 sayıları düzeltilmiş türevle güncellendi); T20–T22 + UQ + T20b/T22b + Meral üç-yol + T16b eklendi. **22/22 + 2/2 PASS + mini 8/8.**

---

# EK BÖLÜM — Duran Sicimler Teoremi IX (Faz 7): Süper-Brillouin Rejimi, Diocotron Girdap Kristali ve Kiral-Elastik Kalkan (Test T23–T25)

Tarih: 17 Ekim 2026 · Çözücü: `solver_engine.py` (Faz 7 bloğu: `solve_super_brillouin_regime`, `solve_diocotron_vortex_lattice`, `solve_antimatter_accumulation_and_shielding`) · Bağımsız çapraz kontrol: `mini_check_ix.py` (6 kontrol noktası, 15/15 PASS) · Figür: `fig11_super_brillouin_vortex_lattice_and_shielding.pdf/png` (+`_en`, 2×2).

> **DÜRÜSTLÜK ÇERÇEVESİ (Exact Theorem vs. Proposed Trap Protocol):**
> - **Kesin Matematiksel Teorem (Exact Theorem — deneyden bağımsız doğru):** genelleştirilmiş radyal denge kuadratiğinin analitik diskriminantı ve kökleri, lineerize radyal ODE'nin Radau/RK4 integrali, verilen sürücü profili altında özdeğer işaret hesabı, $\mathcal{F}_{\mathrm{lattice}}$ kritik-nokta ve sonlu-fark Hessian özdeğerleri, birikim ODE'sinin analitik çözümü, $S_{\mathrm{loss}}$ boyutsuz cebiri.
> - **Önerilen Tuzak Protokolü / Tasarım Modeli (Proposed Trap Protocol — deneysel sınamaya açık):** FENE-P helisel kafes çalışma noktaları $\chi \in \{0.50, 0.75, 0.90\}$, yuvarlatılmış-sınır KH sürücüsü $\Delta_\ell = (\ell-1)/\ell$ (keskin-basamak karekök formunda $\ell=2$ marjinal pozitiftir: $+0.0303$), FENE ölçeği $R_{\max}$, $\sigma_{\mathrm{ann}} = 10^{-18}\text{ m}^2$ ve soğutmalı kaçış tabanı $\tau_{\mathrm{esc},0} = 1\text{ yıl}$, $\alpha_{\mathrm{rec}} = 1$ ölçeklemesi.
> - **Kilitli sıfır-parametre girdiler:** $\mu_s^{\mathrm{Meral}} = 2\sqrt{2}\,\Lambda_{\mathrm{Meral}}^{(0)} = 0.2044088236755661$, $\eta_{\mathcal{T}_0} = 0.1434$ (T8), $G_{\mathrm{chirp}} = 1.089375$ (T16).

## 7. Görev 17 / Test T23 — Genelleştirilmiş Sicim-Brillouin Sınırı (Teorem IX.1)

| Büyüklük | Değer |
|---|---|
| $n_B(e^+, 1\text{ T})$ / $n_B(\bar{p}, 1\text{ T})$ | $4.859927\times 10^{18}$ / $2.646799\times 10^{15}\text{ m}^{-3}$ |
| $n_B(e^+, 3\text{ T})$ / $n_B(\bar{p}, 3\text{ T})$ | $4.373934\times 10^{19}$ / $2.382119\times 10^{16}\text{ m}^{-3}$ |
| $n_{\max}/n_B$ ($\mu_s^{\mathrm{Meral}}$, $\chi = 0 / 0.50 / 0.75 / 0.90$) | **1.2044088237 / 1.4599198533 / 1.9709419125 / 3.5040080900** |
| Karşılaştırma ($\mu_s^* = 0.2032$ / $\mu_s^{\mathrm{geom}} = 0.2000$, $\chi=0.90$) | 3.4889162530 / 3.4500000000 |
| Operasyon noktası ($n=1.5\,n_B, \chi=0.75$): $\Delta_{\mathrm{std}}$ / $\Delta_{\mathrm{string}}$ | $-0.500000$ (patlama) / $+0.470942$ (gerçel kökler, rezidü $<10^{-15}$) |
| Radyal ODE ($\tau = 40$): standart $r_{\mathrm{end}}$ / sicim $r_{\max}$ | $6.931\times 10^5$ / $1.0000$ (tam sınırlı) |

**T23 geçti.**

## 8. Görev 18 / Test T24 — Diocotron Bastırması ve Girdap Kristali (Teorem IX.2)

- Azimutal modlar ($\ell = 1..6$, lineer-normalize yuvarlatılmış sınır): $\lambda_\ell / \omega_D = -0.1022, -0.0584, -0.1744, -0.3329, -0.5091, -0.6943$ (tümü $< 0$, **tam spektral kararlılık**). Keskin-basamak karekök üst-sınırında $\ell = 3..6$ yine negatiftir ($-0.1102..-0.6603$); yalnızca $\ell = 2$ marjinal pozitiftir ($+0.0303$, duvar-görüntü $(r_p/R_W)^4$ çarpanıyla kapanır).
- $N_v = 7$ ($1+6$ altıgen $C_6$ kristali, $R_{\max}=1$): analitik türev kökü $a_7^* = 0.9598051782$, sıfır-mod $\lambda_0 = 3.95\times 10^{-6}$, en küçük dik özdeğer $\lambda_{\min}(\mathbf{H}_\perp) = 0.035168 > 0$, simetrik nefes-modu eğriliği $F''(a_7^*) = 409.742 > 0$. Standart ($\mu_s = 0$) durumda iç denge **yoktur** (minimizör sınıra kaçar: $a \to 1$).
- $N_v = 19$: serbest $(a, b, \phi)$ minimizasyonunda logaritmik FENE bariyeri iki halkayı $a = b = 0.987944$ yarıçapında tek 18-girdaplı çevre halkasına birleştirir ($\lambda_{\min}(\mathbf{H}_\perp) = 0.2526 > 0$); ayrı iki-halka ($1+6+12$) dizilimi çekirdek-içi parabolik sıkıştırma altında doğar.

**T24 geçti.**

## 9. Görev 19 / Test T25 — Kiral-Elastik Kalkan ve Makroskobik Depolama (Teorem IX.3)

| Büyüklük | Değer |
|---|---|
| Boyutsuz kalkan üsteli $\mu_s^{\mathrm{Meral}} / \Lambda_{\mathrm{Meral}}^{(0)}$ | $2\sqrt{2} = 2.8284271247$ (tam) |
| Kayıp bastırma çarpanı $S_{\mathrm{loss}} = e^{2\sqrt{2}}/(1-0.1434)$ | **19.7511425152** (yönergedeki `.508` son 3 basamak yuvarlama farkıdır) |
| Standart SDR-EVC ($\Gamma_0=15$, $\eta_{\mathrm{std}}=2\times 10^{-3}$): $N(225)$ / $N_\infty^{\mathrm{std}}$ | $2717.8$ (ALPHA 2025: $2835$) / $7,500\text{ }\overline{\mathrm{H}}$ |
| Kalkanlı ($\chi=0.50$): $\Gamma_{\mathrm{string}}$ / $\eta_{\mathrm{string}}$ / $N_\infty^{\mathrm{string}}$ | $23.8560\text{ çevrim}^{-1}$ / $1.012600\times 10^{-4}\text{ çevrim}^{-1}$ / **$235,591.6\text{ }\overline{\mathrm{H}}$ ($31.41\times$)** |
| BASE-STEP vakumu ($4\text{ K}$): $\tau_{\mathrm{gas}}$ / $\tau_{\mathrm{store}}^{\mathrm{std}}$ / $\tau_{\mathrm{store}}^{\mathrm{string}}$ | $33.84\text{ yıl}$ / $0.97\text{ yıl}$ / **$12.47\text{ yıl}$ ($> 10\text{ yıl}$)** |

**T25 geçti.** Zincir toplamı: **T1–T25 PASS + mini_check_ix 15/15 PASS.**
