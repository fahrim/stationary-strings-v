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
