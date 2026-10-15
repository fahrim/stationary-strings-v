# Duran Sicimler Teoremi V: Navier–Stokes Girdap Tekilliğinin Elastik Sicim Gerilimiyle Durdurulması

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.YYYYYYY.svg)](https://doi.org/10.5281/zenodo.YYYYYYY)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Durum:** Yayımlanmamış ön-baskı (preprint, v3, 15 Ekim 2026). Hakem denetiminden geçmemiştir.
> Sayısal sonuçların tamamı bu repodaki kodla birebir yeniden üretilebilir (`python duran_sicimler_sim/solver_engine.py` — T1–T12).

Eylül 2026 Navier–Stokes iç-çekirdek denklemlerine ([1], Bölüm 3.1 ve 4.1) 1 boyutlu
elastik sicim (Oldroyd-B / Polyakov filament) gerilim dinamiğini ekleyen **model çalışması**
ve yüksek hassasiyetli sayısal doğrulama (`SciPy Radau`, `DOP853`, `brentq`, $10^{-14}$ tolerans).

**Ana bulgular (ayrıntılar `duran_sicimler_sim/SONUC_RAPORU.md` dosyasında):**

| Bulgu | Sonuç |
|---|---|
| Açık modelde tekillik durdurma | $s_{\mathrm{kritik}} \in [16.12,\, 58.93]$, sonlu $u_{\max} \in [5.9\times10^{2},\, 1.0\times10^{12}]$ (16 parametre çifti) |
| Tam geri-tepkili model | Sonlu-$s$ sıfır geçişi yok; $s=40$'ta $\Pi_{\mathrm{eff}} \approx 0.181$–$0.187$ asimptotik doyum |
| Teorem 2: $X \to 0$ eksen düzenlemesi | $\ell_\theta=\sqrt{X/X_*}\,\tilde{\ell}$ ile $1/X$ kutbu sadeleşir; $\Pi_{\mathrm{reg}}(0)=0.952236996205853$ (Test T6) |
| Teorem 3: denge-çıkarmalı küresel model (T7) | $[0,\infty)$ kesmesiz yakınsaklık; çapraz kontrol $2.20\times10^{-14}$ |
| Önerme 4.2 taşıma operatörleri (T8) | Teğetsel stres kalıntısı $\|T_0\|_\infty$: $0.5447 \to 0.4666$ (%14.34 azalma) |
| Tam $(X,s)$ PDE + FENE-P (T9) | fig5; tablo 8 ODE–PDE tutarlılığı |
| Teorem 4 rasyonel eğrilik (T10) | PDE izi yakınsaması |
| Teorem 5 iki-alanlı PDE + Lyapunov (T11) | Noktasal güç iptalleşmesi tam sıfır; bastırma + pozitiflik |
| Lagrange spektral aktarım (T12) | $k_{\max}==n$ 4/4, 12/12 geçti |
| Akustik olay ufku | $q_{\mathrm{ufuk}} \in [9.8\times10^{-4},\, 1.2\times10^{-2}]$ |
| Topolojik değişmez $\mathcal{I}_n = n c_s$ | $\max\|\mathcal{I}_n - n c_s\| \le 7.11\times10^{-15}$ (makine hassasiyeti) |

## Repo yapısı

```
.
├── paper/
│   ├── duran_sicimler_teoremi_V.tex      # Makale kaynağı (Türkçe)
│   ├── duran_sicimler_teoremi_V.pdf     # Derlenmiş makale, TR (v1)
│   ├── duran_sicimler_teoremi_V_EN.tex  # Makale kaynağı (İngilizce)
│   └── duran_sicimler_teoremi_V_EN.pdf  # Derlenmiş makale, EN (v1)
├── duran_sicimler_sim/
│   ├── solver_engine.py               # Sayısal çözücü (Radau/brentq/DOP853)
│   ├── generate_figures.py            # 4 yayın figürünü üretir
│   ├── SONUC_RAPORU.md                # Sayısal bulgular + LaTeX tabloları
│   └── fig1_fig4.{pdf,png}            # Üretilmiş figürler
├── CITATION.cff
└── LICENSE
```

## Tekrar-üretim

```bash
pip install numpy scipy matplotlib
python duran_sicimler_sim/solver_engine.py      # doğrulama testleri (T1–T6)
python duran_sicimler_sim/generate_figures.py   # 4 figür (PDF + 300 DPI PNG)
```

Makale derleme (MiKTeX / TeX Live):

```bash
cd paper
pdflatex -interaction=nonstopmode duran_sicimler_teoremi_V.tex
pdflatex -interaction=nonstopmode duran_sicimler_teoremi_V.tex
```

Derleme logunda `!` hatası ve `Overfull` uyarısı yoktur.

## Atıf

```bibtex
@misc{meral2026duranV,
  author       = {Fahri Meral},
  title        = {Duran Sicimler Teoremi V: Eyl\"ul 2026 Navier--Stokes
                  Girdap Tekilli\u{g}inin Elastik Sicim Gerilimiyle
                  Durdurulmas\i, \"Oz-Tutarl\i\ Doyum ve Akustik Olay
                  Ufku Analizi},
  year         = {2026},
  month        = oct,
  doi          = {10.5281/zenodo.YYYYYYY},
  url          = {https://github.com/KULLANICI/duran-sicimler-V}
}
```

`CITATION.cff` dosyasına da bakınız. (DOI, ilk Zenodo sürümü yayımlanınca işlenecektir.)

## Lisans

- **Kod** (`duran_sicimler_sim/`): [MIT](LICENSE)
- **Makale metni ve figürler** (`paper/`, `*.pdf`, `*.png`, `SONUC_RAPORU.md`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — kaynak gösterilerek paylaşılabilir ve uyarlanabilir.

## Not

Bu repo bir **öncelik ve şeffaflık kaydıdır**: v1 olduğu gibi arşivlenir, düzeltmeler yeni sürüm (v2, v3…) olarak eklenir; v1 kaydı silinmez. İçerik hakem denetiminden geçmemiştir; bulgular belirtilen model varsayımları ($F$ referans profili, $X_*=1.0$, $X \ge 0.05$) altındaki sayısal sonuçlardır.
