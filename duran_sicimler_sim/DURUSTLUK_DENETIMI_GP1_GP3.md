# DÜRÜSTLÜK DENETİMİ — Grand Proof I–III (GP-I/II/III, Faz 9–11)

> Bağımsız gözden geçirme. Hatalar gizlenmemiş, düzeltilmiş ve kayıt altına
> alınmıştır. Bu dosya makale metnine girmez; okuyucu dürüstlük çerçevesi
> için kanıttır.

## A. Düzeltilmiş GERÇEK HATA

**A1. T37 Ore–Powell form etiketleri TERS'ti (düzeltildi: `f8a41c3`, `95de962`).**
Kaynak denetimi: `arXiv:2605.21289` Denk.(12)'nin **yayımlanan** formunda
3. terim `2m(m−E)/E²` paydası taşır (1/E² kutuplu, ∫₁ = 2343.47); yalnız
`E²→(2m−E)²` ikame edilen varyant `2m(m−E)/(2m−E)²` taşır (∫₁ = 27.147).
GP-III'nin ilk sürümünde bu ikisi ters etiketlenmişti (`B_misprint = 9.31`,
`B_pole = 159.37`). Doğrusu: **yayımlanan form 159.37, ikame form 9.31.**
Ayrıca `ore_prompt_form` docstring'i ("ln'siz basım hatası") düzeltildi ve
`ore_powell_published` eklendi. Sayısal argüman değişmedi (ikisi de >500σ ile
çürütülüyor) ama etiketler artık kaynakla birebir tutuyor.

## B. YAPISAL ZAYIFLIKLAR (düzeltilmedi — bilinçli olarak açık bırakıldı)

**B1. GP-I: `a_Meral` doğrudan `H_0` ile orantılı — test gücü zayıf.**
`a_Meral = κ·c·H_0`. Test edilen `a_0 ≈ 1.2e-10` ile karşılaştırma:

| H_0 | a_Meral | a_0 farkı |
|---|---|---|
| Planck 2018 (67.4) | 1.1466e-10 | %4.45 |
| SH0ES/JWST (70.5) | 1.1993e-10 | **%0.06** |
| CCHP (73.0) | 1.2419e-10 | %3.49 |

Yani "tahmin"in `a_0` ile uyumu esasen **H_0 seçimine** bağlıdır; modelin
`a_0` hakkında `H_0`'dan bağımsız içeriği yoktur. Bu, GP-I'nin en zayıf
noktasıdır ve makalede açıkça yazılmalıdır.

**B2. GP-II.1 ve GP-II.3'te iki çarpan post-hoc (sapma sonrası) eklendi.**
(i) `0^{-+}` için paranteze `+μ_s` (Teorem VI "topolojik" gerekçesiyle):
hata %4.68 → %1.03. (ii) kiral köprü için `(1+Λ_0^{(0)}/3)`: köprü
%1.23 → %0.049. İkisi de türetilmedi; kilitli sabitleri sapmayı kapatmak için
yeniden devreye sokar. Kiral çarpan "ek model yorumu" etiketiyle işaretlendi;
`0^{-+}` topolojik terimi için aynı etiket **yapılmalıdır**.

**B3. GP-II.2 "Temple–Kato kesin alt sınır" totolojiktir.**
`μ = v_0ᵀSv_0 = λ_0` (çünkü `S v_0 = λ_0 v_0`) ve rezidü ≈ 1e-11
(makine epsilonu) → alt sınır = üst sınır = sayısal özdeğer. Bu **bağımsız
alt sınır değildir**; "bilgisayar destekli kesin ispat" ifadesi abartılıdır.
Bağımsız olan tek nicel Gauss varyasyonel **üst** sınırıdır (5.6467531).
Düzgün ifade: "λ_0 < 5.6468 (varyasyonel) ve λ_0 = 5.6102 (FV, alt-yaklaşık)".

**B4. GP-III T35: üç öngörünün de bant merkezine düşmesi yapısal işarettir.**
Bantlar ~%10–15 genişlikte; modelin üç ayrı oranı (`C_univ`, `C_univ^{-1}`,
`[C²(1+μ)]^{-1}`) üçü de tam merkeze düşüyor. Bu, formların bandı hedefleyerek
seçildiği izlenimini verir. En azından "band merkezinde uyum, ansatz seçiminin
doğrulanmışlığı değil, tutarlılığıdır" cümlesi yazılmalıdır.

**B5. GP-III T36 gevşeme ODE'sinin zaman ölçekleri keyfidir** (τ_f = 6 gün,
τ_s = 60 gün) ve çapraz doğrulama trivialdir (aynı ifadenin iki yazımı,
resid = 0). Bunlar gözlemsel olarak kalibre edilmelidir.

**B6. GP-III T36'da "base" zaten gözlemle uyumludur.** Standart rezervuar
ilişkisi 1.9777e-6 verir; gözlem (2.25 ± 0.35)e-6 → base **0.77σ** içindedir.
Meral çarpanı (1+μ_s) bunu %5.86'ya (0.38σ) çeker. Yani çarpan, zaten uyumlu
bir sayıyı daha da uyumlu hale getirir; modelin **katkısı sıfırdır**.

**B7. "Bağımsız veri" bacağı kaynak-denetimsizdir — KISMEN ÇÖZÜLDÜ (19 Ekim 2026).**
Groisman–Steinberg (2000), Vela kataloğu ve Link+1999 sayıları yönerge
metninden gelmiş ve hiçbir kaynaktan teyit edilmemişti. **19 Ekim 2026'da
dört birincil kaynak indirildi, tam metin ve LaTeX kaynakları tarandı ve
kaynak deposuna eklendi** (`src/08`–`11`, `src/tex/…`, `kaynaklarim.md` Bölüm 8):

| İddia | Sonuç |
|---|---|
| PEO bantları [1.55,1.70] / [0.30,0.35] | ❌ **doğrulanamadı** — Groisman–Steinberg'de "core/funnel/dimple" yok; deney PAAm+şurup, iki disk. "Bandın merkezinde uyum" **geri çekildi**; T35 barajı artık bant kullanmaz. |
| $I_{\rm crust}/I = 0.0160$ | ✅ **düzeltildi → 0.0140** (Link+1999: $\Delta I/I \ge 0.014$) |
| $(2.25\pm0.35)\times10^{-6}$ bandı | ❌ doğrulanamadı → **2000 olayı 3.0858e-6** (Dodson+2002 Tablo 1), **2004 olayı 2.08e-6** (Dodson+2007) kullanıldı |
| $Q\in[0.05,0.18]$ bandı | ❌ doğrulanamadı → **ölçülen $Q_{2000}=0.02198$** raporlandı |
| Gevşeme zaman ölçekleri (keyfi 6/60 gün) | ✅ **ölçülen: 1.2 dk, 0.53, 3.29, 19.07 gün** (Dodson+2002) |
| T36 uyumu (%5.86, 0.38σ) | ✅ kaynak-doğrulanmış: **2004 olayı %0.43**, 2000 olayı %32.9 (açıkça raporlanır) |

**B8. "Sıfır serbest parametre" iddiası ansatz seviyesindedir.** Sayısal
olarak yeni sabit yoktur; ancak model içeriği **form seçimlerinde** gizlidir
(hangi faktörün çarpıldığı, hangisinin paranteze girdiği, üs alınan
fonksiyonlar). Bunlar sayılmayan model dereceleri. "Sıfır parametre" ifadesi
bu nedenle "yeni kalibrasyon sabiti yok" olarak okunmalı, "türetilmiş" olarak
okunmamalıdır.

## E. KAYNAK DENETİMİ SONUÇLARI VE SERİ DENETİMİ (19 Ekim 2026)

**E1. PEO bandı aranması — sonuç: bulunamadı, ve bulunamayacağı
gösterildi.** Üç kaynak indirilip tam metin + LaTeX taranarak denetlendi:
- Groisman & Steinberg 2000 (arXiv:nlin/0104052): `1.55`, `1.70`, `0.30`,
  `0.35`, "funnel", "dimple", "core radius", "free surface" **hiç geçmiyor**.
  Deney: PAAm+şurup, iki paralel disk, $d/R=0.263$; ölçüm: akış direnci
  ~20 kat, elastik gerilim 2 büyüklük mertebesi artışı.
- van Buel & Stark 2022 (arXiv:2202.12714): Oldroyd-B VKS (iki paralel plaka),
  **$\mathrm{Wi}_c = 12$**, akış direnci artışı, histerezis.
- Guido+ 2021 (arXiv:2110.03004): **PAAm+gliserin, eş eksenli silindirler,
  DPV** — GP-III protokolüne en yakın *gerçek* yayımlanmış masaüstü düzenek.
- Burghelea+ 2006 (arXiv:physics/0609112): elastik turbulans, yeni uzunluk
  ölçeği = sınır tabakası kalınlığı.
→ **Sonuç: altta yatan fizik gerçek ve kaynaklanabilir; ama (1.55–1.70) /
(0.30–0.35) sayı çifti hiçbir kaynakta izlenebilir değil.** Bu yüzden
GP-III T35'te "bandın merkezinde uyum" iddiası geri çekildi; deneyin kendisi
falsifiable test olarak sunuluyor.

**E2. Teoremler I–X etkileniyor mu? — HAYIR (somut denetimle).**
GP-II.2'de bulunan `u = √x ψ` hatası yalnızca GP-II'nin YM akı-tüpü
operatöründe (T32) vardı; Teoremler I–X hiçbir yerde bu dönüşümü
kullanmaz (açık ℓ, Radau ODE, 1D/2D PDE, quad/GL integralleri).
Ama denetim sırasında **asıl boşluk** bulundu: `C_univ` dört mini betiğinde
(`mini_check_gp1/gp2/gp3/x`) **sabit kopyalanmıştı**, yani bağımsız olarak
hiç türetilmemişti. Bu kapatıldı: yeni `duran_sicimler_sim/mini_check_audit.py`
(solver'ı import etmez; ℓ̃'yi GL-256 yerine DOP853 ODE ile çözer, `l_shear_reg`
ve `G_drive`'ı ayrı yazar) **tüm serinin zincirini bağımsız türetir**:
**AUDIT 8/8 PASS** — `C_univ = 1.6260379365517` (çözücüyle fark $5.6\times10^{-13}$),
$\Pi_{\mathrm{Meral}} = \mu_s$ (fark $5.6\times10^{-17}$), $I_{\mathrm{shield}} =
2\sqrt{2}$, $S_{\mathrm{loss}}$, $a_{\mathrm{Meral}}$, gap ve kiral köprü doğrulandı.

## C. SAĞLAM OLAN KISIMLAR

- **Kilitli sabitler** (`Λ_0^{(0)}`, `μ_s^Meral`, `C_univ`, `S_loss`,
  `I_shield = 2√2`): kapalı-form/kahrakter integrallerinden; bağımsız mini
  denetimlerle 14–17 basamak doğrulanıyor **ve `mini_check_audit.py` ile
  tüm zincir bağımsız olarak yeniden türetiliyor (AUDIT 8/8)**.
- **FV ayrıklaştırması (GP-II.2):** `p = 2.0000`; `u`-formunun **varyasyonel
  üst-sınırı ihlal ettiği** (λ_0 = 5.9158 > 5.6468) fark edilip düzeltildi
  ve geri çekme açıkça yazıldı. Bu, serinin en güçlü dürüstlük anıdır.
- **Ore–Powell düzeltmesi (T18) ve 511 keV bütçesi:** kaynak denetimli.
- **GP-III T37 çürütme argümanı:** yayımlanan formun 1/E² kutuplu olması
  gerçek ve ölçümle ayırt edilebilir; bu, GP-III'ün en sağlam parçası.
- **Mühendislik disiplini:** mini denetimler bağımsız; TR/EN eşitliği;
  `Overfull = 0`; her sapma rapor/makalede kayıtlı.

## D. ÖNERİLEN SONRAKİ ADIMLAR

1. ✅ GP-I makalesine **B1** uyarısı eklendi (H_0 bağımlılığı).
2. ✅ GP-II makalesinde **B2** etiketlendi (0⁻⁺ topolojik terimi "ek model
   yorumu"), **B3** "bilgisayar destekli kesin ispat" ifadesi düzeltildi.
3. ✅ GP-III için **B7** çözüldü: dört birincil kaynak indirildi, tarandı,
   kaynak deposuna eklendi; doğrulanamayan bantlar geri çekildi, doğrulanmış
   sayılarla değiştirildi.
4. ⬜ GP-III'ün "sıfır parametre" dilini "yeni kalibrasyon sabiti yok" olarak
   yumuşatmak.
5. ✅ PEO bandının gerçek kaynağı arandı (E1): **bulunamadı**, gösterildi.
6. ⬜ T36 modelinin 2000 olayındaki %32.9 sapması için ek fizik
   (büyük glitch'lerde daha büyük $Q$ / daha güçlü kabuk-girdap pinlemesi)
   araştırılmalıdır.
7. ✅ `C_univ` sabit-kopya boşluğu kapatıldı: `mini_check_audit.py` (AUDIT 8/8).
