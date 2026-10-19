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

**B7. "Bağımsız veri" bacağı kaynak-denetimsizdir.** Kaynak deposunda
Groisman–Steinberg (2000) PEO bantları, Vela kataloğu ve Link+1999 sayıları
**yoktur**; bu sayılar görev metninden gelmiş ve hiçbir kaynaktan teyit
edilmemiştir. T35'in "bandın merkezinde" iddiası ve T36'nın "0.38σ" iddiası
doğrulanmamış sayılara dayanır. GP-III'ün en zayıf bacağı budur; sayılar
kaynaklarıyla birlikte doğrulanmalıdır.

**B8. "Sıfır serbest parametre" iddiası ansatz seviyesindedir.** Sayısal
olarak yeni sabit yoktur; ancak model içeriği **form seçimlerinde** gizlidir
(hangi faktörün çarpıldığı, hangisinin paranteze girdiği, üs alınan
fonksiyonlar). Bunlar sayılmayan model dereceleri. "Sıfır parametre" ifadesi
bu nedenle "yeni kalibrasyon sabiti yok" olarak okunmalı, "türetilmiş" olarak
okunmamalıdır.

## C. SAĞLAM OLAN KISIMLAR

- **Kilitli sabitler** (`Λ_0^{(0)}`, `μ_s^Meral`, `C_univ`, `S_loss`,
  `I_shield = 2√2`): kapalı-form/kahrakter integrallerinden; bağımsız mini
  denetimlerle 14–17 basamak doğrulanıyor.
- **FV ayrıklaştırması (GP-II.2):** `p = 2.0000`; `u`-formunun **varyasyonel
  üst-sınırı ihlal ettiği** (λ_0 = 5.9158 > 5.6468) fark edilip düzeltildi
  ve geri çekme açıkça yazıldı. Bu, serinin en güçlü dürüstlük anıdır.
- **Ore–Powell düzeltmesi (T18) ve 511 keV bütçesi:** kaynak denetimli.
- **GP-III T37 çürütme argümanı:** yayımlanan formun 1/E² kutuplu olması
  gerçek ve ölçümle ayırt edilebilir; bu, GP-III'ün en sağlam parçası.
- **Mühendislik disiplini:** mini denetimler bağımsız; TR/EN eşitliği;
  `Overfull = 0`; her sapma rapor/makalede kayıtlı.

## D. ÖNERİLEN SONRAKİ ADIMLAR

1. GP-I makalesine **B1** uyarısını (H_0 bağımlılığı) ekle.
2. GP-II makalesinde **B2**'yi etiketle (0⁻⁺ topolojik terimi "ek model
   yorumu"), **B3**'te "bilgisayar destekli kesin ispat" ifadesini
   "varyasyonel üst sınır + monoton alt-approach" olarak düzelt.
3. GP-III için **B7**: PEO/Vela sayılarını birincil kaynaklardan doğrula.
4. GP-III'ün "sıfır parametre" dilini "yeni kalibrasyon sabiti yok" olarak
   yumuşat.
