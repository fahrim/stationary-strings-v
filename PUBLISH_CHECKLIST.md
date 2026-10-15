# Yayınlama Kontrol Listesi (öncelik kaydı için)

Sırayla ilerleyin; 1–3 arası bugün bitirilebilir.

## 1. GitHub reposu (bu klasör hazır)

- [ ] `github.com` üzerinde boş repo açın (örn. `duran-sicimler-V`), README ile başlatmayın.
- [ ] Aşağıdaki komutlarla ilk sürümü gönderin (`v1` etiketiyle):

```bash
cd G:\navier-stokes-fahri
git init
git add .
git commit -m "v1: Duran Sicimler Teoremi V preprint + simülasyon paketi (15 Ekim 2026)"
git branch -M main
git remote add origin https://github.com/KULLANICI/duran-sicimler-V.git
git push -u origin main
git tag v1 && git push origin v1
```

- [ ] `README.md`, `CITATION.cff` dosyalarındaki `KULLANICI` ve `YYYYYYY` yer tutucularını gerçek değerlerle değiştirip ikinci commit atın.

## 2. Zenodo DOI (öncelik için en güçlü adım, ~15 dk)

- [ ] `zenodo.org` hesabıyla giriş yapın → **New upload**.
- [ ] Bu klasörün tamamını (veya `paper/*.pdf` + kod arşivini) yükleyin, başlık + yazar + özeti girin, lisansları işaretleyin (kod: MIT, metin: CC-BY-4.0).
- [ ] **Publish** → verilen DOI'yi `README.md` ve `CITATION.cff` dosyalarına işleyin.
- [ ] (Opsiyonel ama önerilir) GitHub → Zenodo entegrasyonunu açın; sonraki release'ler otomatik DOI alır.

## 3. arXiv (1–3 gün)

- [ ] `arxiv.org` hesabı + (gerekirse) onay (endorsement) alın.
- [ ] `paper/duran_sicimler_teoremi_V.tex` dosyasını kaynak olarak gönderin (figürler tikz ile çizildiği için ek dosya gerekmez).
- [ ] arXiv kimliğini (örn. `arXiv:2610.xxxxx`) README'ye ekleyin.

## 4. Dergi (haftalar–aylar)

- [ ] Hedef dergi seçilip hakem sürecine gönderilir; hakem düzeltmeleri repo ve Zenodo'da **yeni sürüm (v2…)** olarak yayımlanır. v1 kaydı silinmez.

## Notlar

- Başkalarına ait PDF'leri (Unruh 1981, Oldroyd 1950, 166 sayfalık Navier-Stokes metni) repoya **yüklemeyin**; kaynakçada künye + link yeterlidir.
- I–IV çalışmaları için aynı adımları tekrarlayın; V numaralı makale onlara DOI ile atıf vermelidir.
