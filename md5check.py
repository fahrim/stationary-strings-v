import re
import zlib
f = 'duran_sicimler_sim/fig7_dirac_majorana_and_meral_constant_en.pdf'
d = open(f, 'rb').read()
ms = list(re.finditer(rb'stream\r?\n', d))
print('stream sayisi:', len(ms))
n = 0
for m in ms:
    s = m.end()
    e = d.find(b'endstream', s)
    raw = d[s:e].rstrip(b'\r\n')
    try:
        t = zlib.decompress(raw).decode('latin-1')
        n += 1
        if n == 1:
            print('ornek:', t[:300])
    except Exception as ex:
        pass
print('cozulen:', n)
