import sys, pdfplumber, collections
for f in sys.argv[1:]:
    with pdfplumber.open(f) as p:
        pg=p.pages[0]; sc=183/ (pg.width/72*25.4)
        ch=pg.chars; import math
        for c in ch: c['size']=c['size'] if c['upright'] else c['x1']-c['x0']
        sz=[round(c['size']*sc,1) for c in ch]
        fonts=collections.Counter(c['fontname'].split('+')[-1] for c in ch)
        bold=[c for c in ch if 'Bold' in c['fontname']]
        bs=sorted(set(round(c['size']*sc,1) for c in bold))
        print(f.split('/')[-1], 'w=%.0fmm'%(pg.width/72*25.4), 'min %.1f med %.1f max %.1f'%(min(sz),sorted(sz)[len(sz)//2],max(sz)), dict(fonts), 'bold:',''.join(c['text'] for c in bold)[:30], bs)
