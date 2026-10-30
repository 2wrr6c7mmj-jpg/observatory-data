"""Word lists shaped like pdfplumber output of the JCG monthly PDFs."""


def word(t, x0, top, w=None, h=9.0, p=0):
    w = w if w is not None else 6.0 * len(t)
    return [p, t, x0, x0 + w, top, top + h]


def table(title="令和8年10月", as_of="( 10/28 現在)", vals=None, totals=True, headers=True, last=31):
    """vals: {day: (contiguous, territorial or None)}; days not in vals are blank."""
    vals = vals if vals is not None else {1: (8, 4), **{d: (4, None) for d in range(2, 29)}}
    ws = [word("中国海警局に所属する船舶等による尖閣諸島接近(接続水域入域・領海侵入)状況", 100, 20, 400),
          word("−" + title + "−", 520, 20)]
    if as_of:
        ws.append(word(as_of, 330, 45))
    if headers:
        ws += [word("年", 40, 70), word("月", 70, 70), word("日", 100, 70), word("接続水域入域", 130, 70),
               word("領海侵入", 220, 70), word("(接続水域入域の内数)", 205, 81)]
    for d in range(1, last + 1):
        top = 95 + d * 14
        ws.append(word(str(d), 108 - 6 * len(str(d)), top))
        if d == 1:
            ws += [word("R8", 40, top), word("10", 70, top)]
        if d in vals:
            cz, ts = vals[d]
            ws.append(word(str(cz), 165 - 6 * len(str(cz)), top))
            if ts is not None:
                ws.append(word(str(ts), 245 - 6 * len(str(ts)), top))
    if totals:
        cz = [v[0] for v in vals.values()]
        ts = [v[1] for v in vals.values() if v[1]]
        ws += [word("領海侵入(接続水域入域の内数);", 330, 560, 170), word(str(len(ts)), 505, 559, h=11),
               word("日、のべ", 520, 560), word(str(sum(ts)), 570, 559, h=11), word("隻", 590, 560),
               word("接続水域入域;", 330, 575, 80), word(str(sum(1 for v in cz if v)), 415, 574, h=11),
               word("日、のべ", 430, 575), word(str(sum(cz)), 480, 574, h=11), word("隻", 500, 575)]
    for d in range(1, 32):  # chart axis labels elsewhere on the page
        ws.append(word(str(d), 600 + d * 7, 520))
    return ws


PAGE = """<html><body><a href="/top.pdf">x</a>
<table><tr><th>平成２４年</th><td><a href="data_h24_09.pdf">９月</a>　<a href="./data_h24_10.pdf">10月</a></td></tr>
<tr><td>平成30年</td><td><a href="data_h30_06.pdf"><span>6</span></a></td></tr>
<tr><td>平成31年</td><td><a href="data_h31_04.pdf">4月</a></td></tr>
<tr><td>令和元年</td><td><a href="data_R_05.pdf">5月</a></td></tr>
<tr><td>令和8年</td><td><a href="https://www.kaiho.mlit.go.jp/mission/senkaku/data_R8_10.pdf">10月</a>
<a href="data_R8_10.pdf">10月</a><a href="odd.pdf">資料</a></td></tr></table>
<p>平成24年9月以降の状況</p></body></html>"""
