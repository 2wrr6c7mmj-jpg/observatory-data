import json,re,collections,datetime as dt,sys
from jsoparse import parse_release
docs=json.load(open('data/jso_texts.json',encoding='utf-8'))
def cat(t):
    if re.search(r'中国.*艦艇',t) and 'ロシア' in t: return 'CNRU','ship'
    if re.search(r'中国.*艦艇',t): return 'CN','ship'
    if re.search(r'ロシア.*艦艇',t): return 'RU','ship'
    if re.search(r'中国.*(軍機|機)',t) and 'ロシア' in t: return 'CNRU','air'
    if re.search(r'中国.*(軍機|機)',t): return 'CN','air'
    if re.search(r'ロシア.*(軍機|機)',t): return 'RU','air'
    if '訂正' in t: return 'OTHER','correction'
    return 'OTHER','other'
out=[]; stats=collections.Counter(); problems=[]
for d in docs:
    k=d['key']; m=re.search(r'p(\d{4})(\d{2})(\d{2})',k); pub=dt.date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
    nation,domain=cat(d['t'])
    text=d['x'].replace('=====PAGE=====','\n')
    if len(text.strip())<80: stats['scanned']+=1; out.append({'release':{'release_id':k,'pub_date':pub.isoformat(),'nation':nation,'domain':domain,'parse_status':'scanned'},'sightings':[],'jp_units':[]}); continue
    try:
        r=parse_release(k,pub,text,domain_hint=domain,nation_hint=nation)
    except Exception as e:
        stats['error']+=1; problems.append((k,'ERR',str(e)[:80])); continue
    r['release']['title']=d['t']; r['release']['pages']=d['n']; r['release']['url']='https://www.mod.go.jp/js/pdf/'+k+'.pdf'
    s=r['sightings']
    if not s: stats['no_sightings']+=1; problems.append((k,'nosight',text[:120]))
    else:
        stats['ok']+=1
        if domain=='ship':
            if not any(x['units'] for x in s): stats['no_units']+=1; problems.append((k,'nounits',s[0]['sentence'][:100]))
            if not any(x['strait'] for x in s): stats['no_strait']+=1
            if not any(x['observed_date'] for x in s): stats['no_date']+=1; problems.append((k,'nodate',s[0]['sentence'][:100]))
    out.append(r)
json.dump(out,open('data/parsed.json','w',encoding='utf-8'),ensure_ascii=False)
print(stats)
for p in problems[:40]: print(p)
