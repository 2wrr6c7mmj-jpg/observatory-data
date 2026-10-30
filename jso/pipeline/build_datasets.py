import sqlite3,csv,datetime as dt,json,collections
con=sqlite3.connect('data/jso.sqlite'); c=con.cursor()
D=lambda s: dt.date.fromisoformat(s)
NAMES={'miyako':'宮古海峡','osumi':'大隅海峡','tsushima':'対馬海峡','tsugaru':'津軽海峡','soya':'宗谷海峡','yonaguni_taiwan':'与那国－台湾','yonaguni_iriomote':'与那国－西表','amami_yokoate':'奄美－横当','tokara':'トカラ','okinawa_kume':'沖縄－久米','ishigaki_iriomote':'石垣－西表','miyako_ishigaki':'宮古－石垣','izu_mikura_hachijo':'伊豆諸島','ogasawara':'小笠原','nemuro':'根室海峡','etorofu':'択捉・国後','okinawa_iheya':'沖縄－伊平屋','kikai':'喜界','yakushima_tanegashima':'屋久－種子'}
ev=[(D(a),D(b or a),ac,cat,n,reg) for a,b,ac,cat,n,reg in c.execute("select date_start,date_end,actor,category,name,region from context_events")]
def nearest(d,f):
    best=None
    for ds,de,a,cat,n,reg in ev:
        if not f(a,cat,n,reg): continue
        off=0 if ds<=d<=de else ((ds-d).days if d<ds else -(d-de).days)
        if best is None or abs(off)<abs(best[0]): best=(off,n)
    return best or ('','')
pla=lambda a,cat,n,reg: a=='CN' and cat=='exercise' and reg in('taiwan_strait','multi')
allied=lambda a,cat,n,reg: a in('JP','US','USJP','MULTI','AU','PH','KR','UK','FR','DE','IN') and cat=='exercise'
pol=lambda a,cat,n,reg: cat in('election','inauguration','summit','visit','statement','policy','sanction','legislation','incident')
rows=c.execute("""select s.sighting_id,r.release_id,r.nation,r.pub_date,s.observed_date,s.observed_date_end,s.strait,s.strait_dir,s.waters_from,s.waters_to,s.strait_inferred,s.activity,s.unit_count,r.title,r.url,
 (select group_concat(class_jp||coalesce('('||hull_no||')','')||case when count>1 then '×'||count else '' end,' / ') from sighting_units u where u.sighting_id=s.sighting_id),
 (select group_concat(hull_no,' ') from sighting_units u where u.sighting_id=s.sighting_id and hull_no is not null),
 (select count(*) from sighting_units u join vessel_classes v on v.class_key=u.class_key where u.sighting_id=s.sighting_id and v.type_cat='CV'),
 (select count(*) from sighting_units u join vessel_classes v on v.class_key=u.class_key where u.sighting_id=s.sighting_id and v.type_cat in('AOR','AKS')),
 (select count(*) from sighting_units u join vessel_classes v on v.class_key=u.class_key where u.sighting_id=s.sighting_id and v.type_cat in('AGI','AGS','AGM')),
 (select count(*) from sighting_units u where u.sighting_id=s.sighting_id and u.kind='sub'),
 (select count(*) from sighting_units u join vessel_classes v on v.class_key=u.class_key where u.sighting_id=s.sighting_id and v.type_cat in('DDG','DD','CG','CGN','FFG','FF','FFL','PGG')),
 (select p.days_away from transit_pairs p where p.in_sighting_id=s.sighting_id limit 1),
 s.same_as_release,
 (select group_concat(unit_name,'/') from jp_units j where j.release_id=r.release_id)
 from sightings s join releases r using(release_id) where r.domain='ship' and s.strait is not null and s.observed_date is not None order by s.observed_date, s.sighting_id""".replace('is not None','is not null')).fetchall()
seen={}
for row in rows:
    sid,rid,nat,pub,d,de,st,sd,wf,wt,inf,act,n,title,url,units,hulls,cv,aor,agi,sub,comb,days,same,jp=row
    key=(rid,st,d)
    if key in seen and not (seen[key]['inferred'] and not inf): continue
    seen[key]=dict(sighting_id=sid,release_id=rid,nation=nat,pub_date=pub,date=d,date_end=de or '',year=d[:4],month=int(d[5:7]),strait=st,strait_jp=NAMES.get(st,st),dir=sd or '',waters_from=wf or '',waters_to=wt or '',inferred=inf or 0,activity=act or '',n_units=n or '',units=units or '',hulls=hulls or '',has_cv=cv,has_aor=aor,has_agi=agi,has_sub=sub,n_combat=comb,days_away=days if days is not None else '',linked_release=same or '',jp_units=jp or '',title=title,url=url)
pas=list(seen.values())
for p in pas:
    d=D(p['date'])
    o,nm=nearest(d,pla); p['pla_ex_offset'],p['pla_ex_name']=o,nm
    o,nm=nearest(d,allied); p['allied_ex_offset'],p['allied_ex_name']=o,nm
    o,nm=nearest(d,pol); p['pol_offset'],p['pol_name']=o,nm
cols=list(pas[0].keys())
w=csv.DictWriter(open('dash/passages.csv','w',encoding='utf-8',newline=''),fieldnames=cols); w.writeheader(); w.writerows(pas)
print('passages',len(pas),collections.Counter(p['nation'] for p in pas))
w=csv.writer(open('dash/context.csv','w',encoding='utf-8',newline='')); w.writerow(['date_start','date_end','actor','category','name','region','notes','source_url'])
for r in c.execute("select date_start,coalesce(date_end,date_start),actor,category,name,region,notes,source_url from context_events order by date_start"): w.writerow(r)
w=csv.writer(open('dash/round_trips.csv','w',encoding='utf-8',newline='')); w.writerow(['out_date','out_strait','out_strait_jp','in_date','in_strait','in_strait_jp','days_away','nation','units','with_aor','has_cv','out_release','in_release','in_url'])
n=0
for r in c.execute("""select o.observed_date,o.strait,i.observed_date,i.strait,p.days_away,r.nation,
 (select group_concat(class_jp||coalesce('('||hull_no||')',''),' / ') from sighting_units u where u.sighting_id=o.sighting_id),
 (select count(*) from sighting_units u join vessel_classes v on v.class_key=u.class_key where u.sighting_id=o.sighting_id and v.type_cat in('AOR','AKS')),
 (select count(*) from sighting_units u join vessel_classes v on v.class_key=u.class_key where u.sighting_id=o.sighting_id and v.type_cat='CV'),
 o.release_id,i.release_id,r.url from transit_pairs p join sightings o on o.sighting_id=p.out_sighting_id join sightings i on i.sighting_id=p.in_sighting_id join releases r on r.release_id=i.release_id where p.days_away between 0 and 200 order by 1"""):
    w.writerow([r[0],r[1],NAMES.get(r[1],r[1]),r[2],r[3],NAMES.get(r[3],r[3]),r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11]]); n+=1
print('round_trips',n)
