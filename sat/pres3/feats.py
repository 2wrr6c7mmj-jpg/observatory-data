"""Step 3 (library): features of a cell from the 5-band rectified patches. No answer is used.

The reference of a section (quantiles over its clear days) is made, with mode='past' (the default and the only mode the
published numbers use), from the clear days strictly before the cell's day: the cells of a section are sorted by date and
cell j only sees cells [:j]. numpy only.
"""
import numpy as np
ZB=[(-80,-60),(-60,-40),(-40,-20),(-20,0),(0,15),(15,30),(30,50),(50,80)]
def water(P,offs,C):
    """[N,5]: for every cell, the water colour of its scene (port x day) = the median over ALL the port's sections that day of
    the pixels 70 m and more offshore. P may be float16 (the patch store): each scene is converted before the median."""
    key=np.array([c['port']+c['date'] for c in C]); W=np.zeros((len(C),5),np.float32)
    for k in np.unique(key):
        m=key==k; W[m]=np.median(np.asarray(P[m][:,:,offs>=70,:],np.float32).reshape(-1,5),0)
    return W
def chans(P,offs,C,W=None):
    """W: the water colour per cell computed beforehand (water()) on the whole port; needed when C is a subset of the port's
    sections (the daily scoring), where the median over the subset would be another number."""
    if W is None: W=water(P,offs,C)
    else: W=np.asarray(W,np.float32)
    key=np.array([c['port']+c['date'] for c in C]); E=np.empty_like(P)
    for k in np.unique(key):
        m=key==k; E[m]=P[m]-W[m][0]
    vis=E[...,:3].mean(-1); nir=E[...,3]; sw=E[...,4]
    absd=np.sqrt((E[...,:3]**2).mean(-1)+E[...,3]**2+E[...,4]**2)
    return dict(vis=vis,nir=nir,sw=sw,absd=absd,rb=E[...,2]-E[...,0]),W
def smooth(x):   # 沿岸 3 点の移動平均
    return (np.roll(x,1,1)+x+np.roll(x,-1,1))/3
def good_day(C):
    key=np.array([c['port']+c['date'] for c in C]); cl=np.array([c['cloud']+c['shadow'] for c in C]); nd=np.array([c['nodata'] for c in C])
    g=np.zeros(len(C),bool)
    for k in np.unique(key):
        m=key==k; g[m]=(np.mean(cl[m]>0.2)<0.15)
    return g&(cl<=0.2)&(nd==0)
def features(P,offs,C,nmin=6,mode='past',only=None,W=None,good=None):
    """only: when given (a set of row numbers), the reference of a section is made only for these cells; the reference
    features of the other rows stay NaN (the daily scoring wants the day's cells and the earlier answered cells, not the whole
    history). W, good: the water colour and the clear-day flags computed beforehand on the whole port (water(), good_day()),
    for a C that is a subset of the port's sections. The defaults are the research code's behaviour, unchanged."""
    X,W=chans(P,offs,C,W); N=len(C); F={}
    good=good_day(C) if good is None else np.asarray(good,bool)
    core=(offs>=-50)&(offs<=50)
    scal={}
    for nm in ('vis','nir','sw','absd','rb'):
        x=X[nm]
        for lo,hi in ZB:
            m=(offs>=lo)&(offs<hi); v=x[:,:,m].mean(2)
            F[f'{nm}_mean_{lo}']=v.mean(1); F[f'{nm}_p10_{lo}']=np.percentile(v,10,axis=1); F[f'{nm}_p90_{lo}']=np.percentile(v,90,axis=1)
        xs=smooth(x)[:,:,core].reshape(N,-1)
        for q in (2,10,50,90,98): scal[f'{nm}_q{q}']=np.percentile(xs,q,axis=1)
    def box(a): return (a[:,:-1,:-1]+a[:,1:,:-1]+a[:,:-1,1:]+a[:,1:,1:])/4
    for zn,(lo,hi) in (('L',(-50,10)),('M',(-20,40)),('W',(10,70))):
        m=(offs>=lo)&(offs<=hi)
        for nm in ('vis','nir','sw','absd'):
            b=box(X[nm][:,:,m]).reshape(N,-1)
            scal[f'blob_{nm}_min_{zn}']=b.min(1); scal[f'blob_{nm}_max_{zn}']=b.max(1)
            scal[f'blob_{nm}_fdark_{zn}']=(b<-0.012).mean(1); scal[f'blob_{nm}_fbright_{zn}']=(b>0.02).mean(1)
    F.update(scal)
    # 参照（区間ごと、過去の晴れの日）
    groups={}
    for i,c in enumerate(C): groups.setdefault((c['port'],c['section']),[]).append(i)
    R={nm:{q:np.full(X[nm].shape,np.nan,np.float32) for q in (10,50,90)} for nm in ('vis','nir','absd')}
    NW={'a02':X['absd']>0.02,'a04':X['absd']>0.04,'a08':X['absd']>0.08,'dk':X['vis']<-0.012,'nr':X['nir']>0.03}
    LD={k:{f:np.full(N,np.nan,np.float32) for f in ('occ','occn','ext50','ext75','ext90','run10','run20','area','new','gone','vfrac')} for k in NW}
    A=P.shape[1]; step=np.array([c['length_m'] for c in C],np.float32)/A; dz=np.where((offs>=-60)&(offs<=40))[0]; i0=int(np.searchsorted(offs,0))
    def lrun(m):
        out=0;cur=0
        for v in m:
            cur=cur+1 if v else 0; out=max(out,cur)
        return out
    nref=np.zeros(N,np.float32); rank={k:np.full(N,np.nan,np.float32) for k in scal}; zsc={k:np.full(N,np.nan,np.float32) for k in scal}
    for g in groups.values():
        g=sorted(g,key=lambda i:C[i]['date'])
        gg=np.array(g); gd=good[gg]
        for j,i in enumerate(g):
            if mode=='past': r=gg[:j][gd[:j]]
            else: r=np.r_[gg[:j][gd[:j]],gg[j+1:][gd[j+1:]]]
            nref[i]=len(r)
            if len(r)<nmin or (only is not None and i not in only): continue
            for k,nw in NW.items():
                fq=nw[r].mean(0); t=nw[i]; L=LD[k]
                V=(fq>0.08)&(fq<0.92)&core[None,:]; S=(fq>=0.92)
                L['vfrac'][i]=V.mean()
                if V.sum()>=4:
                    L['occ'][i]=t[V].mean(); L['occn'][i]=(t[V].mean()-fq[V].mean())
                L['new'][i]=(t&(fq<0.3)&core[None,:]).sum()*step[i]*5/100.0; L['gone'][i]=((~t)&(fq>0.7)&core[None,:]).sum()*step[i]*5/100.0
                ext=np.zeros(A,np.float32)
                for a in range(A):
                    ds=dz[S[a,dz]]; st=(ds.max()+1) if len(ds) else i0
                    n=0
                    while st+n<len(offs) and t[a,st+n]: n+=1
                    ext[a]=n*5.0
                L['ext50'][i]=np.median(ext); L['ext75'][i]=np.percentile(ext,75); L['ext90'][i]=np.percentile(ext,90)
                L['run10'][i]=lrun(ext>=10)*step[i]; L['run20'][i]=lrun(ext>=20)*step[i]; L['area'][i]=ext.sum()*step[i]/100.0
            for nm in R:
                qs=np.percentile(X[nm][r],[10,50,90],axis=0)
                for q,a in zip((10,50,90),qs): R[nm][q][i]=a
            for k,v in scal.items():
                h=v[r]; rank[k][i]=(h<v[i]).mean(); s=np.percentile(h,[25,50,75]); zsc[k][i]=(v[i]-s[1])/(s[2]-s[0]+0.004)
    for k in scal: F['rank_'+k]=rank[k]; F['z_'+k]=zsc[k]
    for k in LD:
        for f,v in LD[k].items(): F[f'ld_{k}_{f}']=v
    for nm in R:
        x=X[nm]; q10,q50,q90=R[nm][10],R[nm][50],R[nm][90]; sp=q90-q10
        d=x-q50; zz=d/(sp+0.01)
        for zn,(lo,hi) in (('L',(-50,10)),('M',(-20,40)),('W',(10,70))):
            m=(offs>=lo)&(offs<=hi); b=box(d[:,:,m]).reshape(N,-1)
            F[f'dblob_{nm}_min_{zn}']=b.min(1); F[f'dblob_{nm}_max_{zn}']=b.max(1)
        for lo,hi in ZB:
            m=(offs>=lo)&(offs<hi)
            F[f'{nm}_d50_mean_{lo}']=d[:,:,m].mean((1,2)); F[f'{nm}_d50_abs_{lo}']=np.abs(d[:,:,m]).mean((1,2))
            F[f'{nm}_d10_{lo}']=(x-q10)[:,:,m].mean((1,2)); F[f'{nm}_d90_{lo}']=(x-q90)[:,:,m].mean((1,2))
        # 可変画素（区間の中で時間の変動が大きい所＝艦が出入りする所）での当日の位置
        spc=np.where(core[None,None,:],sp,np.nan).reshape(N,-1); thr=np.nanpercentile(spc,80,axis=1)
        V=(sp>=thr[:,None,None])&core[None,None,:]
        with np.errstate(invalid='ignore'):
            pos=np.where(V,(x-q10)/(sp+0.004),np.nan).reshape(N,-1)
            F[f'{nm}_vpos_mean']=np.nanmean(pos,1); F[f'{nm}_vpos_med']=np.nanmedian(pos,1); F[f'{nm}_vpos_p80']=np.nanpercentile(pos,80,axis=1); F[f'{nm}_vpos_p20']=np.nanpercentile(pos,20,axis=1)
            F[f'{nm}_vspread']=np.nanmean(np.where(V,sp,np.nan).reshape(N,-1),1)
            F[f'{nm}_vabs']=np.nanmean(np.where(V,x,np.nan).reshape(N,-1),1); F[f'{nm}_vq50']=np.nanmean(np.where(V,q50,np.nan).reshape(N,-1),1)
            F[f'{nm}_zmax']=np.nanpercentile(np.where(core[None,None,:],smooth(zz),np.nan).reshape(N,-1),97,axis=1)
            F[f'{nm}_zmin']=np.nanpercentile(np.where(core[None,None,:],smooth(zz),np.nan).reshape(N,-1),3,axis=1)
    F['nref']=nref; F['len']=np.array([c['length_m'] for c in C],np.float32); F['pier']=np.array([c['kind']=='pier' for c in C],np.float32)
    for k in range(5): F[f'water{k}']=W[:,k]
    F['cloud']=np.array([c['cloud'] for c in C],np.float32); F['shadow']=np.array([c['shadow'] for c in C],np.float32)
    names=list(F); return np.stack([np.asarray(F[k],np.float32) for k in names],1),names,good
