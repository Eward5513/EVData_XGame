# -*- coding: utf-8 -*-
"""
WEST stop-line detection (Q90 strategy):
- Same data flow as v2 (strict dwell + two-pass refinement),
  but final s_stop = Q=0.90 percentile of per-vehicle "last dwell s"
  computed on the refined (line-front-only) evidence.
- This ensures ~90% 的“每车最后一次驻停”都在 s_stop 之前/到达处。

Outputs:
  /home/tzhang174/EVData_XGame/data/west_stopline.json
  /home/tzhang174/EVData_XGame/data/west_stopline_analyse.json
  (same filenames; this script overwrites with Q90 strategy results)
"""
import os, csv, json, math, datetime
import numpy as np

DATA_DIR = "/home/tzhang174/EVData_XGame/data"
DIR_CSV  = os.path.join(DATA_DIR, "direction.csv")
FILES    = [("A0003_refined.csv","A0003"), ("A0008_refined.csv","A0008")]
CL_JSON  = os.path.join(DATA_DIR, "intersection_A.json")

OUT_GEOM    = os.path.join(DATA_DIR, "west_stopline.json")
OUT_ANALYSE = os.path.join(DATA_DIR, "west_stopline_analyse.json")

CENTER_BY_ROAD = {
    "A0003": (123.15253329, 32.34513595),
    "A0008": (123.18126882, 32.32708227),
}
VALID_DIRS = {"A1-1","B2-1","B3-2"}

# Params (same as v2)
SEARCH_BACK_M   = 40.0
STOP_HALF_M     = 10.0
LOW_SPEED_KMH   = 8.0
MIN_STOP_SEC    = 2.5
REFINE_BUFFER_M = 1.5
Q_ALPHA         = 0.10   # 90th percentile

# ---------- IO helpers ----------
def _read_csv_rows(path, encodings=("utf-8","gb18030","latin-1")):
    last=None
    for enc in encodings:
        try:
            with open(path,"r",newline="",encoding=enc) as f:
                return list(csv.DictReader(f))
        except Exception as e:
            last=e
    raise last

def _norm(s): return (s or "").strip()

def _parse_epoch_ts(row):
    def _num(s):
        try: return float(str(s).strip())
        except: return None
    coll = row.get('collectiontime') or row.get('collection_time') or row.get('collecttime')
    if coll is not None and str(coll).strip()!='':
        v = _num(coll)
        if v is not None:
            if v > 1e10: return v/1000.0
            if v > 1e6:  return v
    date_s=(row.get('date') or '').strip()
    time_s=(row.get('time_stamp') or row.get('timestamp') or '').strip()
    if date_s and time_s:
        try:
            import datetime
            return datetime.datetime.strptime(date_s+' '+time_s, '%Y-%m-%d %H:%M:%S').timestamp()
        except: pass
    v = _num(row.get('time_stamp'))
    return v if v is not None else 0.0

def _parse_end_ts(row):
    def _num(s):
        try: return float(str(s).strip())
        except: return None
    end_raw = row.get('end_time')
    if end_raw is None or str(end_raw).strip()=='' : return None
    v = _num(end_raw)
    if v is not None:
        if v > 1e10: return v/1000.0
        if v > 1e6:  return v
    date_s=(row.get('date') or '').strip()
    end_s =str(end_raw).strip()
    if date_s and len(end_s)>=5:
        import datetime
        try: return datetime.datetime.strptime(date_s+' '+end_s, '%Y-%m-%d %H:%M:%S').timestamp()
        except: return None
    return None

def lonlat_to_local_xy(lon, lat, lon0=None, lat0=None):
    import numpy as np
    lon = np.asarray(lon,float); lat=np.asarray(lat,float)
    if lon0 is None: lon0=float(np.nanmean(lon))
    if lat0 is None: lat0=float(np.nanmean(lat))
    R=6371000.0
    x=R*np.cos(np.deg2rad(lat0))*np.deg2rad(lon-lon0)
    y=R*np.deg2rad(lat-lat0)
    return x,y,lon0,lat0

def local_xy_to_lonlat(x,y,lon0,lat0):
    import numpy as np
    R=6371000.0
    lon=lon0+np.rad2deg(x/(R*np.cos(np.deg2rad(lat0))))
    lat=lat0+np.rad2deg(y/R)
    return lon,lat

# ---------- Polyline ----------
import numpy as np, json
def load_centerline(path, road):
    with open(path,"r",encoding="utf-8") as f:
        obj=json.load(f)
    entry=obj.get(road,{})
    cand=entry.get("lane_divider",[])
    CL=[]
    for p in cand:
        if isinstance(p,dict) and "lon" in p: CL.append([float(p["lon"]),float(p["lat"])])
        else: CL.append([float(p[0]),float(p[1])])
    if len(CL)<2: raise ValueError("lane_divider too short")
    return np.array(CL,float)

def polyline_metrics_xy(lonlat):
    lon=lonlat[:,0]; lat=lonlat[:,1]
    x,y,lon0,lat0=lonlat_to_local_xy(lon,lat)
    P=np.column_stack([x,y])
    segs=P[1:]-P[:-1]
    seg_len=np.linalg.norm(segs,axis=1)
    s_vertices=np.concatenate([[0.0],np.cumsum(seg_len)])
    return P,segs,seg_len,s_vertices,lon0,lat0

def project_point_to_polyline_xy(px,py,P,segs,seg_len,s_vertices):
    best=(None,None,None,None,float("inf"))
    for i,(v,w,L) in enumerate(zip(P[:-1],segs,seg_len)):
        if L<=1e-6:
            qx,qy=v[0],v[1]; s_here=s_vertices[i]
            d2=(px-qx)**2+(py-qy)**2
            if d2<best[-1]: best=(s_here,qx,qy,i,0.0,d2)
            continue
        t=((px-v[0])*w[0]+(py-v[1])*w[1])/(L*L)
        t=max(0.0,min(1.0,t))
        qx,qy=v[0]+t*w[0], v[1]+t*w[1]
        s_here=s_vertices[i]+t*L
        d2=(px-qx)**2+(py-qy)**2
        if d2<best[-1]: best=(s_here,qx,qy,i,t,d2)
    return best[:5]

def normal_at_s(i,segs):
    w=segs[i]; L=float(np.hypot(w[0],w[1])) or 1.0
    nx,ny=-w[1]/L, w[0]/L
    return nx,ny,L

# ---------- Data loading ----------
def load_directions_keys(path):
    rows=_read_csv_rows(path)
    allow={}
    for r in rows:
        d=_norm(r.get("direction"))
        if d not in VALID_DIRS: continue
        k=(_norm(r.get("vehicle_id")), _norm(r.get("date")),
           _norm(r.get("seg_id")), _norm(r.get("road_id")))
        allow[k]=d
    return allow

def load_points_for_road(csv_path, road, allow_map):
    rows=_read_csv_rows(csv_path)
    by_vid_dir={}; all_lonlat=[]
    for r in rows:
        if _norm(r.get("road_id"))!=road: continue
        k=(_norm(r.get("vehicle_id")), _norm(r.get("date")),
           _norm(r.get("seg_id")), _norm(r.get("road_id")))
        d=allow_map.get(k)
        if d is None: continue
        try:
            lon=float(_norm(r.get("longitude"))); lat=float(_norm(r.get("latitude")))
            spd_kmh=float(_norm(r.get("speed")) or 0.0)
        except: continue
        ts_start=_parse_epoch_ts(r) or 0.0
        ts_end  =_parse_end_ts(r)
        by_vid_dir.setdefault((k[0],d), []).append((ts_start, ts_end, lon, lat, spd_kmh))
        all_lonlat.append((lon,lat))
    for key in by_vid_dir: by_vid_dir[key].sort(key=lambda x: x[0])
    import numpy as np
    return by_vid_dir, (np.array(all_lonlat) if all_lonlat else np.zeros((0,2))), rows

# ---------- Dwell ----------
def is_dwell(ts_start, ts_end, spd_kmh):
    if ts_end is None: return False
    dur=max(0.0, ts_end-ts_start)
    return (dur>=MIN_STOP_SEC) and (spd_kmh<=LOW_SPEED_KMH)

def per_vehicle_rightmost_stop(seq,P,segs,seg_len,s_vertices,lon0,lat0,s_lo,s_hi,s_max=None):
    rightmost=None
    for ts_start, ts_end, lon, lat, spd_kmh in seq:
        if not is_dwell(ts_start, ts_end, spd_kmh): continue
        x,y,_,_=lonlat_to_local_xy(np.array([lon]),np.array([lat]),lon0,lat0)
        s_proj,*_=project_point_to_polyline_xy(x[0],y[0],P,segs,seg_len,s_vertices)
        if s_proj is None: continue
        if s_proj<s_lo or s_proj>s_hi: continue
        if (s_max is not None) and (s_proj> s_max): continue
        if (rightmost is None) or (s_proj>rightmost): rightmost=s_proj
    return rightmost

# ---------- Road pipeline ----------
def compute_for_road(road, csv_path, allow_map, centerline_lonlat, center_lonlat):
    P,segs,seg_len,s_vertices,lon0,lat0 = polyline_metrics_xy(centerline_lonlat)
    by_vid_dir, all_lonlat, _ = load_points_for_road(csv_path, road, allow_map)
    if len(by_vid_dir)==0:
        return ({"s_stop_m":None,"stopline_segment":[]},{"error":"no data"})

    # Orientation
    Xall,Yall,_,_=lonlat_to_local_xy(all_lonlat[:,0],all_lonlat[:,1],lon0,lat0)
    s_all=[]
    for (x,y) in zip(Xall,Yall):
        s_proj,*_=project_point_to_polyline_xy(x,y,P,segs,seg_len,s_vertices)
        s_all.append(s_proj if s_proj is not None else np.nan)
    s_all=np.array(s_all,float)
    if np.nanstd(s_all)<1e-9 or np.sum(~np.isnan(s_all))<3: east_sign=+1.0
    else:
        c=np.corrcoef(s_all[~np.isnan(s_all)], all_lonlat[~np.isnan(s_all),0])[0,1]
        east_sign=+1.0 if c>=0 else -1.0

    # s0 from intersection center
    cx,cy,_,_=lonlat_to_local_xy(np.array([center_lonlat[0]]),np.array([center_lonlat[1]]),lon0,lat0)
    s0,*_=project_point_to_polyline_xy(cx[0],cy[0],P,segs,seg_len,s_vertices)

    # WEST window
    if east_sign>0: s_lo,s_hi = s0-SEARCH_BACK_M, s0
    else:           s_lo,s_hi = s0, s0+SEARCH_BACK_M

    def gather_all_dwells(s_max=None):
        per_class_stop_s={"A1-1":[],"B2-1":[],"B3-2":[]}
        all_stop_s=[]
        for (vid,d),seq in by_vid_dir.items():
            rs=per_vehicle_rightmost_stop(seq,P,segs,seg_len,s_vertices,lon0,lat0,s_lo,s_hi,s_max=s_max)
            if (rs is not None) and (d in per_class_stop_s):
                per_class_stop_s[d].append(float(rs))
                all_stop_s.append(float(rs))
        s_all_low=[]; post_line_slow=0
        for (vid,d),seq in by_vid_dir.items():
            for ts_start,ts_end,lon,lat,spd_kmh in seq:
                if not is_dwell(ts_start,ts_end,spd_kmh): continue
                x,y,_,_=lonlat_to_local_xy(np.array([lon]),np.array([lat]),lon0,lat0)
                s_proj,*_=project_point_to_polyline_xy(x[0],y[0],P,segs,seg_len,s_vertices)
                if s_proj is None: continue
                if s_proj<s_lo or s_proj>s_hi: continue
                if (s_max is not None) and (s_proj> s_max):
                    post_line_slow += 1
                    continue
                s_all_low.append(float(s_proj))
        # histogram (diagnostic only)
        if len(s_all_low)==0:
            s_peak=float(s_hi); edges=[]; counts=[]
        else:
            s_all_low=np.array(s_all_low,float)
            nb=max(10,int((s_hi-s_lo)/2.0))
            counts,edges=np.histogram(s_all_low, bins=nb, range=(s_lo,s_hi))
            k=int(np.argmax(counts)); s_peak=float(0.5*(edges[k]+edges[k+1]))
        return per_class_stop_s, all_stop_s, s_all_low, (edges,counts), s_peak, post_line_slow

    # Pass1
    per1, all1, s_low1, (edges1,counts1), peak1, post1 = gather_all_dwells(s_max=None)
    if len(all1)==0:
        s_stop0=peak1
    else:
        s_stop0=float(np.quantile(np.array(all1,float), 0.50))  # coarse = median (only for s_max)
    # Pass2 (refined)
    s_max_refine = s_stop0 + REFINE_BUFFER_M
    per2, all2, s_low2, (edges2,counts2), peak2, post2 = gather_all_dwells(s_max=s_max_refine)

    # ---- Q90 strategy ----
    if len(all2)==0:
        s_stop = float(s_max_refine)  # fallback: practically near coarse
        qvals = {}
    else:
        arr = np.array(all2, float)
        q50 = float(np.quantile(arr, 0.50))
        q80 = float(np.quantile(arr, 0.80))
        q90 = float(np.quantile(arr, Q_ALPHA))
        q95 = float(np.quantile(arr, 0.95))
        s_stop = q90  # core: 90% before this line
        qvals = {"q50":q50,"q80":q80,"q90":q90,"q95":q95}
    # 20 m segment at s_stop
    s_vertices=np.concatenate([[0.0], np.cumsum(seg_len)])
    si=int(np.searchsorted(s_vertices, s_stop)-1); si=max(0,min(len(segs)-1,si))
    nx,ny,Lseg=normal_at_s(si,segs)
    s_base=s_vertices[si]
    t=(s_stop - s_base)/(Lseg if Lseg>1e-6 else 1.0)
    vx,vy=P[si]; wx,wy=segs[si]
    px,py=vx+t*wx, vy+t*wy
    ax,ay=px-STOP_HALF_M*nx, py-STOP_HALF_M*ny
    bx,by=px+STOP_HALF_M*nx, py+STOP_HALF_M*ny
    seg_lon,seg_lat=local_xy_to_lonlat(np.array([ax,bx]), np.array([ay,by]), lon0, lat0)

    geom = {
        "s_stop_m": float(s_stop),
        "stopline_segment": [[float(seg_lon[0]), float(seg_lat[0])],
                             [float(seg_lon[1]), float(seg_lat[1])]]
    }
    analyse = {
        "params": {
            "search_back_m": SEARCH_BACK_M,
            "stop_segment_half_m": STOP_HALF_M,
            "low_speed_kmh": LOW_SPEED_KMH,
            "min_stop_sec": MIN_STOP_SEC,
            "refine_buffer_m": REFINE_BUFFER_M,
            "q_alpha": Q_ALPHA
        },
        "east_sign": 1 if east_sign>0 else -1,
        "s0_center_m": float(s0),
        "window": {"s_lo": float(s_lo), "s_hi": float(s_hi)},
        "pass1": {
            "hist_edges": edges1.tolist() if len(edges1)>0 else [],
            "hist_counts": [int(x) for x in (counts1.tolist() if len(edges1)>0 else [])],
            "s_peak_right": float(peak1),
            "s_stop0": float(s_stop0),
            "per_class_stop_s": {k: [float(x) for x in v] for k,v in per1.items()},
            "postline_slowroll_count": int(post1)
        },
        "pass2_refined": {
            "hist_edges": edges2.tolist() if len(edges2)>0 else [],
            "hist_counts": [int(x) for x in (counts2.tolist() if len(edges2)>0 else [])],
            "s_peak_right": float(peak2),
            "quantiles": qvals,
            "s_stop": float(s_stop),
            "per_class_stop_s": {k: [float(x) for x in v] for k,v in per2.items()},
            "postline_slowroll_count": int(post2)
        }
    }
    return geom, analyse

def main():
    allow_map = load_directions_keys(DIR_CSV)
    out_geom={}; out_ana={}
    for fname, road in FILES:
        csv_path=os.path.join(DATA_DIR, fname)
        if not os.path.exists(csv_path):
            out_geom[road]={"error":"file not found"}; out_ana[road]={"error":"file not found"}; continue
        try: CL=load_centerline(CL_JSON, road)
        except Exception as e:
            out_geom[road]={"error":f"centerline error: {e}"}; out_ana[road]={"error":f"centerline error: {e}"}; continue
        center=CENTER_BY_ROAD.get(road)
        if center is None:
            out_geom[road]={"error":"missing intersection center"}; out_ana[road]={"error":"missing intersection center"}; continue
        geom, ana = compute_for_road(road, csv_path, allow_map, CL, center)
        out_geom[road]=geom; out_ana[road]=ana
    with open(OUT_GEOM,"w",encoding="utf-8") as f: json.dump(out_geom,f,ensure_ascii=False,indent=2)
    with open(OUT_ANALYSE,"w",encoding="utf-8") as f: json.dump(out_ana,f,ensure_ascii=False,indent=2)
    print("Wrote", OUT_GEOM); print("Wrote", OUT_ANALYSE)

if __name__ == "__main__":
    main()
