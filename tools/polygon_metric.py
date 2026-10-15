"""Independent XML parsing, OpenCV polygon intersection and VOC07 AP.
Does not import the project evaluator, matching, IoU, model or dataset parser.
"""
import os,sys,json,time,hashlib,fcntl,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import cv2,torch,yaml
from PIL import Image
HERE=Path(__file__).resolve().parent
CLASSES=('airplane','airport','baseballfield','basketballcourt','bridge','chimney','expressway-service-area','expressway-toll-station','dam','golffield','groundtrackfield','harbor','overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill')
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def save(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def polygons(b):
 return np.asarray([cv2.boxPoints(((float(x),float(y)),(float(w),float(h)),float(a*180/np.pi))) for x,y,w,h,a in b],np.float32).reshape(-1,4,2)
def overlaps(a,b):
 z=np.zeros((len(a),len(b)),np.float32)
 if not len(a) or not len(b):return z
 pa,pb=polygons(a),polygons(b);amin,amax=pa.min(1),pa.max(1);bmin,bmax=pb.min(1),pb.max(1)
 aa=a[:,2].astype(float)*a[:,3];ab=b[:,2].astype(float)*b[:,3]
 for i in range(len(a)):
  candidates=np.flatnonzero(np.all(amax[i]>bmin,axis=1)&np.all(bmax>amin[i],axis=1))
  for j in candidates:
   area=float(cv2.intersectConvexConvex(pa[i],pb[j],handleNested=True)[0]);area=max(0,min(area,aa[i],ab[j]));z[i,j]=area/max(aa[i]+ab[j]-area,1e-12)
 return z

def ap(tp,fp,scores,total):
 if not len(scores):return 0.
 order=np.argsort(scores)[::-1];t=np.cumsum(np.asarray(tp,np.float32)[order]);f=np.cumsum(np.asarray(fp,np.float32)[order]);rec=t/(total+1e-16);prec=t/(t+f+1e-16)
 return float(sum(float(prec[rec>=v].max()) if np.any(rec>=v) else 0. for v in np.linspace(0,1,11))/11)
def flags(z,scores,threshold):
 tp=np.zeros(len(scores),np.float32);fp=np.ones(len(scores),np.float32)
 if z.shape[1]==0:return tp,fp
 used=set();mx=z.max(1);ix=z.argmax(1)
 for i in np.argsort(np.asarray(scores,np.float32))[::-1]:
  if mx[i]>=threshold and int(ix[i]) not in used:tp[i]=1;fp[i]=0;used.add(int(ix[i]))
 return tp,fp

def unit():
 a=np.array([[10,10,8,4,.3],[100,100,3,2,.9]],np.float32);np.testing.assert_allclose(overlaps(a,a),np.eye(2),atol=2e-5)
 b=a.copy();b[:,2:4]=b[:,[3,2]];b[:,4]+=np.pi/2;np.testing.assert_allclose(overlaps(a,b),np.eye(2),atol=2e-5)
 z=np.array([[1.],[1.],[0.]],np.float32);t,f=flags(z,np.array([.9,.8,.7]),.5);assert t.tolist()==[1,0,0] and f.tolist()==[0,1,1]
 assert ap(t,f,np.array([.9,.8,.7]),1)==1.;assert ap([0,1],[1,0],[.9,.8],1)==.5
 save(HERE/'independent_unit.json',dict(status='PASS',same_box=True,width_height_swap=True,disjoint=True,duplicate_matching=True,ap_hand_cases=True))

def evaluate(name,path,expected,gt):
 out=HERE/('independent_'+name);out.mkdir(exist_ok=True);preds=torch.load(path,map_location='cpu',weights_only=False);assert set(preds)==set(range(11738));outcomes={};started=time.monotonic()
 buckets={c:{'scores':[],'tp':[[],[]],'fp':[[],[]],'gt':0} for c in range(20)}
 for idx in range(11738):
  pred=preds[idx];b=np.asarray(pred['bboxes'],np.float32);scores=np.asarray(pred['scores']);labels=np.asarray(pred['labels']);g=gt[idx]
  assert len(b)==len(scores)==len(labels)<=300 and np.all(np.isfinite(b)) and np.all(np.isfinite(scores));assert np.all((labels>=0)&(labels<20)) and np.all(b[:,2:4]>0)
  for c in range(20):
   pb=b[labels==c];sc=scores[labels==c];gb=g['boxes'][g['labels']==c];bucket=buckets[c];bucket['gt']+=len(gb)
   if not len(sc):continue
   z=overlaps(pb,gb);bucket['scores'].append(sc)
   for ti,thr in enumerate([.5,.75]):
    t,f=flags(z,sc,thr);bucket['tp'][ti].append(t);bucket['fp'][ti].append(f)
  if idx%100==0:save(out/'status.json',dict(status='RUNNING',images=idx+1,heartbeat=now(),elapsed_seconds=time.monotonic()-started))
 for ti,key in enumerate(['ap50','ap75']):
  per={}
  for c,b in buckets.items():
   per[CLASSES[c]]=100*ap(np.concatenate(b['tp'][ti]),np.concatenate(b['fp'][ti]),np.concatenate(b['scores']),b['gt']) if b['scores'] else 0.
  mean=float(np.mean(list(per.values())));outcomes[key]=dict(value=mean,expected=expected[key],delta=mean-expected[key],per_class=per)
 save(out/'results.json',outcomes);good=all(abs(x['delta'])<=.01 for x in outcomes.values());save(out/'status.json',dict(status='PASS' if good else 'REVIEW_REQUIRED',images=11738,at=now(),prediction_sha256=sha(path),elapsed_seconds=time.monotonic()-started,tolerance_ap_points=.01,disclosure='Independently written AP and OpenCV polygon intersection; same XML annotation rectangle convention and VOC07 metric. Not proof of exact paper protocol. Numerical disagreements are reviewed, not tuned away.'))
 return good
