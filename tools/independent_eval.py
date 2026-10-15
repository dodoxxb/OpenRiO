"""Independently parse XML and verify native predictions with polygon/VOC07 AP."""
import argparse, hashlib, json, os, time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import polygon_metric as metric


def annotations(root):
    names=(root/'ImageSets/Main/test.txt').read_text().split()
    assert len(names)==len(set(names))==11738
    digest=hashlib.sha256();targets=[];difficult={}
    for name in names:
        raw=(root/'Annotations/Oriented Bounding Boxes'/(name+'.xml')).read_bytes()
        digest.update(name.encode()+raw);tree=ET.fromstring(raw);boxes=[];labels=[]
        for obj in tree.findall('object'):
            category=obj.findtext('name','').lower()
            assert category in metric.CLASSES, category
            flag=obj.findtext('difficult',obj.findtext('difficulty','missing'))
            difficult[flag]=difficult.get(flag,0)+1
            obb=obj.find('robndbox')
            if obb is not None:
                points=np.array([[float(obb.findtext('x_'+key)),float(obb.findtext('y_'+key))]
                    for key in ['left_top','right_top','right_bottom','left_bottom']],np.float32)
                (cx,cy),(w,h),angle=metric.cv2.minAreaRect(points)
                if min(w,h)<=1e-6:continue
                if w<h:w,h=h,w;angle+=90
                angle=(angle%180)/180
            else:
                box=obj.find('bndbox')
                if box is None:continue
                x0,y0,x1,y1=[float(box.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
                cx,cy=(x0+x1)/2,(y0+y1)/2;w,h=x1-x0,y1-y0;angle=0.
                if w<h:w,h=h,w;angle=.5
            boxes.append([cx,cy,w,h,angle]);labels.append(metric.CLASSES.index(category))
        b=np.asarray(boxes,np.float32).reshape(-1,5);b[:,4]*=np.pi
        targets.append(dict(boxes=b,labels=np.asarray(labels,np.int64)))
    return targets,dict(images=len(names),test_annotation_sha256=digest.hexdigest(),difficulty_counts=difficult,
        protocol='Native DIOR convention: fit minimum-area rectangles, include parser-valid objects, VOC07 AP50/AP75. No silent difficulty filtering.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True,help='Trusted predictions.pt written by this package')
    p.add_argument('--native-metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);metric.cv2.setNumThreads(1);metric.HERE=args.output
    metric.unit();gt,identity=annotations(args.data_root)
    expected=json.loads(args.native_metrics.read_text())
    expected={k.lower():v for k,v in expected.items() if k.lower() in ['ap50','ap75']}
    assert set(expected)=={'ap50','ap75'}
    good=metric.evaluate('predictions',args.predictions,expected,gt)
    assert not torch.cuda.is_initialized()
    (args.output/'annotation_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(dict(status='PASS' if good else 'REVIEW_REQUIRED',images=11738,cuda_used=False)))
    if not good:raise SystemExit(1)


if __name__=='__main__':main()
