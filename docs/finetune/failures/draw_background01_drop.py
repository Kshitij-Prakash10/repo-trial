# Run from the repo root: python docs/finetune/failures/draw_background01_drop.py DATA_HOME
# Blue = base model, red = fine-tuned (full). 10 background01 frames spread over the IoU drops.
import sys
DATA = sys.argv[1]
import json, pandas as pd
from PIL import Image, ImageDraw
b=pd.read_csv('models/finetune/eval/base__smartdoc.csv.gz');f=pd.read_csv('models/finetune/eval/ft_full__smartdoc.csv.gz')
m=b.merge(f,on='id',suffixes=('_b','_f'));m=m[m.id.str.startswith('background01')]
m['d']=m.iou_doc_f-m.iou_doc_b;w=m.sort_values('d').iloc[::25].head(10)  # spread over the drops, not only the extreme
print(w[['id','iou_doc_b','iou_doc_f']].to_string()); print('bg01 median drop',m.d.median())
sheet=Image.new('RGB',(5*384,2*216))
for i,(_,r) in enumerate(w.iterrows()):
    im=Image.open(DATA+'/smartdoc_ch1_home/frames/'+r.id).convert('RGB');d=ImageDraw.Draw(im)
    for p,c in ((r.pred_b,(0,90,255)),(r.pred_f,(230,0,0))):
        q=[tuple(x) for x in json.loads(p)];d.line(q+[q[0]],fill=c,width=8)
    sheet.paste(im.resize((384,216)),((i%5)*384,(i//5)*216))
sheet.save('docs/finetune/failures/ft_full_vs_base_background01.jpg')
