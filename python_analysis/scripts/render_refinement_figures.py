"""Editable sources for report figures; run from Mashine_Vision_Balls."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
base=Path(__file__).resolve().parents[1]
root=base/'validation/refinement_20261001'
out=base/'report/figures';out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(6.5,2.6));ax.set_xlim(0,16.5);ax.set_ylim(0,6.6);ax.axis('off')
labels=['Исходное видео\nSHA-256','Декодирование\nвремя кадра','Кандидаты Хафа\nобласть поиска','Уточнение края\nотказ / допуск','CSV, JSON\nSQL-графики','ClickHouse\nпротокол завершения','Диаметр, масштаб\nмм или NULL','Сопровождение\nID по времени']
positions=[(.1,4.1),(4.25,4.1),(8.4,4.1),(12.55,4.1),(.1,1),(4.25,1),(8.4,1),(12.55,1)]
for (x,y),label in zip(positions,labels):
 ax.add_patch(FancyBboxPatch((x,y),3.8,1.65,boxstyle='round,pad=.03',facecolor='#edf2f5',edgecolor='#243746',linewidth=1))
 ax.text(x+1.9,y+.825,label,ha='center',va='center',fontsize=8.2,fontfamily='DejaVu Sans')
for a,b in [(0,1),(1,2),(2,3),(3,7),(7,6),(6,5),(5,4)]:
 x,y=positions[a];xx,yy=positions[b]
 if y==yy:
  direction=1 if xx>x else -1
  start=(x+3.8 if direction>0 else x,y+.825);end=(xx if direction>0 else xx+3.8,yy+.825)
 else:start=(x+1.9,y);end=(xx+1.9,yy+1.65)
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,linewidth=1,color='#243746'))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
for ext in ['pdf','svg','png']:fig.savefig(out/f'pipeline_20261001.{ext}',dpi=180)
plt.close(fig)
paths={f'E3V{i}':root/f'E3V{i}' for i in range(6)}
fig,axes=plt.subplots(3,2,figsize=(6.5,7.0))
sources=[]
for i,ax in enumerate(axes.flat):
 name=f'E3V{i}'
 for label,path,color in [('Алгебраическая',(base/'validation/phone_20260930')/name/'measurements.csv','#999999'),('Радиальная',Path(paths[name])/'measurements.csv','#1965a1')]:
  with path.open() as f:rows=list(csv.DictReader(f))
  ax.plot([float(r['time_s']) for r in rows],[float(r['diameter_mm']) for r in rows],'.',color=color,markersize=1.8,label=label)
  sources.append({'series':name,'method':label,'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'run_id':json.loads(path.with_name('summary.json').read_text('utf8'))['run_id']})
 summary=json.loads(Path(paths[name],'summary.json').read_text('utf8'))
 ax.set_title(f"{name}: приняты {summary['measurements']}/{summary['sampled_frames']}",fontsize=9)
 ax.set_xlabel('Время, с',fontsize=8);ax.set_ylabel('Условный диаметр, мм',fontsize=8)
 ax.tick_params(labelsize=7);ax.grid(alpha=.2)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=8)
fig.subplots_adjust(left=.12,right=.98,bottom=.08,top=.93,hspace=.65,wspace=.40)
for ext in ['pdf','svg','png']:fig.savefig(out/f'phone_comparison_20261001.{ext}',dpi=180)
plt.close(fig)
(root/'figure_data_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf8')
print('Two vector figures rebuilt from measured CSVs')
