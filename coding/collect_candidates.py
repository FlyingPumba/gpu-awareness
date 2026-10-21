import collections,csv,json,re,statistics
from pathlib import Path
from bs4 import BeautifulSoup
from common import has_language_hint
root=Path('/workspace/coding-awareness/source-data/metadata-extract/Project_CodeNet')
rows=[]; missing=collections.Counter()
for problem in csv.DictReader((root/'metadata/problem_list.csv').open()):
 pid=problem['id']; path=root/'problem_descriptions'/f'{pid}.html'
 if not path.exists(): missing['no_description']+=1; continue
 soup=BeautifulSoup(path.read_text(),'html.parser')
 en=soup.select_one('.lang-en') or soup
 text=en.get_text(' ',strip=True)
 if en.find('img'): missing['has_images']+=1; continue
 if re.search('[\u3040-\u30ff\u4e00-\u9fff]',text): missing['non_english']+=1; continue
 if has_language_hint(text): missing['language_hint']+=1; continue
 groups={'C':[],'Python':[]}
 for sub in csv.DictReader((root/'metadata'/f'{pid}.csv').open()):
  lang=sub['language']
  if sub['status']!='Accepted' or lang not in groups: continue
  if lang=='Python' and not re.search(r'(?:Python3|PyPy3|Python \(3\.)',sub['original_language']): continue
  groups[lang].append(sub)
 if not all(groups.values()): missing['missing_pair']+=1; continue
 for lang,subs in groups.items(): subs.sort(key=lambda x:(int(x['code_size']),x['submission_id']))
 score=re.search(r'Score\s*:\s*(\d+)',text)
 picks={lang:subs[len(subs)//4] for lang,subs in groups.items()}
 rows.append(dict(problem=problem,description_words=len(text.split()),score=int(score.group(1)) if score else None,
                  picked=picks,max_bytes=max(int(x['code_size']) for x in picks.values()),counts={k:len(v) for k,v in groups.items()}))
print('paired',len(rows),'excluded',dict(missing),flush=True)
print('scores',collections.Counter(x['score'] for x in rows),flush=True)
for threshold in (300,400,500,700,1000,1500,2000):
 small=[x for x in rows if x['max_bytes']<=threshold and x['description_words']<=600]
 easy=[x for x in small if x['score'] is None or x['score']<=300]
 print('bytes',threshold,'small',len(small),'score<=300_or_AOJL',len(easy),'sources',collections.Counter(x['problem']['dataset'] for x in easy),flush=True)
Path('/workspace/coding-awareness/source-data/candidates.json').write_text(json.dumps(rows,indent=2))
print('first',[(x['problem']['id'],x['problem']['name'],x['max_bytes'],x['score']) for x in sorted(rows,key=lambda x:x['max_bytes'])[:30]],flush=True)
