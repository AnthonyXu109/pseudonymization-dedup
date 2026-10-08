import subprocess
out = subprocess.run(['python3', 'tables_v4.py'], capture_output=True, text=True, cwd='./xcur').stdout
rec = out.split('% RECALL\n')[1].split('% MERGE')[0].strip(); mer = out.split('% MERGE\n')[1].strip()
t = open('./main_v4.tex').read().replace('@RECALL@', rec).replace('@MERGE@', mer).replace('@ARTIFACTURL@', (lambda u: '\\url{'+u+'}' if u else '\\pend{repository URL}')(__import__('os').environ.get('ARTIFACT_URL','')))
t = t.replace('@HUMANREVIEW@', __import__('os').environ.get('HUMAN_REVIEW', ''))
open('./main.tex', 'w').write(t)
