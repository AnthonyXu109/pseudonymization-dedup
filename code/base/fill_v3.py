import subprocess, re
out = subprocess.run(['python3', 'tables_v3.py'], capture_output=True, text=True, cwd='./xcur').stdout
rec = out.split('% recall table\n')[1].split('% merge table')[0].strip()
mer = out.split('% merge table\n')[1].split('%')[0].strip()
extra = out.split('% extra\n')[1] if '% extra\n' in out else ''
t = open('./main_v3.tex').read()
parts = t.split('PENDING')
assert len(parts) == 3, len(parts)
t = parts[0] + rec + parts[1] + mer + parts[2]
open('./main.tex', 'w').write(t)
print(extra)
