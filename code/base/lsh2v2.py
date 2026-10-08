import json, sys
from common import *
from lsh2 import run
from build_corpus import build
NORM['mode'] = sys.argv[1]
json.dump(run(build()), open(f'results/v2/lsh_tab_gold_{sys.argv[1]}.json', 'w'), indent=1)
