"""Copy released Multipole sources and fix its final-window variable typo."""
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('destination', type=Path)
args = parser.parse_args()
source = args.source / 'multipoleattention'
destination = args.destination / 'multipoleattention'
if destination.exists():
    raise SystemExit(f'refusing to overwrite {destination}')
shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__'))
path = destination / 'clustering.py'
text = path.read_text()
old = 'cent_interim, num_keys_interim, labels_interim = block_kmeans(data_new, num_clusters)'
new = 'cent_interim, num_keys_interim, labels_interim = block_kmeans(window_data, num_clusters)'
if text.count(old) != 1:
    raise SystemExit('expected upstream final-window typo not found exactly once')
path.write_text(text.replace(old, new))
