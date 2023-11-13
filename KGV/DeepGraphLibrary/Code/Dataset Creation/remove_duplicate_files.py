import os
import shutil

PATH = '/mydata/dgl/Data/Data_1'
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.ttl']

files_set = set()
duplicates = []

for i in result:
    accession_id = i.split('/')[-1].split('.')[0]
    if accession_id not in files_set:
        files_set.add(accession_id)
    else:
        duplicates.append(i)

print(len(duplicates))
for f in duplicates:
    shutil.move(f, '/mydata/dgl/Duplicates')