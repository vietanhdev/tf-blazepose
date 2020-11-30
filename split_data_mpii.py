import json
jsonfile = "data/mpii_annotations.json"

# load train or val annotation
with open(jsonfile) as anno_file:
    anno = json.load(anno_file)

val_anno, train_anno = [], []
for idx, val in enumerate(anno):
    if val['isValidation'] == True:
        val_anno.append(anno[idx])
    else:
        train_anno.append(anno[idx])

with open("data/mpii/train.json", "w") as anno_file:
    json.dump(train_anno, anno_file)
    
with open("data/mpii/val.json", "w") as anno_file:
    json.dump(val_anno, anno_file)