import json
import os

with open("data/pushup/train.json", "r") as fp:
    train1 = json.load(fp)
    for l in train1:
        l["is_pushing_up"] = True
with open("data/lsp_lspet_7points/train.json", "r") as fp:
    train2 = json.load(fp)
    for l in train2:
        l["is_pushing_up"] = False

with open("data/pushup/val.json", "r") as fp:
    val1 = json.load(fp)
    for l in val1:
        l["is_pushing_up"] = True
with open("data/lsp_lspet_7points/val.json", "r") as fp:
    val2 = json.load(fp)
    for l in val2:
        l["is_pushing_up"] = False

with open("data/pushup/test.json", "r") as fp:
    test1 = json.load(fp)
    for l in test1:
        l["is_pushing_up"] = True
with open("data/lsp_lspet_7points/test.json", "r") as fp:
    test2 = json.load(fp)
    for l in test2:
        l["is_pushing_up"] = False

with open("data/lsp_lspet_pushup/train.json", "w") as fp:
    json.dump(train1 + train2, fp)
with open("data/lsp_lspet_pushup/val.json", "w") as fp:
    json.dump(val1 + val2, fp)
with open("data/lsp_lspet_pushup/test.json", "w") as fp:
    json.dump(test1 + test2, fp)

os.system("mkdir -p data/lsp_lspet_pushup/images")
os.system("cp data/pushup/images/* data/lsp_lspet_pushup/images")
os.system("cp data/lsp_lspet_7points/images/* data/lsp_lspet_pushup/images")