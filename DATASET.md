# DATASET

### I. Our custom dataset format

All data should be converted to our custom dataset format before being used for training. Our format has this folder structure:
```
dataset_name/
    images/
    train.json
    val.json
    test.json
```

- `images` is a folder containing image files.
- `train.json`, `val.json`, `test.json` are annotation files. Here are an example of labels in these files:

```
[
    {
        "image": "001.png",
        "points": [[280, 540], [315, 468], [356, 354], [354, 243], [471, 331], [514, 440], [546, 540]],
        "visibility": [1, 1, 1, 1, 0, 0, 1]
    }
    {
        "image": "002.png",
        "points": [[269, 529], [289, 465], [305, 410], [310, 309], [455, 358], [542, 429], [560, 542]],
        "visibility": [1, 0, 0, 1, 1, 1, 1]
    },
    ...
]
```

### II. LSP and LSPET

- Link to LSP dataset: <https://sam.johnson.io/research/lsp.html>.
- Link to LSPET dataset: <https://sam.johnson.io/research/lspet.html>.

#### 1. Convert annotation to JSON format

- The annotation contains x and y locations and a binary value indicating the visbility of joints.
- Use `tools/lsp_data_to_json.py` to convert LSP and LSPET annotation files to json format:
- **NOTE:** We removed 6061 images from LSPET dataset due to missing points.

```
python tools/lsp_data_to_json.py --image_folder=data/lsp_dataset/images --input_file data/lsp_dataset/joints.mat --output_file data/lsp_dataset/labels.json
python tools/lsp_data_to_json.py --image_folder=data/lspet_dataset/images --input_file data/lspet_dataset/joints.mat --output_file data/lspet_dataset/labels.json
```

#### 2. Merge 2 dataset and divide into subsets

+ Training: 3739 from LSPET and 1800 from LSP.
+ Validation: 100 from LSPET and 100 from LSP.
+ Test: 100 from LSPET and 100 from LSP.

Please update paths to LSP and LSPET in `tools/split_lsp_lspet.py` and run:

```
python tools/split_lsp_lspet.py
```


### III. MPII Humanpose

- We only use images with numOtherPeople = 0. The original dataset are divided into 3 subsets:

+ Training: 9503 images.
+ Validation: 1000 images.
+ Test: 1000 images.


### IV. PushUp dataset

We have push-up 420 videos, divided in 3 sets:

+ Training: 8837 images from 317 videos.
+ Validation: 1189 images from 41 videos.
+ Test: 1013 images from 62 videos.