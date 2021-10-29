# BlazePose Tensorflow 2.x

This is an implementation of Google BlazePose in Tensorflow 2.x. The original paper is "BlazePose: On-device Real-time Body Pose tracking" by Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann, which is available on [arXiv](https://arxiv.org/abs/2006.10204). You can find some demonstrations of BlazePose from [Google blog](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html).

Currently, the model being developed in this repo is based on TFLite (.tflite) model from [here](https://github.com/PINTO0309/PINTO_model_zoo/tree/master/058_BlazePose_Full_Keypoints/01_Accurate). I use [Netron.app](https://netron.app/) to visualize the architecture and try to mimic that architecture in my implementation. The visualized model architecture can be found [here](images/blazepose_full.png). Other architectures will be added in the future.

**Note:** This repository is still under active development.

**Update 14/12/2020:** Our PushUp Counter App is using this BlazePose model to count pushups from videos/webcam. [***Read more.***](https://github.com/vietanhdev/pushup-counter-app)

## TODOs

- [ ] Implementation

    - [x] Initialize code for model from .tflite file.

    - [x] Basic dataset loader

    - [x] Implement loss function.

    - [x] Implement training code.

    - [x] Advanced augmentation: Random occlusion (BlazePose paper)

    - [x] Implement demo code for video and webcam.

    - [x] Support PCK metric.

    - [ ] Implement testing code.

    - [ ] Add training graph and pretrained models.

    - [ ] Support offset maps.

    - [ ] Experiment with other loss functions.

    - [ ] Workout counting from keypoints.

    - [ ] Rewrite in eager mode.

- [ ] Datasets

    - [x] Support LSP dataset and LSPET dataset (partially). [More](DATASET.md).

    - [x] Support PushUps dataset.

    - [x] Support MPII dataset.

    - [ ] Support YOGA-82 dataset.

    - [ ] Custom dataset.

- [ ] Convert and run model in TF Lite format.

- [ ] Convert and run model in TensorRT.

- [ ] Convert and run model in Tensorflow.js.

## Demo

- Download pretrained model for PushUp dataset [here](https://1drv.ms/u/s!Av71xxzl6mYZgddJ7IdF0wfjwI3sgw?e=l94WL5) and put into `trained_models/blazepose_pushup_v1.h5`. Test with your webcam:

```
python run_video.py -c configs/mpii/config_blazepose_mpii_pushup_heatmap_bce_regress_huber.json  -m trained_models/blazepose_pushup_v1.h5 -v webcam --confidence 0.3
```

The pretrained model is only in experimental state now. It only detects 7 keypoints for Push Up counting and it may not produce a good result now. I will update other models in the future.

## Training

**NOTE:** Currently, I only focus on PushUp datase, which contains 7 keypoints. Due to the copyright of this dataset, I don't have permission to publish it on the Internet. You can read the instruction and try with your own dataset.

- Prepare dataset using instruction from [DATASET.md](DATASET.md).

- Training heatmap branch:

```
python train.py -c configs/mpii/config_blazepose_mpii_pushup_heatmap_bce.json
```

- After heatmap branch converged, set `load_weights` to `true` and update the `pretrained_weights_path` to the best model, and continue with the regression branch:

```
python train.py -c configs/mpii/config_blazepose_mpii_pushup_heatmap_bce_regress_huber.json
```

## Reference

- Cite the original paper:

```tex
@article{Bazarevsky2020BlazePoseOR,
  title={BlazePose: On-device Real-time Body Pose tracking},
  author={Valentin Bazarevsky and I. Grishchenko and K. Raveendran and Tyler Lixuan Zhu and Fangfang Zhang and M. Grundmann},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.10204}
}
```

This source code uses some code and ideas from these repos:

- https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html
- https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras

## Contributions

Please feel free to [submit an issue](https://github.com/vietanhdev/tf-blazepose/issues) or [pull a request](https://github.com/vietanhdev/tf-blazepose/pulls).

