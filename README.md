# BlazePose Tensorflow 2.x

This is an implementation of Google BlazePose in Tensorflow 2.x. The original paper is "BlazePose: On-device Real-time Body Pose tracking" by Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann, which is available on [arXiv](https://arxiv.org/abs/2006.10204). You can find some demonstrations of BlazePose from [Google blog](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html).

Currently, the model being developed in this repo is based on TFLite (.tflite) file from [here](https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_upper_body.tflite). I use [Netron.app](https://netron.app/) to visualize the architecture and try to mimic that architecture in my implementation. The visualized model architecture can be found [here](pose_landmark_upper_body.tflite.png). Other architectures will be considered in the future.

**Note:** This repository is still under active development.

## TODOs

- [ ] Implementation

    - [x] Initialize code for model from .tflite file.

    - [ ] Complete dataset loader.

    - [ ] Implement loss function.

    - [ ] Integrate PCKH metric.

    - [ ] Implement PCKS metric for PushUp challenge.

    - [ ] Implement training code.

    - [ ] Implement testing code.

    - [ ] Add training graph and pretrained models.

    - [ ] Support offset maps.

    - [ ] Experiment with other loss functions.

    - [ ] Workout counting from keypoints.

- [ ] Datasets

    - [ ] Support MPII dataset.

    - [ ] Support YOGA-82 dataset.

    - [ ] Support PushUps dataset.

    - [ ] Custom dataset.

- [ ] Convert and run model in TF Lite format.

- [ ] Convert and run model in TensorRT.

- [ ] Convert and run model in Tensorflow.js.

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

