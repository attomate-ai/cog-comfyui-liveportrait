# cog-comfyui-liveportrait

> Efficient Portrait Animation with Stitching and Retargeting Control

A cog implementation of LivePortrait using the ComfyUI custom node, with the
**MediaPipe** face cropper instead of InsightFace, so it can be used
commercially.

- Paper: https://arxiv.org/pdf/2407.03168
- Website: https://liveportrait.github.io/

Forked from [fofr/cog-comfyui-liveportrait](https://github.com/fofr/cog-comfyui-liveportrait)
([Replicate demo](https://replicate.com/fofr/live-portrait)). The inputs and
the output (a list holding one MP4) are unchanged.

## Example driving videos

Try these videos:

https://github.com/KwaiVGI/LivePortrait/tree/main/assets/examples/driving

## License

This model does **not** download, install or run InsightFace or its
`buffalo_l` models, which are for non-commercial research only. Faces are
detected and cropped with Google's MediaPipe Face Landmarker instead, as
LivePortrait's licence asks for commercial use ("remove and replace
InsightFace's detection models"). Every component it uses is under a licence
that allows commercial use:

| Component | Used for | Licence |
|---|---|---|
| This repository (fofr's wrapper and our changes) | cog predictor, workflow | MIT ([LICENSE](LICENSE)) |
| [LivePortrait](https://github.com/KwaiVGI/LivePortrait) code and weights (`appearance_feature_extractor`, `motion_extractor`, `warping_module`, `spade_generator`, `stitching_retargeting_module`) | animation | MIT |
| LivePortrait `landmark.onnx` (203-point landmark model from [KwaiVGI/LivePortrait](https://huggingface.co/KwaiVGI/LivePortrait); byte-identical to Kijai's copy) | landmarks inside the face crop | MIT |
| [Kijai's safetensors conversion](https://huggingface.co/Kijai/LivePortrait_safetensors) of the LivePortrait weights | weight format only | follows the LivePortrait weights (MIT) |
| [ComfyUI-LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ) | ComfyUI nodes | MIT |
| MediaPipe `face_landmarker_v2_with_blendshapes.task` and `blaze_face_short_range.tflite` (shipped inside ComfyUI-LivePortraitKJ; byte-identical to Google's published models) | face detection and landmarks | Apache-2.0 |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) Python package | runs the models above | Apache-2.0 |
| ONNX Runtime | runs `landmark.onnx` | MIT |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | graph runner | GPL-3.0 |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | video load and encode | GPL-3.0 |

ComfyUI and VideoHelperSuite are GPL-3.0. Running them in a hosted model is
fine and their source is public; the GPL does not cover the videos this model
makes.

The InsightFace cropper node still exists in the pinned ComfyUI-LivePortraitKJ
source, but this model never uses it: the `insightface` package is not
installed, its weights are not downloaded, and the graph only uses
`LivePortraitLoadMediaPipeCropper`.

## Implementation

This model uses the ComfyUI custom node created by Kijai, pinned to
[`4d9dc62`](https://github.com/kijai/ComfyUI-LivePortraitKJ/commit/4d9dc6205b793ffd0fb319816136d9b8c0dbfdff):

https://github.com/kijai/ComfyUI-LivePortraitKJ

And the safetensor weights they converted:

https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main

The graph is `workflow_api.json`; `workflow.py` applies each prediction's
inputs to it:

1. `LoadImage` (the face) and `VHS_LoadVideo` (the driving video).
2. `DownloadAndLoadLivePortraitModels` loads the LivePortrait weights (fp16).
3. `LivePortraitLoadMediaPipeCropper` → `LivePortraitCropper` finds and crops
   the face (`dsize`, `scale`, `vx_ratio`, `vy_ratio`).
4. `LivePortraitProcess` animates the crop with the driving frames.
5. `LivePortraitComposite` pastes the animated crop back into the full image.
6. `VHS_VideoCombine` writes the MP4 (24 fps).

With eye or lip retargeting on, the driving video is also cropped with
MediaPipe and a `LivePortraitRetargeting` node feeds the process node.

## Checking the graph without a GPU

```bash
git clone https://github.com/kijai/ComfyUI-LivePortraitKJ /tmp/kj
git -C /tmp/kj checkout 4d9dc6205b793ffd0fb319816136d9b8c0dbfdff
python scripts/validate_workflow.py --kj /tmp/kj
```

It checks that every node exists at the pinned commit, every link points at a
real output of a real node, and every input matches the node's `INPUT_TYPES`,
for both the default graph and the retargeting graph.
