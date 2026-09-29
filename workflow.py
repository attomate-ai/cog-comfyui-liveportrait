"""Builds the ComfyUI API graph for one prediction.

Pure: no cog, ComfyUI or network imports, so scripts/validate_workflow.py can
check the exact graph predict.py sends against the pinned node sources.

The graph uses ComfyUI-LivePortraitKJ's MediaPipe cropper (Apache-2.0 models)
instead of InsightFace (non-commercial models). InsightFace is never loaded.
"""

import copy
import json

API_JSON_FILE = "workflow_api.json"

# Node ids in workflow_api.json.
MODELS = "1"
LOAD_IMAGE = "4"
LOAD_VIDEO = "8"
VIDEO_COMBINE = "23"
PROCESS = "30"
CROPPER_LOADER = "31"
SOURCE_CROPPER = "32"
COMPOSITE = "33"
# Added only when eye or lip retargeting is on.
DRIVING_CROPPER = "34"
RETARGETING = "35"


def load_base_workflow(path=API_JSON_FILE):
    with open(path, "r") as file:
        return json.loads(file.read())


def relative_motion_mode(relative):
    # The old all-in-one node's relative=False used the driving frame's
    # rotation and expression with the source scale: "single_frame" here.
    return "relative" if relative else "single_frame"


def build_workflow(base, **kwargs):
    """Returns a new graph with the prediction's inputs applied."""
    workflow = copy.deepcopy(base)

    load_video = workflow[LOAD_VIDEO]["inputs"]
    load_video["video"] = kwargs["driving_filename"]
    load_video["frame_load_cap"] = kwargs["frame_load_cap"]  # 0 = every frame
    load_video["select_every_nth"] = kwargs["select_every_n_frames"]

    workflow[LOAD_IMAGE]["inputs"]["image"] = kwargs["face_filename"]

    crop = {
        "dsize": kwargs["dsize"],
        "scale": kwargs["scale"],
        "vx_ratio": kwargs["vx_ratio"],
        "vy_ratio": kwargs["vy_ratio"],
    }
    workflow[SOURCE_CROPPER]["inputs"].update(crop)

    process = workflow[PROCESS]["inputs"]
    process["lip_zero"] = kwargs["lip_zero"]
    process["stitching"] = kwargs["stitching"]
    process["relative_motion_mode"] = relative_motion_mode(kwargs["relative"])

    if kwargs["eye_retargeting"] or kwargs["lip_retargeting"]:
        # Retargeting needs landmarks for every driving frame, so the driving
        # video is cropped too (MediaPipe again). Off by default: it costs a
        # face detection per driving frame.
        workflow[DRIVING_CROPPER] = {
            "inputs": {
                "pipeline": [MODELS, 0],
                "cropper": [CROPPER_LOADER, 0],
                "source_image": [LOAD_VIDEO, 0],
                **crop,
                "face_index": 0,
                "face_index_order": "large-small",
                "rotate": True,
            },
            "class_type": "LivePortraitCropper",
            "_meta": {"title": "LivePortrait Cropper (driving)"},
        }
        workflow[RETARGETING] = {
            "inputs": {
                "driving_crop_info": [DRIVING_CROPPER, 1],
                "eye_retargeting": kwargs["eye_retargeting"],
                "eyes_retargeting_multiplier": kwargs["eyes_retargeting_multiplier"],
                "lip_retargeting": kwargs["lip_retargeting"],
                "lip_retargeting_multiplier": kwargs["lip_retargeting_multiplier"],
            },
            "class_type": "LivePortraitRetargeting",
            "_meta": {"title": "LivePortrait Retargeting"},
        }
        process["opt_retargeting_info"] = [RETARGETING, 0]

    return workflow
