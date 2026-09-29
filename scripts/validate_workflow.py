#!/usr/bin/env python3
"""Static check of the ComfyUI graph that predict.py sends. Needs no GPU,
no ComfyUI install and no network.

For the default graph and the retargeting graph (built by workflow.py) it
checks that:
  - every node's class_type exists in the given node sources;
  - every [node_id, output_index] link points at a real node and a real
    output whose type matches the input's declared type;
  - every required input is present, and no input is unknown;
  - combo values are allowed and numbers are inside min/max.

Node schemas are read from the sources with `ast` (INPUT_TYPES, RETURN_TYPES
and NODE_CLASS_MAPPINGS), so they come from the pinned commits, not guesses.

Usage:
  python scripts/validate_workflow.py --kj PATH [--vhs PATH] [--comfy PATH]

--kj is a ComfyUI-LivePortraitKJ checkout, --vhs a ComfyUI-VideoHelperSuite
checkout, --comfy a ComfyUI checkout (or its nodes.py). Nodes whose source
isn't given are reported as skipped, not passed.
"""

import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workflow import build_workflow, load_base_workflow  # noqa: E402

UNKNOWN = object()

# Inputs the ComfyUI front end adds that the server ignores.
UI_EXTRAS = {"LoadImage": {"upload"}}


def py_files(path):
    if os.path.isfile(path):
        yield path
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in (".git", "web", "node_modules")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return UNKNOWN


def parse_spec(value):
    """(type, options) from an INPUT_TYPES entry like ("INT", {...})."""
    if not isinstance(value, ast.Tuple) or not value.elts:
        return UNKNOWN, {}
    first = value.elts[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        kind = first.value
    elif isinstance(first, ast.List):
        kind = literal(first)  # a combo: list of allowed values
        if kind is UNKNOWN:
            kind = "COMBO?"
    else:
        kind = "COMBO?"  # computed at runtime (file lists, formats)
    options = {}
    if len(value.elts) > 1 and isinstance(value.elts[1], ast.Dict):
        for k, v in zip(value.elts[1].keys, value.elts[1].values):
            key = literal(k)
            if isinstance(key, str):
                options[key] = literal(v)
    return kind, options


def parse_class(cls):
    schema = {"required": {}, "optional": {}, "returns": None}
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
            for ret in ast.walk(item):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Dict):
                    for k, v in zip(ret.value.keys, ret.value.values):
                        section = literal(k)
                        if section in ("required", "optional") and isinstance(v, ast.Dict):
                            for ik, iv in zip(v.keys, v.values):
                                name = literal(ik)
                                if isinstance(name, str):
                                    schema[section][name] = parse_spec(iv)
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "RETURN_TYPES":
                    schema["returns"] = literal(item.value)
    return schema


def load_schemas(paths):
    classes, mappings = {}, {}
    for path in paths:
        for f in py_files(path):
            try:
                tree = ast.parse(open(f, encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.setdefault(node.name, parse_class(node))
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "NODE_CLASS_MAPPINGS"
                            and isinstance(node.value, ast.Dict)
                        ):
                            for k, v in zip(node.value.keys, node.value.values):
                                key = literal(k)
                                if isinstance(key, str) and isinstance(v, ast.Name):
                                    mappings[key] = v.id
    return {name: classes[cls] for name, cls in mappings.items() if cls in classes}


def vhs_format_widgets(vhs_path, fmt):
    """Extra inputs a VHS video format adds (e.g. pix_fmt, crf)."""
    if not vhs_path or not fmt.startswith("video/"):
        return {}
    path = os.path.join(vhs_path, "video_formats", fmt[len("video/"):] + ".json")
    if not os.path.exists(path):
        return {}
    widgets = {}

    def walk(x):
        if isinstance(x, list):
            if len(x) >= 2 and isinstance(x[0], str) and isinstance(x[1], list):
                widgets[x[0]] = (x[1], {})
            elif len(x) >= 2 and isinstance(x[0], str) and x[1] in ("INT", "FLOAT", "BOOLEAN", "STRING"):
                widgets[x[0]] = (x[1], x[2] if len(x) > 2 else {})
            else:
                for y in x:
                    walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)

    walk(json.load(open(path)))
    return widgets


def is_link(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def check_value(errors, where, kind, options, value):
    if isinstance(kind, list):
        if value not in kind:
            errors.append(f"{where}: {value!r} is not one of {kind}")
        return
    if kind == "BOOLEAN" and not isinstance(value, bool):
        errors.append(f"{where}: expected BOOLEAN, got {value!r}")
    if kind == "INT" and (isinstance(value, bool) or not isinstance(value, int)):
        errors.append(f"{where}: expected INT, got {value!r}")
    if kind == "FLOAT" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        errors.append(f"{where}: expected FLOAT, got {value!r}")
    if kind == "STRING" and not isinstance(value, str):
        errors.append(f"{where}: expected STRING, got {value!r}")
    if kind in ("INT", "FLOAT") and isinstance(value, (int, float)):
        lo, hi = options.get("min"), options.get("max")
        if isinstance(lo, (int, float)) and value < lo:
            errors.append(f"{where}: {value} < min {lo}")
        if isinstance(hi, (int, float)) and value > hi:
            errors.append(f"{where}: {value} > max {hi}")


def validate(workflow, schemas, vhs_path):
    errors, skipped = [], set()
    for node_id, node in workflow.items():
        cls = node.get("class_type")
        if cls not in schemas:
            skipped.add(cls)
            continue
        schema = schemas[cls]
        spec = {**schema["required"], **schema["optional"]}
        if cls == "VHS_VideoCombine":
            spec.update(vhs_format_widgets(vhs_path, node["inputs"].get("format", "")))
        for name in schema["required"]:
            if name not in node["inputs"]:
                errors.append(f"node {node_id} ({cls}): missing required input {name!r}")
        for name, value in node["inputs"].items():
            where = f"node {node_id} ({cls}).{name}"
            if name in UI_EXTRAS.get(cls, set()):
                continue
            if name not in spec:
                errors.append(f"{where}: not an input of {cls}")
                continue
            kind, options = spec[name]
            if is_link(value):
                src_id, index = value
                src = workflow.get(src_id)
                if src is None:
                    errors.append(f"{where}: links to missing node {src_id}")
                    continue
                src_schema = schemas.get(src["class_type"])
                if src_schema is None or src_schema["returns"] in (None, UNKNOWN):
                    continue
                returns = src_schema["returns"]
                if not 0 <= index < len(returns):
                    errors.append(f"{where}: {src['class_type']} has no output {index}")
                elif isinstance(kind, str) and kind != "COMBO?" and returns[index] != kind:
                    errors.append(
                        f"{where}: expects {kind}, but node {src_id} output {index} is {returns[index]}"
                    )
            elif kind is not UNKNOWN and kind != "COMBO?":
                check_value(errors, where, kind, options, value)
    return errors, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kj", required=True, help="ComfyUI-LivePortraitKJ checkout")
    parser.add_argument("--vhs", help="ComfyUI-VideoHelperSuite checkout")
    parser.add_argument("--comfy", help="ComfyUI checkout or its nodes.py")
    args = parser.parse_args()

    sources = [args.kj] + [p for p in (args.vhs, args.comfy) if p]
    if args.comfy and os.path.isdir(args.comfy):
        sources[-1] = os.path.join(args.comfy, "nodes.py")
    schemas = load_schemas(sources)

    base = load_base_workflow(os.path.join(os.path.dirname(__file__), "..", "workflow_api.json"))
    common = dict(
        face_filename="face.jpg",
        driving_filename="driving.mp4",
        frame_load_cap=0,  # VidStudio sends 0: every frame
        select_every_n_frames=1,
        dsize=512,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.12,
        lip_zero=True,
        eyes_retargeting_multiplier=1.0,
        lip_retargeting_multiplier=1.0,
        stitching=True,
    )
    variants = {
        "default": dict(common, eye_retargeting=False, lip_retargeting=False, relative=True),
        "retargeting": dict(common, eye_retargeting=True, lip_retargeting=True, relative=True),
        "absolute": dict(common, eye_retargeting=False, lip_retargeting=False, relative=False),
    }

    failed = False
    for name, kwargs in variants.items():
        workflow = build_workflow(base, **kwargs)
        errors, skipped = validate(workflow, schemas, args.vhs)
        status = "FAIL" if errors else "ok"
        print(f"[{status}] {name}: {len(workflow)} nodes")
        for e in errors:
            print(f"    {e}")
        if skipped:
            print(f"    skipped (no source given): {sorted(skipped)}")
        failed = failed or bool(errors)

    classes = {n["class_type"] for n in build_workflow(base, **variants["retargeting"]).values()}
    banned = sorted(c for c in classes if c in ("LivePortraitLoadCropper",))
    if banned:
        print(f"[FAIL] InsightFace cropper in graph: {banned}")
        failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
