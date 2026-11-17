"""Compose the floating X2 body with the bundled articulated hands, without copying meshes."""

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/robots/x2/x2_omnihand.xml"


def build(output=OUTPUT):
    body_path = ROOT / "assets/robots/x2/x2.xml"
    hand_path = ROOT / "third_party/xrobotoolkit_teleop/assets/agibot/x2/x2_omnihand.xml"
    root, donor = (ET.parse(path).getroot() for path in (body_path, hand_path))
    for document, source in ((root, body_path), (donor, hand_path)):
        meshdir = document.find("compiler").get("meshdir", ".")
        for mesh in document.findall("asset/mesh"):
            mesh.set("file", os.path.relpath((source.parent / meshdir / mesh.get("file")).resolve(), output.parent))
    root.find("compiler").set("meshdir", ".")
    root.find("option").set("integrator", "implicitfast")
    for item in donor.find("default"):
        if item.get("class") != "x2":
            root.find("default").append(deepcopy(item))
    assets = root.find("asset")
    existing = {item.get("name"): item for item in assets}
    for item in donor.find("asset"):
        if item.get("name") not in existing:
            assets.append(deepcopy(item))
    # The original keyframe is ordered by body traversal, not by configured DoFs.
    joints = root.findall(".//body/joint")
    key = root.find("keyframe/key")
    values = key.get("qpos").split()
    floating = values[:7]
    home = dict(zip((j.get("name") for j in joints), values[7:], strict=True))
    for side, prefix in (("left", "L"), ("right", "R")):
        name = f"{side}_wrist_roll_link"
        target, source = (doc.find(f".//body[@name='{name}']") for doc in (root, donor))
        target.remove(target.find("inertial"))
        target.insert(0, deepcopy(source.find("inertial")))
        # Replace wrist mesh while retaining body joints and frame definitions.
        old = existing[name]
        old.set("file", donor.find(f"asset/mesh[@name='{name}']").get("file"))
        target.append(deepcopy(source.find(f"body[@name='{prefix}_palm']")))
    for section in ("actuator", "equality"):
        target = root.find(section)
        if target is None:
            target = ET.SubElement(root, section)
        target.extend(
            deepcopy(item)
            for item in donor.find(section)
            if section == "equality" or item.get("name", "").startswith(("L_", "R_"))
        )
    key.set("qpos", " ".join([*floating, *(home.get(j.get("name"), "0") for j in root.findall(".//body/joint"))]))
    for attr in ("ctrl", "qvel"):
        key.attrib.pop(attr, None)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    build()
