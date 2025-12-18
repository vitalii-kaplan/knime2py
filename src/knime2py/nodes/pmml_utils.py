#!/usr/bin/env python3

"""
Utilities for building PMML documents used by generated code.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional
import xml.etree.ElementTree as ET

PMML_NS = "http://www.dmg.org/PMML-4_4"

__all__ = [
    "build_pmml_document",
    "pmml_to_string",
    "build_normalizer_pmml",
]


def build_pmml_document(
    *,
    version: str = "4.4",
    application: str = "knime2py",
    app_version: str = "1.0",
) -> ET.Element:
    """Create a PMML root element with a header."""
    root = ET.Element("PMML", version=version, xmlns=PMML_NS)
    header = ET.SubElement(root, "Header")
    ET.SubElement(header, "Application", name=application, version=app_version)
    return root


def pmml_to_string(root: ET.Element) -> str:
    """Serialize an ElementTree root to a unicode string with XML declaration."""
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")


def _ensure_transformation_dict(root: ET.Element) -> ET.Element:
    trans = root.find("./TransformationDictionary")
    if trans is None:
        trans = ET.SubElement(root, "TransformationDictionary")
    return trans


def build_normalizer_pmml(
    columns: Iterable[str],
    stats: Dict[str, Dict[str, Optional[float]]],
    mode: str,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> str:
    """
    Build a PMML TransformationDictionary describing normalization of columns.
    """
    cols = [c for c in columns if c]
    root = build_pmml_document()
    if not cols:
        return pmml_to_string(root)

    trans = _ensure_transformation_dict(root)
    upper_mode = (mode or "MINMAX").upper()

    for col in cols:
        info = stats.get(col) or {}
        if upper_mode == "MINMAX":
            mn = info.get("min")
            mx = info.get("max")
            if mn is None or mx is None:
                continue
            derived = ET.SubElement(
                trans,
                "DerivedField",
                name=f"{col}_norm",
                optype="continuous",
                dataType="double",
            )
            norm = ET.SubElement(derived, "NormContinuous", origField=str(col))
            ET.SubElement(norm, "LinearNorm", orig=str(mn), norm=str(new_min))
            ET.SubElement(norm, "LinearNorm", orig=str(mx), norm=str(new_max))
        else:
            mu = info.get("mean")
            sd = info.get("std")
            if sd in (None, 0):
                continue
            derived = ET.SubElement(
                trans,
                "DerivedField",
                name=f"{col}_z",
                optype="continuous",
                dataType="double",
            )
            apply_div = ET.SubElement(derived, "Apply", function="/")
            apply_sub = ET.SubElement(apply_div, "Apply", function="-")
            ET.SubElement(apply_sub, "FieldRef", field=str(col))
            const_mu = ET.SubElement(apply_sub, "Constant", dataType="double")
            const_mu.text = "0.0" if mu is None else str(mu)
            const_sd = ET.SubElement(apply_div, "Constant", dataType="double")
            const_sd.text = str(sd)

    return pmml_to_string(root)
