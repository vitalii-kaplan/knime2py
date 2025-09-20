# xml_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Tuple
from lxml import etree as ET

__all__ = ["XML_PARSER", "parse_settings_xml"]

XML_PARSER = ET.XMLParser(
    remove_comments=True,
    resolve_entities=False,
    no_network=True,
    ns_clean=True,
    recover=True,
)

def _findall_any(parent, names: Tuple[str, ...]):
    """
    Namespace-agnostic search using XPath local-name().
    Returns a list of elements whose local-name is in `names`.
    """
    expr = " | ".join([f".//*[local-name()='{n}']" for n in names])
    return parent.xpath(expr)

def _get_entry_value_by_key(config_el, key_name: str) -> Optional[str]:
    vals = config_el.xpath(".//*[local-name()='entry' and @key=$k]/@value", k=key_name)
    return vals[0] if vals else None

def parse_settings_xml(node_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns: (name, factory) from settings.xml when present.
    Accepts a node directory or a direct settings.xml path.
    """
    settings = node_dir / "settings.xml"
    if not settings.exists():
        if node_dir.name.endswith(".xml") and node_dir.exists():
            settings = node_dir
        else:
            return (None, None)

    root = ET.parse(str(settings), parser=XML_PARSER).getroot()

    # name candidates
    name_vals = root.xpath(
        ".//*[local-name()='entry' and (@key='name' or @key='label' or @key='node_name')]/@value"
    )
    # factory candidates
    fac_vals = root.xpath(
        ".//*[local-name()='entry' and (@key='factory' or @key='node_factory')]/@value"
    )

    return (name_vals[0] if name_vals else None, fac_vals[0] if fac_vals else None)
