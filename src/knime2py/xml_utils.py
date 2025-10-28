# xml_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Tuple
from lxml import etree as ET

"""
XML Utilities for KNIME to Python Conversion.

Overview
----------------------------
This module provides utilities for parsing XML configuration files used in the 
knime2py generator pipeline, specifically for extracting node settings from 
settings.xml files.

Runtime Behavior
----------------------------
Inputs:
- The module reads XML configuration files, specifically settings.xml, to 
  extract node names and factory identifiers.

Outputs:
- The parsed node name and factory are returned as a tuple, which can be 
  utilized in the knime2py workflow context.

Key algorithms:
- The module employs XPath expressions to perform namespace-agnostic searches 
  for specific entry elements within the XML structure.

Edge Cases
----------------------------
The code handles cases where the settings.xml file may not exist or when 
expected entries are missing, returning None for such cases.

Generated Code Dependencies
----------------------------
The generated code depends on the lxml library for XML parsing. Note that 
these dependencies are required by the generated code, not by this utility module.

Usage
----------------------------
This module is typically invoked by the emitter when processing KNIME nodes 
to extract necessary configuration details. An example of expected context 
access might look like:
```python
node_name, factory = parse_settings_xml(node_directory)
```

Node Identity
----------------------------
The module generates code based on the settings.xml file, which may include 
factory identifiers such as FACTORY constants.

Configuration
----------------------------
The settings are extracted using XPath queries targeting specific entry keys 
like 'name', 'label', and 'factory'. The module does not define a 
@dataclass for settings but retrieves values directly from the XML.

Limitations
----------------------------
The module does not support all possible configurations that may exist in 
KNIME and may approximate behavior in certain cases.

References
----------------------------
For more information, refer to the KNIME documentation and the 
HUB_URL constant for additional resources.
"""

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

    Args:
        parent: The parent element to search within.
        names: A tuple of local names to search for.

    Returns:
        A list of matching elements.
    """
    expr = " | ".join([f".//*[local-name()='{n}']" for n in names])
    return parent.xpath(expr)

def _get_entry_value_by_key(config_el, key_name: str) -> Optional[str]:
    """
    Retrieves the value of an entry element by its key.

    Args:
        config_el: The configuration element to search within.
        key_name: The key of the entry to find.

    Returns:
        The value of the entry if found, otherwise None.
    """
    vals = config_el.xpath(".//*[local-name()='entry' and @key=$k]/@value", k=key_name)
    return vals[0] if vals else None

def parse_settings_xml(node_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses the settings.xml file to extract the node name and factory.

    Returns:
        A tuple containing the node name and factory. If not found, returns (None, None).
        Accepts a node directory or a direct settings.xml path.

    Args:
        node_dir: The directory containing the settings.xml file or the path to the settings.xml file.

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
