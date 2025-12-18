#!/usr/bin/env python3

"""
Gradient Boosted Trees → PMML exporter.

Consumes the bundle emitted by the Gradient Boosting learner and serializes it as a
PMML document containing metadata plus a base64-encoded pickle of the model bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lxml import etree as ET

from ..xml_utils import XML_PARSER
from .node_utils import collect_module_imports, normalize_in_ports, split_out_imports

FACTORY = (
    "org.knime.base.node.mine.treeensemble2.node.gradientboosting.pmml.exporter."
    "GBTPMMLExporterNodeFactory"
)
HUB_URL = (
    "https://hub.knime.com/knime/extensions/org.knime.features.ensembles/latest/"
    "org.knime.base.node.mine.treeensemble2.node.gradientboosting.pmml.exporter."
    "GBTPMMLExporterNodeFactory"
)


@dataclass
class GBTPMMLSettings:
    # Currently no user configuration needed; placeholder for future enhancements.
    pass


def parse_settings(node_dir: Optional[Path]) -> GBTPMMLSettings:
    # Reserved for future options (e.g., metadata toggles). For now nothing to parse.
    _ = node_dir
    return GBTPMMLSettings()


def generate_imports() -> List[str]:
    return ["import json", "import base64", "import pickle"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[tuple[str, str]],
) -> List[str]:
    _ = parse_settings(Path(node_dir) if node_dir else None)

    lines: List[str] = []
    lines.append(f"# {HUB_URL}")

    src_id, in_port = normalize_in_ports(in_ports)[0]
    lines.append(f"bundle = context['{src_id}:{in_port}']")
    lines.append("if isinstance(bundle, dict):")
    lines.append("    model = bundle.get('model') or bundle.get('estimator') or bundle")
    lines.append("else:")
    lines.append("    model = bundle")

    lines.append("feature_cols = []")
    lines.append("classes = []")
    lines.append("params = {}")
    lines.append("if isinstance(bundle, dict):")
    lines.append("    feature_cols = list(bundle.get('features') or bundle.get('feature_cols') or [])")
    lines.append("    classes = list(bundle.get('classes') or [])")
    lines.append("    params = {k: bundle.get(k) for k in ('n_estimators','learning_rate','max_depth',"
                 "'min_samples_split','min_samples_leaf','subsample','max_features','random_state') if k in bundle}")
    lines.append("meta = {")
    lines.append("    'feature_columns': feature_cols,")
    lines.append("    'classes': classes,")
    lines.append("    'params': params,")
    lines.append("}")
    lines.append("meta_json = json.dumps(meta, ensure_ascii=False)")

    lines.append("try:")
    lines.append("    pickled = pickle.dumps(model)")
    lines.append("except Exception:")
    lines.append("    pickled = pickle.dumps(bundle)")
    lines.append("pickle_b64 = base64.b64encode(pickled).decode('ascii')")

    lines.append("pmml_lines = [")
    lines.append("    \"<?xml version='1.0' encoding='UTF-8'?>\",")
    lines.append("    \"<PMML version='4.4' xmlns='http://www.dmg.org/PMML-4_4'>\",")
    lines.append("    \"  <Header>\",")
    lines.append("    \"    <Application name='knime2py' version='1.0'/>\",")
    lines.append("    \"  </Header>\",")
    lines.append("    \"  <Extension extender='knime2py' name='gbt_metadata'>\",")
    lines.append("    f\"    {meta_json}\",")
    lines.append("    \"  </Extension>\",")
    lines.append("    \"  <Extension extender='knime2py' name='gbt_pickle'>\",")
    lines.append("    f\"    {pickle_b64}\",")
    lines.append("    \"  </Extension>\",")
    lines.append("    \"</PMML>\",")
    lines.append("]")
    lines.append("pmml_text = \"\\n\".join(pmml_lines)")
    lines.append(f"context['{node_id}:1'] = pmml_text")
    return lines


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src, str(getattr(edge, "source_port", "") or "1")) for src, edge in (incoming or [])]
    node_lines = generate_py_body(nid, npath, in_ports)
    out_ports = [str(getattr(edge, "source_port", "") or "1") for _, edge in (outgoing or [])]  # unused but kept for API symmetry
    _ = out_ports
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
