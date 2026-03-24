#!/usr/bin/env python3

"""
Gradient Boosted Trees → PMML exporter.

Consumes the bundle emitted by the Gradient Boosting learner and serializes it
as a PMML document featuring explicit TreeModel segments (no binary payloads).
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
    pass


def parse_settings(node_dir: Optional[Path]) -> GBTPMMLSettings:
    _ = node_dir
    return GBTPMMLSettings()


def generate_imports() -> List[str]:
    return ["import json", "import xml.etree.ElementTree as ET"]


TREE_HELPERS = [
    "def _gbt_make_node(tree, node_id, feature_cols, lr_scale, predicate=None):",
    "    value = float(tree.value[node_id][0][0]) * lr_scale",
    "    node = ET.Element('Node', id=str(node_id), score=f\"{value:.12g}\")",
    "    if predicate is None:",
    "        ET.SubElement(node, 'True')",
    "    else:",
    "        node.append(predicate)",
    "    left = tree.children_left[node_id]",
    "    right = tree.children_right[node_id]",
    "    if left != -1 and right != -1:",
    "        feat_idx = tree.feature[node_id]",
    "        if 0 <= feat_idx < len(feature_cols):",
    "            field = str(feature_cols[feat_idx])",
    "        else:",
    "            field = f'f{feat_idx}'",
    "        threshold = tree.threshold[node_id]",
    "        left_pred = ET.Element('SimplePredicate', field=field, operator='lessOrEqual', value=f\"{threshold:.12g}\")",
    "        node.append(_gbt_make_node(tree, left, feature_cols, lr_scale, left_pred))",
    "        right_pred = ET.Element('SimplePredicate', field=field, operator='greaterThan', value=f\"{threshold:.12g}\")",
    "        node.append(_gbt_make_node(tree, right, feature_cols, lr_scale, right_pred))",
    "    return node",
    "",
    "def _gbt_make_tree_model(tree, feature_cols, lr_scale, tree_id, target_name):",
    "    tm = ET.Element('TreeModel', modelName=f'tree_{tree_id}', functionName='regression', splitCharacteristic='binarySplit')",
    "    schema = ET.SubElement(tm, 'MiningSchema')",
    "    for col in feature_cols:",
    "        ET.SubElement(schema, 'MiningField', name=str(col), usageType='active', invalidValueTreatment='asIs')",
    "    ET.SubElement(schema, 'MiningField', name=str(target_name), usageType='target', invalidValueTreatment='asIs')",
    "    tm.append(_gbt_make_node(tree.tree_, 0, feature_cols, lr_scale))",
    "    return tm",
    "",
    "def _gbt_collect_estimators(model):",
    "    est = getattr(model, 'estimators_', None)",
    "    if est is None:",
    "        raise ValueError('Gradient Boosting model has no estimators_ attribute; cannot export PMML.')",
    "    return est",
]


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
    lines.append("if not isinstance(bundle, dict):")
    lines.append("    bundle = {'model': bundle}")

    lines.extend(TREE_HELPERS)
    lines.append("model = bundle.get('model') or bundle.get('estimator')")
    lines.append("if model is None:")
    lines.append("    raise ValueError('Gradient Boosted Trees bundle missing trained model')")
    lines.append("feature_cols = list(bundle.get('features') or bundle.get('feature_cols') or [])")
    lines.append("if not feature_cols:")
    lines.append("    feature_cols = [f'feature_{i}' for i in range(len(getattr(model, 'feature_importances_', []) or []))]")
    lines.append("classes = list(bundle.get('classes') or getattr(model, 'classes_', []) or [])")
    lines.append("target_name = bundle.get('target') or bundle.get('target_name') or (classes[-1] if classes else 'prediction')")
    lines.append("learning_rate = getattr(model, 'learning_rate', bundle.get('learning_rate', 1.0)) or 1.0")
    lines.append("try:")
    lines.append("    estimators = _gbt_collect_estimators(model)")
    lines.append("except ValueError:")
    lines.append("    estimators = None")
    lines.append("n_stages = len(estimators) if estimators is not None else 0")
    lines.append("class_dim = len(estimators[0]) if n_stages else 0")
    lines.append("class_count = len(classes) if classes else class_dim or 1")

    # Metadata for extensions
    lines.append("params = {k: bundle.get(k) for k in (")
    lines.append("    'n_estimators','learning_rate','max_depth','min_samples_split','min_samples_leaf',")
    lines.append("    'subsample','max_features','random_state')")
    lines.append("    if k in bundle}")
    lines.append("meta_json = json.dumps({")
    lines.append("    'feature_columns': feature_cols,")
    lines.append("    'classes': classes,")
    lines.append("    'params': params,")
    lines.append("}, ensure_ascii=False)")
    lines.append("bundle_json = json.dumps(bundle, default=str, ensure_ascii=False)")

    lines.append("root = ET.Element('PMML', version='4.2', xmlns='http://www.dmg.org/PMML-4_2')")
    lines.append("header = ET.SubElement(root, 'Header')")
    lines.append("ET.SubElement(header, 'Application', name='knime2py', version='1.0')")
    lines.append("n_fields = len(feature_cols) + (1 if classes else 1)")
    lines.append("data_dict = ET.SubElement(root, 'DataDictionary', numberOfFields=str(max(1, n_fields)))")
    lines.append("for col in feature_cols:")
    lines.append("    df = ET.SubElement(data_dict, 'DataField', name=str(col), optype='continuous', dataType='double')")
    lines.append("    ET.SubElement(df, 'Interval', closure='closedClosed', leftMargin='0.0', rightMargin='1.0')")
    lines.append("target_field = ET.SubElement(data_dict, 'DataField', name=str(target_name), optype='categorical' if classes else 'continuous', dataType='string' if classes else 'double')")
    lines.append("if classes:")
    lines.append("    for cls in classes:")
    lines.append("        ET.SubElement(target_field, 'Value', value=str(cls))")

    lines.append("mining_model = ET.SubElement(root, 'MiningModel', functionName='classification' if classes else 'regression', modelName='GradientBoostedTrees')")
    lines.append("schema = ET.SubElement(mining_model, 'MiningSchema')")
    lines.append("for col in feature_cols:")
    lines.append("    ET.SubElement(schema, 'MiningField', name=str(col), usageType='active', invalidValueTreatment='asIs')")
    lines.append("usage = 'target' if classes else 'predicted'")
    lines.append("ET.SubElement(schema, 'MiningField', name=str(target_name), usageType=usage, invalidValueTreatment='asIs')")

    lines.append("if estimators is None or not n_stages:")
    lines.append("    seg = ET.SubElement(mining_model, 'Segmentation', multipleModelMethod='sum')")
    lines.append("    seg_segment = ET.SubElement(seg, 'Segment', id='1')")
    lines.append("    ET.SubElement(seg_segment, 'True')")
    lines.append("    default_tree = ET.SubElement(seg_segment, 'TreeModel', modelName='placeholder', functionName='regression')")
    lines.append("    default_schema = ET.SubElement(default_tree, 'MiningSchema')")
    lines.append("    for col in feature_cols:")
    lines.append("        ET.SubElement(default_schema, 'MiningField', name=str(col), usageType='active', invalidValueTreatment='asIs')")
    lines.append("    ET.SubElement(default_schema, 'MiningField', name=str(target_name), usageType='target', invalidValueTreatment='asIs')")
    lines.append("    node = ET.SubElement(default_tree, 'Node', id='1', score='0.0')")
    lines.append("    ET.SubElement(node, 'True')")
    lines.append("else:")
    lines.append("    top_seg_method = 'modelChain' if classes and class_count > 1 else 'sum'")
    lines.append("    top_seg = ET.SubElement(mining_model, 'Segmentation', multipleModelMethod=top_seg_method)")
    lines.append("    for class_idx in range(class_count):")
    lines.append("        segment = ET.SubElement(top_seg, 'Segment', id=str(class_idx + 1))")
    lines.append("        ET.SubElement(segment, 'True')")
    lines.append("        inner_model = ET.SubElement(segment, 'MiningModel', functionName='regression')")
    lines.append("        inner_schema = ET.SubElement(inner_model, 'MiningSchema')")
    lines.append("        for col in feature_cols:")
    lines.append("            ET.SubElement(inner_schema, 'MiningField', name=str(col), usageType='active', invalidValueTreatment='asIs')")
    lines.append("        ET.SubElement(inner_schema, 'MiningField', name=str(target_name), usageType='target', invalidValueTreatment='asIs')")
    lines.append("        seg = ET.SubElement(inner_model, 'Segmentation', multipleModelMethod='sum')")
    lines.append("        for stage_idx in range(n_stages):")
    lines.append("            trees = estimators[stage_idx]")
    lines.append("            tree_idx = class_idx if class_idx < len(trees) else len(trees) - 1")
    lines.append("            tree = trees[tree_idx]")
    lines.append("            lr_scale = float(learning_rate)")
    lines.append("            if len(trees) == 1 and class_count == 2 and class_idx == 1:")
    lines.append("                lr_scale *= -1.0")
    lines.append("            tm = _gbt_make_tree_model(tree, feature_cols, lr_scale, f'{class_idx+1}_{stage_idx+1}', target_name)")
    lines.append("            tree_segment = ET.SubElement(seg, 'Segment', id=str(stage_idx + 1))")
    lines.append("            ET.SubElement(tree_segment, 'True')")
    lines.append("            tree_segment.append(tm)")

    lines.append("ET.SubElement(mining_model, 'Extension', extender='knime2py', name='gbt_metadata').text = meta_json")
    lines.append("ET.SubElement(mining_model, 'Extension', extender='knime2py', name='gbt_bundle_json').text = bundle_json")
    lines.append("pmml_text = ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')")
    lines.append(f"context['{node_id}:1'] = pmml_text")
    return lines



def get_name() -> str:
    """Return name of the node in KNIME workflow."""
    return "Gradient Boosted Trees to PMML"


def handle(ntype, nid, npath, incoming, outgoing):
    explicit_imports = collect_module_imports(generate_imports)
    in_ports = [(src, str(getattr(edge, 'source_port', '') or '1')) for src, edge in (incoming or [])]
    node_lines = generate_py_body(nid, npath, in_ports)
    out_ports = [str(getattr(edge, 'source_port', '') or '1') for _, edge in (outgoing or [])]
    _ = out_ports
    found_imports, body = split_out_imports(node_lines)
    imports = sorted(set(explicit_imports) | set(found_imports))
    return imports, body
