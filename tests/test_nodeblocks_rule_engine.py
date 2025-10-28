# tests/test_nodeblocks_rule_engine.py
import re
import sys
from pathlib import Path

# Make package importable from repo root
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
from knime2py.parse_knime import WorkflowGraph, Node, Edge
from knime2py.emitters import build_workbook_blocks

"""
Test the generation of a NodeBlock for the Rule Engine node.

Overview
----------------------------
This module tests the generation of a NodeBlock for the Rule Engine node in the knime2py
pipeline, ensuring that the generated code correctly implements the specified rules and
outputs the expected results.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context key corresponding to the CSV Reader node.

Outputs:
- Writes results to the context, specifically to the 'Dependent' column of the output DataFrame.

Key algorithms:
- Implements conditional logic for rules defined in the Rule Engine node, mapping them to
  Python expressions.

Edge Cases
----------------------------
The code handles scenarios such as empty or constant columns, ensuring that the generated
code does not fail under these conditions.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas

These dependencies are required by the generated code, not by this test module.

Usage
----------------------------
This module is typically invoked by the emitter during the conversion of KNIME workflows to
Python code. An example of expected context access is:
```python
df = context['1393:1']
```

Node Identity
----------------------------
KNIME factory id(s):
- org.knime.base.node.rules.engine.RuleEngineNodeFactory

Configuration
----------------------------
The settings are parsed using the `parse_smote_settings` function, which extracts values
from the settings.xml file.

Limitations
----------------------------
This module does not support all possible configurations of the Rule Engine node and may
approximate certain behaviors.

References
----------------------------
For more information, refer to the KNIME documentation on Rule Engine nodes.
"""

def test_rule_engine_generates_compare_and_default(node_csv_reader_dir: Path):
    """
    Test the generation of a NodeBlock for the Rule Engine node.

    This test builds a minimal workflow graph consisting of a CSV Reader node
    followed by a Rule Engine node. It verifies that the Rule Engine NodeBlock
    correctly implements the simple rule:
      $in_amount$ > 0 => "yes"
      TRUE => "no"
    and that it writes the results to the 'Dependent' column when append-column
    is set to true.
    
    Args:
        node_csv_reader_dir (Path): The directory containing the CSV Reader node's test data.
    """
    node_rule_engine_dir = repo_root / "tests" / "data" / "Node_rule_engine"
    assert node_rule_engine_dir.joinpath("settings.xml").exists(), "Missing Rule Engine settings.xml test data"

    # Node ids & types
    reader_id = "1393"
    re_id = "4001"
    reader_type = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"
    rule_engine_type = "org.knime.base.node.rules.engine.RuleEngineNodeFactory"

    # Minimal workflow graph
    g = WorkflowGraph(
        workflow_id="test_rule_engine_block",
        workflow_path=str((repo_root / "tests" / "data" / "dummy" / "workflow.knime").resolve()),
        nodes={
            reader_id: Node(id=reader_id, name="CSV Reader", type=reader_type, path=str(node_csv_reader_dir)),
            re_id: Node(id=re_id, name="Rule Engine", type=rule_engine_type, path=str(node_rule_engine_dir)),
        },
        edges=[Edge(source=reader_id, target=re_id, source_port="1", target_port="1")],
    )

    blocks, _imports = build_workbook_blocks(g)
    assert blocks, "Expected NodeBlocks to be created"

    # Find the Rule Engine block
    re_block = next((b for b in blocks if b.nid == re_id), None)
    assert re_block is not None, "Rule Engine NodeBlock not found"

    code = "\n".join(re_block.code_lines)

    # 1) Pulls df from the upstream context key "1393:1"
    assert "df = context['1393:1']" in code

    # 2) Condition for $in_amount$ > 0 is compiled into a condN line
    #    e.g., cond0 = (out_df['in_amount'] > 0)
    assert re.search(r"cond\d+\s*=\s*\(out_df\['in_amount'\]\s*>\s*0\)", code), \
        f"Expected comparison rule for in_amount > 0 in code:\n{code}"

    # 3) A mask applying 'yes' on the condition
    assert re.search(r"res\s*=\s*res\.mask\(\s*cond\d+\s*,\s*['\"]yes['\"]\s*\)", code), \
        "Expected mask to set 'yes' when condition holds"

    # 4) Default TRUE => "no" via fillna('no')
    assert re.search(r"res\s*=\s*res\.fillna\(\s*['\"]no['\"]\s*\)", code), \
        "Expected default outcome fillna('no')"

    # 5) Target column should be 'Dependent' (append-column = true, new-column-name = Dependent)
    assert re.search(r"out_df\[['\"]Dependent['\"]\]\s*=\s*res", code), \
        "Expected assignment to new column 'Dependent'"
