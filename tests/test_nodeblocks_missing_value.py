# tests/test_nodeblocks_missing_value.py
"""
Test for the Missing Value NodeBlock in the knime2py generator.

Overview
----------------------------
This module tests the functionality of the Missing Value NodeBlock, ensuring it correctly
imputes missing integer values in a DataFrame.

Runtime Behavior
----------------------------
Inputs:
- Reads from the context key corresponding to the upstream node's output DataFrame.

Outputs:
- Writes the imputed DataFrame back to the context, specifically to the key associated with
  the Missing Value node's output port.

Key algorithms or mappings:
- The module selects integer columns and imputes missing values with 0.

Edge Cases
----------------------------
The code handles cases where columns may be empty or contain constant values, ensuring
robustness against NaNs and class imbalance.

Generated Code Dependencies
----------------------------
The generated code requires pandas for DataFrame manipulation. These dependencies are
specific to the generated code, not this testing module.

Usage
----------------------------
Typically invoked by the knime2py emitter when processing a KNIME workflow. An example of
context access would be:
```python
df = context['<upstream_node_id>:<port>']
```

Node Identity
----------------------------
KNIME factory id:
- org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory

Configuration
----------------------------
The settings are parsed using the `parse_smote_settings` function, which extracts relevant
configuration from the settings.xml file.

Limitations
----------------------------
This module does not support advanced imputation strategies available in KNIME.

References
----------------------------
Refer to the KNIME documentation for more details on the Missing Value node and its
configuration options.
"""

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

@pytest.fixture(scope="session")
def node_missing_value_dir(node_dir):
    """
    Fixture that provides the directory for the Missing Value node.

    Args:
        node_dir (function): A function that returns the directory path for a given node.

    Returns:
        Path: The path to the Missing Value node directory.
    """
    return node_dir("Node_missing_value")

def test_missing_value_block_imputes_integers_to_zero(node_csv_reader_dir: Path):
    """
    Test that the Missing Value NodeBlock correctly imputes integer-typed columns with 0.

    This test builds a minimal workflow graph consisting of a CSV Reader node and a Missing Value node.
    It verifies that the Missing Value NodeBlock reads from the correct context and emits code that
    imputes missing integer values with 0, based on the settings defined in
    tests/data/Node_missing_value/settings.xml.

    Args:
        node_csv_reader_dir (Path): The directory path for the CSV Reader node.
    """
    # Path to the Missing Value node settings
    node_missing_value_dir = repo_root / "tests" / "data" / "Node_missing_value"
    assert node_missing_value_dir.joinpath("settings.xml").exists(), "Missing Missing Value settings.xml test data"

    print("test_missing_value_block_imputes_integers_to_zero dir is: ", node_missing_value_dir)

    # Node ids & types
    reader_id = "1393"
    mv_id = "3001"
    reader_type = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"
    mv_type = "org.knime.base.node.preproc.pmml.missingval.compute.MissingValueHandlerNodeFactory"

    # Minimal workflow graph
    g = WorkflowGraph(
        workflow_id="test_missing_value_block",
        workflow_path=str((repo_root / "tests" / "data" / "dummy" / "workflow.knime").resolve()),
        nodes={
            reader_id: Node(id=reader_id, name="CSV Reader", type=reader_type, path=str(node_csv_reader_dir)),
            mv_id: Node(id=mv_id, name="Missing Value", type=mv_type, path=str(node_missing_value_dir)),
        },
        edges=[Edge(source=reader_id, target=mv_id, source_port="1", target_port="1")],
    )

    blocks, _ = build_workbook_blocks(g)
    assert blocks, "Expected NodeBlocks to be created"

    # Find the Missing Value block
    mv_block = next((b for b in blocks if b.nid == mv_id), None)
    assert mv_block is not None, "Missing Value NodeBlock not found"

    code = "\n".join(mv_block.code_lines)

    # 1) Pulls df from the upstream context key "1393:1"
    assert "df = context['1393:1']" in code

    # 2) It should select integer columns in some reasonable way
    #    Accept either a select_dtypes(include=['Int...']) approach or an is_integer_dtype check.
    assert (
        ("select_dtypes" in code and re.search(r"include\s*=\s*\[?['\"]Int", code, re.I))
        or ("is_integer_dtype" in code)
    ), "Expected integer column selection in generated code"

    # 3) It should impute missing integers with 0 (FixedIntegerValueMissingCellHandlerFactory -> 0)
    assert re.search(r"\.fillna\(\s*0\s*\)", code), "Expected integer NA imputation to 0"
