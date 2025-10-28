# tests/test_nodeblocks_partitioning.py
"""
Test for the Partitioning NodeBlock in the knime2py generator.

Overview
----------------------------
This module tests the generation of code for the Partitioning NodeBlock, which is part of the 
knime2py pipeline. It verifies that the generated code correctly implements a stratified 
train/test split based on the settings defined in the provided settings.xml file.

Runtime Behavior
----------------------------
Inputs:
- Reads a DataFrame from the context, specifically the 'Dependent' column for stratification.

Outputs:
- Writes to context keys for train and test DataFrames, mapping them to the appropriate ports:
  - `context['4001:1']` for the training DataFrame.
  - `context['4001:2']` for the testing DataFrame.

Key algorithms:
- Utilizes `train_test_split` from sklearn for stratified sampling based on the 'Dependent' column.

Edge Cases
----------------------------
The code handles cases such as:
- Empty or constant columns in the input DataFrame.
- NaN values in the 'Dependent' column, replacing them with a placeholder.
- Class imbalance by ensuring stratified sampling.

Generated Code Dependencies
----------------------------
The generated code requires the following external libraries:
- pandas
- sklearn
These dependencies are required by the generated code, not by this test module.

Usage
----------------------------
This module is typically invoked by the emitter during the generation of the KNIME workflow. 
An example of expected context access is:
```python
train_df = context['4001:1']
test_df = context['4001:2']
```

Node Identity
----------------------------
The file generates code based on the settings.xml for the Partitioning node:
- KNIME factory id: `org.knime.base.node.preproc.partition.PartitionNodeFactory`
- Special flags: None.

Configuration
----------------------------
The settings are parsed using the `parse_smote_settings` function, which extracts values 
from the settings.xml file. Important fields include:
- `seed`: The random seed for reproducibility (default: 1).
- `fraction`: The fraction of data to be used for training (default: 0.7).

Limitations
----------------------------
Currently, the implementation does not support advanced options available in KNIME, such as 
custom stratification logic or multiple input DataFrames.

References
----------------------------
For more information, refer to the KNIME documentation on the Partitioning node and 
stratified sampling techniques.
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


def test_partitioning_block_emits_stratified_train_lines(node_csv_reader_dir: Path):
    """
    Test that the Partitioning NodeBlock emits the expected stratified train/test split code.
    
    This test builds a minimal workflow graph consisting of a CSV Reader node and a Partitioning node.
    It verifies that the generated code for the Partitioning NodeBlock correctly implements the 
    stratified train/test split based on the settings defined in the provided settings.xml file.
    
    Args:
        node_csv_reader_dir (Path): The directory containing the CSV reader node.
    """
    node_partitioning_dir = repo_root / "tests" / "data" / "Node_partitioning"
    assert node_partitioning_dir.joinpath("settings.xml").exists(), "Missing Partitioning settings.xml test data"

    # Node ids & types
    reader_id = "1393"
    part_id = "4001"
    reader_type = "org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory"
    part_type = "org.knime.base.node.preproc.partition.PartitionNodeFactory"

    # Minimal workflow graph
    g = WorkflowGraph(
        workflow_id="test_partitioning_block",
        workflow_path=str((repo_root / "tests" / "data" / "dummy" / "workflow.knime").resolve()),
        nodes={
            reader_id: Node(id=reader_id, name="CSV Reader", type=reader_type, path=str(node_csv_reader_dir)),
            part_id: Node(id=part_id, name="Partitioning", type=part_type, path=str(node_partitioning_dir)),
        },
        edges=[Edge(source=reader_id, target=part_id, source_port="1", target_port="1")],
    )

    blocks, _imports = build_workbook_blocks(g)
    assert blocks, "Expected NodeBlocks to be created"

    # Find the Partitioning block
    p_block = next((b for b in blocks if b.nid == part_id), None)
    assert p_block is not None, "Partitioning NodeBlock not found"

    code = "\n".join(p_block.code_lines)

    # Seed and fraction derived from settings.xml
    assert "_seed = 1" in code
    assert "_frac = 0.7" in code

    # Stratification helper column on 'Dependent'
    assert "_y = df['Dependent'].astype('object').where(pd.notna(df['Dependent']), '__NA__')" in code

    # Primary stratified split line (inside try)
    assert "train_test_split(df, train_size=_frac, random_state=_seed, stratify=_y)" in code

    # Fallback split (without stratify) in except
    assert "train_test_split(df, train_size=_frac, random_state=_seed)" in code

    # Port assignment rule: train_df to the numerically smaller port, test_df to the other
    assert f"context['{part_id}:1'] = train_df" in code
    assert f"context['{part_id}:2'] = test_df" in code
