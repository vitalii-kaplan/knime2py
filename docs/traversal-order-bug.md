# Traversal Order Bug: When a Consumer Appeared Before Its Input

This note describes a traversal bug found while converting the `INZ_visa_decisions_combined` workflow. It is a useful example because the generated Python looked structurally correct at the node-comment level, but the linear execution order was wrong.

## What Happened

In the generated workbook, `Column Appender (#1556)` had three inputs:

* `1547:1` from `Joiner (#1547)`
* `1551:1` from `Joiner (#1551)`
* `1592:1` from `Joiner (#1592)`

That dependency list was correct. It matched the KNIME graph.

The problem was the position of the generated code. `Column Appender (#1556)` appeared before `Joiner (#1547)`. The generated Python tried to read:

```python
context["1547:1"]
```

before `Joiner (#1547)` had written that value.

So the graph metadata was correct, but the linearized workbook order was not dependency-safe.

## Why This Workflow Exposed the Bug

Earlier workflows were mostly simple chains, forks, or joins where the traversal order happened to stay valid. This workflow had a more demanding structure:

* multiple root-like input branches,
* three Joiner nodes feeding one Column Appender,
* one shared input node feeding several downstream nodes,
* a consumer node that required all three join outputs before it could run.

The important difference is the three-input `Column Appender (#1556)`. It forced the traversal to prove that all predecessors had actually been emitted, not merely discovered or temporarily visited.

The previous depth-first traversal could enter one dependency branch, encounter another branch through a shared predecessor, and then treat an unfinished node as if it was already handled because it was on the recursion stack. That was enough to let `Column Appender (#1556)` appear before `Joiner (#1547)`.

## Why It Happened

The old traversal was based on depth-first search with a recursion-stack guard.

Depth-first search is useful for graph exploration, but it is not automatically a safe linear execution order for generated code. A code generator needs a stronger guarantee:

> A node must be emitted only after every upstream node that produces its inputs has been emitted.

In this case, the DFS recursion-stack guard avoided infinite recursion, but it also hid the fact that one predecessor was still unfinished.

## How It Was Fixed

The traversal was changed from DFS-style ordering to deterministic topological ordering.

The current implementation uses Kahn's algorithm with stable numeric node-id tie-breaking. For acyclic workflow graphs, this guarantees that every node is emitted after all of its predecessors. If a cycle is ever present, remaining cyclic nodes are still emitted deterministically, but a fully dependency-ready order is only guaranteed for the acyclic part.

The fix was added in:

* `src/knime2py/traverse.py`
* `tests/test_traverse_order.py`

The regression test models the same shape as the failing workflow: three upstream nodes feeding one downstream consumer. The test asserts that all three Joiner-like predecessors are ordered before the Column-Appender-like consumer.

## Result

After the fix, the generated workbook order became:

1. `Joiner (#1547)`
2. `Joiner (#1551)`
3. `Joiner (#1592)`
4. `Column Appender (#1556)`

The generated Python now follows the KNIME data-flow dependencies.

More detailed algorithm notes can be added later. For now, the key point is that graph traversal for inspection and graph linearization for execution are related problems, but they are not the same problem.

## Follow-up: When Topological Order Is Not Enough

The `HW_Churn_test` workflow exposed a second, more subtle point.

After switching from DFS ordering to topological ordering, normal table-flow dependencies were handled better. However, the workflow output changed for some machine-learning branches, including Decision Tree and Logistic Regression results. At first this looked strange: changing traversal order should not change model results if the graph dependencies are preserved.

The reason is that `HW_Churn_test` is not only a table-flow graph. It contains several parallel X-validation branches:

* `X_Partitioner`
* learner nodes
* predictor nodes
* `X_Aggregator`
* scorer and writer nodes

In the current emitter, `X_Partitioner` opens a Python `for` loop, and `X_Aggregator` closes the indentation level. That means traversal order is also being used to infer Python control-flow structure.

The old DFS order accidentally kept each X-validation branch mostly contiguous:

```text
X_Partitioner -> sampler -> learner -> predictor -> X_Aggregator -> scorer
```

The new topological order is dependency-correct for ordinary DAG nodes, but it may interleave independent validation branches:

```text
X_Partitioner #60 -> LR nodes -> X_Partitioner #61 -> DT nodes -> X_Aggregator #62
```

That sequence is valid as a plain graph order, but it is not valid for the current indentation-based loop emission. It can accidentally nest one cross-validation loop inside another. It can also make an `X_Aggregator` bind to the wrong active loop state.

So the model outputs did not change because Decision Tree or Logistic Regression became different algorithms. They changed because the generated Python control flow became different.

This shows an important boundary:

> Topological sorting solves dependency ordering for ordinary nodes, but it does not by itself solve structured control-flow generation.

For workflows with loop nodes, the generator needs an additional layer. Possible directions were:

* loop-aware traversal,
* structured control-flow scheduling,
* region-based graph linearization,
* collapsing each `X_Partitioner ... X_Aggregator` pair into a block before ordering the outer graph.

## Implemented Change: Structured X-Validation Regions

The implemented fix is the first version of loop-aware traversal.

The traversal now treats each `X_Partitioner ... X_Aggregator` span as a structured region. The region starts at the `X_Partitioner`, follows the downstream graph, and stops at the matching `X_Aggregator`. Nodes after the aggregator, such as scorer and writer nodes, remain outside the region.

The scheduler still uses deterministic topological ordering as the baseline. That preserves the fix for ordinary multi-input nodes, including consumers with three or more incoming flows. A node like `Column Appender (#1556)` cannot be emitted until all of its upstream Joiners have been emitted.

The additional rule is applied only when an X-validation region is open. After an `X_Partitioner` is selected, the scheduler prioritizes dependency-ready nodes inside that partitioner region until it reaches the `X_Aggregator`. If no node inside the region is ready, it may emit ordinary prerequisite work, but it avoids starting another independent `X_Partitioner` while the current region is open.

This keeps generated Python loop bodies contiguous:

```text
X_Partitioner -> loop body nodes -> X_Aggregator -> scorer/writer
```

instead of allowing independent validation regions to interleave:

```text
X_Partitioner A -> nodes from A -> X_Partitioner B -> nodes from B -> X_Aggregator A
```

The important point is that this is not a replacement for topological sorting. It is a layer on top of it. Topological ordering handles data dependencies, including three-input and larger joins. Structured-region scheduling handles the Python control-flow shape required by KNIME loop nodes.

This is a separate problem from the original three-input `Column Appender` bug. The `INZ_visa_decisions_combined` workflow needed correct dependency ordering. The `HW_Churn_test` workflow needs correct dependency ordering plus preservation of control-flow regions.
