# Read Me

**Paper:** Boosting Logical Fallacy Reasoning in LLMs via Logical Structure Tree<br/>
**Accepted:** Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang<br/>
**Paper Link:** https://aclanthology.org/2024.emnlp-main.730/

<br/>

## Task Description

The paper works logical fallacy detection and classification.

<br/>

## Dataset Description

We use four datasets for experiments: **Argotario** (Habernal et al., 2017), **Reddit** (Sahai et al., 2021), **Climate** (Alhindi et al., 2022), **Logic** (Jin et al., 2022). The datasets after processing and their train/dev/test splits are saved in the _processed_datasets_ folder.

<br/>

## Code Description

* **construct_logical_structure_tree.py**: the code for constructing the logical structure tree
