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
* **llama_identify_baseline.py**: the code for the baseline of logical fallacy detection using Llama-2 model
* **llama_identify_structure.py**: the code for logical fallacy detection via logical structure tree using Llama-2 model
* **llama_classify_baseline.py**: the code for the baseline of logical fallacy classification using Llama-2 model
* **llama_classify_structure.py**: the code for logical fallacy classification via logical structure tree using Llama-2 model
* **flan_t5_identify_baseline.py**: the code for the baseline of logical fallacy detection using Flan_T5 model
* **flan_t5_identify_structure.py**: the code for logical fallacy detection via logical structure tree using Flan_T5 model
* **flan_t5_classify_baseline.py**: the code for the baseline of logical fallacy classification using Flan_T5 model
* **flan_t5_classify_structure.py**: the code for logical fallacy classification via logical structure tree using Flan_T5 model


<br/>


## Citation

If you are going to cite this paper, please use the form:

Yuanyuan Lei and Ruihong Huang. 2024. Boosting Logical Fallacy Reasoning in LLMs via Logical Structure Tree. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13157–13173, Miami, Florida, USA. Association for Computational Linguistics.

```bibtex
@inproceedings{lei-huang-2024-boosting,
    title = "Boosting Logical Fallacy Reasoning in {LLM}s via Logical Structure Tree",
    author = "Lei, Yuanyuan  and
      Huang, Ruihong",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.730",
    pages = "13157--13173",
    abstract = "Logical fallacy uses invalid or faulty reasoning in the construction of a statement. Despite the prevalence and harmfulness of logical fallacies, detecting and classifying logical fallacies still remains a challenging task. We observe that logical fallacies often use connective words to indicate an intended logical relation between two arguments, while the argument semantics does not actually support the logical relation. Inspired by this observation, we propose to build a logical structure tree to explicitly represent and track the hierarchical logic flow among relation connectives and their arguments in a statement. Specifically, this logical structure tree is constructed in an unsupervised manner guided by the constituency tree and a taxonomy of connectives for ten common logical relations, with relation connectives as non-terminal nodes and textual arguments as terminal nodes, and the latter are mostly elementary discourse units. We further develop two strategies to incorporate the logical structure tree into LLMs for fallacy reasoning. Firstly, we transform the tree into natural language descriptions and feed the textualized tree into LLMs as a part of the hard text prompt. Secondly, we derive a relation-aware tree embedding and insert the tree embedding into LLMs as a soft prompt. Experiments on benchmark datasets demonstrate that our approach based on logical structure tree significantly improves precision and recall for both fallacy detection and fallacy classification.",
}

```











