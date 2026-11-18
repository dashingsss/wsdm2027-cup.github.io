# Third-party sources and licenses

Agentic Search builds on ALFWorld, WebShop, and HotpotQA. We selected development tasks, generated or collected trajectories, and standardized the released records. Runtime resources were prepared as described below. Please acknowledge the original projects when using their materials.

| Dataset | Sources and paper | Preparation in this project |
| --- | --- | --- |
| ALFWorld | [Official project](https://github.com/alfworld/alfworld); Shridhar et al., [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768), ICLR 2021. | Official games and logic files, selected development tasks, and generated or collected action–observation trajectories. |
| WebShop | [Official project](https://github.com/princeton-nlp/WebShop); Yao et al., [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206), NeurIPS 2022. Raw files were obtained from the [versioned data mirror](https://huggingface.co/datasets/HongbangYuan/webshop/tree/0129d4a81dbdb827e76afd20a1e2c38b61098613). | Product records converted to SQLite, a prepared search index, fixed price and task mappings, selected development tasks, and standardized trajectories. The upstream reward function is unchanged. |
| HotpotQA | [Official project and data](https://hotpotqa.github.io/); [dataset distribution](https://huggingface.co/datasets/hotpotqa/hotpot_qa); Yang et al., [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/), EMNLP 2018. | Article passages from the training and development contexts collected into a local SQLite search database, selected development tasks, and generated and cleaned trajectories. This is a fixed document collection, not the full Wikipedia corpus. |

## License texts

- **ALFWorld upstream code:** [MIT License](ALFWorld-MIT.txt), copyright Mohit Shridhar and others.
- **WebShop upstream code:** [MIT License](WebShop-MIT.txt), copyright Princeton Natural Language Processing. The included text matches the source revision used by this project.
- **HotpotQA data and processed Wikipedia:** [CC BY-SA 4.0](CC-BY-SA-4.0.txt), as stated by the [original project](https://hotpotqa.github.io/). Our adaptations of those materials are shared under the same license; the processing changes are described above. See the [license summary](https://creativecommons.org/licenses/by-sa/4.0/) for attribution and sharing requirements.

These notices apply to the corresponding materials, rather than assigning a single license to the entire project. The original license texts are reproduced without modification.
