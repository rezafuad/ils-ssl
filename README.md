<h1 align="center"> Intermittent Learning Strategy via Sketch-Visible Person Reidentification </h1>

Implementation of our paper entitled "Intermittent Learning Strategy via Sketch-Visible Person Reidentification."

## Results

PKU-Sketch-ReID Dataset
| Methods | mAP | R@1 | R@5 | R@10 |
|---------|-----|-----|-----|------|
| MSIF [1] |91.12|87.00|96.80|98.70|
| DALNet [2] |86.20|90.00|98.60|100.0|
| AIO [3] |93.70|93.80| - | - |
| **SV2L-SSL-Res384 (Our)** |90.12|94.00|99.60|100.0|

MaSk1K Dataset
| Methods | mAP | R@1 | R@5 | R@10 |
|---------|-----|-----|-----|------|
| DEEN [4] |12.62|12.11|25.44|30.94|
| BDG [5] |19.61|18.10|38.95|50.75|
| **SV2L-SSL (Our)** |29.72|33.42|58.23|69.01|

## Model Weight

will be released soon

## References

1. Q. Chen, Z. Quan, Y. Zheng, Y. Li, Z. Liu, M.G. Mozerov. *MSIF: Multi-Spectrum Image Fusion Method for Cross-Modality Person Re-identification*, International Journal of Machine Learning and Cybernetics, 2024.
2. X. Liu, X. Cheng, H. Chen, H. Yu, G. Zhao. *Differentiable Auxiliary Learning for Sketch Re-identification*, Proceedings of AAAI Conference on Artificial Intelligence, 2024.
3. H. Li, M. Ye, M. Zhang, B. Du. *All in One Framework for Multimodal Re-identification in The Wild*, Proceedings of CVPR, 2024.
4. Y. Zhang, H. Wang. *Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification*, Proceedings of CVPR 2023.
5. K. Lin, Z. Wang, Z. Wang, Y. Zheng, D. Satoh. *Beyond Domain Gap: Exploiting Subjectivity in Sketch-based Person Retrieval*, Proceedings of ACM Multimedia 2023. 

## Acknowledgement

This repository is based on [layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) 
