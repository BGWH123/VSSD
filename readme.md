# Visual Saliency Steering Distillation for Multimodal Chain-of-Thought Reasoning

## Environment
```bash
pip install -r requirements.txt
```

## Hyper-parameters Anlysis
### 1. Analysis of $\alpha$
<img src="https://raw.githubusercontent.com/BGWH123/VSSD/main/image/all_vs_alpha.png" width="600" alt="Analysis of alpha">

In the proposed Visual Saliency-guided Steering Distillation (VSSD) framework, the hyperparameter α controls the strength with which the steering vector—derived from the feature difference between the original and perturbed images—is injected into the model’s intermediate layers. Experimental results and the accompanying analysis plot demonstrate that α significantly affects model performance. Specifically, α = 0.2 yields the best overall accuracy on the ScienceQA dataset, along with peak performance across most subcategories such as NAT, TXT, and G1–6, indicating an optimal balance between enhancing sensitivity to critical visual semantic differences and preserving the model’s intrinsic semantic stability. When α is too small (e.g., 0.1), the steering signal is insufficient to effectively activate fine-grained discriminative capabilities. Conversely, when α is too large (e.g., ≥0.3), the excessive signal distorts feature representations, notably degrading performance on complex categories like SOC. Therefore, α = 0.2 is established as the optimal setting, underscoring the importance of precise calibration of steering intensity in this distillation paradigm.


### 2. Analysis of $\beta$
<img src="https://raw.githubusercontent.com/BGWH123/VSSD/main/image/rouge_vs_beta.png" width="600" alt="Analysis of beta">
