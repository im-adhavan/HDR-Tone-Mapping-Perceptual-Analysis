# HDR Tone Mapping Perceptual Analysis

A GPU-accelerated experimental framework for analysing how radiometric structure of HDR scenes influences tone-mapping instability, exposure robustness, and optimal parameter selection.

---

## 1. Motivation

High Dynamic Range (HDR) imaging is now standard in computational photography and display pipelines. However, tone-mapping operators (TMOs) remain largely heuristic and often behave unpredictably across diverse scenes.

This project investigates:

- Do HDR scenes exhibit structured radiometric geometry?
- Can tone-mapping instability be predicted from scene statistics?
- How robust are tone-mapping operators to exposure shifts?
- Can optimal tone-mapping parameters be predicted adaptively?

Rather than evaluating operators qualitatively, this framework models tone mapping as a measurable, statistically analyzable system.

---

## 2. Dataset

- 105 HDR scenes
- Generated from multi-exposure RAW (.NEF) stacks
- Merged into linear EXR radiance maps
- Stored in: `data/hdr_exr/`

Each scene is processed as full floating-point radiance data (no SDR compression).

---

## 3. Radiometric Scene Descriptors

For each HDR scene, the following descriptors are computed:

- **Dynamic Range** (log10 percentile-based)
- **Log-luminance standard deviation**
- **Highlight energy ratio**
- **Shadow mass ratio**
- **Skewness of luminance distribution**
- **Kurtosis of luminance distribution**

These descriptors characterize the geometric structure of scene luminance and form the basis for predictive modeling.

---

## 4. Tone Mapping Operators

Implemented as parametric GPU operators:

- Reinhard (global)
- Reinhard (local Gaussian surround)
- Drago
- Filmic
- Gamma baseline

All operators are applied in floating-point on GPU using PyTorch.

---

## 5. Perceptual Stability Metrics

For each HDR → LDR mapping, we compute:

- Contrast loss (log-domain compression)
- Clipping ratio
- Entropy
- Composite Stability Index:

  Stability = |ContrastLoss| + Clipping + Entropy

Higher stability index indicates stronger perceptual degradation.

---

## 6. Statistical Analysis

### 6.1 Multivariate Regression

We model:

Scene Radiometric Features → Tone Mapping Instability

Observed:

- Multivariate R² = 0.41  
- Cross-validated R² = 0.37  

Interpretation:

Radiometric geometry explains approximately 37–41% of instability variance after cross-validation. Tone-mapping behaviour is therefore partially predictable from measurable scene structure.

---

### 6.2 PCA of Radiometric Features

- PC1 ≈ 62%
- PC2 ≈ 27%
- ~90% variance explained by 2 components

HDR scenes lie on a low-dimensional radiometric manifold, indicating structured luminance geometry rather than arbitrary distribution.

---

### 6.3 Scene Clustering

Unsupervised clustering reveals distinct scene classes:

- Highlight-dominated
- Shadow-heavy
- Balanced mid-tone scenes

Operator instability varies systematically across these clusters.

---

## 7. Exposure Robustness Analysis

Operators are evaluated under exposure shifts:

- −2 EV
- −1 EV
- 0 EV
- +1 EV
- +2 EV

Define:

Exposure Sensitivity Index (ESI) = std(stability across EV shifts)

Findings:

- **Most exposure-robust operator:** Drago  
- **Most exposure-sensitive operator:** Gamma  
- Local operators show improved robustness over global variants.

Exposure robustness varies systematically with highlight geometry and luminance spread.

---

## 8. Optimal Parameter Search

For the Reinhard exposure parameter:

- Grid search per scene
- Minimize stability index
- Compute optimal exposure per scene

We then model:

Scene Features → Optimal Exposure

Results indicate a measurable correlation between tone-mapping compression strength and radiometric scene geometry.

---

## 9. Adaptive Prediction

Regression model:

Radiometric Descriptors → Optimal Reinhard Exposure

Outputs:

- Coefficient analysis
- Cross-validated R²
- Prediction scatter plot

This establishes feasibility of scene-aware adaptive tone mapping.

---

## 10. Experimental Results Snapshot

- Multivariate R²: 0.41  
- Cross-validated R²: 0.37  
- PCA variance explained (PC1 + PC2): ~90%  
- Most exposure-robust operator: Drago  
- Most exposure-sensitive operator: Gamma  

These results demonstrate that tone-mapping instability and exposure robustness are partially determined by measurable radiometric geometry.

---

## 11. Architecture

src/
├── dataset_loader.py  
├── gpu_utils.py  
├── tone_mapping.py  
├── perceptual_metrics.py  
├── scene_features.py  
├── operator_analysis.py  
├── exposure_analysis.py  
├── optimization.py  
├── adaptive_prediction.py  
└── advanced_analysis.py  

scripts/  
└── run_analysis.py  

All experiments are orchestrated via:

python -m scripts.run_analysis

---

## 12. Reproducibility

Create environment:

python -m venv hdr_env  
hdr_env\Scripts\activate  
python -m pip install -r requirements.txt  

Run full experimental pipeline:

python -m scripts.run_analysis  

Outputs stored in:

results/

Includes:

- Full results CSV
- Stability ranking
- Statistical tests
- PCA plot
- Clustering plot
- Exposure sensitivity metrics
- Optimal parameter search results
- Adaptive prediction metrics

---

## 13. Limitations

- Perceptual model is luminance-based (no full HVS modeling)
- No psychophysical validation
- No display-dependent modeling
- No learned tone-mapping operators

---

## 14. Future Directions

- Chromaticity distortion modeling
- Perceptual color constancy integration
- Local adaptive parameter prediction
- Neural surrogate modeling
- Scene-aware tone-mapping networks

---

## 15. Conclusion

This framework demonstrates that HDR radiometric structure significantly influences tone-mapping behavior, instability, and parameter sensitivity.

Rather than treating tone mapping as a black-box heuristic, this work models it as a predictable function of scene geometry and provides a structured foundation for adaptive, scene-aware HDR compression systems.

---

Author: Adhavan Murugaiyan
