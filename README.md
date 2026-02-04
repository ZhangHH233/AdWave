This is the implementation of the paper "AdWave: Adaptive Wavelet Attention with Multi-scale Fusion for Noise-Robust OCT Segmentation".

Abstract:
In ophthalmic clinical practice, accurate segmentation of anatomical layers and pathological lesions in Optical Coherence Tomography (OCT) is essential for identifying structural abnormalities. However, the inherent speckle noise in OCT images introduces stochastic intensity variations that obscure fine-grained structural details and degrade segmentation accuracy. To address these challenges, we propose Ad-Wave, a novel segmentation framework that adaptively enhances structural discriminability and suppresses speckle interference through frequency-spatial collaborative learning. Specifically, we introduce an Adaptive Wavelet Attention (AWA) module that selectively attenuates noise-sensitive high-frequency components, while enhancing relevant structural cues that guide the model's attention to target regions; and a Multi-scale Dual-domain Fusion Skip Connection (MDF) that aggregates complementary frequency and spatial features across scales to reduce boundary ambiguity and improve contextual understanding. Comprehensive evaluations on GOALS (healthy/glaucoma) and DUKEBOE (diabetic macular edema) datasets validate AdWave's effectiveness. The results demonstrated superior performance in retinal layer and lesion segmentation, along with strong robustness under speckle noise interference. Code is available at https://github.com/ZhangHH233/AdWave.

@INPROCEEDINGS{11356869,
  author={Huihong Zhang and Sanqian Li  and Bing Yang, Xiaoqing Zhang and Risa Higashita and Jiang Liu},
  booktitle={2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={AdWave: Adaptive Wavelet Attention with Multi-Scale Fusion for Noise-Robust OCT Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={6323-6330},
  keywords={Noise;Image segmentation;Speckle;Discrete wavelet transforms;Accuracy;Lesions;Frequency-domain analysis;Decoding;Image edge detection;Wavelet analysis;OCT;retinal layer segmentation;lesion segmentation;speckle noise;discrete wavelet transform;multi-scale},
  doi={10.1109/BIBM66473.2025.11356869}}
