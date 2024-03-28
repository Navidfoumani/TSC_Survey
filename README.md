# Time Series Classification and Extrinsic Regression Papers
Deep Learning for Time Series Classification and Extrinsic Regression: A Current Survey

#### ✨ **News:** This work has been accepted for publication in [ACM Computing Survey ](https://dl.acm.org/doi/10.1145/3649448).
## Citation
<p align="center">
    <img src="Taxonomy.png">
</p>

### CNN Models TSC/TSER

|Abbreviation| Title                                                                                 | Venue       | Year | Code |
|------------| --------------------------------------------------------------------------------------| ------------| ---- | ---- |
|MC-DCNN| [Time series classification using multi-channels deep convolutional neural networks](https://link.springer.com/chapter/10.1007/978-3-319-08010-9_33) | WAIM | 2014 | - |
|MC-CNN| [Deep convolutional neural networks on multichannel time series for human activity recognition](https://www.ijcai.org/Proceedings/15/Papers/561.pdf)                                                        | IJCAI | 2015 | - |
|-| [Convolutional neural networks for time series classification](https://ieeexplore.ieee.org/document/7870510)| J. Syst. Eng. Electron | 2017 | - |
|FCN| [Time series classification from scratch with deep neural networks: A strong baseline (FCN, ResNet, Disjoint-CNN)](https://ieeexplore.ieee.org/document/7966039)| IJCNN | 2017 | [code](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline) |
|Res-CNN| [Integration of residual network and convolutional neural network along with various activation functions and global pooling for time series classification](https://www.sciencedirect.com/science/article/abs/pii/S0925231219311506) | Neurocomputing | 2019 | - |
|DCNNs|[Multivariate Time Series Classification using Dilated Convolutional Neural Network](https://arxiv.org/abs/1905.01697)| arXiv | 2019 | - |
|Disjoint-CNN|[Disjoint-cnn for multivariate time series classification](https://ieeexplore.ieee.org/abstract/document/9679860)|ICDMW|2021|[code](https://github.com/Navidfoumani/Disjoint-CNN)|
|-|[Encoding time series as images for visual inspection and classification using tiled convolutional neural networks](http://coral-lab.umbc.edu/wp-content/uploads/2015/05/10179-43348-1-SM1.pdf) | AAAI | 2015 | - |
|-|[Classification of time-series images using deep convolutional neural networks](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10696/2309486/Classification-of-time-series-images-using-deep-convolutional-neural-networks/10.1117/12.2309486.short?SSO=1#_=_) |ICMV | 2017 | - |
|-| [Scalable classification of univariate and multivariate time series](https://ieeexplore.ieee.org/abstract/document/8621889) | IEEE Big Data | 2018 | - |
|-| [Classify multivariate time series by deep neural network image classification](https://ieeexplore.ieee.org/document/8901913) | CCHI | 2019 | - |
|RPMCNN| [A deep learning framework for time series classification using relative position matrix and convolutional neural network](https://www.sciencedirect.com/science/article/abs/pii/S0925231219308598) | Neurocomputing | 2019 | - |
|-| [Sensor classification using convolutional neural network by encoding multivariate time series as two-dimensional colored images](https://www.mdpi.com/1424-8220/20/1/168) | Sensors | 2019 | - |
|MCNN| [Multi-scale convolutional neural networks for time series classification](https://arxiv.org/abs/1603.06995)| arXiv | 2016 | - |
|t-LetNet| [Data augmentation for time series classification using convolutional neural networks](https://shs.hal.science/halshs-01357973/document) | AALTD | 2016 | - |
|MVCNN| [Time series classification with multivariate convolutional neural network](https://ieeexplore.ieee.org/document/8437249) | IEEE Trans. Ind. Electron | 2018 | - |
|-| [A cnn adapted to time series for the classification of supernovae](https://www.lirmm.fr/~chaumont/publications/IST_ELECTRONIC_IMAGING_Color_Imaging_2019_BRUNEL_PASQUET_RODRIGUEZ_COMBY_FOUCHEZ_CHAUMONT_Deep_Learning_Supernovae_Ia_vs_Not_Ia.pdf) | Electronic Imaging | 2019 | [code](https://github.com/anthonybrunel/SupernovaeClassification) |
|InceptionTime| [Inceptiontime: Finding alexnet for time series classification](https://link.springer.com/article/10.1007/s10618-020-00710-y) | DMKD | 2020 | [code](https://github.com/hfawaz/InceptionTime) |
|EEG-inception| [Prototypical inception network with cross branch attention for time series classification](https://ieeexplore.ieee.org/document/9533440) | IJCNN | 2021 | - |
|Inception-FCN| [Time Series Classification with InceptionFCN](https://www.mdpi.com/1424-8220/22/1/157)| Sensors | 2021 | - |
|KDCTime| [KDCTime: Knowledge distillation with calibration on InceptionTime for time-series classification](https://www.sciencedirect.com/science/article/abs/pii/S0020025522009434) | J. Inf. Sci | 2022 | - |
|LITE| [LITE: Light Inception with boosTing tEchniques for Time Series Classification](https://ieeexplore.ieee.org/document/10302569) | DSAA | 2023 | [code](https://github.com/MSD-IRIMAS/LITE) |

## Attention-based Models TSC/TSER
|Abbreviation| Title                                                                                 | Venue       | Year | Code |
|------------| --------------------------------------------------------------------------------------| ------------| ---- | ---- |
|MuVAN|[MuVAN: A Multi-view Attention Network for Multivariate Temporal Data](https://ieeexplore.ieee.org/document/8594896)|ICDM|2018|-|
|ChannelAtt|[A novel channel-aware attention framework for multi-channel eeg seizure detection via multi-view deep learning](https://ieeexplore.ieee.org/document/8333405)|IEEE EMBS- BHI|2018|-|
|GeoMAN |[Geoman: Multi-level attention networks for geo-sensory time series prediction](https://www.ijcai.org/Proceedings/2018/0476.pdf)|IJCAI|2018|[code](https://github.com/yoshall/GeoMAN)|
|Multi-Stage-Att|[Multistage attention network for multivariate time series prediction](https://www.sciencedirect.com/science/article/abs/pii/S0925231219316625)|Neurocomputing|2020|-|
|CT_CAM|[A novel channel and temporal-wise attention in convolutional networks for multivariate time series classification](https://ieeexplore.ieee.org/document/9269975)|IEEE Acesss |2020|-|
|CA-SFCN|[A new attention mechanism to classify multivariate time series](https://www.ijcai.org/Proceedings/2020/277)|IJCAI|2020|[code](https://github.com/huipingcao/nmsu_yhao_ijcai2020)|
|RTFN|[RTFN: A Robust Temporal Feature Network for Time Series Classification](https://www.sciencedirect.com/science/article/abs/pii/S0020025521003820)|Information Sciences|2021|-|
|LAXCAT|[Explainable Multivariate Time Series Classification: A Deep Neural NetworkWhich Learns To Attend To Important Variables AsWell As Informative Time Intervals](https://dl.acm.org/doi/10.1145/3437963.3441815)|ACM- WSDM|2021|-|
|MACNN|[Multi-scale attention convolutional neural network for time series classification](https://www.sciencedirect.com/science/article/abs/pii/S0893608021000010)|Neural Networks|2021|[code](https://github.com/wesley1001/MACNN)|
|WHEN|[WHEN: A Wavelet-DTW Hybrid Attention Network for Heterogeneous Time Series Analysis](https://dl.acm.org/doi/10.1145/3580305.3599549)|ACM KDD|2023|-|
|SAnD|[Attend and Diagnose: Clinical Time Series Analysis Using Attention Models](https://ojs.aaai.org/index.php/AAAI/article/view/11635)|AAAI|2018|-|
|T2|[Paying Attention to Astronomical Transients: Introducing the Time-series Transformer for Photometric Classification](https://arxiv.org/abs/2105.06178)|arXiv|2021|[code](https://github.com/tallamjr/astronet)|
|GTN|[Gated Transformer Networks for Multivariate Time Series Classification](https://arxiv.org/abs/2103.14438v1)|arXiv|2021|[code](https://github.com/ZZUFaceBookDL/GTN)|
|TRANS_tf|[An end-to-end framework combining time–frequency expert knowledge and modified transformer networks for vibration signal classification](https://www.sciencedirect.com/science/article/abs/pii/S0957417421000117) |Expert Syst. Appl.|2021|-|
|FMLA|[Rethinking Attention Mechanism in Time Series Classification](https://arxiv.org/abs/2207.07564)|arXiv|2022|-|
|AutoTransformer|[AutoTransformer: Automatic Transformer Architecture Design for Time Series Classification](https://link.springer.com/chapter/10.1007/978-3-031-05933-9_12)|PAKDD|2022|-|
|ConvTran|[Improving position encoding of transformers for multivariate time series classification](https://link.springer.com/article/10.1007/s10618-023-00948-2)|Data Min. Knowl. Discov|2023|[code](https://github.com/Navidfoumani/ConvTran)|

## Graph Neural Network Models TSC/TSER

|Abbreviation| Title                                                                                 | Venue       | Year | Code |
|------------| --------------------------------------------------------------------------------------| ------------| ---- | ---- |
| TGCN |[Temporal Graph Convolutional Networks for Automatic Seizure Detection](https://proceedings.mlr.press/v106/covert19a)| PMLR | 2019 | - |
| DGCNN |[EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks](https://ieeexplore.ieee.org/document/8320798)| IEEE Trans. Affect. Comput | 2020 | - |
| GraphSleepNet |[GraphSleepNet: Adaptive Spatial-Temporal Graph Convolutional Networks for Sleep Stage Classification](https://www.ijcai.org/Proceedings/2020/184)| IJCAI | 2020 | [code](https://github.com/jingwang2020/GraphSleepNet) |
| T-GCN |[A deep learning approach using graph convolutional networks for slope deformation prediction based on time-series displacement data](https://link.springer.com/article/10.1007/s00521-021-06084-6)|  Neural Computing and Applications | 2021 | - |
| MRF-GCN |[Multireceptive Field Graph Convolutional Networks for Machine Fault Diagnosis](https://ieeexplore.ieee.org/document/9280401)| IEEE Trans. Ind. Electron | 2021 | - |
| - |[Graph Convolutional Network For Generalized Epileptiform Abnormality Detection On EEG](https://ieeexplore.ieee.org/abstract/document/9672293)| SPMB | 2021 | - |
| DCRNN |[Self-Supervised Graph Neural Networks for Improved Electroencephalographic Seizure Analysi](https://iclr.cc/virtual/2022/poster/7027)| ICLR | 2022 | [code](https://github.com/tsy935/eeg-gnn-ssl) |
| Time2Graph+ |[Time2Graph+: Bridging Time Series and Graph Representation Learning via Multiple Attentions](https://ieeexplore.ieee.org/document/9477138)| IEEE Trans. Knowl | 2023 | [code](https://github.com/petecheng/Time2GraphPlus)|
| RAINDROP |[Graph-Guided Network for Irregularly Sampled Multivariate Time Series](https://iclr.cc/virtual/2022/poster/6409)| ICLR | 2022 | [code](https://github.com/mims-harvard/Raindrop) |
| STEGON |[Attentive Spatial Temporal Graph CNN for Land Cover Mapping From Multi Temporal Remote Sensing Data]()| IEEE Access | 2021 | - |
| - |[A deep graph neural network architecture for modelling spatio-temporal dynamics in resting-state functional MRI data](https://www.sciencedirect.com/science/article/pii/S1361841522001189)| Medical Image Analysis | 2022 | [code](https://github.com/tjiagoM/spatio-temporal-brain) |
| MTPool |[Multivariate time-series classification with hierarchical variational graph pooling](https://www.sciencedirect.com/science/article/abs/pii/S0893608022002970)| Neural Networks | 2022 | [code](https://github.com/RRRussell/MTPool) |
| SimTSC |[Towards Similarity-Aware Time-Series Classification](https://epubs.siam.org/doi/10.1137/1.9781611977172.23)| SIAM- SDM | 2022 | [code](https://github.com/daochenzha/SimTSC) |
| - |[Graph Neural Networks Extract High-Resolution Cultivated Land Maps From Sentinel-2 Image Series](https://ieeexplore.ieee.org/document/9803235)| IEEE Geosci | 2022 | - |
| C-DGAM |[Class-driven Graph Attention Network for Multi-label Time Series Classification in Mobile Health Digital Twins](https://ieeexplore.ieee.org/abstract/document/10234411)| IEEE J. Sel. Areas Commun | 2023 | - |
| - |[Graph Dynamic Earth Net: Spatio-Temporal Graph Benchmark for Satellite Image Time Series](https://ieeexplore.ieee.org/document/10281458)| IGARSS | 2023 | [code](https://github.com/corentin-dfg/graph-dynamic-earth-net) |
| TISER-GCN |[Graph neural networks for multivariate time series regression with application to seismic data](https://link.springer.com/article/10.1007/s41060-022-00349-6)| Int. j. data sci. anal | 2023 | [code](https://link.springer.com/article/10.1007/s41060-022-00349-6) |
| TodyNet |[TodyNet: Temporal Dynamic Graph Neural Network for Multivariate Time Series Classification](https://arxiv.org/abs/2304.05078)| arXiv | 2023 | [code](https://github.com/liuxz1011/TodyNet) |
| LB-SimTSC |[LB-SimTSC: An Efficient Similarity-Aware Graph Neural Network for Semi-Supervised Time Series Classification](https://arxiv.org/abs/2301.04838)| arXiv | 2023 | - |

## Self-supervised Models TSC/TSER
|Abbreviation| Title                                                                                 | Venue       | Year | Code |
|------------| --------------------------------------------------------------------------------------| ------------| ---- | ---- |
| TCL | [Unsupervised feature extraction by time-contrastive learning and nonlinear ica](https://proceedings.neurips.cc/paper_files/paper/2016/file/d305281faf947ca7acade9ad5c8c818c-Paper.pdf) | NIPS |2016 | -|
| T-Loss/SRL | [***Unsupervised Scalable Representation Learning for Multivariate Time Series](https://papers.neurips.cc/paper_files/paper/2019/hash/53c6de78244e9f528eb3e1cda69699bb-Abstract.html) | NeurIPS |2019 | [code](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries?tab=readme-ov-file)|
| TNC | [Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding](https://arxiv.org/abs/2106.00750) | arXiv |2021 |[code](https://github.com/sanatonek/TNC_representation_learning)|
| TS-TCC | [Time-Series Representation Learning via Temporal and Contextual Contrasting](https://www.ijcai.org/proceedings/2021/0324.pdf) | IJCAI |2021 | [code](https://github.com/emadeldeen24/TS-TCC)|
| MCL | [Mixing up contrastive learning: Self-supervised representation learning for time series](https://www.sciencedirect.com/science/article/pii/S0167865522000502) | Pattern Recognit Lett |2022 | -|
| TimeCLR | [TimeCLR: A self-supervised contrastive learning framework for univariate time series representation](https://www.sciencedirect.com/science/article/abs/pii/S0950705122002726) |  Knowl.-Based Syst |2022 | -|
| TS2Vec | [TS2Vec: Towards Universal Representation of Time Series](https://aaai.org/papers/08980-ts2vec-towards-universal-representation-of-time-series/) | AAAI |2022 | [code](https://github.com/zhihanyue/ts2vec)|
| BTSF | [Unsupervised Time-Series Representation Learning with Iterative Bilinear Temporal-Spectral Fusion](https://icml.cc/virtual/2022/spotlight/16052) | ICML |2022 | -|
| TF-C | [Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency](https://papers.nips.cc/paper_files/paper/2022/file/194b8dac525581c346e30a2cebe9a369-Paper-Conference.pdf) | NeurIPS |2022 | [code](https://github.com/mims-harvard/TFC-pretraining)|
| MHCCL | [MHCCL: Masked Hierarchical Cluster-Wise Contrastive Learning for Multivariate Time Series]() | AAAI |2023 | [code](https://github.com/mqwfrog/MHCCL)|
| BENDR | [BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data](https://pubmed.ncbi.nlm.nih.gov/34248521/) | Front Hum Neurosci |2021 | [code](https://github.com/SPOClab-ca/BENDR)|
| Voice2Series | [Voice2Series: Reprogramming Acoustic Models for Time Series Classification](https://proceedings.mlr.press/v139/yang21j.html) | PMLR |2021 | [code](https://github.com/huckiyang/Voice2Series-Reprogramming)|
| TST | [A Transformer-based Framework for Multivariate Time Series Representation Learning](https://dl.acm.org/doi/10.1145/3447548.3467401) | ACM KDD |2021 |-|
| TARNet | [TARNet: Task-Aware Reconstruction for Time-Series Transformer](https://dl.acm.org/doi/10.1145/3534678.3539329) | ACM KDD |2022 | [code](https://github.com/ranakroychowdhury/TARNet)|
| TimeMAE | [TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders](https://arxiv.org/abs/2303.00320) | arXiv |2023 | [code](https://github.com/Mingyue-Cheng/TimeMAE)|
| CRT | [Self-Supervised Time Series Representation Learning via Cross Reconstruction Transformer](https://ieeexplore.ieee.org/document/10190201) | IEEE Trans. Neural Netw. Learn. Syst |2023 | -|
| PHIT | [Finding Foundation Models for Time Series Classification with a PreText Task](https://arxiv.org/html/2311.14534v2) | arXiv |2023 | [code](https://github.com/msd-irimas/domainfoundationmodelstsc)|
| Series2Vec | [Series2Vec: Similarity-based Self-supervised Representation Learning for Time Series Classification](https://arxiv.org/abs/2312.03998) | arXiv |2023 | [code](https://github.com/Navidfoumani/Series2Vec)|


If you find **This Survey** useful for your research, please consider citing this paper using the following information:

````
```
@article{Survey24TS,
author = {Foumani, Navid Mohammadi and Miller, Lynn and Tan, Chang Wei and Webb, Geoffrey I. and Forestier, Germain and Salehi, Mahsa},
title = {Deep Learning for Time Series Classification and Extrinsic Regression: A Current Survey},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {0360-0300},
url = {https://doi.org/10.1145/3649448},
doi = {10.1145/3649448},
note = {Just Accepted},
journal = {ACM Comput. Surv.},
}

```
````

