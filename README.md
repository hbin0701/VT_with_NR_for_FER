# Vision Transformer Equipped with Neural Resizer on Facial Expression Recognition Task 
---
Paper: https://arxiv.org/abs/2204.02181

**Pytorch implemenatation of Vision Transformer Equipped with Neural Resizer on Facial Expression Recognition Task**

Hyeonbin Hwang, Soyeon Kim, Wei-Jin Park, Jiho Seo, Kyungtae Ko, Hyeon Yeo <br/>
KAIST and Acryl, Republic of Korea

**Accepted to IEEE ICASSP 2022.**


**Abstract**: When it comes to wild conditions, Facial Expression Recognition is often challenged with low-quality data and imbalanced, ambiguous labels. This field has much benefited from CNN based approaches; however, CNN models have structural limitation to see the facial regions in distant. As a remedy, Transformer has been introduced to vision fields with global receptive field, but requires adjusting input spatial size to the pretrained models to enjoy their strong inductive bias at hands. We herein raise a question whether using the deterministic interpolation method is enough to feed low-resolution data to Transformer. In this work, we propose a novel training framework, Neural Resizer, to support Transformer by compensating information and downscaling in a data-driven manner trained with loss function balancing the noisiness and imbalance. Experiments show our Neural Resizer with F-PDLS loss function improves the performance with Transformer variants in general and nearly achieves the state-of-the-art performance.


##### Prepare Dataset 📙
1. First  downlaod images from [FER+](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
2. We have used [DIC](https://github.com/Maclory/Deep-Iterative-Collaboration) model for SR module of our paper, so apply super-resolution **(x4)** to each image with the pretrained model of `DIC_HELEN.pth`.
3. Finally, put the SR images accordingly in the ``/data/train`` and ``/data/test`` directory.


##### Run Experiment 💻
After, Change accordingly `config.json` to customize the setting for the test.
Current setting is the one that yielded the best result on ``FERPlus``, with ``89.50%.``

Simply run below command:
```
python test.py -c config.json
```

and Results and model checkpoint will be saved in **.txt file** format in `/experiments/{exp_name}` folder.

