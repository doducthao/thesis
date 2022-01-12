# RMCosSSL and RLMSoftmaxSSL for training model semi-supervied learning base on GAN

## Relativistic Marin Cosine semi-supervised Learning (RMCosSSL)
- Move to `rmcos-gan-ssl` folder
- To train with change number of labeled data in [50, 100, 600, 1000, 3000]
python run.py --epoch 40 --change_label True

- To train with change values of alpha in [0.1, 0.3, 0.5, 0.7, 0.9]
python run.py --epoch 40 --change_alpha True

## Relativistic Large Marin Softmax semi-supervised Learning (RLMSoftmaxSSL)
- Move to `rlmsoftmax-gan-ssl` folder
- To train with change number of labeled data in [50, 100, 600, 1000, 3000]
python run.py --epoch 40 --change_label True

- To train with change values of alpha in [0.1, 0.3, 0.5, 0.7, 0.9]
python run.py --epoch 40 --change_alpha True
