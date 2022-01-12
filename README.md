# RMCosSSL and RLMSoftmaxSSL losses for training semi-supervied taxonomy model based on GAN

## Relativistic Marin Cosine semi-supervised Learning (RMCosSSL)
- Move to `rmcos-gan-ssl` folder
- Train the model when changing the amount of labeled data [50, 100, 600, 1000, 3000]

  `python run.py --epoch 40 --change_label True`

- Train the model when changing the values of alpha [0.1, 0.3, 0.5, 0.7, 0.9]

  `python run.py --epoch 40 --change_alpha True`

## Relativistic Large Marin Softmax semi-supervised Learning (RLMSoftmaxSSL)
- Move to `rlmsoftmax-gan-ssl` folder
- Train the model when changing the amount of labeled data [50, 100, 600, 1000, 3000]

  `python run.py --epoch 40 --change_label True`
  
- Train the model when changing the values of alpha [0.1, 0.3, 0.5, 0.7, 0.9]

  `python run.py --epoch 40 --change_alpha True`

