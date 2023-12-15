# Recursive-Generalization-Transformer
Unoffical implementation of the paper https://openreview.net/pdf?id=owziuM1nsR.
This architecture is specialized for the Super Resolution domain. 

# Implemntation for x2 factor
We only used [DIV2k](https://www.kaggle.com/datasets/joe1995/div2k-dataset/code) dataset instead of [DIV2K](https://www.kaggle.com/datasets/joe1995/div2k-dataset/code) and [Flickr2k](https://www.kaggle.com/datasets/hliang001/flickr2k) since the resource for computation is limited. For the same reason, we reduced the number of Residual Blocks (RG) from 6 to 2, but kept the same number of Transformer blocks as 12 per RG block. The model was trained for 1 day on 1 GPU NVIDIA A100 40GB.

![Capture d’écran 2023-12-15 à 15 03 27](https://github.com/nhs2828/Recursive-Generalization-Transformer_/assets/78078713/46fb8b05-fc9d-4da4-9b6b-d53ea6dd5bb2)




# Highlights of RGT'sarchitecture:
- Rectangle-window self-attention in L-SA block [Rwin-SA](https://arxiv.org/pdf/2211.13654.pdf)
- Recursive-generalization self-attention (RG-SA block), this allows each token in the input image features can obtain a global receptive field by aggregating the information of the whole image features.
- Hybrid adaptive intergration, this learnable parameter encourages more information flows to the deep network layers.
  
![Capture d’écran 2023-12-15 à 15 04 38](https://github.com/nhs2828/Recursive-Generalization-Transformer_/assets/78078713/c60b98c4-34f4-4973-ab69-da36c7a70876)


# Results
After 5600 itérations, the model yields 39.036 PSNR (Peak Signal to Noise Ratio) score and 0.9334 SSIM (Structural similarity index measure) score.

<figcaption>Reconstructed images from Low resolution</figcaption>

![Capture d’écran 2023-12-15 à 15 06 19](https://github.com/nhs2828/Recursive-Generalization-Transformer_/assets/78078713/b27f911a-81f1-4ebd-8922-8443f0c3e65c)


<figcaption>True High Resolution</figcaption>

![Capture d’écran 2023-12-15 à 15 06 37](https://github.com/nhs2828/Recursive-Generalization-Transformer_/assets/78078713/519d3435-ddbf-4d49-92e5-a0ec7749311d)
