## Case 1

Question:How does the azimuth-controllable generative adversarial network control the azimuth angle of generated SAR target images?

Retrieval Query:How does the azimuth-controllable generative adversarial network control the azimuth angle of generated SAR target images?

Top-k:5

Answer:
Based on the retrieved content, the azimuth-controllable generative adversarial network (GAN) controls the azimuth angle of generated SAR target images by taking two SAR images with given azimuths as inputs and generating a SAR image with an intermediate azimuth between the two input images’ azimuths [1][3][4].

Specifically, the generator uses a dual-input parallel SAR image input block to extract and fuse the optimal target features from the two input SAR images [1][3]. The azimuth of the generated SAR image is then controlled by the azimuths of the two input images [4]. The network is described as containing a generator, discriminator, and predictor [3], but the retrieved content does not provide further details on the exact role of the predictor or the precise internal mechanism for azimuth interpolation.

Citation check:
正确


## Case 2

Question:What dataset does it use?

Retrieval Query:
What dataset does the azimuth-controllable generative adversarial network use for SAR target image generation?

Top-k:5

Answer:
根据当前论文知识库中的检索内容，无法可靠回答该问题。检索内容中仅提及了 SAR 图像切片（如从原始 128 × 128 SAR 图像中随机采样十个 88 × 88 的切片）[4]，但未明确说明所使用的具体数据集名称。

## Multi-turn Case

Previous question:

Follow-up:

Rewritten query:

人工评价：
正确 / 基本正确 / 错误


## No-answer Case

Question:法国的首都是哪里？

Answer:根据当前论文知识库中的检索内容，无法可靠回答该问题。

是否拒答：
 否