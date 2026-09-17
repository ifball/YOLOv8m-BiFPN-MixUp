# BiFPN Fusion Weight Analysis

Model: `/root/autodl-tmp/ifball_uav/runs/detect_sci4/sci4_B4_bifpn_mixup/weights/best.pt`

The BiFPN module uses ReLU-normalized learnable weights: `weight_i = ReLU(w_i) / (sum(ReLU(w)) + epsilon)`. Initial weights are uniform, i.e., 0.5 and 0.5 for each two-input fusion node.

## Key Findings

- Node 1 and Node 2 belong to the top-down pathway. Their learned weights assign larger contributions to the lower-level backbone features (`P4` and `P3`), with an average Input-2 contribution of 0.5346.
- Node 3 and Node 4 belong to the bottom-up pathway. Their learned weights assign larger contributions to the downsampled refined features from the previous lower-resolution output, with an average Input-1 contribution of 0.6097.
- Compared with the initial uniform fusion weights, the trained BiFPN no longer performs simple equal fusion. It adaptively emphasizes high-resolution spatial details in the top-down pathway and refined semantic-spatial features in the bottom-up pathway.

## Suggested Paper Text

The learned BiFPN weights indicate that the proposed network does not simply concatenate multi-scale features with equal importance. In the top-down pathway, the model assigns higher weights to shallow and middle-level backbone features, which preserve richer spatial details for small UAV objects. In the bottom-up pathway, the model places higher weights on the downsampled refined features, suggesting that the detector further propagates enhanced localization cues to deeper semantic layers. This adaptive distribution is consistent with the characteristics of aerial object detection, where targets are small, densely distributed, and frequently occluded.
