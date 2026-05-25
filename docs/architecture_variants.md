# Architecture Variants

This repo documents only the supported mainline variants that matter for the paper surface.

## VQ macro-action path

- The hierarchical trainer supports a VQ latent-action encoder through `latent_action_encoder.type=vq`
- This remains a supported mainline variant, not a separate public package surface

## Latent-action-dimension ablations

- The repo retains the latent-action-dimension ablation family as a supported architecture variant
- These ablations stay behind the canonical experiment system rather than separate root-level scripts

## Empirical-macro / Samuele CEM constraint

- The evaluation stack keeps the empirical-macro option as a supported variant of hierarchical planning
- This variant lives inside the hierarchical evaluation seam rather than as a separate public workflow
