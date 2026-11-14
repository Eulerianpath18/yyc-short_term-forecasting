# Sequence-model comparison

The flattened MLP and compact temporal convolutional network share the same 24-hour multistation input, four cyclic calendar predictors, missingness masks and five jointly predicted lead times. Four variables at each of six stations supply 24 value channels. Every value channel has a matching mask. Both networks use AdamW, standardized targets, a shared batch schedule and a 15-epoch cap. Three seeds were run, and all six runs selected epoch 15.

The fitted TCN uses one causal kernel-width-3 convolution per residual block with dilations 1, 2, 4 and 8. Its 31-hour receptive field covers the input window. It has 4,757 trainable parameters. The flattened MLP has 156,677 parameters. The code also defines a larger two-convolution block variant for future experiments.

Both sequence models were trained on their own mask-compatible cohort. Their common-case scores use 8,102 matched origins in 2025. The reported seed mean averages three separately computed RMSEs. Comparing the two sequence models controls the input representation and training rule. Comparisons with the engineered MLP also change the variables, training cases and model design.
