# AI4I Task Redefinition Notice

Hi Naman,

While validating the full ML pipeline against the AI4I dataset, we discovered a data leak that required us to redefine how we treat this dataset on the ML end. 

The AI4I target_degradation (which maps to Tool wear [min]) was also present as an input feature (tool_wear). This caused the RUL regression head to trivially learn an identity function, returning a near-perfect (but invalid) training score because it was predicting a feature using itself.

Furthermore, looking closely at the dataset card, AI4I doesn't actually have a genuine continuous RUL (Remaining Useful Life) signal in the same way CMAPSS does. Tool wear [min] is a direct sensor reading, not a ground-truth RUL target. 

Because of this, we have permanently disabled the RUL regression head for the AI4I model. Going forward, AI4I is treated strictly as a classification-only task in the ML pipeline:
- Binary Failure (Anomaly)
- Fault Type (Classification)

What this means for the data export:
Since we no longer use target_degradation for RUL regression, it is strictly used as the binary failure flag for the Anomaly BCE head (derived from your target_binary_failure). Does this change anything about how you should structure the AI4I dataset export? Should tool_wear remain strictly as a feature modality, or are there any other adjustments you'd like to make to the AI4I export format given this new classification-only scope?

Let us know your thoughts!

Best,
ML Team
