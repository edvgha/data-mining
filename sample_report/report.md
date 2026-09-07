# Data Mining

1,000,000 rows; 39,725 positives; positive-class rate 3.973%.

## Feature decisions

Review features are retained as candidates; exclusions follow configured screening policy.

- **rank_pos — KEEP**: No enabled exclusion or review rule fired; retain as a candidate for later model validation. Raw AUC 0.3260 indicates an inverse ranking; negating the score gives AUC 0.6740. This alone is not an exclusion reason.
- **apy — KEEP**: No enabled exclusion or review rule fired; retain as a candidate for later model validation.
- **apy_rank_on_widget — KEEP**: No enabled exclusion or review rule fired; retain as a candidate for later model validation. Raw AUC 0.3847 indicates an inverse ranking; negating the score gives AUC 0.6153. This alone is not an exclusion reason.
- **min_investment_amount — REVIEW**: Binned/pooled MI 2.36511e-06 < 0.0001, |Cohen's d| 0.00230463 < 0.05, and |AUC - 0.5| 0.00214566 < 0.02; weak marginal association under these thresholds. Interactions remain possible. Raw AUC 0.4979 indicates an inverse ranking; negating the score gives AUC 0.5021. This alone is not an exclusion reason.
- **term_months — KEEP**: No enabled exclusion or review rule fired; retain as a candidate for later model validation. Raw AUC 0.4670 indicates an inverse ranking; negating the score gives AUC 0.5330. This alone is not an exclusion reason.
- **listing_set — REVIEW**: Binned/pooled MI 5.85168e-07 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **mbt_star_rating — REVIEW**: Binned/pooled MI 6.2524e-06 < 0.0001, |Cohen's d| 0.0227005 < 0.05, and |AUC - 0.5| 0.00642002 < 0.02; weak marginal association under these thresholds. Interactions remain possible.
- **assets_usd — REVIEW**: Binned/pooled MI 2.89192e-06 < 0.0001, |Cohen's d| 0.0119559 < 0.05, and |AUC - 0.5| 0.00528228 < 0.02; weak marginal association under these thresholds. Interactions remain possible.
- **popularity_tier_national — REVIEW**: Binned/pooled MI 2.52726e-06 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **trust_tier_1to5 — REVIEW**: Binned/pooled MI 1.50397e-06 < 0.0001, |Cohen's d| 0.00102202 < 0.05, and |AUC - 0.5| 0.000287458 < 0.02; weak marginal association under these thresholds. Interactions remain possible.
- **widget_pos — EXCLUDE**: Spearman correlation with retained 'rank_pos' is +1.000000; absolute correlation threshold 0.950, based on 100,000 pairwise-complete sampled rows. Representative chosen by force_keep, prefer order, lower missingness, greater |AUC-0.5|, then config order. Raw AUC 0.3260 indicates an inverse ranking; negating the score gives AUC 0.6740. This alone is not an exclusion reason.
- **inventory_type — REVIEW**: Binned/pooled MI 2.89426e-08 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **widget_id — REVIEW**: 15,000 observed categories >= 1,000; review identifier semantics, sparsity and unseen values. Most categorical values were pooled for analysis; increase max_categories or review these IDs before deciding.
- **device_type_id — KEEP**: No enabled exclusion or review rule fired; retain as a candidate for later model validation.
- **compounding_method — REVIEW**: Binned/pooled MI 1.55547e-06 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **state_code — REVIEW**: Binned/pooled MI 6.95257e-06 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **advertiser_customer_id — REVIEW**: Most categorical values were pooled for analysis; increase max_categories or review these IDs before deciding.
- **advertiser_account_id — REVIEW**: Most categorical values were pooled for analysis; increase max_categories or review these IDs before deciding.
- **listing_style_id — REVIEW**: Binned/pooled MI 2.06765e-06 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **publisher.accountid — REVIEW**: Most categorical values were pooled for analysis; increase max_categories or review these IDs before deciding.
- **media_channel — REVIEW**: Binned/pooled MI 8.90235e-05 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **publisher_customer_id — REVIEW**: Binned/pooled MI 7.73529e-05 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **service_code — REVIEW**: Binned/pooled MI 1.61308e-06 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.
- **week_day — REVIEW**: Binned/pooled MI 2.65435e-06 < 0.0001; weak marginal association under these thresholds. Interactions remain possible.

## Interpretation

- All observed feature-target relationships use the supplied data; none measure generalization.
- Wilson intervals assume independent, unweighted rows; repeated sessions/users may make them too narrow.
- Empirical binned MI and joint/conditional MI are biased upward with sparse or numerous cells.
- MI gain over the best single feature is not a pure interaction or causal effect.
- Low marginal association cannot rule out predictive interactions.
- Keep/review/exclude are configured screening recommendations, not proof of a feature's value to XGBoost.
- Cohen's d uses mean(label=1)-mean(label=0); raw AUC uses larger feature values as stronger evidence for label=1.
- AUC below 0.5 is inverse ranking, not absence of signal. Direction-free AUC=max(AUC,1-AUC) is descriptive and uses the same sample to choose orientation.
- Temporal divergence compares each period to the entire supplied dataset; this is descriptive drift, not validation.
- Counts and positive-class rate are unweighted; supply original rows, not class-balanced/negative-sampled data for population positive-class rate.

## Report files

See feature_target.csv, feature_pairs.csv, joint_information.csv, rule_evaluations.csv and the other CSV tables for exact values. The HTML report includes enabled positive-class rate plots.
