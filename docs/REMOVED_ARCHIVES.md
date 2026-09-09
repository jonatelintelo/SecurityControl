# Removed result archives — index

Two archived result roots were deleted on 2026-09-09 as stale. They held
outputs of superseded runs and were never a source of numbers. This file is
the record of what existed, so the deletion is documented rather than silent.

The scientific record of what changed and why is in
`results/RQ1_FINDINGS.md` §6 (eight corrections, each with the before/after)
and `slurm/1_run_phase/KEPT.md` (job -> artifact mapping). The code state that
produced them is in git history.

## results_ARCHIVED_v1_invalid

Reason it was invalid, as recorded at the time:

> # Archived — the original attempt (pre-2026-09-07)
> 
> Outputs of the first experimental programme, retained unmodified as a record of
> what was run before the restart. **Not valid, not a specification, not a source of
> numbers.**
> 
> Reasons for the reset, as recorded at the time:
> 
> * attack evaluation ran on 10–15 prompts;
> * harmfulness and refusal were read at the **same** token position, rather than
>   `t_inst` / `t_post-inst` respectively;
> * `late_layer` (21) was earlier than `mid_layer` (23), so part of the
>   cross-intervention matrix read *upstream* of its own intervention and produced
>   exact `0.0` rows;
> * `k_50` was undefined in every row;
> * the RQ5 phase only ever ran under `FAST_DEV`.
> 
> The current programme is documented in `PLAN.md`, `ENVIRONMENT.md` and
> `EXPERIMENTS.md`; RQ1's findings are in `results/RQ1_FINDINGS.md`.

```
./neurostrike_reproduction/neurostrike_reproduction.log
./neurostrike_reproduction/reproduction.csv
./phase1_geometry/concept_dimensionality.csv
./phase1_geometry/control_transfer_test.csv
./phase1_geometry/directions.pt
./phase1_geometry/geometry_cosine.csv
./phase1_geometry/phase1_geometry.log
./phase1_geometry/probe_accuracy.csv
./phase1_geometry/projection_correlations.csv
./phase1_geometry/projections.csv
./phase1_geometry/run_manifest.json
./phase1_geometry/selected_layers.json
./phase2_causal_structure/cross_intervention_matrix.csv
./phase2_causal_structure/phase2_causal_structure.log
./phase2_causal_structure/run_manifest.json
./phase3_components/ablation_curves.csv
./phase3_components/architecture_metrics.csv
./phase3_components/mediation_rescue_results.json
./phase3_components/mediation_rescue_samples.txt
./phase3_components/neuron_functional_classification.csv
./phase3_components/neuron_ranking_overlap.csv
./phase3_components/neuron_rankings.pt
./phase3_components/neuron_read_write_roles.csv
./phase3_components/neurostrike_neuron_functions.csv
./phase3_components/neurostrike_neurons.pt
./phase3_components/neurostrike_weights.pt
./phase3_components/phase3_components.log
./phase3_components/run_manifest.json
./phase4_architecture_prediction/alpha_star_curves.csv
./phase4_architecture_prediction/k_star_curves.csv
./phase4_architecture_prediction/neurostrike_prefix_curve.csv
./phase4_architecture_prediction/phase4_architecture_prediction.log
./phase4_architecture_prediction/prediction_correlations.csv
./phase4_architecture_prediction/prediction_table.csv
./phase4_architecture_prediction/proxy_validation.csv
./phase4_architecture_prediction/run_manifest.json
./phase5_context_and_attacks/attack_stage_diagnosis.csv
./phase5_context_and_attacks/causal_transfer_matrix.csv
./phase5_context_and_attacks/component_stability.csv
./phase5_context_and_attacks/context_stability.csv
./phase5_context_and_attacks/phase5_context_and_attacks.log
./phase5_context_and_attacks/reorganization_gap_ci.csv
./phase5_context_and_attacks/run_manifest.json
./phase6_feature_interactions/persona_interaction.csv
./phase6_feature_interactions/persona_interaction_samples.txt
./phase6_feature_interactions/phase6_feature_interactions.log
./phase6_feature_interactions/role_to_harm_interaction.csv
./phase6_feature_interactions/run_manifest.json
./probe_attack_Qwen2.5-7B-Instruct/probe_attack.csv
./probe_attack_Qwen2.5-7B-Instruct/probe_attack_Qwen2.5-7B-Instruct.log
./probe_attack_Qwen2.5-7B-Instruct/probe_weights.pt
./probe_attack_Qwen2.5-7B-Instruct/safety_neurons.pt
./probe_attack_Qwen3.5-9B/probe_attack.csv
./probe_attack_Qwen3.5-9B/probe_attack_Qwen3.5-9B.log
./probe_attack_Qwen3.5-9B/probe_weights.pt
./probe_attack_Qwen3.5-9B/safety_neurons.pt
./README.md
```

## results_ARCHIVED_v2_pre_rebuild

Reason it was invalid, as recorded at the time:

> # Archived — pre-rebuild RQ1 artifacts (2026-09-08)
> 
> Produced before the PLAN/ENVIRONMENT/EXPERIMENTS rebuild, by code that has since
> been corrected. **Not valid, not a specification, not a source of numbers.**
> 
> Known defects in the code that produced these:
> 
> * `R_control` was fitted **role-pooled**, so it carried a role-composition
>   component and was partly a role direction. Refusal rate varies strongly by role.
> * layer selection used the **test** AUC in several places, so reported held-out
>   numbers were partly selected on the evaluation split.
> * the refusal-prefix list was hand-assembled rather than transcribed from Arditi
>   et al., and matched over a 280-character window rather than their unbounded
>   substring rule.
> * the generation budget was fixed at 48 tokens rather than chosen by the
>   pre-registered ladder rule.
> * no per-model `run_manifest.json` — these outputs are not traceable to a commit.
> 
> Kept only as evidence of what was run.

```
./e0_corpus_prerename/attack_intents.jsonl
./e0_corpus_prerename/corpus_meta.json
./e0_corpus_prerename/e0_corpus.log
./e0_corpus_prerename/instructions.jsonl
./e0_corpus_prerename/rendering_report.csv
./e0_corpus_prerename/run_manifest.json
./e0_corpus_prerename/tokenisation_mismatches.csv
./e0_corpus_prerename/verification.json
./gate_badgap_0904/causal_gate.json
./gate_badgap_0904/causal_layer_profile.csv
./gate_badgap_0904/causal_matrix.csv
./gate_badgap_0904/steering_sanity.json
./gate_marginproxy_1113/causal_gate.json
./gate_marginproxy_1113/causal_layer_profile.csv
./gate_marginproxy_1113/causal_matrix.csv
./gate_marginproxy_1113/steering_sanity.json
./gate_vacuous_0854/causal_gate.json
./gate_vacuous_0854/causal_layer_profile.csv
./gate_vacuous_0854/causal_matrix.csv
./gate_vacuous_0854/steering_sanity.json
./README.md
./results_phase1/phase1_execution.log
./rq1_2100/qwen2.5-7b/capability_sweep.csv
./rq1_2100/qwen2.5-7b/causal_gate.json
./rq1_2100/qwen2.5-7b/causal_matrix.csv
./rq1_2100/qwen2.5-7b/degeneracy_calibration.json
./rq1_2100/qwen2.5-7b/dimensionality.csv
./rq1_2100/qwen2.5-7b/directions.pt
./rq1_2100/qwen2.5-7b/direction_validation.csv
./rq1_2100/qwen2.5-7b/emergence_curves.csv
./rq1_2100/qwen2.5-7b/fidelity.json
./rq1_2100/qwen2.5-7b/geometry_cosines.csv
./rq1_2100/qwen2.5-7b/geometry_null_band.json
./rq1_2100/qwen2.5-7b/labels_checks.json
./rq1_2100/qwen2.5-7b/null_distributions.csv
./rq1_2100/qwen2.5-7b/projection_correlations.csv
./rq1_2100/qwen2.5-7b/projections.csv
./rq1_2100/qwen2.5-7b/refusal_labels.csv
./rq1_2100/qwen2.5-7b/role_probe.csv
./rq1_2100/qwen2.5-7b/role_probes.pt
./rq1_2100/qwen2.5-7b/steering_sanity.json
./rq1_2100/qwen3.5-9b/capability_sweep.csv
./rq1_2100/qwen3.5-9b/causal_gate.json
./rq1_2100/qwen3.5-9b/causal_matrix.csv
./rq1_2100/qwen3.5-9b/degeneracy_calibration.json
./rq1_2100/qwen3.5-9b/dimensionality.csv
./rq1_2100/qwen3.5-9b/directions.pt
./rq1_2100/qwen3.5-9b/direction_validation.csv
./rq1_2100/qwen3.5-9b/emergence_curves.csv
./rq1_2100/qwen3.5-9b/fidelity.json
./rq1_2100/qwen3.5-9b/geometry_cosines.csv
./rq1_2100/qwen3.5-9b/geometry_null_band.json
./rq1_2100/qwen3.5-9b/labels_checks.json
./rq1_2100/qwen3.5-9b/null_distributions.csv
./rq1_2100/qwen3.5-9b/projection_correlations.csv
./rq1_2100/qwen3.5-9b/projections.csv
./rq1_2100/qwen3.5-9b/refusal_labels.csv
./rq1_2100/qwen3.5-9b/role_probe.csv
./rq1_2100/qwen3.5-9b/role_probes.pt
./rq1_2100/qwen3.5-9b/steering_sanity.json
./rq1_2100/rq1.log
./rq1_2100/run_manifest.json
./rq1_guardtrunc_2114/qwen2.5-7b/budget_ladder.json
./rq1_guardtrunc_2114/qwen2.5-7b/degeneracy_calibration.json
./rq1_guardtrunc_2114/qwen2.5-7b/label_rule_disagreement.json
./rq1_guardtrunc_2114/qwen2.5-7b/labels_checks.json
./rq1_guardtrunc_2114/qwen2.5-7b/refusal_labels.csv
./rq1_guardtrunc_2114/qwen3.5-9b/budget_ladder.json
./rq1_guardtrunc_2114/qwen3.5-9b/degeneracy_calibration.json
./rq1_guardtrunc_2114/qwen3.5-9b/label_rule_disagreement.json
./rq1_guardtrunc_2114/qwen3.5-9b/labels_checks.json
./rq1_guardtrunc_2114/qwen3.5-9b/refusal_labels.csv
./rq1_guardtrunc_2114/rq1.log
./rq1_guardtrunc_2114/run_manifest.json
```

