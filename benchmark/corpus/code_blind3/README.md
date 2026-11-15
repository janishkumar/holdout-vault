# code_blind3 corpus

This corpus was written from the code definitions alone, without access to the checker, its source, its tests or its benchmark. Labels follow the written definitions of HV-COD-001 to HV-COD-006; a case carries every code that applies anywhere in it. About 40% of cases are clean but tricky negatives.

| Case | Kind | Codes |
|---|---|---|
| case01_te_pipeline_cv.py | script | none |
| case02_get_dummies_align.py | script | none |
| case03_config_test_features.py | script | none |
| case04_smote_imblearn_gridsearch.py | script | none |
| case05_final_scaler_full.py | script | none |
| case06_optuna_pipeline.py | script | none |
| case07_ttr_columntransformer.py | script | none |
| case08_split_then_scale.py | script | none |
| case09_docker_readonly.py | script | none |
| case10_generator_folds.py | script | none |
| case11_match_model_choice.py | script | none |
| case12_smote_after_split.py | script | none |
| case13_global_state.py | script | none |
| case14_pca_in_pipeline.py | script | none |
| case15_shutil_outputs.py | script | none |
| case16_pkg_clean | multi-file | none |
| case17_scripts_clean | multi-file | none |
| case18_nb_clean.ipynb | notebook | none |
| case19_chain_assign_clean.py | script | none |
| case20_imputer_train_only_test_transform.py | script | none |
| case21_scaler_all_then_split.py | script | HV-COD-001 |
| case22_selectk_then_cv.py | script | HV-COD-001 |
| case23_te_all_then_cv.py | script | HV-COD-001 |
| case24_pca_all_split.py | script | HV-COD-001 |
| case25_coltransformer_full_split.py | script | HV-COD-001 |
| case26_imputer_full_transform_parts.py | script | HV-COD-001 |
| case27_tfidf_all_cv.py | script | HV-COD-001 |
| case28_scaler_concat_test.py | script | HV-COD-002 |
| case29_scaler_fit_val.py | script | HV-COD-002 |
| case30_resample_val.py | script | HV-COD-002 |
| case31_pca_fit_test_features.py | script | HV-COD-002 |
| case32_optuna_imputer_concat.py | script | HV-COD-002 |
| case33_target_in_X.py | script | HV-COD-003 |
| case34_churn_rate_by_region.py | script | HV-COD-003 |
| case35_lambda_assign_target.py | script | HV-COD-003 |
| case36_target_in_json_features.py | script | HV-COD-003 |
| case37_raw_test_read.py | script | HV-COD-004 |
| case38_raw_test_environ.py | script | HV-COD-004 |
| case39_labels_parquet.py | script | HV-COD-004 |
| case40_sealed_ledger_read.py | script | HV-COD-004 |
| case41_smote_before_split.py | script | HV-COD-005 |
| case42_undersample_before_cv.py | script | HV-COD-005 |
| case43_adasyn_before_cv.py | script | HV-COD-005 |
| case44_ros_before_split_argparse.py | script | HV-COD-005 |
| case45_scorer_write.py | script | HV-COD-006 |
| case46_docker_exec.py | script | HV-COD-006 |
| case47_docker_cp.py | script | HV-COD-006 |
| case48_monkeypatch_roc_auc.py | script | HV-COD-006 |
| case49_rmtree_ledger.py | script | HV-COD-006 |
| case50_patch_vault_module.py | script | HV-COD-006 |
| case51_nb_smote_scale.ipynb | notebook | HV-COD-001, HV-COD-005 |
| case52_nb_raw_test.ipynb | notebook | HV-COD-004 |
| case53_pkg_target_feature | multi-file | HV-COD-003 |
