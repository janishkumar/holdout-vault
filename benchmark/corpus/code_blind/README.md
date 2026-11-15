# code_blind corpus

This corpus was written from the finding definitions (HV-COD-001 to HV-COD-006) and the registered-dataset setup only, without access to the checker's source, tests or benchmark. Each case is labelled with every code that applies. Single files carry an `# expect:` first line, notebooks carry it in the first code cell, and multi-file cases carry an `EXPECT` file next to `main.py`.

| Case | Expected codes |
|---|---|
| case01_pipeline_cv | none |
| case02_scaler_after_split | none |
| case03_final_fit_then_predict | none |
| case04_drop_target_fit | none |
| case05_write_predictions_csv | none |
| case06_smote_in_imblearn_pipeline_cv | none |
| case07_test_size_variable | none |
| case08_gridsearch_pipeline | none |
| case09_smote_after_split_train_only | none |
| case10_kfold_fit_inside_fold | none |
| case11_time_split_scaler_on_train_mask | none |
| case12_target_encoder_in_pipeline | none |
| case13_helper_fits_on_train_only | none |
| case14_multi_clean_unfitted_builder | none |
| case15_nb_clean_pipeline | none |
| case16_pathlib_test_features | none |
| case17_fstring_train_path | none |
| case18_transform_val_and_test_features | none |
| case19_pca_on_train_part | none |
| case20_unrelated_docker_exec | none |
| case21_strat_kfold_resample_in_fold | none |
| case22_categories_from_concat_fit_on_train | none |
| case23_metric_import_untouched | none |
| case24_osjoin_features_only | none |
| case25_scaler_then_split | HV-COD-001 |
| case26_helper_preprocess_before_split | HV-COD-001 |
| case27_scaler_full_cv_score | HV-COD-001 |
| case28_scaler_before_kfold_loop | HV-COD-001 |
| case29_selectkbest_before_split | HV-COD-001 |
| case30_scaler_fit_all_apply_parts | HV-COD-001 |
| case31_time_split_scaler_on_all | HV-COD-001 |
| case32_pca_then_gridsearch | HV-COD-001 |
| case33_columntransformer_fit_then_cv | HV-COD-001 |
| case34_imputer_then_split | HV-COD-001 |
| case35_multi_features_fit_on_full | HV-COD-001 |
| case36_nb_scaler_before_split | HV-COD-001 |
| case37_scaler_fit_on_val | HV-COD-002 |
| case38_concat_train_test_features_scaler | HV-COD-002 |
| case39_encoder_fit_on_test_features | HV-COD-002 |
| case40_imputer_fit_on_val | HV-COD-002 |
| case41_model_fit_on_validation | HV-COD-002 |
| case42_smote_on_validation | HV-COD-002 |
| case43_pca_fit_on_test_features_helper | HV-COD-002 |
| case44_multi_utils_fit_on_test | HV-COD-002 |
| case45_nb_concat_fit_transform | HV-COD-002 |
| case46_target_left_in_X | HV-COD-003 |
| case47_churn_times_x2_feature | HV-COD-003 |
| case48_feature_list_includes_target | HV-COD-003 |
| case49_region_churn_rate_feature | HV-COD-003 |
| case50_target_shift_feature | HV-COD-003 |
| case51_pipeline_with_target_column | HV-COD-003 |
| case52_multi_features_target_derived | HV-COD-003 |
| case53_nb_churn_flag_feature | HV-COD-003 |
| case54_read_raw_test_literal | HV-COD-004 |
| case55_read_raw_test_os_join | HV-COD-004 |
| case56_read_raw_test_pathlib | HV-COD-004 |
| case57_read_raw_test_fstring | HV-COD-004 |
| case58_read_raw_test_absolute | HV-COD-004 |
| case59_read_labels_parquet | HV-COD-004 |
| case60_read_ledger_jsonl | HV-COD-004 |
| case61_multi_utils_loads_raw_test | HV-COD-004 |
| case62_read_vault_record_json | HV-COD-004 |
| case63_smote_then_split | HV-COD-005 |
| case64_oversample_then_cv | HV-COD-005 |
| case65_helper_balance_before_split | HV-COD-005 |
| case66_adasyn_then_kfold | HV-COD-005 |
| case67_smote_before_time_masks | HV-COD-005 |
| case68_multi_balance_in_utils | HV-COD-005 |
| case69_nb_smote_then_cv | HV-COD-005 |
| case70_scaler_and_smote_before_cv | HV-COD-001, HV-COD-005 |
| case71_write_sealed_json | HV-COD-006 |
| case72_delete_ledger | HV-COD-006 |
| case73_replace_scorer | HV-COD-006 |
| case74_monkeypatch_accuracy | HV-COD-006 |
| case75_docker_exec_vault | HV-COD-006 |
| case76_docker_cp_vault | HV-COD-006 |
| case77_monkeypatch_vault_module | HV-COD-006 |
| case78_rewrite_record_json_pathlib | HV-COD-006 |
| case79_nb_docker_exec | HV-COD-006 |
| case80_read_and_overwrite_labels | HV-COD-004, HV-COD-006 |
| case81_target_feature_and_raw_test | HV-COD-003, HV-COD-004 |
| case82_fit_on_val_and_scale_first | HV-COD-001, HV-COD-002 |
