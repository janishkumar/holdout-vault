# code_blind2 corpus

This corpus of labelled Python ML scripts was written from the HV-COD-001 to HV-COD-006 definitions only, without any access to the checker, its source, its tests or its benchmark. Labels reflect the definitions as written; a case carries every code that applies. Single files carry an `# expect:` first line, notebooks carry it in the first code cell, and multi-file folders carry an `EXPECT` file.

| Case | Codes |
|---|---|
| case01_trainer_valsplit.py | none |
| case02_scale_then_split.py | HV-COD-001 |
| case03_final_fit_all.py | none |
| case04_raw_test_read.py | HV-COD-004 |
| case05_numpy_np_split.py | none |
| case06_smote_before_split.py | HV-COD-005 |
| case07_target_in_features.py | HV-COD-003 |
| case08_pipeline_cv.py | none |
| case09_shutil_scorer.py | HV-COD-006 |
| case10_concat_test_feats.py | HV-COD-002 |
| case11_lgbm_columntransformer_pipeline.py | none |
| case12_trainer_preprocess_all.py | HV-COD-001 |
| case13_holdout_preds.py | none |
| case14_abs_path_cfg.py | HV-COD-004 |
| case15_docker_exec.py | HV-COD-006 |
| case16_smote_after_split.py | none |
| case17_groupby_target_mean.py | HV-COD-003 |
| case18_trainer_adasyn.py | HV-COD-005 |
| case19_statsmodels.py | none |
| case20_onehot_cv.py | HV-COD-001 |
| case21_fit_on_val.py | HV-COD-002 |
| case22_argparse_te_pipeline.py | none |
| case23_labels_parquet.py | HV-COD-004 |
| case24_monkeypatch_metric.py | HV-COD-006 |
| case25_raw_in_exclude_list.py | none |
| case26_concat_encoder.py | HV-COD-001, HV-COD-002 |
| case27_numpy_target_col.py | HV-COD-003 |
| case28_kfold_fold_scaler.py | none |
| case29_undersample_kfold.py | HV-COD-005 |
| case30_selectkbest_cv.py | HV-COD-001 |
| case31_pca_train_only.py | none |
| case32_pathlib_join.py | HV-COD-004 |
| case33_delete_ledger.py | HV-COD-006 |
| case34_docker_logs.py | none |
| case35_test_features_fit_transform.py | HV-COD-002 |
| case36_lambda_flag.py | HV-COD-003 |
| case37_label_encoder_y.py | none |
| case38_scale_smote_split.py | HV-COD-001, HV-COD-005 |
| case39_labels_and_record.py | HV-COD-004, HV-COD-006 |
| case40_other_config_dir.py | none |
| case41_partial_fit_val.py | HV-COD-002 |
| case42_dataclass_cols.py | HV-COD-003 |
| case43_partial_smote.py | HV-COD-005 |
| case44_write_sealed.py | HV-COD-006 |
| case45_ledger_read.py | HV-COD-004 |
| case46_selectkbest_gridsearch.py | none |
| case47_xgb_undersample_train.py | none |
| case48_catboost_cols.py | none |
| case49_mf_clean | none |
| case50_mf_global_scaler | HV-COD-001 |
| case51_mf_raw_loader | HV-COD-004 |
| case52_nb_clean.ipynb | none |
| case53_nb_smote_first.ipynb | HV-COD-005 |
| case54_nb_concat_pca.ipynb | HV-COD-002 |
