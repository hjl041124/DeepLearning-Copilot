# Baseline Error Analysis

Model: Qwen3-4B-Instruct-2507 Base Model

## standard_test

### total_samples

480

### parse_fail

0

### task_type_error

388

### primary_issue_error

468

### evidence_error

437

### recommendation_error

480

### complete_match_error

480

### prediction_primary_issue_distribution

{'类别不平衡导致模型对少数类性能较差': 10, '类别不平衡导致模型在少数类上表现不佳': 18, '类别不平衡导致模型对少数类预测能力差': 3, '类别不平衡导致模型对少数类预测能力不足': 7, '类别不平衡导致模型在少数类上表现差': 7, '类别不平衡导致模型对少数类性能严重不足': 4, '类别不平衡导致模型对少数类性能差': 4, '类别不平衡导致模型在少数类上表现严重不足': 7, 'distribution_shift': 9, 'distribution_shift_detected': 1, '高重复样本率导致训练数据分布不均衡': 5, 'high_duplicate_rate_indicating_data_overfitting_or_data_issues': 2, 'high_duplicate_rate': 3, '高标签噪声导致模型性能严重下降': 5, 'high_label_noise_rate': 4, 'high_label_noise_rate_degrading_model_performance': 1, 'preprocessing_mismatch': 9, '数据划分重叠导致训练数据污染': 1, '数据划分重叠导致训练数据不独立': 2, '低数据多样性导致模型过拟合风险': 1, '数据集划分重叠严重，可能导致训练数据泄露': 1, '低数据覆盖导致模型过拟合风险高': 2, '低数据多样性导致模型过拟合': 2, '类别不平衡导致模型性能波动': 7, '类别不平衡导致模型性能受限': 2, '类别不平衡导致模型对少数类性能不足': 3, '类别不平衡问题': 1, '类别不平衡导致的模型性能波动': 1, '类别不平衡导致的模型性能偏差': 1, '类别不平衡导致模型对少数类表现不佳': 2, '类别不平衡导致模型性能偏差': 4, '类别不平衡导致模型对少数类表现不足': 1, '模型过拟合': 10, '验证集性能提升缓慢且存在过拟合风险': 5, '验证集性能提升缓慢且与训练集差距明显': 3, '验证集性能提升缓慢且与训练集差距持续存在': 2, '验证集性能提升缓慢，存在过拟合风险': 4, 'validation performance plateauing with no significant improvement': 1, '严重过拟合': 2, '过拟合': 8, '模型在验证集上表现显著低于训练集，存在明显过拟合': 1, 'overfitting': 12, '验证指标持续下降后回升，存在过拟合风险': 10, '验证指标远低于训练指标，存在明显的过拟合现象': 1, '验证指标显著低于训练指标，存在明显的过拟合现象': 1, '验证指标显著高于训练指标，存在明显的过拟合现象': 1, '验证指标远高于训练指标，存在明显的过拟合现象': 1, '验证指标远低于训练指标，存在显著的过拟合现象': 1, '验证指标在训练初期显著下降，存在过拟合风险': 2, '验证指标在训练过程中出现下降后又回升，存在过拟合风险': 1, 'validation_metric_not_improving_after_initial_drop': 2, '验证指标在训练过程中出现显著波动且最终值偏低，未达到预期性能': 2, '验证指标在训练过程中出现显著波动且最终未收敛，存在过拟合风险': 1, '验证指标在训练过程中出现显著下降后又回升，表明模型存在过拟合风险': 1, 'nan_or_inf': 24, '训练损失和验证损失波动较大，未呈现收敛趋势': 1, '训练损失下降缓慢，收敛速度差': 3, '训练损失持续上升，验证损失缓慢上升，模型可能陷入过拟合或训练不稳定': 1, '训练损失下降缓慢且波动较大': 2, '训练损失下降缓慢，验证损失未明显改善': 2, '训练损失持续上升，验证损失缓慢增加，模型可能陷入过拟合或训练不稳定': 1, '训练损失持续下降但验证损失未明显改善，可能存在过拟合风险': 1, '训练损失和验证损失持续上升，模型性能下降': 2, 'loss_convergence_issue': 4, 'learning_rate_is_too_high': 4, 'learning_rate_too_high': 1, '训练损失收敛缓慢且波动较大': 1, '训练损失波动大且未收敛': 1, '模型性能显著低于参考性能，存在严重过拟合或训练不足问题': 2, '模型性能显著低于参考性能，存在过拟合风险': 13, '模型性能显著低于参考性能，存在严重过拟合或欠拟合问题': 3, '模型性能显著低于参考性能，存在严重过拟合或欠拟合风险': 1, 'underperformance compared to reference performance': 24, '模型性能未达到预期目标，存在明显性能差距': 1, '模型性能远低于参考性能，存在严重欠拟合': 2, '模型性能未达到预期，存在明显性能差距': 1, '模型性能未达到预期，验证指标显著低于参考性能': 1, '类别不平衡导致宏平均 F1 值偏低': 2, '类别不平衡导致模型在少数类上的表现较差': 3, '类别不平衡导致宏平均F1分数偏低': 3, '类别不平衡导致的F1分数偏低': 2, '类别不平衡导致宏平均 F1 值偏低，尽管整体准确率较高': 1, '类别不平衡导致模型在少数类上的表现严重不足': 1, '类别不平衡导致模型性能严重偏倚': 1, '类别不平衡导致模型性能严重偏向多数类': 1, '类别 B 的表现显著低于其他类别，存在严重类别不平衡问题': 8, '类别不平衡导致性能严重不均，特别是 class_B 的表现显著低于其他类别': 1, '类别B的性能显著低于其他类别，存在严重类别不平衡问题': 1, '类别B的性能显著低于其他类别': 1, '类别 B 的表现显著低于其他类别，存在严重的类别不平衡问题': 1, 'Class_B performance is significantly lower than other classes': 3, 'Class_B性能严重不足': 3, 'Class_B性能显著低于其他类别': 6, '召回率偏低，存在漏检风险': 5, '较低的精确率与较高的召回率之间存在显著不匹配': 1, '较高的召回率与较低的精确率表明模型倾向于预测为正类，可能导致大量误报。': 1, '召回率高但精确率低，存在较高的误报风险': 1, '高召回率伴随较低精确率': 3, '较高的召回率与中等精确率之间的不匹配': 1, '召回率远高于精确率，存在明显的假阳性问题': 1, '低精确率与高召回率的不匹配': 1, '较低的召回率导致模型可能漏检大量正类样本': 1, '高召回率伴随较低精确率，表明模型存在较多误报': 2, '低召回率导致潜在误报风险': 2, '高召回率伴随较低精确率，表明模型存在较多误报（false positives）': 1, '高召回率伴随低精确率，表明模型存在大量误报（false positives）': 1, '低召回率导致潜在正类样本漏检': 1, '低召回率导致潜在正例漏检': 1, '低召回率导致模型漏检严重': 1, '训练集表现显著优于验证集，存在过拟合风险': 5, '显著的训练集与验证集性能差距': 5, '训练集表现显著优于验证集，存在明显的过拟合现象': 2, 'significant_gap_between_train_and_validation_metrics': 5, 'performance_gap_between_train_and_validation': 6, 'validation performance significantly lags behind training performance': 1, 'Model A outperforms Model B in macro_f1': 1, 'Model B outperforms Model A in macro_f1': 8, 'Model B underperforms compared to Model A': 3, 'Model B achieves a higher macro_f1 score than Model A': 3, 'Model B achieves a significantly higher macro_f1 score than Model A': 1, 'Model A and Model B show significant differences in performance on the primary metric': 1, 'Model A outperforms Model B on the primary metric': 12, 'Model A and Model B performance difference': 1, 'Model A performs worse than Model B on the primary metric': 3, 'Model A has higher accuracy but lower macro-F1 than Model B, indicating poorer performance in balanced class prediction.': 1, 'Model B outperforms Model A in both accuracy and macro-F1, indicating better overall performance.': 1, 'Model B has a higher macro-F1 score than Model A, indicating better balance across classes, despite similar accuracy.': 1, 'Model B has higher accuracy but lower macro-F1 than Model A, indicating a potential imbalance in class performance.': 1, 'Model B has a higher macro-F1 score than Model A, indicating better balance across classes.': 1, 'Model B has a lower macro-F1 score compared to Model A despite having a slightly higher accuracy': 1, 'Model A has higher accuracy but lower macro-F1 than Model B, indicating potential imbalance in class performance.': 2, 'Model A has higher accuracy but lower macro-F1 than Model B, indicating a potential imbalance in class performance.': 1, 'Model B has a lower macro-F1 score compared to Model A, indicating poorer performance in handling class imbalance.': 1, 'Model A has lower macro-F1 than Model B despite similar accuracy': 2, 'Model B achieves higher accuracy and macro-F1 than Model A, indicating better overall performance.': 1, 'Model A has higher accuracy and macro-F1 than Model B, indicating better overall performance.': 1, 'Model B has a higher macro-F1 score than Model A, indicating better performance in handling class imbalance.': 1, 'Model B has a higher accuracy but lower macro-F1 compared to Model A, indicating potential imbalance in class performance.': 1, 'Model B has a significantly lower macro-F1 score despite higher accuracy, indicating poor performance on class imbalance or minority classes': 3, 'Model B has a significantly lower macro-F1 score despite a higher accuracy, indicating poor class balance handling and potential overfitting to majority classes.': 1, 'Model B has a significantly lower macro-F1 score compared to Model A, indicating poorer performance in handling class imbalance or diverse classes.': 1, 'Model B has higher accuracy and macro-F1, indicating better overall performance': 1, 'Model B has higher accuracy and macro-F1, indicating better overall performance.': 1, 'Model A and Model B show different performance across accuracy and macro-F1, indicating a trade-off between precision and recall in classification.': 1, 'Model A achieves a higher macro_f1 score than Model B': 9, 'Model A has a slightly higher macro_f1 score than Model B': 2, 'Model A and Model B have very similar performance with a small difference in macro_f1': 1, 'Model A achieves a slightly higher macro_f1 score than Model B': 3, 'Model B has a higher macro_f1 score but significantly higher latency than Model A': 1, 'Model B has a higher macro_f1 score but significantly higher latency': 2, 'Model B has a higher macro_f1 score but significantly higher latency compared to Model A': 1, 'Model A has a higher macro_f1 score but significantly higher latency compared to Model B': 8, 'Model B has a significantly higher macro_f1 score than Model A, indicating better performance on the quality metric, but at the cost of much higher latency.': 1, 'Model A has higher macro_f1 (0.918) than Model B (0.811), indicating better performance on the quality metric, but Model B has significantly lower latency (49.264 ms vs. 117.799 ms).': 1, 'Model B has a significantly higher macro_f1 score than Model A, indicating better performance on the quality metric.': 1, 'Model A has a higher macro_F1 score than Model B, indicating better overall performance in terms of classification accuracy, but at the cost of significantly higher inference latency.': 1, 'Model B has a significantly higher macro_f1 score than Model A, indicating better performance on the quality metric, but at the cost of higher latency.': 1, 'Model B has higher quality but significantly higher latency': 1, 'Model A has superior quality but higher latency compared to Model B': 2, 'Model B has significantly higher quality but much higher latency': 1, 'Model B has significantly higher quality metric but much higher latency': 1, 'Model A has significantly higher macro_f1 but much higher latency compared to Model B': 1, 'Model B has significantly higher quality metric but also much higher latency': 1}

### ground_truth_primary_issue_distribution

{'class_imbalance': 48, 'data_quality_issue': 48, 'no_clear_issue': 48, 'overfitting': 48, 'optimization_problem': 48, 'underfitting': 48, 'not_applicable': 192}

### hard_property_distribution

{}

### hard_property_accuracy

{}

## hard_test

### total_samples

240

### parse_fail

0

### task_type_error

220

### primary_issue_error

238

### evidence_error

240

### recommendation_error

240

### complete_match_error

240

### prediction_primary_issue_distribution

{'Class imbalance due to uneven class distribution': 1, 'Class imbalance in the dataset': 2, 'Class imbalance in training data': 1, 'class imbalance': 4, '类不平衡导致模型在少数类上表现不佳': 8, '类不平衡导致的模型性能偏差': 4, 'Class imbalance in the dataset leading to biased model performance': 1, 'Class imbalance leading to biased model performance': 2, 'Validation accuracy plateauing and then dropping after initial improvement': 4, 'overfitting': 3, '过拟合': 3, 'Validation accuracy is lower than training accuracy, indicating overfitting': 2, 'Overfitting': 6, 'validation accuracy plateauing and dropping after initial improvement': 2, 'Validation accuracy is significantly lower than training accuracy, indicating overfitting': 3, 'validation performance drops significantly after training stabilization': 1, 'validation performance degradation after training stabilization': 1, 'validation accuracy plateauing and then dropping after initial improvement': 1, '高标签噪声导致模型学习偏差': 1, '标签噪声严重影响模型性能': 1, '高标签噪声导致模型训练不稳定': 16, 'label_noise': 2, 'high_label_noise': 2, 'label noise': 3, 'high label noise': 16, 'high label noise in training data': 1, '高标签噪声率导致模型训练不稳定': 5, 'High label noise in training data': 1, 'model_a_outperforms_model_b_on_primary_metric': 6, 'model_b_outperforms_model_a_on_primary_metric': 9, 'none': 1, 'model_b_performs_significantly_better_than_model_a_on_primary_metric': 1, 'model_a_performs_worse_than_model_b_on_primary_metric': 1, 'model_a_lags_behind_model_b_in_primary_metric': 1, 'model_b_outperforms_model_a_in_primary_metric': 1, '严重的过拟合': 4, '严重过拟合': 21, 'Validation accuracy is significantly lower than training accuracy, indicating poor generalization': 5, '严重的验证集性能过低，存在明显过拟合风险': 1, 'Validation accuracy is significantly lower than training accuracy, indicating severe overfitting': 6, '训练集准确率远高于验证集准确率，存在明显过拟合现象': 1, 'High training accuracy with low validation accuracy indicating overfitting': 2, 'Validation accuracy significantly drops after training stabilizes, indicating overfitting': 1, 'Validation accuracy significantly lower than training accuracy, indicating severe overfitting': 4, '严重的验证集性能与训练集性能不匹配，存在明显过拟合': 1, 'Validation accuracy significantly lower than training accuracy, indicating overfitting': 2, '严重的验证集性能过低，存在明显过拟合现象': 1, '严重的验证集性能下降，存在明显的过拟合现象': 1, 'Class imbalance leading to poor validation accuracy and overfitting': 1, 'validation performance significantly drops after initial improvement, indicating overfitting': 1, 'validation accuracy significantly lower than training accuracy with clear overfitting signs': 1, 'validation performance significantly drops after training stabilization, indicating overfitting': 2, 'Class imbalance leading to poor validation accuracy and overfitting on majority classes': 1, 'validation performance significantly drops after training stabilizes, indicating overfitting': 1, '严重的验证集性能下降（过拟合）': 1, 'high label noise rate leading to poor performance on minority class': 3, 'high label noise leading to poor performance on minority class': 2, '类不平衡导致模型在少数类上表现严重不足': 2, 'high label noise rate leading to unreliable model performance': 2, 'high label noise leading to poor model performance on minority classes': 1, 'high label noise rate leading to poor performance on minority classes': 2, 'Class imbalance and high label noise': 1, 'high label noise leading to poor performance on minority classes': 1, '数据中存在 NaN 或无穷大值': 2, '严重标签噪声和数据缺失': 5, 'label noise and data quality issues': 2, '数据质量严重受损，存在大量 NaN 或无穷值': 2, 'high label noise and data quality issues': 1, '数据质量严重受损，存在大量无效值（NaN或Inf）': 1, '高标签噪声': 1, 'label noise due to high data quality label noise rate': 1, 'Validation accuracy significantly drops after training stabilizes, indicating poor generalization despite high training accuracy': 1, 'High label noise leading to poor generalization on validation set': 2, 'Validation accuracy drops significantly after training stabilizes, indicating poor generalization': 1, 'High label noise leading to poor generalization': 1, 'High training accuracy with low validation accuracy indicating significant overfitting': 1, 'Validation accuracy plateauing and underperformance compared to training accuracy': 1, 'Validation accuracy drops significantly after training stabilizes, indicating overfitting': 1, 'Validation accuracy significantly lower than training accuracy with signs of overfitting and NaN/inf values': 1, 'Validation accuracy significantly drops after training stabilizes, indicating overfitting or data leakage': 1, 'Validation performance is significantly lower than training performance, indicating severe overfitting': 1, 'Validation accuracy significantly drops after training stabilizes, indicating poor generalization and potential overfitting or data issues': 2, 'Validation performance significantly drops after training stabilization, indicating overfitting or poor generalization': 1, 'validation performance drops significantly after initial improvement, indicating overfitting or instability in training': 1, '训练集准确率远高于验证集准确率，存在明显过拟合': 1, '模型性能显著低于参考性能，验证集准确率远低于预期': 1, '训练和验证准确率远低于参考性能，且存在 NaN 或无穷值': 11, '模型性能远低于参考性能，存在严重过拟合或模型能力不足问题': 2, '模型性能远低于参考性能，训练和验证准确率均显著偏低': 1, 'Model underperforms significantly compared to reference performance': 1, '模型性能严重低于预期，训练和验证准确率均远低于参考性能': 1, '严重性能不足，远低于参考性能': 1, '模型性能远低于参考性能，且存在 NaN 或无穷值': 1, '模型性能严重低于参考性能，存在无效训练或数据问题': 1}

### ground_truth_primary_issue_distribution

{'class_imbalance': 30, 'no_clear_issue': 30, 'overfitting': 30, 'data_quality_issue': 90, 'not_applicable': 20, 'optimization_problem': 40}

### hard_property_distribution

{}

### hard_property_accuracy

{}

