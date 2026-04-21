# m5-s5b-Hyperparameter-Tuning-Nested-Cross-Validation-alialquraan

Part 1 Analysis — GridSearchCV
The grid search results indicate that max_depth is the hyperparameter with the most significant impact on the F1 score. We observed a clear "sweet spot" at max_depth: 5 and n_estimators: 100, yielding the peak F1 score of 0.4887. As shown in the heatmap, increasing the depth to 20 or leaving it unrestricted (nan) caused the performance to drop (darker purple regions), which is a classic sign of overfitting—where the model memorizes noise instead of learning patterns. Conversely, at a depth of 3, the model slightly underfits. The performance plateaus when increasing from 100 to 200 estimators, suggesting that the model complexity is well-captured at 100 trees. I would not recommend expanding the grid toward deeper trees; instead, I would explore regularization parameters like min_samples_leaf to further stabilize the model.


Part 2 Analysis — Nested Cross-ValidationModel 

Metric,Random Forest,Decision Tree

 Metric  Random Forest  Decision Tree
    Inner best_score_ (Mean)       0.493024       0.475511
Outer nested CV score (Mean)       0.491177       0.467685
        Gap (Selection Bias)       0.001847       0.007827



Insights
The Decision Tree model family shows a significantly larger gap (0.0078) between the inner and outer scores compared to the Random Forest (0.0018). This makes perfect sense: individual decision trees are high-variance estimators that are prone to overfitting the specific nuances of the training data. Consequently, the GridSearchCV "picks" hyperparameters that capitalize on random fluctuations in the inner folds, leading to a more optimistic (biased) score. Random Forests, through the power of bagging (Bootstrap Aggregating), naturally reduce variance and provide a more stable, honest estimation of performance.

The GridSearchCV.best_score_ from Part 1 is quite trustworthy for the Random Forest given the tiny gap, but it is notably over-optimistic for the Decision Tree. This demonstrates the core principle of Nested Cross-Validation: data used to inform a decision (tuning hyperparameters) cannot be used to objectively evaluate that same decision. Just as a held-out test set provides an unbiased final grade for a model, the outer loop of Nested CV provides an unbiased evaluation of the entire tuning process, protecting us from "cherry-picking" results that look good only on the training data.