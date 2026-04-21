import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

# ==========================================
# دالة تحميل البيانات
# ==========================================
def load_and_prep_data(path):
    df = pd.read_csv(path)
    
    # 1. تحديد العمود المستهدف
    target_column = 'churned'
    
    # 2. حذف أي أعمدة تعريفية (Identifiers) 
    # سنحذف أي عمود يحتوي على 'id' أو 'customer' لتجنب خطأ النصوص
    cols_to_drop = [col for col in df.columns if 'id' in col.lower() or 'customer' in col.lower()]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped identification columns: {cols_to_drop}")

    # 3. التعامل مع القيم المفقودة
    df = df.fillna(0) 

    # 4. فصل الأهداف عن الميزات
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # 5. تحويل الأعمدة النصية المتبقية (مثل Contract أو Gender) إلى أرقام
    # هذا السطر ضروري جداً لتحويل الفئات إلى صيغة رياضية
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Final features shape: {X.shape}")
    return X, y

# ==========================================
# الجزء الأول: Systematic Hyperparameter Tuning
# ==========================================
def run_part1_grid_search(X, y):
    print("--- Running Part 1: GridSearchCV ---")
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid_rf,
        cv=StratifiedKFold(5),
        scoring='f1',
        n_jobs=-1
    )
    rf_grid.fit(X, y)

    print(f"Best Params: {rf_grid.best_params_}")
    print(f"Best F1 Score: {rf_grid.best_score_:.4f}\n")

    # توليد الرسم البياني
    results = pd.DataFrame(rf_grid.cv_results_)
    # تبرير: نثبت min_samples_split عند الأفضل لنرى تأثير العمق وعدد الأشجار بوضوح
    best_mss = rf_grid.best_params_['min_samples_split']
    pivot_table = results[results['param_min_samples_split'] == best_mss].pivot(
        index='param_max_depth', columns='param_n_estimators', values='mean_test_score'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".4f")
    plt.title(f'Random Forest F1-Score (min_samples_split={best_mss})')
    plt.savefig('heatmap.png')
    plt.show()
    
    return rf_grid.best_params_

# ==========================================
# الجزء الثاني: Nested Cross-Validation
# ==========================================
def run_nested_cv(model_obj, grid, X, y):
    # استخدام random_state مختلف عن الداخلي لضمان عدم تطابق الـ folds
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    
    inner_scores = []
    outer_scores = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        # البحث الداخلي (Hyperparameter Selection)
        grid_search = GridSearchCV(model_obj, grid, cv=inner_cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_outer, y_train_outer)
        inner_scores.append(grid_search.best_score_)
        
        # التقييم الخارجي (Honest Performance Estimation)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_outer)
        outer_scores.append(f1_score(y_test_outer, y_pred))
        
    return np.mean(inner_scores), np.mean(outer_scores)

# ==========================================
# تنفيذ البرنامج (Execution)
# ==========================================
if __name__ == "__main__":
    # 1. تحميل البيانات
    X, y = load_and_prep_data('data/telecom_churn.csv')
    
    # 2. تنفيذ الجزء الأول
    best_params_rf = run_part1_grid_search(X, y)
    
    # 3. تنفيذ الجزء الثاني (Nested CV)
    print("--- Running Part 2: Nested Cross-Validation ---")
    
    # شبكة Random Forest (نفس الجزء الأول)
    rf_grid_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_inner, rf_outer = run_nested_cv(
        RandomForestClassifier(class_weight='balanced', random_state=42), 
        rf_grid_params, X, y
    )
    
    # شبكة Decision Tree
    dt_grid_params = {
        'max_depth': [3, 5, 10, 20, None], 
        'min_samples_split': [2, 5, 10]
    }
    dt_inner, dt_outer = run_nested_cv(
        DecisionTreeClassifier(class_weight='balanced', random_state=42), 
        dt_grid_params, X, y
    )
    
    # 4. عرض جدول المقارنة النهائي
    comparison_data = {
        'Metric': ['Inner best_score_ (Mean)', 'Outer nested CV score (Mean)', 'Gap (Selection Bias)'],
        'Random Forest': [rf_inner, rf_outer, rf_inner - rf_outer],
        'Decision Tree': [dt_inner, dt_outer, dt_inner - dt_outer]
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\nFinal Nested CV Comparison Table:")
    print(comparison_df.to_string(index=False))