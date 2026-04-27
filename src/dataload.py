# ─────────────────────────────────────────
# 04. CLASSIFICAZIONE — PIPELINE COMPLETA
# MODELLI: RandomForest + SVM linear + SVM rbf
# Lasso e Ridge → solo feature selection
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GridSearchCV,
                                     learning_curve)
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, matthews_corrcoef,
                             ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────
# CONFIGURAZIONE MODELLI
# ─────────────────────────────────────────
MODELLI = {
    'random_forest': {
        'model': RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'param_grid': {
            'model__n_estimators':      [100, 200, 300],
            'model__max_depth':         [5, 10, 15, 20],
            'model__min_samples_split': [5, 10, 20],
            'model__min_samples_leaf':  [4, 8, 16],
            'model__max_features':      ['sqrt', 'log2']
        },
        'usa_feature_lasso': False
    },
    'svm_linear': {
        'model': SVC(
            kernel='linear',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        'param_grid': {
            'model__C': [0.01, 0.1, 1, 10]
        },
        'usa_feature_lasso': True
    },
    'svm_rbf': {
        'model': SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        'param_grid': {
            'model__C':     [0.1, 1, 10],
            'model__gamma': ['scale', 'auto']
        },
        'usa_feature_lasso': True
    }
}

# ─────────────────────────────────────────
# CREA CARTELLE
# ─────────────────────────────────────────
def crea_cartelle(base_dir, model_name):
    cartelle = [
        f'{base_dir}/figures/{model_name}/feature_selection',
        f'{base_dir}/figures/{model_name}/cross_validation',
        f'{base_dir}/figures/{model_name}/confusion_matrix',
        f'{base_dir}/figures/{model_name}/feature_importance',
        f'{base_dir}/figures/{model_name}/curve',
        f'{base_dir}/data/train_test',
        f'{base_dir}/data/{model_name}/cv_results',
        f'{base_dir}/data/{model_name}/predictions',
        f'{base_dir}/reports',
    ]
    for cartella in cartelle:
        os.makedirs(cartella, exist_ok=True)

# ─────────────────────────────────────────
# FEATURE SELECTION — Lasso e Ridge
# ─────────────────────────────────────────
def feature_selection(X_train, y_train, X, random_state, base_dir):
    os.makedirs(f'{base_dir}/figures/feature_selection', exist_ok=True)
    os.makedirs(f'{base_dir}/data/feature_selection', exist_ok=True)

    scaler_temp    = RobustScaler()
    X_train_scaled = scaler_temp.fit_transform(X_train)

    lasso = LassoCV(cv=5, random_state=random_state, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coef_lasso    = pd.Series(np.abs(lasso.coef_), index=X.columns)
    feature_lasso = coef_lasso[coef_lasso > 0].index.tolist()

    ridge = RidgeCV(cv=5)
    ridge.fit(X_train_scaled, y_train)
    coef_ridge = pd.Series(np.abs(ridge.coef_), index=X.columns)

    print(f"\n[FEATURE SELECTION]")
    print(f"    Lasso — feature selezionate ({len(feature_lasso)}): {feature_lasso}")
    print(f"    Lasso — feature eliminate: {coef_lasso[coef_lasso == 0].index.tolist()}")
    print(f"    Ridge — top 5: {coef_ridge.sort_values(ascending=False).head(5).index.tolist()}")

    pd.DataFrame({
        'feature':        X.columns,
        'lasso_coef':     coef_lasso.values,
        'ridge_coef':     coef_ridge.values,
        'lasso_selected': coef_lasso.values > 0
    }).sort_values('lasso_coef', ascending=False).to_csv(
        f'{base_dir}/data/feature_selection/feature_selection.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Feature Selection — Lasso e Ridge',
                 fontsize=13, fontweight='bold')
    coef_lasso.sort_values().plot(
        kind='barh', ax=axes[0], color='#4878CF', edgecolor='white')
    axes[0].set_title('Lasso (L1) — 0 = feature eliminata')
    axes[0].axvline(0, color='black', linewidth=0.8)
    coef_ridge.sort_values().plot(
        kind='barh', ax=axes[1], color='#E07B54', edgecolor='white')
    axes[1].set_title('Ridge (L2) — solo visualizzazione')
    plt.tight_layout()
    plt.savefig(f'{base_dir}/figures/feature_selection/lasso_ridge.png')
    plt.show()

    return feature_lasso, coef_lasso, coef_ridge

# ─────────────────────────────────────────
# LEARNING CURVE
# ─────────────────────────────────────────
def plot_learning_curve(pipeline, X_train, y_train,
                        model_name, base_dir, cv, random_state=42):
    """
    Mostra come train e validation score cambiano
    al crescere del numero di esempi.
    Curve vicine = ok | Curva train alta e validation bassa = overfitting
    """
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=cv, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, random_state=random_state
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    test_mean  = test_scores.mean(axis=1)
    test_std   = test_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-',
             color='#2E86C1', label='Train F1', linewidth=2)
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.15, color='#2E86C1')
    plt.plot(train_sizes, test_mean, 'o-',
             color='#E07B54', label='Validation F1', linewidth=2)
    plt.fill_between(train_sizes,
                     test_mean - test_std,
                     test_mean + test_std,
                     alpha=0.15, color='#E07B54')
    plt.title(f'Learning Curve — {model_name}\n'
              f'Curve vicine = ok | Train alta e Validation bassa = overfitting',
              fontsize=13, fontweight='bold')
    plt.xlabel('Numero di esempi di training')
    plt.ylabel('F1 Score')
    plt.legend(fontsize=11)
    plt.ylim(0.5, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f'{base_dir}/figures/{model_name}/cross_validation/learning_curve.png')
    plt.show()

    pd.DataFrame({
        'train_size': train_sizes,
        'train_mean': train_mean.round(3),
        'train_std':  train_std.round(3),
        'test_mean':  test_mean.round(3),
        'test_std':   test_std.round(3)
    }).to_csv(
        f'{base_dir}/data/{model_name}/cv_results/learning_curve.csv',
        index=False)

# ─────────────────────────────────────────
# ROC CURVE + PRECISION-RECALL CURVE
# ─────────────────────────────────────────
def plot_curve(y_test, y_proba, model_name, base_dir, roc_auc):
    """
    ROC Curve: mostra il tradeoff tra TPR e FPR.
    Più la curva è vicina all'angolo in alto a sinistra, meglio è.
    AUC = 1.0 → perfetto | AUC = 0.5 → casuale

    Precision-Recall Curve: più utile con dataset sbilanciati.
    Mostra il tradeoff tra precision e recall al variare della soglia.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Curve di Performance — {model_name}',
                 fontsize=14, fontweight='bold')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, color='#2E86C1', linewidth=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1,
                 label='Classificatore casuale (AUC = 0.5)')
    axes[0].fill_between(fpr, tpr, alpha=0.1, color='#2E86C1')
    axes[0].set_title('ROC Curve\n'
                       'Più vicina all\'angolo in alto a sinistra = meglio')
    axes[0].set_xlabel('False Positive Rate (FPR)\n'
                        '= falsi allarmi / tutti i no-tsunami')
    axes[0].set_ylabel('True Positive Rate (TPR)\n'
                        '= tsunami trovati / tutti i tsunami reali')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    axes[1].plot(recall, precision, color='#E07B54', linewidth=2,
                 label=f'PR curve (AP = {ap:.3f})')
    axes[1].axhline(y_test.mean(), color='gray', linestyle='--',
                    linewidth=1, label=f'Baseline ({y_test.mean():.2f})')
    axes[1].fill_between(recall, precision, alpha=0.1, color='#E07B54')
    axes[1].set_title('Precision-Recall Curve\n'
                       'Più vicina all\'angolo in alto a destra = meglio')
    axes[1].set_xlabel('Recall\n= tsunami trovati / tutti i tsunami reali')
    axes[1].set_ylabel('Precision\n= tsunami predetti corretti / tutti i predetti')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(f'{base_dir}/figures/{model_name}/curve/roc_pr_curve.png')
    plt.show()

# ─────────────────────────────────────────
# CALIBRATION CURVE
# ─────────────────────────────────────────
def plot_calibration(y_test, y_proba, model_name, base_dir):
    """
    Calibration curve: mostra se le probabilità predette
    corrispondono alle probabilità reali.

    Esempio: se il modello predice 0.8 (80% di probabilità tsunami),
    realmente l'80% di quei casi dovrebbe essere tsunami.

    Curva vicina alla diagonale = probabilità ben calibrate
    Curva sopra la diagonale = modello sottostima
    Curva sotto la diagonale = modello sovrastima
    """
    prob_true, prob_pred = calibration_curve(
        y_test, y_proba, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-',
             color='#2E86C1', linewidth=2, label=model_name)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1,
             label='Calibrazione perfetta')
    plt.fill_between(prob_pred, prob_true, prob_pred,
                     alpha=0.1, color='#2E86C1')
    plt.title(f'Calibration Curve — {model_name}\n'
              f'Vicina alla diagonale = probabilità affidabili',
              fontsize=13, fontweight='bold')
    plt.xlabel('Probabilità predetta dal modello')
    plt.ylabel('Proporzione reale di tsunami')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/figures/{model_name}/curve/calibration.png')
    plt.show()

# ─────────────────────────────────────────
# FUNZIONE SINGOLO MODELLO
# ─────────────────────────────────────────
def addestra_modello(
    model_name, model_config,
    X_train, X_test,
    y_train, y_test,
    X, feature_lasso,
    cv, n_splits,
    random_state, base_dir
):
    print(f"\n{'='*60}")
    print(f"MODELLO: {model_name.upper()}")
    print(f"{'='*60}")

    if model_config['usa_feature_lasso']:
        print(f"\n    Usando {len(feature_lasso)} feature selezionate da Lasso")
        X_train_m     = X_train[feature_lasso]
        X_test_m      = X_test[feature_lasso]
        feature_names = feature_lasso
    else:
        print(f"\n    Usando tutte le {X_train.shape[1]} feature")
        X_train_m     = X_train
        X_test_m      = X_test
        feature_names = X.columns.tolist()

    # ── GRIDSEARCH ──
    print(f"\n[1] GRIDSEARCH")

    pipeline_gs = Pipeline([
        ('scaler', RobustScaler()),
        ('model',  model_config['model'])
    ])

    grid_search = GridSearchCV(
        pipeline_gs, model_config['param_grid'],
        cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train_m, y_train)

    print(f"    Migliori parametri: {grid_search.best_params_}")
    print(f"    Miglior F1 (CV):    {grid_search.best_score_:.3f}")

    pd.DataFrame(grid_search.cv_results_).to_csv(
        f'{base_dir}/data/{model_name}/cv_results/gridsearch.csv', index=False)

    best_params = {
        k.replace('model__', ''): v
        for k, v in grid_search.best_params_.items()
    }
    best_model = model_config['model'].__class__(
        **{**model_config['model'].get_params(), **best_params}
    )
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model',  best_model)
    ])

    # ── CROSS VALIDATION ──
    print(f"\n[2] CROSS VALIDATION ({n_splits}-Fold Stratified)")

    scoring = {
        'accuracy':  'accuracy',
        'f1':        'f1',
        'roc_auc':   'roc_auc',
        'precision': 'precision',
        'recall':    'recall'
    }

    cv_results = cross_validate(
        pipeline, X_train_m, y_train,
        cv=cv, scoring=scoring,
        return_train_score=True
    )

    print(f"\n    {'Metrica':<12} {'Train':>8} {'Test':>8}")
    print(f"    {'-'*30}")
    for metrica in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
        train = cv_results[f'train_{metrica}'].mean()
        test  = cv_results[f'test_{metrica}'].mean()
        print(f"    {metrica:<12} {train:>8.3f} {test:>8.3f}")

    pd.DataFrame({
        m: cv_results[f'test_{m}']
        for m in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
    }).to_csv(
        f'{base_dir}/data/{model_name}/cv_results/cv_results.csv', index=False)

    # ── LEARNING CURVE ──
    print(f"\n[3] LEARNING CURVE")
    plot_learning_curve(
        pipeline, X_train_m, y_train,
        model_name, base_dir, cv, random_state
    )

    # ── VALUTAZIONE SUL TEST SET ──
    print(f"\n[4] VALUTAZIONE SUL TEST SET")

    pipeline.fit(X_train_m, y_train)
    y_pred  = pipeline.predict(X_test_m)
    y_proba = pipeline.predict_proba(X_test_m)[:, 1]

    accuracy = (y_pred == y_test).mean()
    mcc      = matthews_corrcoef(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_proba)

    print(f"\n    Accuracy:  {accuracy:.3f}")
    print(f"    MCC:       {mcc:.3f}")
    print(f"    ROC-AUC:   {roc_auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Tsunami', 'Tsunami'])}")

    pred_df = pd.DataFrame(X_test_m)
    pred_df['y_true']  = y_test.values
    pred_df['y_pred']  = y_pred
    pred_df['y_proba'] = y_proba
    pred_df.to_csv(
        f'{base_dir}/data/{model_name}/predictions/predictions.csv', index=False)

    # ── CONFUSION MATRIX ──
    print(f"\n[5] CONFUSION MATRIX")

    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Confusion Matrix — {model_name}\n'
                 f'FN = tsunami mancato | FP = falso allarme',
                 fontsize=13, fontweight='bold')

    ConfusionMatrixDisplay(
        cm, display_labels=['No Tsunami', 'Tsunami']
    ).plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Valori assoluti')

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ConfusionMatrixDisplay(
        cm_norm.round(2), display_labels=['No Tsunami', 'Tsunami']
    ).plot(ax=axes[1], colorbar=False, cmap='Blues')
    axes[1].set_title('Valori normalizzati')

    fig.text(0.5, -0.02,
             f'MCC: {mcc:.3f}  |  ROC-AUC: {roc_auc:.3f}  |  Accuracy: {accuracy:.3f}',
             ha='center', fontsize=12, fontweight='bold', color='#2E86C1')
    plt.tight_layout()
    plt.savefig(
        f'{base_dir}/figures/{model_name}/confusion_matrix/confusion_matrix.png')
    plt.show()

    # ── ROC + PRECISION-RECALL CURVE ──
    print(f"\n[6] ROC E PRECISION-RECALL CURVE")
    plot_curve(y_test, y_proba, model_name, base_dir, roc_auc)

    # ── CALIBRATION CURVE ──
    print(f"\n[7] CALIBRATION CURVE")
    plot_calibration(y_test, y_proba, model_name, base_dir)

    # ── FEATURE IMPORTANCE ──
    print(f"\n[8] FEATURE IMPORTANCE")

    model_step = pipeline.named_steps['model']
    importanza = None

    if hasattr(model_step, 'feature_importances_'):
        print("    Metodo: Gini impurity (RandomForest)")
        importanza = pd.Series(
            model_step.feature_importances_, index=feature_names
        ).sort_values(ascending=True)
    elif hasattr(model_step, 'coef_'):
        print("    Metodo: coefficienti assoluti (SVM linear)")
        importanza = pd.Series(
            np.abs(model_step.coef_).flatten(), index=feature_names
        ).sort_values(ascending=True)
    else:
        print(f"    {model_name} non supporta feature importance diretta")

    if importanza is not None:
        plt.figure(figsize=(10, 7))
        colors = ['#E07B54' if v >= importanza.mean()
                  else '#4878CF' for v in importanza]
        importanza.plot(kind='barh', color=colors, edgecolor='white')
        plt.title(f'Feature Importance — {model_name}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Importanza')
        plt.axvline(importanza.mean(), color='black', linestyle='--',
                    linewidth=1, label=f'Media ({importanza.mean():.3f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f'{base_dir}/figures/{model_name}/feature_importance/feature_importance.png')
        plt.show()

        importanza.sort_values(ascending=False).to_csv(
            f'{base_dir}/data/{model_name}/cv_results/feature_importance.csv',
            header=['importance'])

        print("\n    Top 5 feature:")
        print(importanza.sort_values(ascending=False).head(5).to_string())

    # ── REPORT ──
    report = {
        'model':           model_name,
        'accuracy':        round(accuracy, 3),
        'mcc':             round(mcc, 3),
        'roc_auc':         round(roc_auc, 3),
        'cv_f1':           round(cv_results['test_f1'].mean(), 3),
        'cv_roc_auc':      round(cv_results['test_roc_auc'].mean(), 3),
        'cv_precision':    round(cv_results['test_precision'].mean(), 3),
        'cv_recall':       round(cv_results['test_recall'].mean(), 3),
        'best_params':     str(grid_search.best_params_),
        'n_features_used': len(feature_names),
        'usa_lasso_feat':  model_config['usa_feature_lasso']
    }

    pd.DataFrame([report]).to_csv(
        f'{base_dir}/reports/{model_name}_report.csv', index=False)

    return report, cv_results

# ─────────────────────────────────────────
# FUNZIONE PRINCIPALE
# ─────────────────────────────────────────
def classifica_terremoti(
    df,
    target='tsunami',
    modelli=None,
    test_size=0.2,
    n_splits=5,
    random_state=42,
    base_dir='outputs'
):
    if modelli is None:
        modelli = MODELLI

    print("=" * 60)
    print("CLASSIFICAZIONE TERREMOTI — PIPELINE COMPLETA")
    print("=" * 60)
    print(f"\nModelli: {list(modelli.keys())}")
    print(f"CV: {n_splits}-Fold Stratified")
    print(f"Test size: {test_size:.0%}")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    print(f"\nTrain: {X_train.shape[0]} esempi ({y_train.mean():.1%} tsunami)")
    print(f"Test:  {X_test.shape[0]} esempi ({y_test.mean():.1%} tsunami)")

    os.makedirs(f'{base_dir}/data/train_test', exist_ok=True)
    os.makedirs(f'{base_dir}/reports', exist_ok=True)
    train_df = X_train.copy(); train_df[target] = y_train.values
    test_df  = X_test.copy();  test_df[target]  = y_test.values
    train_df.to_csv(f'{base_dir}/data/train_test/train.csv', index=False)
    test_df.to_csv(f'{base_dir}/data/train_test/test.csv',   index=False)

    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    feature_lasso, coef_lasso, coef_ridge = feature_selection(
        X_train, y_train, X, random_state, base_dir
    )

    tutti_report = []
    tutti_cv     = {}

    for model_name, model_config in modelli.items():
        crea_cartelle(base_dir, model_name)
        report, cv_results = addestra_modello(
            model_name=model_name,
            model_config=model_config,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            X=X, feature_lasso=feature_lasso,
            cv=cv, n_splits=n_splits,
            random_state=random_state,
            base_dir=base_dir
        )
        tutti_report.append(report)
        tutti_cv[model_name] = cv_results

    # ── CONFRONTO MODELLI ──
    print(f"\n{'='*60}")
    print("CONFRONTO FINALE MODELLI")
    print(f"{'='*60}")

    report_df = pd.DataFrame(tutti_report)
    print(report_df[[
        'model', 'accuracy', 'mcc', 'roc_auc', 'cv_f1', 'cv_roc_auc'
    ]].to_string(index=False))

    # ── ROC CURVE TUTTI I MODELLI ──
    plt.figure(figsize=(9, 7))
    colori = ['#4878CF', '#E07B54', '#639922']
    for (_, row), color in zip(report_df.iterrows(), colori):
        pred = pd.read_csv(
            f"{base_dir}/data/{row['model']}/predictions/predictions.csv")
        fpr, tpr, _ = roc_curve(pred['y_true'], pred['y_proba'])
        plt.plot(fpr, tpr, linewidth=2, color=color,
                 label=f"{row['model']} (AUC={row['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Casuale')
    plt.title('ROC Curve — Confronto Modelli',
              fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/reports/confronto_roc.png')
    plt.show()

    # ── CONFRONTO METRICHE ──
    metriche_conf = ['accuracy', 'mcc', 'roc_auc', 'cv_f1', 'cv_roc_auc']
    fig, axes = plt.subplots(1, len(metriche_conf), figsize=(22, 6))
    fig.suptitle('Confronto Modelli — Tutte le Metriche',
                 fontsize=14, fontweight='bold')

    for i, metrica in enumerate(metriche_conf):
        bars = axes[i].bar(
            report_df['model'], report_df[metrica],
            color=colori[:len(report_df)], edgecolor='white'
        )
        axes[i].set_title(metrica.upper())
        axes[i].set_ylim(0, 1.15)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].axhline(0.9, color='gray', linestyle='--',
                        linewidth=0.8, alpha=0.5)
        for bar, val in zip(bars, report_df[metrica]):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center',
                fontsize=8, fontweight='bold'
            )

    plt.tight_layout()
    plt.savefig(f'{base_dir}/reports/confronto_modelli.png')
    plt.show()

    best = report_df.loc[report_df['mcc'].idxmax()]
    print(f"\n🏆 Miglior modello (MCC): {best['model']}")
    print(f"   MCC:     {best['mcc']:.3f}")
    print(f"   ROC-AUC: {best['roc_auc']:.3f}")
    print(f"   F1 (CV): {best['cv_f1']:.3f}")

    report_df.to_csv(f'{base_dir}/reports/report_finale.csv', index=False)
    print(f"\nTutto salvato in: {base_dir}/")

    return report_df, tutti_cv


# ─────────────────────────────────────────
# ESEGUI
# ─────────────────────────────────────────
df = pd.read_csv('data/processed/earthquake_processed.csv')

report_df, tutti_cv = classifica_terremoti(
    df,
    target='tsunami',
    modelli=MODELLI,
    test_size=0.2,
    n_splits=5,
    random_state=42
)
