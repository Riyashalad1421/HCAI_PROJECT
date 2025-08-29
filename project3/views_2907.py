from sklearn.neighbors import NearestNeighbors
from django.views.decorators.http import require_POST
from sklearn.linear_model import LogisticRegression
import tempfile
import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import sys
import json
import pandas as pd
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
from django.shortcuts import render
import matplotlib
from sklearn.tree import _tree
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot


# Default view for initial page load


def index(request):
    # Get lambda value from request or use default
    lambda_value = request.GET.get('lambda', 0.01)
    try:
        lambda_value = float(lambda_value)
    except ValueError:
        lambda_value = 0.01

    # Load dataset
    penguins_data = load_penguins_data()
    if penguins_data is None:
        return render(request, 'project3/index.html', {
            'error': 'Failed to load the Palmer Penguins dataset. Make sure palmerpenguins is installed.'
        })

    # Prepare (but DO NOT use the pre-scaled X to avoid leakage)
    X_scaled_full, y, feature_names, target_names, X_raw = prepare_data(
        penguins_data)

    # === Split on ORIGINAL features, fit scaler on TRAIN only (no leakage) ===
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # ------------------- Decision Tree (smooth via ccp_alpha) -------------------
    # Build alpha grid from training data, map λ∈[0,1] to a quantile of alphas
    tmp_tree = DecisionTreeClassifier(random_state=42)
    tmp_tree.fit(X_train, y_train)
    path = tmp_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas  # increasing
    # quantile guard for any λ outside [0,1]
    lam01 = min(max(lambda_value, 0.0), 1.0)
    ccp_alpha = float(np.quantile(alphas, lam01))

    tree_model = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=ccp_alpha,
        min_samples_leaf=1 + int(4 * lam01)  # tiny extra smoothness
    )
    tree_model.fit(X_train, y_train)

    tree_y_pred = tree_model.predict(X_test)
    tree_accuracy = accuracy_score(y_test, tree_y_pred)
    tree_leaves = tree_model.get_n_leaves()
    tree_depth = tree_model.get_depth()

    tree_plot = visualize_decision_tree(
        tree_model, feature_names, target_names)
    importance_plot = plot_feature_importance(tree_model, feature_names)

    # ------------------- Logistic Regression (stable) -------------------
    C_value = max(0.001, 1.0 / lambda_value)
    logistic_model = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=C_value,
        max_iter=5000,     # ↑ avoid convergence noise
        tol=1e-4,
        random_state=42
    )
    logistic_model.fit(X_train, y_train)

    logistic_y_pred = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
    logistic_used_features = (logistic_model.coef_ != 0).sum(axis=1).max()
    logistic_plot = plot_logistic_coefficients(logistic_model, feature_names)

    # ------------------- Correlation Heatmap -------------------
    corr_plot = create_correlation_heatmap(penguins_data)

    # ------------------- Context -------------------
    context = {
        # Shared
        'lambda_value': float(lambda_value),
        'features': feature_names,
        'target': 'Species',
        'corr_plot': corr_plot,
        'target_names': target_names,

        # Decision Tree
        'tree_accuracy': round(tree_accuracy * 100, 2),
        'tree_n_leaves': int(tree_leaves),
        'tree_plot': tree_plot,
        'importance_plot': importance_plot,
        'max_depth': int(tree_depth),  # keep this for UI display
        # (optional: you could also show ccp_alpha if you want)
        # 'ccp_alpha': round(ccp_alpha, 6),

        # Logistic Regression
        'logistic_accuracy': round(logistic_accuracy * 100, 2),
        'logistic_n_features': int(logistic_used_features),
        'logistic_plot': logistic_plot,
        'C_value': round(C_value, 4),
    }

    return render(request, 'project3/index.html', context)


@csrf_exempt
def update_model(request):
    """AJAX endpoint to update both Decision Tree and Logistic Regression based on lambda value"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)

    try:
        data = json.loads(request.body)
        lambda_value = float(data.get('lambda', 0.01))

        # Load dataset
        penguins_data = load_penguins_data()
        if penguins_data is None:
            return JsonResponse({'error': 'Failed to load dataset'}, status=400)

        # Prepare (ignore globally scaled X to avoid leakage)
        X_scaled_full, y, feature_names, target_names, X_raw = prepare_data(
            penguins_data)

        # === Split on ORIGINAL features, fit scaler on TRAIN only (no leakage) ===
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # ---------------- Decision Tree (ccp_alpha) ----------------
        tmp_tree = DecisionTreeClassifier(random_state=42)
        tmp_tree.fit(X_train, y_train)
        path = tmp_tree.cost_complexity_pruning_path(X_train, y_train)
        alphas = path.ccp_alphas
        lam01 = min(max(lambda_value, 0.0), 1.0)
        ccp_alpha = float(np.quantile(alphas, lam01))

        tree_model = DecisionTreeClassifier(
            random_state=42,
            ccp_alpha=ccp_alpha,
            min_samples_leaf=1 + int(4 * lam01)
        )
        tree_model.fit(X_train, y_train)

        tree_y_pred = tree_model.predict(X_test)
        tree_accuracy = accuracy_score(y_test, tree_y_pred)
        tree_leaves = tree_model.get_n_leaves()
        tree_depth = tree_model.get_depth()

        tree_plot = visualize_decision_tree(
            tree_model, feature_names, target_names)
        importance_plot = plot_feature_importance(tree_model, feature_names)

        # ---------------- Logistic Regression (stable) ----------------
        C_value = max(0.001, 1.0 / lambda_value)
        logistic_model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=C_value,
            max_iter=5000,
            tol=1e-4,
            random_state=42
        )
        logistic_model.fit(X_train, y_train)

        logistic_y_pred = logistic_model.predict(X_test)
        logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
        logistic_used_features = (logistic_model.coef_ != 0).sum(axis=1).max()

        logistic_plot = plot_logistic_coefficients(
            logistic_model, feature_names)

        # ---------------- Return Both Results ----------------
        return JsonResponse({
            'lambda': float(lambda_value),

            # Decision Tree outputs
            'tree_accuracy': round(tree_accuracy * 100, 2),
            'tree_n_leaves': int(tree_leaves),
            'tree_plot': tree_plot,
            'importance_plot': importance_plot,
            'max_depth': int(tree_depth),
            # (optional) also return ccp_alpha if you want to display it
            # 'ccp_alpha': round(ccp_alpha, 6),

            # Logistic Regression outputs
            'logistic_accuracy': round(logistic_accuracy * 100, 2),
            'logistic_n_features': int(logistic_used_features),
            'logistic_plot': logistic_plot,
            'C_value': round(C_value, 4),
        })

    except Exception as e:
        print(f"Error in update_model: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def load_penguins_data():
    """Load the Palmer Penguins dataset."""
    try:
        from palmerpenguins import load_penguins
        penguins = load_penguins()
        return penguins
    except ImportError:
        # If palmerpenguins is not installed, fall back to a sample dataset
        try:
            import pandas as pd
            import sklearn.datasets

            # Try to get Iris as a fallback (for testing)
            iris = sklearn.datasets.load_iris(as_frame=True)
            df = iris['data']
            df['species'] = pd.Series(iris['target']).map({
                0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'
            })
            return df
        except Exception as e:
            print(f"Error loading fallback dataset: {e}")
            return None


def prepare_data(df):
    """Prepare the data for modeling."""
    # Get feature names
    if 'species' in df.columns:
        target_col = 'species'
    else:
        target_col = 'species'

    # Select only numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Create X and y
    X_raw = df[numeric_cols].copy()  # Keep raw version for GOSDT

    # Handle missing values if any
    X_raw = X_raw.fillna(X_raw.mean())

    # Extract target
    if target_col in df.columns:
        y = df[target_col].copy()
    else:
        # Fallback if column name is different
        y = df.iloc[:, -1].copy()

    # Get unique target names
    target_names = sorted(y.unique())

    # Encode target if needed
    if not pd.api.types.is_numeric_dtype(y):
        y_map = {name: i for i, name in enumerate(target_names)}
        y = y.map(y_map)

    # Scale features for sklearn models
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    return X, y, numeric_cols, target_names, X_raw.values


def visualize_decision_tree(model, feature_names, class_names):
    """Create a visualization of the decision tree."""
    # Adjusted figure size to better fit laptop screens
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8  # Smaller font size for better fit
    )
    plt.title("Decision Tree for Palmer Penguins Dataset")

    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    plt.close()  # Close the figure to free memory

    return base64.b64encode(image_png).decode('utf-8')


def create_correlation_heatmap(df):
    """Create a correlation heatmap of the features."""
    # Adjusted figure size to better fit laptop screens
    plt.figure(figsize=(8, 6))

    # Get only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation
    corr = numeric_df.corr()

    # Create heatmap with adjusted font size
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=0.5, annot_kws={"size": 8})
    plt.title("Feature Correlation Matrix")

    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    plt.close()  # Close the figure to free memory

    return base64.b64encode(image_png).decode('utf-8')


def plot_feature_importance(model, feature_names):
    """Create a bar plot of feature importances."""
    # Adjusted figure size to better fit laptop screens
    plt.figure(figsize=(8, 5))

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [
               feature_names[i] for i in indices], rotation=45, ha='right', fontsize=9)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in Decision Tree')
    plt.tight_layout()

    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    plt.close()  # Close the figure to free memory

    return base64.b64encode(image_png).decode('utf-8')


def plot_logistic_coefficients(model, feature_names):

    import io
    import base64

    # Calculate mean absolute coefficients across classes
    mean_coeffs = np.mean(np.abs(model.coef_), axis=0)  # shape: (n_features,)

    # Sort features by importance
    indices = np.argsort(mean_coeffs)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_coeffs = mean_coeffs[indices]

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(sorted_coeffs)), sorted_coeffs)
    plt.xticks(range(len(sorted_coeffs)),
               sorted_features, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Average Absolute Coefficient')
    plt.title('Feature Importance (Logistic Regression)')
    plt.tight_layout()

    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    # Return base64 string
    return base64.b64encode(image_png).decode('utf-8')


def _fit_models_for_lambda(lambda_value, X_train, y_train, feature_names, target_names):
    """Train both models for a given lambda (shared with counterfactuals)."""
    # Decision Tree
    max_depth = max(2, min(10, int(10 - lambda_value * 9)))
    tree_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree_model.fit(X_train, y_train)

    # Logistic Regression (L1 sparsity)
    C_value = max(0.001, 1.0 / lambda_value)
    logistic_model = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=C_value,
        max_iter=5000,
        random_state=42
    )
    logistic_model.fit(X_train, y_train)

    return tree_model, logistic_model, max_depth, C_value


def _mad_per_feature(X_raw):
    """Median Absolute Deviation per feature (in original/unscaled space)."""
    med = np.median(X_raw, axis=0)
    mad = np.median(np.abs(X_raw - med), axis=0)
    # guard against zeros
    mad[mad == 0] = np.maximum(
        1e-6, np.std(X_raw[:, mad == 0], axis=0, ddof=1))
    return mad


def _mad_weighted_l1(x, c, mad):
    """∑ |c_j - x_j| / MAD_j in original (unscaled) space."""
    return float(np.sum(np.abs(c - x) / mad))


def _clip_to_data_range(cands, mins, maxs):
    return np.clip(cands, mins, maxs)


def _model_code_maps(model, target_names):
    """
    Returns (codes, name_by_code, code_by_name) where:
      - codes: list of class codes in model.classes_ order
      - name_by_code: {code -> human-readable name from target_names}
      - code_by_name: {human-readable name -> code}
    """
    codes = list(getattr(model, "classes_", []))
    name_by_code = {int(c): str(target_names[int(c)]) for c in codes}
    code_by_name = {v: k for k, v in name_by_code.items()}
    return codes, name_by_code, code_by_name


@csrf_exempt
@require_POST
def counterfactuals(request):
    """
    Expects JSON with:
      lambda, instance_index, target_label, model_type ('tree'|'logistic'),
      N, k, immutables[], only_increase[], only_decrease[],
      enforce_plausibility, density_quantile, enforce_diversity, diversity_min_l1
    """
    try:
        data = json.loads(request.body)
        lambda_value = float(data.get("lambda", 0.01))
        instance_index = int(data.get("instance_index", 0))
        target_label = data.get("target_label")
        model_type = str(data.get("model_type", "logistic"))
        N = int(data.get("N", 2000))
        k = int(data.get("k", 5))

        enforce_plaus = bool(data.get("enforce_plausibility", False))
        dens_q = float(data.get("density_quantile", 0.7))
        enforce_div = bool(data.get("enforce_diversity", False))
        div_min_l1 = float(data.get("diversity_min_l1", 0.2))

        # --- Load + prepare data ---
        penguins = load_penguins_data()
        if penguins is None:
            return JsonResponse({"error": "Failed to load dataset"}, status=400)

        X_scaled_full, y, feature_names, target_names, X_raw = prepare_data(
            penguins)
        feature_names = list(map(str, feature_names))
        valid_feats = set(feature_names)

        # Actionability sets (sanitize to valid feature names)
        immutables = set(map(str, data.get("immutables", []))) & valid_feats
        only_inc = {f for f in map(str, data.get(
            "only_increase", [])) if f in valid_feats and f not in immutables}
        only_dec = {f for f in map(str, data.get(
            "only_decrease", [])) if f in valid_feats and f not in immutables}

        # --- Split on ORIGINAL features; fit scaler on TRAIN only ---
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # --- Train tree with smooth pruning (ccp_alpha) ---
        tmp_tree = DecisionTreeClassifier(
            random_state=42).fit(X_train, y_train)
        path = tmp_tree.cost_complexity_pruning_path(X_train, y_train)
        alphas = path.ccp_alphas
        lam01 = min(max(lambda_value, 0.0), 1.0)
        ccp_alpha = float(np.quantile(alphas, lam01))

        tree_model = DecisionTreeClassifier(
            random_state=42, ccp_alpha=ccp_alpha, min_samples_leaf=1 + int(4 * lam01)
        ).fit(X_train, y_train)

        # --- Train logistic (stable) ---
        C_value = max(0.001, 1.0 / lambda_value)
        logistic_model = LogisticRegression(
            penalty='l1', solver='saga', C=C_value, max_iter=5000, tol=1e-4, random_state=42
        ).fit(X_train, y_train)

        # --- Choose model for CF classification ---
        model = tree_model if model_type == "tree" else logistic_model

        # --- Instance selection ---
        n = X_raw.shape[0]
        instance_index = max(0, min(n - 1, instance_index))
        x_orig = X_raw[instance_index].astype(float)
        X1_scaled = scaler.transform(x_orig.reshape(1, -1))

        # --- Class mapping: use model class CODES (critical) ---
        codes, name_by_code, code_by_name = _model_code_maps(
            model, target_names)
        if target_label not in code_by_name:
            return JsonResponse({"error": f"target_label must be one of {list(code_by_name.keys())}"}, status=400)
        target_code = code_by_name[target_label]

        # --- Instance panel data ---
        # Tree predicted name via codes
        t_codes, t_name_by_code, _ = _model_code_maps(tree_model, target_names)
        tree_pred_code = int(tree_model.predict(X1_scaled)[0])
        tree_pred_name = t_name_by_code.get(
            tree_pred_code, str(tree_pred_code))
        # Logistic predicted name via codes
        l_codes, l_name_by_code, _ = _model_code_maps(
            logistic_model, target_names)
        logit_pred_code = int(logistic_model.predict(X1_scaled)[0])
        logit_pred_name = l_name_by_code.get(
            logit_pred_code, str(logit_pred_code))

        instance_summary = {
            "features": {feat: float(val) for feat, val in zip(feature_names, x_orig)},
            "tree_pred": tree_pred_name,
            "tree_proba": _probas_to_list(tree_model, X1_scaled, target_names),
            "logit_pred": logit_pred_name,
            "logit_proba": _probas_to_list(logistic_model, X1_scaled, target_names),
            "tree_path": _tree_decision_path(tree_model, scaler, x_orig, feature_names),
            "logit_top_contributors": _logistic_top_contributors(
                logistic_model, scaler, x_orig, feature_names, top_k=5
            ),
        }

        # --- Deterministic tree recourse (to target label) ---
        tree_recourse = _tree_minimal_recourse(
            tree_model, scaler, x_orig, feature_names, target_id=target_code)

        # --- Local sampling with auto-escalation if zero matches ---
        mins = X_train_raw.min(axis=0)
        maxs = X_train_raw.max(axis=0)
        med = np.median(X_train_raw, axis=0)
        mad = np.median(np.abs(X_train_raw - med), axis=0)
        mad[mad == 0] = 1e-6

        rng = np.random.default_rng(42)
        scales = [0.5, 0.9, 1.3]   # grow radius if needed
        Ns = [N, max(N, 6000), max(N, 12000)]

        selected = None
        tried = []
        for s, n_try in zip(scales, Ns):
            noise = rng.normal(0.0, s * mad, size=(n_try, X_raw.shape[1]))
            # actionability constraints
            feat_idx = {f: j for j, f in enumerate(feature_names)}
            for f in feature_names:
                j = feat_idx[f]
                if f in immutables:
                    noise[:, j] = 0.0
                elif f in only_inc:
                    noise[:, j] = np.abs(noise[:, j])
                elif f in only_dec:
                    noise[:, j] = -np.abs(noise[:, j])

            candidates_orig = np.clip(x_orig + noise, mins, maxs)
            preds = model.predict(scaler.transform(candidates_orig))
            mask = (preds == target_code)
            tried.append(int(np.sum(mask)))
            if np.any(mask):
                selected = candidates_orig[mask]
                break

        if selected is None:
            return JsonResponse({
                "instance_index": instance_index,
                "target_label": target_label,
                "model_type": model_type,
                "lambda": lambda_value,
                "found": 0, "counterfactuals": [],
                "feature_names": list(map(str, feature_names)),
                "instance_summary": instance_summary,
                "tree_action_plan": tree_recourse,
                "debug": {"matched_per_scale": tried, "scales": scales, "Ns": Ns}
            })

        # --- Plausibility (optional kNN density on TRAIN original space) ---
        if enforce_plaus:
            nbrs = NearestNeighbors(n_neighbors=10).fit(X_train_raw)
            dists, _ = nbrs.kneighbors(selected)
            avg_d = dists.mean(axis=1)
            keep = avg_d < np.quantile(avg_d, min(max(dens_q, 0.0), 1.0))
            selected = selected[keep]
            if selected.shape[0] == 0:
                return JsonResponse({
                    "instance_index": instance_index,
                    "target_label": target_label,
                    "model_type": model_type,
                    "lambda": lambda_value,
                    "found": 0, "counterfactuals": [],
                    "feature_names": list(map(str, feature_names)),
                    "instance_summary": instance_summary,
                    "tree_action_plan": tree_recourse,
                    "debug": {"note": "All filtered by plausibility"}
                })

        # --- Rank by MAD-L1 and (optional) enforce diversity ---
        dists_cf = np.sum(np.abs(selected - x_orig) / mad, axis=1)
        order = np.argsort(dists_cf)

        if enforce_div:
            picked, picked_idx = [], []
            for idx in order:
                cand = selected[idx]
                if all(np.sum(np.abs(cand - p) / mad) >= div_min_l1 for p in picked):
                    picked.append(cand)
                    picked_idx.append(idx)
                if len(picked) >= k:
                    break
            chosen = np.array(picked)
            chosen_d = dists_cf[picked_idx] if picked_idx else np.array([])
        else:
            top = order[:k]
            chosen = selected[top]
            chosen_d = dists_cf[top]

        # --- Target probabilities for table (use code's column) ---
        if chosen.size and hasattr(model, "predict_proba"):
            pos = list(model.classes_).index(target_code)
            target_probas = model.predict_proba(
                scaler.transform(chosen))[:, pos]
        else:
            target_probas = np.full(
                (chosen.shape[0] if chosen.size else 0,), np.nan)

        # --- Build payload ---
        feature_keys = list(map(str, feature_names))
        cf_records = []
        for i, (row, dist) in enumerate(zip(chosen, chosen_d)):
            cf_records.append({
                "features": {feat: float(val) for feat, val in zip(feature_keys, row)},
                "mad_weighted_l1": float(dist),
                "target_prob": (None if np.isnan(target_probas[i]) else float(target_probas[i]))
            })

        return JsonResponse({
            "instance_index": instance_index,
            "target_label": target_label,
            "model_type": model_type,
            "lambda": lambda_value,
            "found": len(cf_records),
            "counterfactuals": cf_records,
            "feature_names": feature_keys,
            "instance_summary": instance_summary,
            "tree_action_plan": tree_recourse
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def _probas_to_list(clf, X1, target_names):
    """
    Return [{'label': name, 'p': prob}, ...] using the model's class order.
    Assumes target_names is the label-encoder order used to build y.
    """
    if not hasattr(clf, "predict_proba"):
        return []

    p = clf.predict_proba(X1)[0]
    # model.classes_ are integer codes (e.g., [0,1,2])
    order = list(getattr(clf, "classes_", range(len(p))))
    # Map each class code -> string name using the encoder order in target_names
    names = [str(target_names[i]) for i in order]
    return [{"label": names[i], "p": float(p[i])} for i in range(len(p))]


def _tree_decision_path(model, scaler, x_orig, feature_names):
    """
    Deterministic path that x takes through the tree.
    Each step shows the feature, operator, threshold (scaled & original), and that it's satisfied.
    """
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    left = tree.children_left
    right = tree.children_right

    x_scaled = scaler.transform([x_orig])[0]
    path = []
    node = 0
    while left[node] != _tree.TREE_LEAF:  # not a leaf
        f = feature[node]
        thr = threshold[node]
        thr_orig = thr * scaler.scale_[f] + scaler.mean_[f]
        go_left = x_scaled[f] <= thr
        path.append({
            "feature": str(feature_names[f]),
            "op": "<=" if go_left else ">",
            "threshold_scaled": float(thr),
            "threshold_orig": float(thr_orig),
            "satisfied": True
        })
        node = left[node] if go_left else right[node]
    return path


def _tree_minimal_recourse(model, scaler, x_orig, feature_names, target_id):
    """
    Minimal L1 (in ORIGINAL units) change that sends x to a leaf whose majority class == target_id.
    Returns {'cost_L1': float, 'steps': [{feature, from, to, delta}, ...]} or None.
    """
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    left = tree.children_left
    right = tree.children_right
    value = tree.value

    # collect all target-class leaves with their (node, dir) path
    leaves = []

    def walk(node, path):
        if left[node] == _tree.TREE_LEAF:
            if value[node][0].argmax() == target_id:
                leaves.append(path.copy())
            return
        path.append((node, 'L'))
        walk(left[node], path)
        path.pop()
        path.append((node, 'R'))
        walk(right[node], path)
        path.pop()
    walk(0, [])
    if not leaves:
        return None

    x_scaled = scaler.transform([x_orig])[0]

    def plan_for(path):
        # deltas in scaled units needed to satisfy every split along the path
        deltas = {}
        for node, dir_ in path:
            f = feature[node]
            thr = threshold[node]
            if f < 0:  # safety (shouldn't happen)
                continue
            if dir_ == 'L' and not (x_scaled[f] <= thr):
                deltas[f] = min(deltas.get(f, float('inf')),
                                thr - x_scaled[f] - 1e-6)
            if dir_ == 'R' and not (x_scaled[f] > thr):
                deltas[f] = max(deltas.get(f, float('-inf')),
                                thr - x_scaled[f] + 1e-6)
        # convert to ORIGINAL units
        delta_scaled = np.zeros_like(x_scaled)
        for j, d in deltas.items():
            delta_scaled[j] = d
        delta_orig = delta_scaled * scaler.scale_
        cost = float(np.sum(np.abs(delta_orig)))
        steps = []
        for j, d in enumerate(delta_orig):
            if abs(d) > 1e-9:
                steps.append({
                    "feature": str(feature_names[j]),
                    "from": float(x_orig[j]),
                    "to": float(x_orig[j] + d),
                    "delta": float(d)
                })
        return cost, steps

    plans = [plan_for(p) for p in leaves]
    if not plans:
        return None
    cost, steps = sorted(plans, key=lambda t: t[0])[0]
    return {"cost_L1": cost, "steps": steps}


def _logistic_top_contributors(log_model, scaler, x_orig, feature_names, top_k=5):
    """
    Per-instance contributions for the predicted class: w_cj * x_scaled_j.
    """
    x_scaled = scaler.transform([x_orig])[0]
    # predicted class index (consistent with clf.predict)
    c_idx = int(log_model.predict([x_scaled])[0])
    weights = log_model.coef_[c_idx]  # (n_features,)
    contrib = weights * x_scaled
    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    return [{
        "feature": str(feature_names[i]),
        "contribution": float(contrib[i]),
        "abs_contribution": float(abs(contrib[i])),
        "weight": float(weights[i]),
        "x_scaled": float(x_scaled[i])
    } for i in order]
