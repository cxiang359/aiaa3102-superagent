import os, json, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
import joblib

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

TASK_DIR = r"C:\Users\33075\Desktop\学\课程\python programming\作业\Final Project\SuperAgent-main\SuperAgent-main\runs\coupon_train_val_a79iy05m"
WORK_DIR = r"C:\Users\33075\Desktop\学\课程\python programming\作业\Final Project\SuperAgent-main\SuperAgent-main\runs\task1_campus_trial\run_20261122_132426\workspace"
os.makedirs(WORK_DIR, exist_ok=True)
RS = 42
COLS = ["temperature","time","coupon","expiration","gender","age","maritalStatus","has_children","education","occupation","income","Bar","CoffeeHouse","CarryAway","RestaurantLessThan20","Restaurant20To50","toCoupon_GEQ5min","toCoupon_GEQ15min","toCoupon_GEQ25min","direction_same","direction_opp"]
NOM_COLS = ["coupon","gender","maritalStatus","occupation"]
NUM_COLS = ["has_children","toCoupon_GEQ15min","toCoupon_GEQ25min","direction_same"]
ORD = {
 "temperature": {"30.0":0,"30":0,"55.0":1,"55":1,"80.0":2,"80":2},
 "time": {"7AM":0,"10AM":1,"2PM":2,"6PM":3,"10PM":4},
 "expiration": {"2h":0,"1d":1},
 "age": {"below21":0,"21":1,"26":2,"31":3,"36":4,"41":5,"46":6,"50plus":7},
 "education": {"Some High School":0,"High School Graduate":1,"Some college - no degree":2,"Associates degree":3,"Bachelors degree":4,"Graduate degree (Masters or Doctorate)":5},
 "income": {"Less than $12500":0,"$12500 - $24999":1,"$25000 - $37499":2,"$37500 - $49999":3,"$50000 - $62499":4,"$62500 - $74999":5,"$75000 - $87499":6,"$87500 - $99999":7,"$100000 or More":8},
 "Bar": {"never":0,"less1":1,"1~3":2,"4~8":3,"gt8":4},
 "CoffeeHouse": {"never":0,"less1":1,"1~3":2,"4~8":3,"gt8":4},
 "CarryAway": {"never":0,"less1":1,"1~3":2,"4~8":3,"gt8":4},
 "RestaurantLessThan20": {"never":0,"less1":1,"1~3":2,"4~8":3,"gt8":4},
 "Restaurant20To50": {"never":0,"less1":1,"1~3":2,"4~8":3,"gt8":4},
}

def load(s):
    return pd.DataFrame(np.load(os.path.join(TASK_DIR, f"C_{s}.npy"), allow_pickle=True), columns=COLS).astype(str)

def prep(df, enc=None, fit=False):
    X = pd.DataFrame(index=df.index)
    for c, m in ORD.items():
        X[c] = df[c].map(m).fillna(-1).astype(float)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    nom = df[NOM_COLS]
    if fit:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(nom)
    codes = enc.transform(nom)
    for i, c in enumerate(NOM_COLS):
        X[c] = codes[:, i].astype(float)
    return X, enc

def fit_cat(df, y):
    m = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, cat_features=list(range(df.shape[1])), random_seed=RS, verbose=0, thread_count=-1, allow_writing_files=False)
    m.fit(df, y)
    return m

def fit_lgb(X, y):
    m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.9, colsample_bytree=0.9, random_state=RS, verbose=-1, n_jobs=-1)
    m.fit(X, y)
    return m

def fit_hgb(X, y):
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=RS)
    m.fit(X, y)
    return m

df_tr = load("train")
df_va = load("val")
y_tr = np.load(os.path.join(TASK_DIR, "y_train.npy")).astype(int)
y_va = np.load(os.path.join(TASK_DIR, "y_val.npy")).astype(int)
X_tr, enc = prep(df_tr, fit=True)
X_va, _ = prep(df_va, enc=enc)

models = {}
probs = {}
if CatBoostClassifier is not None:
    try:
        models["cat"] = fit_cat(df_tr, y_tr)
        probs["cat"] = models["cat"].predict_proba(df_va)[:, 1]
    except Exception as e:
        print("catboost skipped:", e)
if lgb is not None:
    try:
        models["lgb"] = fit_lgb(X_tr, y_tr)
        probs["lgb"] = models["lgb"].predict_proba(X_va)[:, 1]
    except Exception as e:
        print("lightgbm skipped:", e)
models["hgb"] = fit_hgb(X_tr, y_tr)
probs["hgb"] = models["hgb"].predict_proba(X_va)[:, 1]
for nm, p in probs.items():
    print(nm, accuracy_score(y_va, (p >= 0.5).astype(int)))

best = None
for r in range(1, len(probs) + 1):
    for combo in itertools.combinations(probs, r):
        p = np.mean([probs[k] for k in combo], axis=0)
        for t in np.arange(0.30, 0.71, 0.02):
            a = accuracy_score(y_va, (p >= t).astype(int))
            if best is None or a > best[0]:
                best = (a, combo, float(t))
best_acc, best_combo, best_thr = best
val_pred = (np.mean([probs[k] for k in best_combo], axis=0) >= best_thr).astype(int)
print("VAL_METRIC:", best_acc)

np.save(os.path.join(WORK_DIR, "val_predictions.npy"), val_pred)
joblib.dump({"combo": best_combo, "threshold": best_thr, "models": {k: models[k] for k in best_combo}, "enc": enc}, os.path.join(WORK_DIR, "model.joblib"))
with open(os.path.join(WORK_DIR, "metrics.json"), "w") as f:
    json.dump({"val_metric": float(best_acc), "combo": list(best_combo), "threshold": best_thr}, f)

if os.environ.get("FINAL") == "1":
    df_te = load("test")
    df_all = pd.concat([df_tr, df_va], ignore_index=True)
    y_all = np.concatenate([y_tr, y_va])
    X_all, enc2 = prep(df_all, fit=True)
    X_te, _ = prep(df_te, enc=enc2)
    ps = []
    for k in best_combo:
        if k == "cat":
            m = fit_cat(df_all, y_all); ps.append(m.predict_proba(df_te)[:, 1])
        elif k == "lgb":
            m = fit_lgb(X_all, y_all); ps.append(m.predict_proba(X_te)[:, 1])
        else:
            m = fit_hgb(X_all, y_all); ps.append(m.predict_proba(X_te)[:, 1])
    p_te = np.mean(ps, axis=0)
    te_pred = (p_te >= best_thr).astype(int)
    np.save(os.path.join(WORK_DIR, "test_predictions.npy"), te_pred)
    yt = os.path.join(TASK_DIR, "y_test.npy")
    if os.path.exists(yt):
        print("TEST_METRIC:", float(accuracy_score(np.load(yt).astype(int), te_pred)))
