import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

TASK_DIR = r"C:\Users\33075\Desktop\学\课程\python programming\作业\Final Project\SuperAgent-main\SuperAgent-main\runs\coupon_train_val_a79iy05m"
WORK_DIR = r"C:\Users\33075\Desktop\学\课程\python programming\作业\Final Project\SuperAgent-main\SuperAgent-main\runs\task1_campus_trial\run_20261122_132426\workspace"
os.makedirs(WORK_DIR, exist_ok=True)
RS = 42

COLS = ["temperature","time","coupon","expiration","gender","age","maritalStatus","has_children","education","occupation","income","Bar","CoffeeHouse","CarryAway","RestaurantLessThan20","Restaurant20To50","toCoupon_GEQ5min","toCoupon_GEQ15min","toCoupon_GEQ25min","direction_same","direction_opp"]
KEEP_RAW = [c for c in COLS if c not in ("toCoupon_GEQ5min","direction_opp")]
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
ORD_COLS = list(ORD.keys())
FEAT_ORDER = ORD_COLS + NUM_COLS + NOM_COLS
CAT_MASK = np.array([c in NOM_COLS for c in FEAT_ORDER])

def to_df(arr):
    X = pd.DataFrame(arr, columns=COLS)
    for c in COLS:
        X[c] = X[c].astype(str)
    return X

def prep_tree(df, enc=None, fit=False):
    X = pd.DataFrame(index=df.index)
    for c in ORD_COLS:
        X[c] = df[c].map(ORD[c]).fillna(-1).astype(float)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    nom = df[NOM_COLS].astype(str)
    if fit:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(nom)
    codes = enc.transform(nom)
    for i, c in enumerate(NOM_COLS):
        X[c] = codes[:, i].astype(float)
    X = X[FEAT_ORDER]
    return X, enc

def raw_view(df):
    return df[KEEP_RAW].copy()

C_tr = np.load(os.path.join(TASK_DIR, "C_train.npy"), allow_pickle=True)
C_va = np.load(os.path.join(TASK_DIR, "C_val.npy"), allow_pickle=True)
y_tr = np.load(os.path.join(TASK_DIR, "y_train.npy")).astype(int)
y_va = np.load(os.path.join(TASK_DIR, "y_val.npy")).astype(int)
df_tr, df_va = to_df(C_tr), to_df(C_va)

tree_tr, enc = prep_tree(df_tr, fit=True)
tree_va, _ = prep_tree(df_va, enc=enc)
raw_tr, raw_va = raw_view(df_tr), raw_view(df_va)

lr = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")),
               ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RS))])
lr.fit(raw_tr, y_tr)
lr_acc = accuracy_score(y_va, lr.predict(raw_va))
print("LR val acc:", lr_acc)

hgb = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
    categorical_features=CAT_MASK, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=50, random_state=RS)
hgb.fit(tree_tr, y_tr)
p_hgb = hgb.predict_proba(tree_va)[:, 1]
hgb_acc = accuracy_score(y_va, (p_hgb >= 0.5).astype(int))
print("HGB val acc:", hgb_acc)

if hgb_acc >= lr_acc:
    best_name = "hgb"
    best_thr, best_acc = 0.5, hgb_acc
    for t in np.arange(0.30, 0.71, 0.02):
        a = accuracy_score(y_va, (p_hgb >= t).astype(int))
        if a > best_acc:
            best_acc, best_thr = a, float(t)
    val_proba, best_model = p_hgb, hgb
else:
    best_name = "lr"
    p_lr = lr.predict_proba(raw_va)[:, 1]
    best_thr, best_acc = 0.5, lr_acc
    for t in np.arange(0.30, 0.71, 0.02):
        a = accuracy_score(y_va, (p_lr >= t).astype(int))
        if a > best_acc:
            best_acc, best_thr = a, float(t)
    val_proba, best_model = p_lr, lr
val_pred = (val_proba >= best_thr).astype(int)
print("best:", best_name, "thr:", best_thr, "acc:", best_acc)

np.save(os.path.join(WORK_DIR, "val_predictions.npy"), val_pred)
joblib.dump({"kind": best_name, "model": best_model, "threshold": best_thr,
             "enc": enc if best_name == "hgb" else None}, os.path.join(WORK_DIR, "model.joblib"))
with open(os.path.join(WORK_DIR, "metrics.json"), "w") as f:
    json.dump({"val_metric": float(best_acc), "model": best_name, "threshold": best_thr}, f)
print("VAL_METRIC:", float(best_acc))

if os.environ.get("FINAL") == "1":
    C_te = np.load(os.path.join(TASK_DIR, "C_test.npy"), allow_pickle=True)
    df_te = to_df(C_te)
    df_all = pd.concat([df_tr, df_va], ignore_index=True)
    y_all = np.concatenate([y_tr, y_va])
    if best_name == "hgb":
        tree_all, enc2 = prep_tree(df_all, fit=True)
        tree_te, _ = prep_tree(df_te, enc=enc2)
        final_m = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            categorical_features=CAT_MASK, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=50, random_state=RS)
        final_m.fit(tree_all, y_all)
        p_te = final_m.predict_proba(tree_te)[:, 1]
    else:
        raw_all = raw_view(df_all)
        raw_te = raw_view(df_te)
        final_m = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")),
                            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RS))])
        final_m.fit(raw_all, y_all)
        p_te = final_m.predict_proba(raw_te)[:, 1]
    te_pred = (p_te >= best_thr).astype(int)
    np.save(os.path.join(WORK_DIR, "test_predictions.npy"), te_pred)
    yt_path = os.path.join(TASK_DIR, "y_test.npy")
    if os.path.exists(yt_path):
        y_te = np.load(yt_path).astype(int)
        print("TEST_METRIC:", float(accuracy_score(y_te, te_pred)))
