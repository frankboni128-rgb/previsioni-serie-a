"""Previsioni Serie A — eseguito automaticamente da GitHub Actions.

Versione 2:
- impara da 5 campionati (Serie A, Premier League, Liga, Bundesliga, Ligue 1), prevede la Serie A
- usa anche le quote Pinnacle (il bookmaker più preciso) e il confronto tra bookmaker
- prova sul passato l'effetto delle formazioni ufficiali (valore degli 11 titolari)
Scrive i risultati in docs/ (pagina web su GitHub Pages)."""
import json, os, time, io, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
from scipy.optimize import minimize_scalar, brentq
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# ==== Impostazioni
CAMPIONATI = {   # codice football-data: (nome, codice Understat, codice Transfermarkt)
    "I1": ("Serie A", "Serie_A", "IT1"),
    "E0": ("Premier League", "EPL", "GB1"),
    "SP1": ("Liga", "La_liga", "ES1"),
    "D1": ("Bundesliga", "Bundesliga", "L1"),
    "F1": ("Ligue 1", "Ligue_1", "FR1"),
}
PRINCIPALE = "I1"
PRIMA_STAGIONE, STAGIONE_CORRENTE = 2008, 2026
INIZIO_MODELLI = "1011"
TEST_STAGIONI = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
PRIMO_TEST = TEST_STAGIONI[0]
PRIMA_XG = 2014
FINESTRA_GG, XI, HALFLIFE_XG, MAXG = 730, 0.002, 8, 10
CLASSI = ["1", "2", "X"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
# Se il calendario automatico manca, partite di Serie A da aggiungere a mano (nomi come su football-data):
# PARTITE_MANUALI = [("2026-10-04", "Inter", "Milan")]
PARTITE_MANUALI = []
NOTE = []
T0 = time.time()

def codice(anno):
    return f"{anno % 100:02d}{(anno + 1) % 100:02d}"
COD_CORRENTE = codice(STAGIONE_CORRENTE)

def tempo(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)

# ==== Dati football-data (5 campionati): risultati e quote di più bookmaker
GRUPPI_QUOTE = [
    (["q1", "qX", "q2"], [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("PSH", "PSD", "PSA")]),
    (["qp1", "qpX", "qp2"], [("PSH", "PSD", "PSA")]),                                   # Pinnacle, in anticipo
    (["qm1", "qmX", "qm2"], [("MaxH", "MaxD", "MaxA"), ("BbMxH", "BbMxD", "BbMxA")]),   # migliore quota sul mercato
    (["q_over", "q_under"], [("B365>2.5", "B365<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("P>2.5", "P<2.5")]),
    (["qs1", "qsX", "qs2"], [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]),  # chiusura precisa
    (["qc1", "qcX", "qc2"], [("B365CH", "B365CD", "B365CA")]),                          # chiusura Bet365
    (["qmc1", "qmcX", "qmc2"], [("MaxCH", "MaxCD", "MaxCA")]),                          # chiusura migliore quota
]
COL_QUOTE = [c for dsts, _ in GRUPPI_QUOTE for c in dsts]

def leggi_csv(contenuto):
    raw = pd.read_csv(io.StringIO(contenuto.decode("latin-1")), on_bad_lines="skip")
    raw.columns = raw.columns.str.replace("ï»¿", "", regex=False).str.strip()
    return raw

def leggi_quote(raw, out):
    for dst in COL_QUOTE:
        out[dst] = np.nan
    for dsts, liste in GRUPPI_QUOTE:
        for cols in liste:
            if all(c in raw for c in cols):
                for dst, src in zip(dsts, cols):
                    out[dst] = out[dst].fillna(pd.Series(pd.to_numeric(raw[src], errors="coerce").values, index=out.index))
    return out

def scarica(div, cod):
    try:
        r = requests.get(f"https://www.football-data.co.uk/mmz4281/{cod}/{div}.csv", timeout=30)
    except Exception:
        return None
    if r.status_code != 200 or len(r.content) < 500:
        return None
    raw = leggi_csv(r.content).dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).reset_index(drop=True)
    out = pd.DataFrame({
        "lega": div, "stagione": cod, "data": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "casa": raw["HomeTeam"].astype(str).str.strip(), "trasf": raw["AwayTeam"].astype(str).str.strip(),
        "gol_c": raw["FTHG"].astype(float), "gol_t": raw["FTAG"].astype(float)})
    return leggi_quote(raw, out)

blocchi = []
for div in CAMPIONATI:
    n = 0
    for anno in range(PRIMA_STAGIONE, STAGIONE_CORRENTE + 1):
        s = scarica(div, codice(anno))
        if s is not None:
            blocchi.append(s); n += len(s)
        time.sleep(0.3)
    tempo(f"{CAMPIONATI[div][0]}: {n} partite")
giocate = pd.concat(blocchi, ignore_index=True)
giocate["futura"] = False

# ==== Prossima giornata di Serie A
oggi = pd.Timestamp.today().normalize()
fut = pd.DataFrame()
try:
    r = requests.get("https://www.football-data.co.uk/fixtures.csv", timeout=30)
    raw = leggi_csv(r.content)
    raw = raw[raw["Div"] == PRINCIPALE].reset_index(drop=True)
    fut = pd.DataFrame({"lega": PRINCIPALE, "stagione": COD_CORRENTE, "ora": raw["Time"].astype(str) if "Time" in raw else "",
                        "data": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
                        "casa": raw["HomeTeam"].astype(str).str.strip(), "trasf": raw["AwayTeam"].astype(str).str.strip()})
    fut = leggi_quote(raw, fut)
except Exception as e:
    print("Calendario automatico non disponibile:", e)
    NOTE.append("Calendario della prossima giornata non disponibile su football-data.")
if PARTITE_MANUALI:
    man = pd.DataFrame(PARTITE_MANUALI, columns=["data", "casa", "trasf"]).assign(
        lega=PRINCIPALE, stagione=COD_CORRENTE, data=lambda x: pd.to_datetime(x.data))
    fut = pd.concat([fut, man], ignore_index=True)
if fut.empty:
    fut = pd.DataFrame(columns=["lega", "stagione", "data", "casa", "trasf"] + COL_QUOTE)
    fut["data"] = pd.to_datetime(fut["data"])
fut = fut[fut.data >= oggi].drop_duplicates(["casa", "trasf"])
fut = fut.merge(giocate[["lega", "stagione", "casa", "trasf"]], how="left", indicator=True)
fut = fut[fut._merge == "left_only"].drop(columns="_merge")
fut["gol_c"] = fut["gol_t"] = np.nan
fut["futura"] = True
print(f"Partite da prevedere: {len(fut)}")

df = pd.concat([giocate, fut], ignore_index=True).sort_values(["data", "lega", "casa"]).reset_index(drop=True)
for c in ["gol_c", "gol_t"] + COL_QUOTE:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["futura"] = df["futura"].astype(bool)
df["casa"], df["trasf"] = df["casa"].astype(object), df["trasf"].astype(object)
df["kc"], df["kt"] = (df.lega + "|" + df.casa).astype(object), (df.lega + "|" + df.trasf).astype(object)
df["esito"] = np.where(df.futura, None, np.select([df.gol_c > df.gol_t, df.gol_c == df.gol_t], ["1", "X"], "2"))
df["match_id"] = df.index

def prob_da_quote(cols):
    inv = 1 / df[cols]
    return inv.div(inv.sum(axis=1), axis=0).values
df[["p_book_1", "p_book_X", "p_book_2"]] = prob_da_quote(["q1", "qX", "q2"])
df[["p_pin_1", "p_pin_X", "p_pin_2"]] = prob_da_quote(["qp1", "qpX", "qp2"])
df[["p_book_over", "p_book_under"]] = prob_da_quote(["q_over", "q_under"])
df["book_1v2"], df["book_X"] = np.log(df.p_book_1 / df.p_book_2), np.log(df.p_book_X)
df["pin_1v2"], df["pin_X"] = np.log(df.p_pin_1 / df.p_pin_2), np.log(df.p_pin_X)
# quanto Pinnacle e gli altri bookmaker non sono d'accordo
df["dis_1v2"] = df.pin_1v2 - df.book_1v2
tempo(f"Totale partite: {len(df)}")

# ==== Elo (separato per campionato)
def elo(d, K, HFA, regressione=0.25):
    rating, attive, righe, stag, base = {}, set(), [], None, 1500.0
    for r in d.itertuples():
        if r.stagione != stag:
            if attive:
                prec = {s: rating[s] for s in attive}
                media = np.mean(list(prec.values()))
                base = float(np.mean(sorted(prec.values())[:3]))
                rating = {s: v * (1 - regressione) + media * regressione for s, v in prec.items()}
            attive, stag = set(), r.stagione
        rc = rating.setdefault(r.casa, base); rt = rating.setdefault(r.trasf, base)
        attive.update([r.casa, r.trasf])
        diff = rc + HFA - rt
        e = 1 / (1 + 10 ** (-diff / 400))
        s = 1.0 if r.gol_c > r.gol_t else 0.5 if r.gol_c == r.gol_t else 0.0
        gd = abs(r.gol_c - r.gol_t)
        delta = K * (1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8) * (s - e)
        rating[r.casa], rating[r.trasf] = rc + delta, rt - delta
        righe.append((rc, rt, diff, e))
    return pd.DataFrame(righe, columns=["elo_c", "elo_t", "elo_diff", "elo_exp"], index=d.index), rating

for c in ["elo_c", "elo_t", "elo_diff", "elo_exp"]:
    df[c] = np.nan
RATING, HFA_LEGA = {}, {}
for L in CAMPIONATI:
    gio = df[(df.lega == L) & ~df.futura]
    if gio.empty:
        continue
    punt = gio.esito.map({"1": 1.0, "X": 0.5, "2": 0.0})
    m_tar = (gio.stagione >= INIZIO_MODELLI) & (gio.stagione < PRIMO_TEST)
    K, H, _ = min(((K, H, ((elo(gio, K, H)[0].elo_exp - punt)[m_tar] ** 2).mean())
                   for K in [10, 15, 20, 25, 30, 35] for H in [30, 50, 70, 90]), key=lambda x: x[2])
    E, RATING[L] = elo(gio, K, H)
    HFA_LEGA[L] = H
    df.loc[E.index, E.columns] = E
for i in df.index[df.futura]:
    L = df.at[i, "lega"]
    rc, rt = RATING.get(L, {}).get(df.at[i, "casa"], np.nan), RATING.get(L, {}).get(df.at[i, "trasf"], np.nan)
    df.loc[i, ["elo_c", "elo_t", "elo_diff"]] = [rc, rt, rc + HFA_LEGA.get(L, 60) - rt]
tempo("Elo calcolato")

# ==== xG da Understat (5 campionati)
HEAD = dict(UA, **{"X-Requested-With": "XMLHttpRequest", "Referer": "https://understat.com/"})
def scarica_xg(lega_u, anno):
    r = requests.get(f"https://understat.com/getLeagueData/{lega_u}/{anno}", headers=HEAD, timeout=30)
    r.raise_for_status()
    j = r.json()
    righe = []
    for m in (j["dates"] if isinstance(j, dict) else j):
        if str(m.get("isResult")).lower() != "true":
            continue
        righe.append({"stagione": codice(anno), "data_u": pd.to_datetime(m["datetime"]).normalize(),
                      "u_casa": m["h"]["title"], "u_trasf": m["a"]["title"],
                      "u_gol_c": float(m["goals"]["h"]), "u_gol_t": float(m["goals"]["a"]),
                      "xg_c": float(m["xG"]["h"]), "xg_t": float(m["xG"]["a"])})
    return pd.DataFrame(righe)

pezzi_xg = []
for L, (nome, lega_u, _) in CAMPIONATI.items():
    bx = []
    for anno in range(PRIMA_XG, STAGIONE_CORRENTE + 1):
        try:
            bx.append(scarica_xg(lega_u, anno))
        except Exception as e:
            print(nome, codice(anno), "xG errore:", e)
        time.sleep(0.8)
    if not bx:
        continue
    xg = pd.concat(bx, ignore_index=True)
    sub = df[df.lega == L]
    m = xg.merge(sub[["stagione", "data", "casa", "trasf", "gol_c", "gol_t"]],
                 left_on=["stagione", "data_u", "u_gol_c", "u_gol_t"], right_on=["stagione", "data", "gol_c", "gol_t"])
    coppie = pd.concat([m[["u_casa", "casa"]].set_axis(["u", "f"], axis=1), m[["u_trasf", "trasf"]].set_axis(["u", "f"], axis=1)])
    mappa = coppie.value_counts().reset_index(name="n").drop_duplicates("u").set_index("u")["f"]
    xg["lega"], xg["casa"], xg["trasf"] = L, xg.u_casa.map(mappa), xg.u_trasf.map(mappa)
    pezzi_xg.append(xg[["lega", "stagione", "casa", "trasf", "xg_c", "xg_t"]].dropna().drop_duplicates(["stagione", "casa", "trasf"]))
    tempo(f"xG {nome}: {len(xg)} partite")
USA_XG = sum(len(p) for p in pezzi_xg) > 1000
if USA_XG:
    df = df.merge(pd.concat(pezzi_xg, ignore_index=True), on=["lega", "stagione", "casa", "trasf"], how="left")
else:
    df["xg_c"] = df["xg_t"] = np.nan
    NOTE.append("Expected goals (Understat) non disponibili in questo aggiornamento: modello un po' meno preciso.")

lungo = pd.concat([
    pd.DataFrame({"match_id": df.match_id, "data": df.data, "squadra": df.kc, "in_casa": 1, "xg_f": df.xg_c, "xg_s": df.xg_t}),
    pd.DataFrame({"match_id": df.match_id, "data": df.data, "squadra": df.kt, "in_casa": 0, "xg_f": df.xg_t, "xg_s": df.xg_c})
]).sort_values(["squadra", "data", "match_id"])
g = lungo.groupby("squadra")
for c in ["xg_f", "xg_s"]:
    lungo[f"ewm_{c}"] = g[c].transform(lambda x: x.shift(1).ewm(halflife=HALFLIFE_XG, min_periods=3).mean())
for pref, flag in [("c_", 1), ("t_", 0)]:
    df = df.join(lungo[lungo.in_casa == flag].set_index("match_id")[["ewm_xg_f", "ewm_xg_s"]].add_prefix(pref), on="match_id")
df["diff_xg"] = (df.c_ewm_xg_f - df.c_ewm_xg_s) - (df.t_ewm_xg_f - df.t_ewm_xg_s)
del lungo

# ==== Valori di mercato storici e formazioni (transfermarkt-datasets, fermo a luglio 2026)
TM = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{}.csv.gz"
def tm(nome, cols):
    r = requests.get(TM.format(nome), headers=UA, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"{nome}: il server risponde {r.status_code}")
    t = pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory=False)
    mancano = [c for c in cols if c not in t.columns]
    if mancano:
        raise ValueError(f"{nome}: colonne mancanti {mancano}. Colonne presenti: {list(t.columns)}")
    tempo(f"{nome}: {t.shape}")
    return t[cols]

TM_LEGA = {v[2]: k for k, v in CAMPIONATI.items()}
games = tm("games", ["game_id", "competition_id", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"])
val = tm("player_valuations", ["player_id", "date", "market_value_in_eur"])
trf = tm("transfers", ["player_id", "transfer_date", "from_club_id", "to_club_id"])

gi = games[games.competition_id.isin(TM_LEGA)].copy()
gi["lega"] = gi.competition_id.map(TM_LEGA)
gi["date"] = pd.to_datetime(gi.date, errors="coerce")
m = gi.merge(df[["lega", "data", "kc", "kt", "gol_c", "gol_t"]],
             left_on=["lega", "date", "home_club_goals", "away_club_goals"], right_on=["lega", "data", "gol_c", "gol_t"])
coppie = pd.concat([m[["home_club_id", "kc"]].set_axis(["id", "chiave"], axis=1),
                    m[["away_club_id", "kt"]].set_axis(["id", "chiave"], axis=1)])
id_chiave = (coppie.value_counts().reset_index(name="n").drop_duplicates("id").drop_duplicates("chiave")
             .set_index("id")["chiave"])
ids = set(id_chiave.index)
tempo(f"Squadre collegate a Transfermarkt: {len(ids)}")

trf = trf.assign(data=pd.to_datetime(trf.transfer_date, errors="coerce")).dropna(subset=["data"])
val = val.assign(data=pd.to_datetime(val.date, errors="coerce")).dropna(subset=["data"])
v_ord = val.sort_values("data")[["player_id", "data", "market_value_in_eur"]].rename(columns={"data": "data_val"})
giocatori = trf.loc[trf.to_club_id.isin(ids) | trf.from_club_id.isin(ids), "player_id"].unique()
foto = pd.date_range("2012-07-01", "2026-07-01", freq="MS")                 # una "fotografia" al mese
griglia = pd.MultiIndex.from_product([giocatori, foto], names=["player_id", "data"]).to_frame(index=False).sort_values("data")
griglia = pd.merge_asof(griglia, trf[trf.player_id.isin(giocatori)].sort_values("data")[["player_id", "data", "to_club_id"]],
                        on="data", by="player_id")
griglia = griglia[griglia.to_club_id.isin(ids)]
griglia = pd.merge_asof(griglia.sort_values("data"), v_ord[v_ord.player_id.isin(giocatori)],
                        left_on="data", right_on="data_val", by="player_id")
griglia = griglia[(griglia.data - griglia.data_val).dt.days <= 400]
griglia = griglia.sort_values(["to_club_id", "data", "market_value_in_eur"], ascending=[True, True, False])
griglia["pos"] = griglia.groupby(["to_club_id", "data"]).cumcount()
rose = (griglia.groupby(["to_club_id", "data"])
        .agg(valore=("market_value_in_eur", "sum"), n=("player_id", "size")).reset_index())
top11 = griglia[griglia.pos < 11].groupby(["to_club_id", "data"]).market_value_in_eur.sum().rename("top11").reset_index()
rose = rose.merge(top11, on=["to_club_id", "data"])
rose = rose[rose.n >= 15]
rose["chiave"] = rose.to_club_id.map(id_chiave).astype(object)
del griglia
tempo("Rose ricostruite")

rs = rose.sort_values("data")[["chiave", "data", "valore"]].rename(columns={"data": "data_foto"})
df = df.sort_values("data")
for lato, pref in [("kc", "c_"), ("kt", "t_")]:
    df = pd.merge_asof(df, rs.rename(columns={"chiave": lato, "valore": pref + "valore", "data_foto": pref + "foto"}),
                       left_on="data", right_on=pref + "foto", by=lato, allow_exact_matches=False)
df = df.sort_values("match_id").reset_index(drop=True)
vecchio = ((df.data - df.c_foto).dt.days > 60) | ((df.data - df.t_foto).dt.days > 60)
df["diff_valore"] = np.where(vecchio, np.nan, np.log(df.c_valore) - np.log(df.t_valore))

# Formazioni ufficiali: valore degli 11 titolari rispetto agli 11 migliori della rosa
USA_XI = False
df["diff_xi"] = np.nan
try:
    lu = tm("game_lineups", ["game_id", "player_id", "club_id", "type"])
    lu = lu[lu.game_id.isin(gi.game_id) & lu.type.astype(str).str.contains("start", case=False)]
    lu = lu.merge(gi[["game_id", "date"]], on="game_id").rename(columns={"date": "data"}).sort_values("data")
    lu = pd.merge_asof(lu, v_ord[v_ord.player_id.isin(lu.player_id.unique())],
                       left_on="data", right_on="data_val", by="player_id")
    lu = lu[(lu.data - lu.data_val).dt.days <= 400]
    xi = lu.groupby(["game_id", "club_id", "data"]).agg(xi=("market_value_in_eur", "sum"), n=("player_id", "size")).reset_index()
    xi = xi[xi.n >= 10].sort_values("data")
    t11 = top11.rename(columns={"to_club_id": "club_id", "data": "data_foto"}).sort_values("data_foto")
    xi = pd.merge_asof(xi, t11, left_on="data", right_on="data_foto", by="club_id", allow_exact_matches=False)
    xi["rapporto"] = (xi.xi / xi.top11).clip(0.2, 1.5)
    gx = gi[["game_id", "date", "home_club_id", "away_club_id"]].copy()
    gx["kc"], gx["kt"] = gx.home_club_id.map(id_chiave), gx.away_club_id.map(id_chiave)
    gx = gx.merge(xi[["game_id", "club_id", "rapporto"]].rename(columns={"club_id": "home_club_id", "rapporto": "r_c"}),
                  on=["game_id", "home_club_id"], how="left")
    gx = gx.merge(xi[["game_id", "club_id", "rapporto"]].rename(columns={"club_id": "away_club_id", "rapporto": "r_t"}),
                  on=["game_id", "away_club_id"], how="left")
    gx["diff_xi_n"] = np.log(gx.r_c) - np.log(gx.r_t)
    gx = gx.dropna(subset=["kc", "kt", "diff_xi_n"]).rename(columns={"date": "data"})
    df = df.drop(columns="diff_xi").merge(gx[["data", "kc", "kt", "diff_xi_n"]].drop_duplicates(["data", "kc", "kt"]),
                                          on=["data", "kc", "kt"], how="left").rename(columns={"diff_xi_n": "diff_xi"})
    df = df.sort_values("match_id").reset_index(drop=True)
    USA_XI = df.diff_xi.notna().sum() > 2000
    del lu, xi
    tempo(f"Formazioni: {df.diff_xi.notna().sum()} partite con valore degli 11 titolari")
except Exception as e:
    print("Formazioni non disponibili:", e)
COPERTURA_XI = {s_: round(float(df.loc[(df.stagione == s_) & (df.lega == PRINCIPALE), "diff_xi"].notna().mean()), 2)
                for s_ in TEST_STAGIONI}

# Valori delle rose di Serie A della stagione in corso: file docs/valori_rose.csv
F_VAL = "docs/valori_rose.csv"
squadre_correnti = set(df.casa[(df.stagione == COD_CORRENTE) & (df.lega == PRINCIPALE)]) | \
                   set(df.trasf[(df.stagione == COD_CORRENTE) & (df.lega == PRINCIPALE)])
valori_ora = pd.Series(dtype=float)
if os.path.exists(F_VAL):
    vf = pd.read_csv(F_VAL)
    vf["squadra"] = vf["squadra"].astype(str).str.strip()
    sconosciute_vf = sorted(set(vf.squadra) - squadre_correnti)
    if sconosciute_vf:
        NOTE.append("Nel file valori_rose.csv questi nomi non corrispondono a squadre della Serie A attuale: "
                    + ", ".join(sconosciute_vf) + ". Controlla come sono scritti.")
    valori_ora = vf.set_index("squadra")["valore_milioni"].astype(float) * 1e6
    agg = pd.to_datetime(vf["aggiornato"], errors="coerce").max()
    if pd.notna(agg) and (oggi - agg).days > 150:
        NOTE.append(f"Il file valori_rose.csv è fermo al {agg.strftime('%d/%m/%Y')}: conviene aggiornarlo.")
manca_val = sorted(squadre_correnti - set(valori_ora.index))
if manca_val:
    NOTE.append("Valore della rosa mancante per: " + ", ".join(manca_val) + ". Le loro partite sono previste senza valori di mercato.")
fm = df.futura
df.loc[fm, "diff_valore"] = (np.log(df.loc[fm, "casa"].map(valori_ora)) - np.log(df.loc[fm, "trasf"].map(valori_ora))).values

# ==== Modello di Poisson (per campionato, su gol e xG)
def stima_lambda(d, col_c, col_t, alpha=0.005):
    lam = np.full((len(d), 2), np.nan)
    for _, idx in d.groupby(d.data.dt.to_period("W")).groups.items():
        inizio = d.data[idx].min()
        if d.stagione[idx[0]] < INIZIO_MODELLI:
            continue
        p = d[(d.data < inizio) & (d.data >= inizio - pd.Timedelta(days=FINESTRA_GG)) & d[col_c].notna()]
        if len(p) < 300:
            continue
        def righe(x):
            X = pd.concat([
                pd.DataFrame({"att": x.casa.values, "dif": x.trasf.values, "casa": 1, "elo": (x.elo_c - x.elo_t).values / 100}),
                pd.DataFrame({"att": x.trasf.values, "dif": x.casa.values, "casa": 0, "elo": (x.elo_t - x.elo_c).values / 100})],
                ignore_index=True)
            return pd.get_dummies(X, columns=["att", "dif"], dtype=float)
        X = righe(p)
        y = np.concatenate([p[col_c].values, p[col_t].values])
        w = np.tile(np.exp(-XI * (inizio - p.data).dt.days.values), 2)
        mdl = PoissonRegressor(alpha=alpha, max_iter=500).fit(X, y, sample_weight=w)
        pr = mdl.predict(righe(d.loc[idx]).reindex(columns=X.columns, fill_value=0.0))
        lam[idx, 0], lam[idx, 1] = pr[:len(idx)], pr[len(idx):]
    return lam

def matrice(lc, lt, rho):
    lc, lt = np.atleast_1d(lc).astype(float), np.atleast_1d(lt).astype(float)
    k = np.arange(MAXG + 1)
    P = poisson.pmf(k[None, :, None], lc[:, None, None]) * poisson.pmf(k[None, None, :], lt[:, None, None])
    P[:, 0, 0] *= 1 - lc * lt * rho; P[:, 0, 1] *= 1 + lc * rho
    P[:, 1, 0] *= 1 + lt * rho;      P[:, 1, 1] *= 1 - rho
    P = np.clip(P, 0, None)
    return P / P.sum(axis=(1, 2), keepdims=True)

def p1x2(P):
    p1 = np.tril(P, -1).sum(axis=(1, 2)); pX = np.trace(P, axis1=1, axis2=2)
    return np.column_stack([p1, 1 - p1 - pX, pX])

RHO = {}
for c in ["lam_c", "lam_t", "lamxg_c", "lamxg_t", "p_pois_1", "p_pois_2", "p_pois_X", "p_xg_1", "p_xg_2", "p_xg_X"]:
    df[c] = np.nan
for L in CAMPIONATI:
    sub = df[df.lega == L].reset_index()
    if sub.empty:
        continue
    for col_c, col_t, nome, dst_l, dst_p in [("gol_c", "gol_t", "gol", ["lam_c", "lam_t"], ["p_pois_1", "p_pois_2", "p_pois_X"]),
                                             ("xg_c", "xg_t", "xg", ["lamxg_c", "lamxg_t"], ["p_xg_1", "p_xg_2", "p_xg_X"])]:
        if nome == "xg" and not USA_XG:
            continue
        lam = stima_lambda(sub, col_c, col_t)
        ok = ~np.isnan(lam[:, 0])
        tr = ok & (sub.stagione < PRIMO_TEST).values & sub.esito.notna().values
        if tr.sum() < 200:
            continue
        rho = minimize_scalar(lambda r: log_loss(sub.esito[tr], np.clip(p1x2(matrice(lam[tr, 0], lam[tr, 1], r)), 1e-9, 1),
                                                 labels=CLASSI), bounds=(-0.3, 0.3), method="bounded").x
        RHO[(L, nome)] = rho
        pr = np.full((len(sub), 3), np.nan)
        pr[ok] = p1x2(matrice(lam[ok, 0], lam[ok, 1], rho))
        df.loc[sub["index"].values, dst_l] = lam
        df.loc[sub["index"].values, dst_p] = pr
    tempo(f"Poisson {CAMPIONATI[L][0]}")
for pref in ["pois"] + (["xg"] if USA_XG else []):
    df[f"{pref}_1v2"] = np.log(df[f"p_{pref}_1"] / df[f"p_{pref}_2"])
    df[f"{pref}_X"] = np.log(df[f"p_{pref}_X"])
RHO_A = RHO.get((PRINCIPALE, "gol"), 0.05)

# ==== Esperimenti: cosa migliora davvero il modello? (test su 7 stagioni)
BASE = ["elo_diff", "pois_1v2", "pois_X"] + (["xg_1v2", "xg_X", "diff_xg"] if USA_XG else [])
BOOK = ["book_1v2", "book_X"]
PIN = ["pin_1v2", "pin_X", "dis_1v2"]
FEAT_A = BASE + ["diff_valore"] + BOOK
prima_train = codice(PRIMA_XG + 1) if USA_XG else "1314"
TUTTE = list(CAMPIONATI)
VERSIONI = {
    "Modello attuale (solo Serie A)": (FEAT_A, [PRINCIPALE]),
    "+ altri 4 campionati": (FEAT_A, TUTTE),
    "+ quote Pinnacle": (FEAT_A + PIN, TUTTE),
}
if USA_XI:
    VERSIONI["+ formazioni ufficiali"] = (FEAT_A + PIN + ["diff_xi"], TUTTE)

def allena(train, cols):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(train[cols], train.esito)

def prepara(d):
    d = d.copy()
    d["diff_xi"] = d["diff_xi"].fillna(0.0)      # formazione non nota = nessuna informazione
    return d

pool = prepara(df[df.esito.notna() & (df.stagione >= prima_train)].dropna(subset=FEAT_A + PIN))
test_all = pool[pool.stagione.isin(TEST_STAGIONI)]
PRED = {}
for nome_v, (cols, leghe) in VERSIONI.items():
    out = []
    for ts in TEST_STAGIONI:
        train = pool[(pool.stagione < ts) & pool.lega.isin(leghe)]
        te = test_all[test_all.stagione == ts] if len(leghe) > 1 else test_all[(test_all.stagione == ts) & (test_all.lega == PRINCIPALE)]
        if len(te) and len(train) > 500:
            out.append(pd.DataFrame(allena(train, cols).predict_proba(te[cols]), index=te.index, columns=CLASSI))
    PRED[nome_v] = pd.concat(out) if out else pd.DataFrame(columns=CLASSI)
    tempo(f"Test: {nome_v}")

def vinte_di(d):
    return d.esito.to_numpy(dtype=object)[:, None] == np.array(CLASSI, dtype=object)[None, :]

def qarr(d, cols):
    return d[cols].to_numpy(dtype=float)

def clv(p, q, qs_):
    inv = 1 / qs_
    pf = inv / inv.sum(axis=1, keepdims=True)
    pk = (p * q - 1 > 0) & np.isfinite(q) & np.isfinite(qs_).all(axis=1)[:, None]
    return (float((q[pk] * pf[pk]).mean() - 1) if pk.sum() else None), int(pk.sum())

def resa(p, q, paga, vinte, soglia=0.05):
    paga = np.where(np.isfinite(paga), paga, q)
    pk = (p * q - 1 > soglia) & np.isfinite(q)
    n = int(pk.sum())
    return (float(np.where(vinte[pk], paga[pk] - 1, -1).sum() / n) if n else None), n

FRAZ_KELLY, MAX_GIOCATA, MAX_GIORNO = 0.25, 0.05, 0.30
def frazioni_kelly(p, quote):
    ev_ = p * quote - 1
    ev_ = np.where(np.isfinite(ev_), ev_, -1)
    k = ev_.argmax(axis=1)
    e = ev_[np.arange(len(p)), k]
    qk = quote[np.arange(len(p)), k]
    f = np.where(e > 0, np.minimum(FRAZ_KELLY * e / (qk - 1), MAX_GIOCATA), 0.0)
    return k, f, e

def simula_kelly(d, p, q, paga, budget=100.0):
    k, f, _ = frazioni_kelly(p, q)
    paga = np.where(np.isfinite(paga), paga, q)
    vinte = vinte_di(d)
    t = pd.DataFrame({"data": d.data.values, "f": f, "vinta": vinte[np.arange(len(k)), k], "quota": paga[np.arange(len(k)), k]})
    banca, picco, ribasso, minimo, n, nv, storia = budget, budget, 0.0, budget, 0, 0, []
    for giorno, g_ in t.groupby("data", sort=True):
        g_ = g_[g_.f > 0]
        if g_.empty:
            continue
        scala = min(1.0, MAX_GIORNO / g_.f.sum())
        puntate = banca * g_.f.values * scala
        banca += np.where(g_.vinta.values, puntate * (g_.quota.values - 1), -puntate).sum()
        n += len(g_); nv += int(g_.vinta.sum())
        picco = max(picco, banca); minimo = min(minimo, banca); ribasso = max(ribasso, 1 - banca / picco)
        storia.append([str(pd.Timestamp(giorno).date()), round(float(banca), 2)])
    return {"budget_iniziale": budget, "finale": round(float(banca), 2), "giocate": n, "vinte": nv,
            "minimo": round(float(minimo), 2), "ribasso_massimo": round(float(ribasso), 3), "andamento": storia}

Q1X2, QS, QC = ["q1", "q2", "qX"], ["qs1", "qs2", "qsX"], ["qc1", "qc2", "qcX"]
ESPERIMENTI = []
for nome_v, pr in PRED.items():
    ta = test_all.loc[pr.index]
    ma = (ta.lega == PRINCIPALE).values
    P_ = pr.values
    riga = {"versione": nome_v}
    for tag, msk in [("serie_a", ma), ("cinque", np.ones(len(ta), bool))]:
        if msk.sum() < 100 or (tag == "cinque" and ma.all()):
            continue
        d_ = ta[msk]; p_ = P_[msk]
        okc = np.isfinite(qarr(d_, QS)).all(axis=1)
        riga[f"errore_{tag}"] = round(float(log_loss(d_.esito, p_, labels=CLASSI)), 4)
        riga[f"errore_chiusura_{tag}"] = round(float(log_loss(d_.esito[okc], qarr(d_, QS)[okc] ** -1 / (qarr(d_, QS)[okc] ** -1).sum(1, keepdims=True), labels=CLASSI)), 4)
        riga[f"errore_modello_su_chiusura_{tag}"] = round(float(log_loss(d_.esito[okc], p_[okc], labels=CLASSI)), 4)
        c_, n_ = clv(p_, qarr(d_, Q1X2), qarr(d_, QS))
        riga[f"clv_{tag}"] = None if c_ is None else round(c_, 4)
        r_, n2 = resa(p_, qarr(d_, Q1X2), qarr(d_, QC), vinte_di(d_))
        riga[f"resa_{tag}"] = None if r_ is None else round(r_, 4); riga[f"giocate_{tag}"] = n2
    d_a = ta[ma]
    riga["kelly_serie_a"] = simula_kelly(d_a, P_[ma], qarr(d_a, Q1X2), qarr(d_a, QC))["finale"]
    ESPERIMENTI.append(riga)
print(pd.DataFrame(ESPERIMENTI).to_string(index=False))

# Versione per le previsioni: la migliore tra quelle usabili prima della partita (le formazioni arrivano troppo tardi)
usabili = [e for e in ESPERIMENTI if e["versione"] != "+ formazioni ufficiali"]
SCELTA = min(usabili, key=lambda e: e["errore_serie_a"])["versione"]
COLS_SCELTA, LEGHE_SCELTA = VERSIONI[SCELTA]
tempo(f"Versione scelta per le previsioni: {SCELTA}")

# ==== Strategia "Pinnacle contro gli altri bookmaker" (nessun modello: solo quote)
def strategia_pinnacle(d, col_q, col_paga):
    pin = qarr(d, ["qp1", "qp2", "qpX"])
    inv = 1 / pin
    pf = inv / inv.sum(axis=1, keepdims=True)
    q = qarr(d, col_q)
    righe = []
    for soglia in (0.0, 0.02, 0.05):
        r_, n_ = resa(pf, q, qarr(d, col_paga), vinte_di(d), soglia)
        c_, _ = clv(np.where(pf * q - 1 > soglia, pf, 0), q, qarr(d, QS))
        righe.append({"soglia": f">{soglia:.0%}", "giocate": n_, "resa": None if r_ is None else round(r_, 4),
                      "clv": None if c_ is None else round(c_, 4)})
    return righe
base_pin = pool[pool.stagione.isin(TEST_STAGIONI)].dropna(subset=["qp1", "qp2", "qpX"])
PINNACLE = {
    "bet365_serie_a": strategia_pinnacle(base_pin[base_pin.lega == PRINCIPALE], Q1X2, QC),
    "migliore_serie_a": strategia_pinnacle(base_pin[base_pin.lega == PRINCIPALE].dropna(subset=["qm1"]), ["qm1", "qm2", "qmX"], ["qmc1", "qmc2", "qmcX"]),
    "bet365_cinque": strategia_pinnacle(base_pin, Q1X2, QC),
    "migliore_cinque": strategia_pinnacle(base_pin.dropna(subset=["qm1"]), ["qm1", "qm2", "qmX"], ["qmc1", "qmc2", "qmcX"]),
}
print(json.dumps(PINNACLE, indent=1))

# ==== Risultati della versione scelta (Serie A): confronti, simulazioni, stagioni, Kelly
pr = PRED[SCELTA]
tt = test_all.loc[pr.index]
tt = tt[tt.lega == PRINCIPALE]
P_comb = pr.loc[tt.index].values
P_book = qarr(tt, ["p_book_1", "p_book_2", "p_book_X"])
q, qc, qs = qarr(tt, Q1X2), qarr(tt, QC), qarr(tt, QS)
vinte = vinte_di(tt)
LL_MOD, LL_BOOK = log_loss(tt.esito, P_comb, labels=CLASSI), log_loss(tt.esito, P_book, labels=CLASSI)
P_senza = PRED["Modello attuale (solo Serie A)"].reindex(tt.index)
LL_SENZA = log_loss(tt.esito, P_senza.values, labels=CLASSI) if P_senza.notna().all().all() else None
ACC_MOD = accuracy_score(tt.esito, np.array(CLASSI)[P_comb.argmax(1)])
ACC_BOOK = accuracy_score(tt.esito, np.array(CLASSI)[P_book.argmax(1)])

def simula(prob, quote, vinte_, soglie=(0.0, 0.05, 0.10, 0.20), paga=None):
    prob, quote, vinte_ = np.asarray(prob, float), np.asarray(quote, float), np.asarray(vinte_, bool)
    paga = quote if paga is None else np.where(np.isfinite(np.asarray(paga, float)), np.asarray(paga, float), quote)
    righe = []
    for s in soglie:
        ev_ = prob * quote - 1
        punta = (ev_ > s) & np.isfinite(quote)
        n = int(punta.sum())
        guad = np.where(vinte_[punta], paga[punta] - 1, -1).sum() if n else 0.0
        righe.append({"soglia_valore": f">{s:.0%}", "giocate": n, "vinte": int(vinte_[punta].sum()),
                      "profitto_unita": round(float(guad), 1), "ROI": f"{guad / n:+.1%}" if n else "—"})
    return pd.DataFrame(righe)

SIM_1X2 = simula(P_comb.ravel(), q.ravel(), vinte.ravel())
okc = np.isfinite(qs).all(axis=1)
p_chius = (1 / qs) / (1 / qs).sum(axis=1, keepdims=True)
CHIUS = {"partite": int(okc.sum())}
SIM_CHIUS = pd.DataFrame()
if okc.sum() >= 100:
    CHIUS["log_loss_modello"] = round(float(log_loss(tt.esito[okc], P_comb[okc], labels=CLASSI)), 4)
    CHIUS["log_loss_chiusura"] = round(float(log_loss(tt.esito[okc], p_chius[okc], labels=CLASSI)), 4)
    c_, n_ = clv(P_comb, q, qs)
    CHIUS["clv_0"], CHIUS["giocate_clv_0"] = (None if c_ is None else round(c_, 4)), n_
    SIM_CHIUS = simula(P_comb.ravel(), q.ravel(), vinte.ravel(), paga=qc.ravel())

PER_STAGIONE = []
ev5 = P_comb * q - 1
for st_ in TEST_STAGIONI:
    m_ = (tt.stagione == st_).values
    if m_.sum() < 50:
        continue
    riga = {"stagione": f"20{st_[:2]}/{st_[2:]}", "partite": int(m_.sum()),
            "errore_modello": round(float(log_loss(tt.esito[m_], P_comb[m_], labels=CLASSI)), 4),
            "errore_bookmaker": round(float(log_loss(tt.esito[m_], P_book[m_], labels=CLASSI)), 4)}
    mc_ = m_ & okc
    if mc_.sum() >= 50:
        riga["errore_chiusura"] = round(float(log_loss(tt.esito[mc_], p_chius[mc_], labels=CLASSI)), 4)
    pk = (ev5 > 0.05) & np.isfinite(q) & m_[:, None]
    n_ = int(pk.sum())
    pay = np.where(np.isfinite(qc), qc, q)
    riga["giocate_valore"] = n_
    riga["resa_valore"] = round(float(np.where(vinte[pk], pay[pk] - 1, -1).sum() / n_), 3) if n_ else None
    PER_STAGIONE.append(riga)
print(pd.DataFrame(PER_STAGIONE).to_string(index=False))

KELLY = {"quote_anticipo": simula_kelly(tt, P_comb, q, q), "quote_chiusura": simula_kelly(tt, P_comb, q, qc)}
PIANO_ATTIVO = (KELLY["quote_chiusura"]["finale"] > 100 and (CHIUS.get("clv_0") or -1) > 0
                and CHIUS.get("log_loss_modello", 9) < CHIUS.get("log_loss_chiusura", 0) + 0.005)
MOTIVO_PIANO = ("Il test sulle stagioni passate è positivo anche contro le quote di chiusura." if PIANO_ATTIVO else
                "Piano disattivato: il test sulle stagioni passate non mostra un vantaggio solido contro le quote di chiusura.")
tempo(MOTIVO_PIANO)

# ==== Under/Over 2.5: probabilità del modello combinata con le quote
def tabella_tarata(tot_gol, p1, p2, rho):
    obiettivo = p1 / (p1 + p2)
    def f(x):
        P = matrice(x, tot_gol - x, rho)[0]
        a, b = np.tril(P, -1).sum(), np.triu(P, 1).sum()
        return a / (a + b) - obiettivo
    lo, hi = 0.02, tot_gol - 0.02
    x = lo if f(lo) > 0 else hi if f(hi) < 0 else brentq(f, lo, hi)
    return matrice(x, tot_gol - x, rho)[0]

def p_over(P, soglia=2.5):
    i, j = np.indices(P.shape)
    return P[(i + j) > soglia].sum()

def tabella_tarata_ou(obiettivo_over, p1, p2, rho):
    f = lambda T: p_over(tabella_tarata(T, p1, p2, rho)) - obiettivo_over
    lo, hi = 0.8, 6.0
    T = lo if f(lo) > 0 else hi if f(hi) < 0 else brentq(f, lo, hi, xtol=1e-3)
    return tabella_tarata(T, p1, p2, rho)

tg = (df.lam_c + df.lam_t).values
tx = (df.lamxg_c + df.lamxg_t).values
T_all = np.where(np.isfinite(tx), (tg + tx) / 2, tg)
df["gol_attesi_mod"] = T_all
df["p_over_mod"] = np.nan
okl = np.isfinite(T_all)
if okl.any():
    k_ = T_all[okl] / tg[okl]
    Pm = matrice(df.lam_c.values[okl] * k_, df.lam_t.values[okl] * k_, RHO_A)
    ii, jj = np.indices(Pm.shape[1:])
    df.loc[okl, "p_over_mod"] = Pm[:, (ii + jj) > 2.5].sum(axis=1)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
df["ou_mod"], df["ou_book"] = logit(df.p_over_mod), logit(df.p_book_over)
df["over"] = np.where(df.esito.notna(), (df.gol_c + df.gol_t > 2.5).astype(float), np.nan)
OU_COLS = ["ou_mod", "ou_book"]
ev_ou = df[(df.stagione >= prima_train) & df.esito.notna()].dropna(subset=OU_COLS)
def allena_ou(train):
    return LogisticRegression(max_iter=1000).fit(train[OU_COLS], train.over.astype(int))
p_blend = pd.Series(np.nan, index=df.index)
for ts in TEST_STAGIONI:
    tr_, te_ = ev_ou[ev_ou.stagione < ts], ev_ou[(ev_ou.stagione == ts) & (ev_ou.lega == PRINCIPALE)]
    if len(te_):
        p_blend[te_.index] = allena_ou(tr_).predict_proba(te_[OU_COLS])[:, 1]
ou = tt.assign(over=(tt.gol_c + tt.gol_t > 2.5).astype(int))
ok_ou = ou.p_book_over.notna().values & p_blend.reindex(ou.index).notna().values
tab_ou = pd.DataFrame({
    "Modello + quote": {"log_loss": log_loss(ou.over[ok_ou], np.clip(p_blend.reindex(ou.index).values[ok_ou], 1e-6, 1 - 1e-6))},
    "Bookmaker": {"log_loss": log_loss(ou.over[ok_ou], ou.p_book_over.values[ok_ou])},
}).T.round(4)
MODO_OU = "Modello + quote"
po = p_blend.reindex(ou.index).values
SIM_OU = simula(np.column_stack([po, 1 - po]).ravel(), qarr(ou, ["q_over", "q_under"]).ravel(),
                np.column_stack([ou.over == 1, ou.over == 0]).ravel())
print(tab_ou.to_string())

# ==== Previsioni per la prossima giornata di Serie A
fin = prepara(df[df.esito.notna() & (df.stagione >= prima_train) & df.lega.isin(LEGHE_SCELTA)])
ALTERNATIVE = [COLS_SCELTA, FEAT_A, BASE + ["diff_valore"], BASE + BOOK, BASE]
MODELLI_FIN = []
for cols in ALTERNATIVE:
    cols = [c for c in cols if c != "diff_xi"]
    if any(cols == c_ for c_, _ in MODELLI_FIN):
        continue
    tr_ = fin.dropna(subset=cols)
    if len(tr_) > 500:
        MODELLI_FIN.append((cols, allena(tr_, cols)))
mod_ou = allena_ou(ev_ou)

def mercati(P):
    i, j = np.indices(P.shape); tot, gd = i + j, i - j
    m = {}
    p1, pX, p2 = P[gd > 0].sum(), P[gd == 0].sum(), P[gd < 0].sum()
    m.update({("1X2", "1"): p1, ("1X2", "X"): pX, ("1X2", "2"): p2,
              ("Doppia chance", "1X"): p1 + pX, ("Doppia chance", "X2"): pX + p2, ("Doppia chance", "12"): p1 + p2})
    for k in [0.5, 1.5, 2.5, 3.5, 4.5]:
        m[("Under/Over", f"Under {k}")] = P[tot < k].sum()
        m[("Under/Over", f"Over {k}")] = P[tot > k].sum()
    gg = P[(i > 0) & (j > 0)].sum()
    m[("Goal/No Goal", "Goal")], m[("Goal/No Goal", "No Goal")] = gg, 1 - gg
    for a, b in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 6)]:
        m[("Multigol", f"{a}-{b}")] = P[(tot >= a) & (tot <= b)].sum()
    for s_ in range(6):
        m[("Somma gol", str(s_))] = P[tot == s_].sum()
    m[("Somma gol", "6+")] = P[tot >= 6].sum()
    for h in [-2, -1, 1, 2]:
        for lab, cond in [("1", gd + h > 0), ("X", gd + h == 0), ("2", gd + h < 0)]:
            m[(f"Handicap casa {h:+d}", lab)] = P[cond].sum()
    ris = sorted(((P[a, b], f"{a}-{b}") for a in range(7) for b in range(7)), reverse=True)[:10]
    for p, lab in ris:
        m[("Risultato esatto", lab)] = p
    return m

def num(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else round(float(x), nd)

partite, righe_csv, righe_storico, piano = [], [], [], []
for r in df[df.futura].itertuples():
    scelto = None
    for cols, mdl in MODELLI_FIN:
        x = pd.DataFrame([{c: getattr(r, c) for c in cols}])
        if not x.isna().values.any():
            scelto = (cols, mdl, x); break
    if scelto is None or pd.isna(r.lam_c):
        print(f"{r.casa}-{r.trasf}: dati insufficienti, salto")
        NOTE.append(f"{r.casa} - {r.trasf}: dati insufficienti per la previsione.")
        continue
    cols, mdl, x = scelto
    usa_val = "diff_valore" in cols
    pc = mdl.predict_proba(x)[0]      # ordine 1, 2, X
    if pd.notna(r.ou_mod) and pd.notna(r.ou_book):
        obiettivo = mod_ou.predict_proba(pd.DataFrame([{"ou_mod": r.ou_mod, "ou_book": r.ou_book}]))[0, 1]
        P = tabella_tarata_ou(obiettivo, pc[0], pc[1], RHO_A)
    else:
        P = tabella_tarata(r.gol_attesi_mod, pc[0], pc[1], RHO_A)
    ms = mercati(P)
    ms.update({("1X2", "1"): pc[0], ("1X2", "X"): pc[2], ("1X2", "2"): pc[1],
               ("Doppia chance", "1X"): pc[0] + pc[2], ("Doppia chance", "X2"): pc[2] + pc[1],
               ("Doppia chance", "12"): pc[0] + pc[1]})
    quote_r = np.array([[r.q1, r.q2, r.qX]], dtype=float)
    if np.isfinite(quote_r).all():
        kk, ff, ee = frazioni_kelly(np.array([pc]), quote_r)
        if ff[0] > 0:
            piano.append({"partita": f"{r.casa} - {r.trasf}", "data": str(r.data.date()),
                          "esito": ["1", "2", "X"][kk[0]], "p": round(float(pc[kk[0]]), 4),
                          "quota": round(float(quote_r[0, kk[0]]), 2), "valore": round(float(ee[0]), 3),
                          "percentuale": float(ff[0])})
    book = {("1X2", "1"): r.q1, ("1X2", "X"): r.qX, ("1X2", "2"): r.q2,
            ("Under/Over", "Over 2.5"): r.q_over, ("Under/Over", "Under 2.5"): r.q_under}
    per_mercato = {}
    for (merc, es), p in ms.items():
        qb = book.get((merc, es), np.nan)
        riga = {"esito": es, "p": num(p), "quota_giusta": num(1 / p, 2) if p > 0 else None,
                "quota_book": num(qb, 2), "valore": num(p * qb - 1, 3) if pd.notna(qb) else None}
        per_mercato.setdefault(merc, []).append(riga)
        righe_csv.append({"data": r.data.date(), "ora": getattr(r, "ora", ""), "partita": f"{r.casa} - {r.trasf}",
                          "mercato": merc, **riga})
    gol_att = float(P.sum(axis=1) @ np.arange(MAXG + 1) + P.sum(axis=0) @ np.arange(MAXG + 1))
    ora = str(getattr(r, "ora", "") or "")
    ora = "" if ora.lower() in ("nan", "none") else ora
    partite.append({"data": str(r.data.date()), "ora": ora, "casa": r.casa, "trasf": r.trasf,
                    "valori_usati": usa_val, "gol_attesi": round(gol_att, 2), "mercati": per_mercato})
    candidati = [(v["valore"], m_, v) for m_, lst in per_mercato.items() for v in lst if v["valore"] is not None]
    miglior = max(candidati, key=lambda t: t[0]) if candidati else None
    ok_pick = bool(miglior and miglior[0] > 0.05)
    righe_storico.append({"stagione": r.stagione, "data": str(r.data.date()), "casa": r.casa, "trasf": r.trasf,
                          "p1": ms[("1X2", "1")], "pX": ms[("1X2", "X")], "p2": ms[("1X2", "2")],
                          "pb1": r.p_book_1, "pbX": r.p_book_X, "pb2": r.p_book_2,
                          "p_over25": ms[("Under/Over", "Over 2.5")], "p_goal": ms[("Goal/No Goal", "Goal")],
                          "pick_mercato": miglior[1] if ok_pick else "", "pick_esito": miglior[2]["esito"] if ok_pick else "",
                          "pick_quota": miglior[2]["quota_book"] if ok_pick else np.nan,
                          "pick_valore": miglior[0] if ok_pick else np.nan})
tot_piano = sum(g_["percentuale"] for g_ in piano)
for g_ in piano:
    g_["percentuale"] = round(g_["percentuale"] * min(1.0, MAX_GIORNO / tot_piano), 4) if tot_piano else 0
tempo(f"Previsioni calcolate: {len(partite)}")

# ==== Storico: salva le previsioni e confrontale con i risultati
os.makedirs("docs", exist_ok=True)
F_STORICO = "docs/storico.csv"
st_old = pd.read_csv(F_STORICO, dtype={"stagione": str, "pick_mercato": str, "pick_esito": str}) if os.path.exists(F_STORICO) else pd.DataFrame()
nuovo = pd.DataFrame(righe_storico)
if len(st_old) and len(nuovo):
    chiave = ["stagione", "casa", "trasf"]
    st_old = st_old.merge(nuovo[chiave], on=chiave, how="left", indicator=True)
    st_old = st_old[st_old._merge == "left_only"].drop(columns="_merge")
storico = pd.concat([st_old, nuovo], ignore_index=True)
if len(storico):
    storico["stagione"] = storico["stagione"].astype(str).str.zfill(4)
    storico = storico.drop(columns=[c for c in ["gol_c", "gol_t"] if c in storico])
    ris = giocate[giocate.lega == PRINCIPALE][["stagione", "casa", "trasf", "gol_c", "gol_t"]]
    storico = storico.merge(ris, on=["stagione", "casa", "trasf"], how="left")
    storico.to_csv(F_STORICO, index=False)

def valuta_storico(s):
    s = s.dropna(subset=["gol_c"]).copy()
    if s.empty:
        return {"partite": 0}, []
    es = np.select([s.gol_c > s.gol_t, s.gol_c == s.gol_t], ["1", "X"], "2")
    pm = s[["p1", "p2", "pX"]].values
    out = {"partite": int(len(s)),
           "esito_azzeccato": round(float((np.array(CLASSI)[pm.argmax(1)] == es).mean()), 3),
           "log_loss_modello": round(float(log_loss(es, pm, labels=CLASSI)), 4)}
    okb = s[["pb1", "pb2", "pbX"]].notna().all(axis=1).values
    if okb.sum() >= 10:
        out["log_loss_bookmaker"] = round(float(log_loss(es[okb], s[["pb1", "pb2", "pbX"]].values[okb], labels=CLASSI)), 4)
        out["log_loss_modello_stesse"] = round(float(log_loss(es[okb], pm[okb], labels=CLASSI)), 4)
    tot = s.gol_c + s.gol_t
    out["over25_azzeccato"] = round(float(((s.p_over25 > 0.5) == (tot > 2.5)).mean()), 3)
    pk = s[s.pick_mercato.fillna("") != ""]
    if len(pk):
        def vinta(r):
            t, gd = r.gol_c + r.gol_t, r.gol_c - r.gol_t
            if r.pick_mercato == "1X2":
                return {"1": gd > 0, "X": gd == 0, "2": gd < 0}[str(r.pick_esito)]
            return t > 2.5 if "Over" in str(r.pick_esito) else t < 2.5
        v = pk.apply(vinta, axis=1)
        prof = float(np.where(v, pk.pick_quota - 1, -1).sum())
        out.update({"giocate_valore": int(len(pk)), "giocate_vinte": int(v.sum()), "profitto_unita": round(prof, 1),
                    "roi": round(prof / len(pk), 3)})
    ultime = []
    s2 = s.sort_values("data", ascending=False).head(40)
    es2 = np.select([s2.gol_c > s2.gol_t, s2.gol_c == s2.gol_t], ["1", "X"], "2")
    for r, e in zip(s2.itertuples(), es2):
        pr_ = {"1": r.p1, "X": r.pX, "2": r.p2}
        pick = f"{r.pick_mercato} {r.pick_esito}".strip() if str(r.pick_mercato) not in ("", "nan") else ""
        ultime.append({"data": r.data, "casa": r.casa, "trasf": r.trasf, "risultato": f"{int(r.gol_c)}-{int(r.gol_t)}",
                       "esito": str(e), "previsto": max(pr_, key=pr_.get), "p_esito": round(float(pr_[str(e)]), 3), "pick": pick})
    return out, ultime

riep_storico, ultime_storico = valuta_storico(storico) if len(storico) else ({"partite": 0}, [])

# ==== Scrittura dei risultati per la pagina
valore = sorted(
    [{"partita": f'{p["casa"]} - {p["trasf"]}', "data": p["data"], "mercato": m_, **v}
     for p in partite for m_, lst in p["mercati"].items() for v in lst if v["valore"] is not None and v["valore"] > 0.05],
    key=lambda v: -v["valore"])
if not partite and not any("Calendario" in n for n in NOTE):
    NOTE.append("Nessuna partita in programma nei prossimi giorni (pausa o calendario non ancora pubblicato).")

def tab_json(t):
    return json.loads(t.to_json(orient="records")) if len(t) else []

dati = {
    "aggiornato": pd.Timestamp.now(tz="Europe/Rome").strftime("%Y-%m-%d %H:%M"),
    "stagione": f"20{COD_CORRENTE[:2]}/{COD_CORRENTE[2:]}",
    "note": NOTE,
    "partite": sorted(partite, key=lambda p: (p["data"], p["ora"])),
    "valore": valore,
    "piano": {"attivo": bool(PIANO_ATTIVO), "motivo": MOTIVO_PIANO, "giocate": piano if PIANO_ATTIVO else [],
              "regole": {"frazione_kelly": FRAZ_KELLY, "max_giocata": MAX_GIOCATA, "max_giorno": MAX_GIORNO}},
    "affidabilita": {
        "stagioni_test": [f"20{s_[:2]}/{s_[2:]}" for s_ in TEST_STAGIONI],
        "partite_test": int(len(tt)),
        "versione": SCELTA,
        "esperimenti": ESPERIMENTI,
        "copertura_formazioni": COPERTURA_XI,
        "pinnacle": PINNACLE,
        "x12": {"log_loss_modello": round(LL_MOD, 4), "log_loss_bookmaker": round(LL_BOOK, 4),
                "log_loss_senza_quote": None,
                "esito_azzeccato_modello": round(ACC_MOD, 3), "esito_azzeccato_bookmaker": round(ACC_BOOK, 3)},
        "ou25": [{"modello": k, "log_loss": float(v.log_loss)} for k, v in tab_ou.iterrows()],
        "ou_scelto": MODO_OU,
        "sim_1x2": tab_json(SIM_1X2), "sim_ou": tab_json(SIM_OU),
        "chiusura": {**CHIUS, "sim": tab_json(SIM_CHIUS)},
        "kelly": KELLY,
        "per_stagione": PER_STAGIONE,
    },
    "storico": {"riepilogo": riep_storico, "ultime": ultime_storico},
}
with open("docs/dati.json", "w", encoding="utf-8") as f:
    json.dump(dati, f, ensure_ascii=False, indent=1, default=str)
pd.DataFrame(righe_csv).to_csv("docs/previsioni_giornata.csv", index=False)
tempo("Scritto docs/dati.json e docs/previsioni_giornata.csv")
