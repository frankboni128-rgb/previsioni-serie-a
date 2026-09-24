"""Previsioni Serie A — eseguito automaticamente da GitHub Actions.
Scarica i dati, rifà i test sul passato, calcola le previsioni della prossima giornata
e scrive i risultati in docs/ (pagina web su GitHub Pages)."""
import json, os

# ==== Impostazioni
import time, io, re, difflib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.stats import poisson
from scipy.optimize import minimize_scalar, brentq
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
PRIMA_STAGIONE, STAGIONE_CORRENTE = 2008, 2026          # 2026 = stagione 2026/27
INIZIO_MODELLI = "1011"
TEST_STAGIONI = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]   # 7 stagioni: dal 2019/20 ci sono anche le quote di chiusura
PRIMO_TEST = TEST_STAGIONI[0]
PRIMA_XG = 2014
FINESTRA_GG, XI, HALFLIFE_XG, MAXG = 730, 0.002, 8, 10
CLASSI = ["1", "2", "X"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
# Se il calendario automatico non si trova, scrivi qui le partite (nomi come su football-data), es.:
# PARTITE_MANUALI = [("2026-09-27", "Inter", "Milan"), ("2026-09-27", "Roma", "Lazio")]
PARTITE_MANUALI = []

NOTE = []   # avvisi mostrati nella pagina

def codice(anno):
    return f"{anno % 100:02d}{(anno + 1) % 100:02d}"
COD_CORRENTE = codice(STAGIONE_CORRENTE)

# ==== Storico Serie A (risultati + quote 1X2 e Under/Over 2.5)
QUOTE_1X2 = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("PSH", "PSD", "PSA")]
QUOTE_OU = [("B365>2.5", "B365<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("P>2.5", "P<2.5")]
# quote di chiusura (subito prima del fischio d'inizio), disponibili dal 2019/20
CHIUSURA_PRECISA = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
CHIUSURA_B365 = [("B365CH", "B365CD", "B365CA")]
COL_QUOTE = ["q1", "qX", "q2", "q_over", "q_under", "qs1", "qsX", "qs2", "qc1", "qcX", "qc2"]

def leggi_quote(raw, out):
    for dst in COL_QUOTE:
        out[dst] = np.nan
    for lista, dsts in [(QUOTE_1X2, ["q1", "qX", "q2"]), (QUOTE_OU, ["q_over", "q_under"]),
                        (CHIUSURA_PRECISA, ["qs1", "qsX", "qs2"]), (CHIUSURA_B365, ["qc1", "qcX", "qc2"])]:
        for cols in lista:
            if all(c in raw for c in cols):
                for dst, src in zip(dsts, cols):
                    out[dst] = out[dst].fillna(pd.Series(pd.to_numeric(raw[src], errors="coerce").values, index=out.index))
    return out

def leggi_csv(contenuto):
    raw = pd.read_csv(io.StringIO(contenuto.decode("latin-1")), on_bad_lines="skip")
    raw.columns = raw.columns.str.replace("ï»¿", "", regex=False).str.strip()
    return raw

def scarica(cod):
    r = requests.get(f"https://www.football-data.co.uk/mmz4281/{cod}/I1.csv", timeout=30)
    if r.status_code != 200 or len(r.content) < 500:
        return None
    raw = leggi_csv(r.content).dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).reset_index(drop=True)
    out = pd.DataFrame({
        "stagione": cod, "data": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "casa": raw["HomeTeam"].str.strip(), "trasf": raw["AwayTeam"].str.strip(),
        "gol_c": raw["FTHG"].astype(float), "gol_t": raw["FTAG"].astype(float)})
    return leggi_quote(raw, out)

blocchi = []
for anno in range(PRIMA_STAGIONE, STAGIONE_CORRENTE + 1):
    s = scarica(codice(anno))
    print(codice(anno), "—" if s is None else len(s))
    if s is not None:
        blocchi.append(s)
    time.sleep(0.4)
giocate = pd.concat(blocchi, ignore_index=True)
giocate["futura"] = False

# ==== Prossima giornata (calendario + quote da football-data)
oggi = pd.Timestamp.today().normalize()
fut = pd.DataFrame()
try:
    r = requests.get("https://www.football-data.co.uk/fixtures.csv", timeout=30)
    raw = leggi_csv(r.content)
    raw = raw[raw["Div"] == "I1"].reset_index(drop=True)
    fut = pd.DataFrame({"stagione": COD_CORRENTE, "ora": raw["Time"].astype(str) if "Time" in raw else "",
                        "data": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
                        "casa": raw["HomeTeam"].str.strip(), "trasf": raw["AwayTeam"].str.strip()})
    fut = leggi_quote(raw, fut)
except Exception as e:
    print("Calendario automatico non disponibile:", e)
    NOTE.append("Calendario della prossima giornata non disponibile su football-data.")
if PARTITE_MANUALI:
    man = pd.DataFrame(PARTITE_MANUALI, columns=["data", "casa", "trasf"]).assign(
        stagione=COD_CORRENTE, data=lambda x: pd.to_datetime(x.data))
    fut = pd.concat([fut, man], ignore_index=True)
if fut.empty:
    fut = pd.DataFrame(columns=["stagione", "data", "casa", "trasf"] + COL_QUOTE)
    fut["data"] = pd.to_datetime(fut["data"])
    print("Nessuna partita da prevedere.")
fut = fut[fut.data >= oggi].drop_duplicates(["casa", "trasf"])
fut = fut.merge(giocate[["stagione", "casa", "trasf"]], how="left", indicator=True)
fut = fut[fut._merge == "left_only"].drop(columns="_merge")          # esclude partite già giocate
fut["gol_c"] = fut["gol_t"] = np.nan
fut["futura"] = True
note = set(giocate.casa) | set(giocate.trasf)
sconosciute = (set(fut.casa) | set(fut.trasf)) - note
print(f"Partite da prevedere: {len(fut)}", "| nomi non riconosciuti:", sconosciute or "nessuno")
print(fut[["data", "casa", "trasf", "q1", "qX", "q2", "q_over", "q_under"]].to_string(index=False))

df = pd.concat([giocate, fut], ignore_index=True).sort_values(["data", "casa"]).reset_index(drop=True)
for c in ["gol_c", "gol_t"] + COL_QUOTE:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["futura"] = df["futura"].astype(bool)
df["esito"] = np.where(df.futura, None,
                       np.select([df.gol_c > df.gol_t, df.gol_c == df.gol_t], ["1", "X"], "2"))
df["match_id"] = df.index
for q, dst in [(["q1", "qX", "q2"], ["p_book_1", "p_book_X", "p_book_2"]), (["q_over", "q_under"], ["p_book_over", "p_book_under"])]:
    inv = 1 / df[q]
    df[dst] = inv.div(inv.sum(axis=1), axis=0).values

# ==== Elo
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

gio = df[~df.futura]
punt = gio.esito.map({"1": 1.0, "X": 0.5, "2": 0.0})
m_tar = (gio.stagione >= INIZIO_MODELLI) & (gio.stagione < PRIMO_TEST)
K, HFA, _ = min(((K, H, ((elo(gio, K, H)[0].elo_exp - punt)[m_tar] ** 2).mean())
                 for K in [10, 15, 20, 25, 30, 35, 40] for H in [30, 50, 70, 90]), key=lambda x: x[2])
E, rating = elo(gio, K, HFA)
df.loc[E.index, E.columns] = E
for i in df.index[df.futura]:
    rc, rt = rating.get(df.at[i, "casa"], np.nan), rating.get(df.at[i, "trasf"], np.nan)
    df.loc[i, ["elo_c", "elo_t", "elo_diff"]] = [rc, rt, rc + HFA - rt]
print(f"Elo: K={K} HFA={HFA}")
print(pd.Series({s: rating.get(s, np.nan) for s in set(df.casa[df.stagione == COD_CORRENTE])})
      .sort_values(ascending=False).round(0).to_string())

# ==== xG da Understat
HEAD = dict(UA, **{"X-Requested-With": "XMLHttpRequest", "Referer": "https://understat.com/"})
def scarica_xg(anno):
    r = requests.get(f"https://understat.com/getLeagueData/Serie_A/{anno}", headers=HEAD, timeout=30)
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

bx = []
for anno in range(PRIMA_XG, STAGIONE_CORRENTE + 1):
    try:
        bx.append(scarica_xg(anno)); print(codice(anno), "xG ok")
    except Exception as e:
        print(codice(anno), "xG errore:", e)
    time.sleep(1)
xg = pd.concat(bx, ignore_index=True) if bx else pd.DataFrame()
USA_XG = len(xg) > 1000
if USA_XG:
    m = xg.merge(df[["stagione", "data", "casa", "trasf", "gol_c", "gol_t"]],
                 left_on=["stagione", "data_u", "u_gol_c", "u_gol_t"], right_on=["stagione", "data", "gol_c", "gol_t"])
    coppie = pd.concat([m[["u_casa", "casa"]].set_axis(["u", "f"], axis=1), m[["u_trasf", "trasf"]].set_axis(["u", "f"], axis=1)])
    mappa = coppie.value_counts().reset_index(name="n").drop_duplicates("u").set_index("u")["f"]
    xg["casa"], xg["trasf"] = xg.u_casa.map(mappa), xg.u_trasf.map(mappa)
    df = df.merge(xg[["stagione", "casa", "trasf", "xg_c", "xg_t"]].drop_duplicates(["stagione", "casa", "trasf"]),
                  on=["stagione", "casa", "trasf"], how="left")
else:
    df["xg_c"] = df["xg_t"] = np.nan
    print("Proseguo SENZA xG")
    NOTE.append("Expected goals (Understat) non disponibili in questo aggiornamento: modello un po' meno preciso.")

lungo = pd.concat([
    pd.DataFrame({"match_id": df.match_id, "data": df.data, "squadra": df.casa, "in_casa": 1, "xg_f": df.xg_c, "xg_s": df.xg_t}),
    pd.DataFrame({"match_id": df.match_id, "data": df.data, "squadra": df.trasf, "in_casa": 0, "xg_f": df.xg_t, "xg_s": df.xg_c})
]).sort_values(["squadra", "data", "match_id"])
g = lungo.groupby("squadra")
for c in ["xg_f", "xg_s"]:
    lungo[f"ewm_{c}"] = g[c].transform(lambda x: x.shift(1).ewm(halflife=HALFLIFE_XG, min_periods=3).mean())
for pref, flag in [("c_", 1), ("t_", 0)]:
    df = df.join(lungo[lungo.in_casa == flag].set_index("match_id")[["ewm_xg_f", "ewm_xg_s"]].add_prefix(pref), on="match_id")
df["diff_xg"] = (df.c_ewm_xg_f - df.c_ewm_xg_s) - (df.t_ewm_xg_f - df.t_ewm_xg_s)

# ==== Valori di mercato STORICI (transfermarkt-datasets, fermo a luglio 2026)
TM = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{}.csv.gz"
def tm(nome, cols):
    r = requests.get(TM.format(nome), headers=UA, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"{nome}: il server risponde {r.status_code}")
    t = pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory=False)
    mancano = [c for c in cols if c not in t.columns]
    if mancano:
        raise ValueError(f"{nome}: colonne mancanti {mancano}. Colonne presenti: {list(t.columns)}")
    print(nome, t.shape)
    return t[cols]

games = tm("games", ["competition_id", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"])
val = tm("player_valuations", ["player_id", "date", "market_value_in_eur"])
trf = tm("transfers", ["player_id", "transfer_date", "from_club_id", "to_club_id"])

gi = games[games.competition_id == "IT1"].copy()
gi["date"] = pd.to_datetime(gi.date, errors="coerce")
m = gi.merge(df[["data", "casa", "trasf", "gol_c", "gol_t"]],
             left_on=["date", "home_club_goals", "away_club_goals"], right_on=["data", "gol_c", "gol_t"])
coppie = pd.concat([m[["home_club_id", "casa"]].set_axis(["id", "nome"], axis=1),
                    m[["away_club_id", "trasf"]].set_axis(["id", "nome"], axis=1)])
id_nome = (coppie.value_counts().reset_index(name="n").drop_duplicates("id").drop_duplicates("nome")
           .set_index("id")["nome"])
ids = set(id_nome.index)

trf = trf.assign(data=pd.to_datetime(trf.transfer_date, errors="coerce")).dropna(subset=["data"])
val = val.assign(data=pd.to_datetime(val.date, errors="coerce")).dropna(subset=["data"])
giocatori = trf.loc[trf.to_club_id.isin(ids) | trf.from_club_id.isin(ids), "player_id"].unique()
foto = pd.date_range("2012-07-01", "2026-07-01", freq="SMS")
griglia = pd.MultiIndex.from_product([giocatori, foto], names=["player_id", "data"]).to_frame(index=False).sort_values("data")
griglia = pd.merge_asof(griglia, trf[trf.player_id.isin(giocatori)].sort_values("data")[["player_id", "data", "to_club_id"]],
                        on="data", by="player_id")
v = val[val.player_id.isin(giocatori)].sort_values("data")[["player_id", "data", "market_value_in_eur"]]
griglia = pd.merge_asof(griglia, v.rename(columns={"data": "data_val"}), left_on="data", right_on="data_val", by="player_id")
griglia = griglia[griglia.to_club_id.isin(ids) & ((griglia.data - griglia.data_val).dt.days <= 400)]
rose = (griglia.groupby(["to_club_id", "data"]).agg(valore=("market_value_in_eur", "sum"), n=("player_id", "size")).reset_index())
rose = rose[rose.n >= 15]
rose["squadra"] = rose.to_club_id.map(id_nome)
rose = rose.sort_values("data")[["squadra", "data", "valore"]].rename(columns={"data": "data_foto"})
del griglia

df = df.sort_values("data")
rose["squadra"] = rose["squadra"].astype(object)
df["casa"], df["trasf"] = df["casa"].astype(object), df["trasf"].astype(object)
for lato, pref in [("casa", "c_"), ("trasf", "t_")]:
    df = pd.merge_asof(df, rose.rename(columns={"squadra": lato, "valore": pref + "valore", "data_foto": pref + "foto"}),
                       left_on="data", right_on=pref + "foto", by=lato, allow_exact_matches=False)
df = df.sort_values("match_id").reset_index(drop=True)
# se l'ultima "fotografia" è vecchia di oltre 60 giorni (es. dopo il mercato estivo 2026) il valore non è affidabile
vecchio = ((df.data - df.c_foto).dt.days > 60) | ((df.data - df.t_foto).dt.days > 60)
df["diff_valore"] = np.where(vecchio, np.nan, np.log(df.c_valore) - np.log(df.t_valore))

# ==== Valori di mercato ATTUALI (Transfermarkt, rose 2026/27)
def num_valore(t):
    t = t.replace("\xa0", " ").lower()
    m = re.search(r"([\d.,]+)\s*(bn|mld|m|mln|k|mila)", t)
    if not m:
        return np.nan
    x = m.group(1)
    x = float(x.replace(".", "").replace(",", ".")) if "," in x else float(x)
    return x * {"bn": 1e9, "mld": 1e9, "m": 1e6, "mln": 1e6, "k": 1e3, "mila": 1e3}[m.group(2)]

valori_ora = pd.Series(dtype=float)
try:
    url = f"https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1/saison_id/{STAGIONE_CORRENTE}"
    r = requests.get(url, headers=dict(UA, **{"Accept": "text/html,application/xhtml+xml",
                                               "Accept-Language": "it-IT,it;q=0.9,en;q=0.8"}), timeout=30)
    r.raise_for_status()
    tab_tm = BeautifulSoup(r.text, "html.parser").select_one("table.items")
    righe = []
    for tr in tab_tm.select("tbody > tr"):
        a = tr.select_one("td.hauptlink a[href*='/verein/']")
        if a is None:
            continue
        celle = tr.find_all("td", recursive=False)
        righe.append({"tm_id": int(re.search(r"/verein/(\d+)", a["href"]).group(1)),
                      "tm_nome": a.get("title") or a.get_text(strip=True),
                      "valore": num_valore(celle[-1].get_text(" ", strip=True))})
    ora = pd.DataFrame(righe)
    squadre_ora = sorted(set(df.casa[df.stagione == COD_CORRENTE]) | set(df.trasf[df.stagione == COD_CORRENTE]))
    def trova(riga):
        if riga.tm_id in id_nome.index:
            return id_nome[riga.tm_id]
        simili = difflib.get_close_matches(riga.tm_nome, squadre_ora, n=1, cutoff=0.4)
        return simili[0] if simili else None
    ora["squadra"] = ora.apply(trova, axis=1)
    print((ora.set_index("tm_nome")[["squadra", "valore"]].assign(valore=lambda x: (x.valore / 1e6).round(1))).to_string())
    valori_ora = ora.dropna(subset=["squadra"]).set_index("squadra").valore
except Exception as e:
    print("Transfermarkt non raggiungibile:", e)

# Integrazione con il file docs/valori_rose.csv (aggiornato a mano dopo ogni sessione di mercato)
F_VAL = "docs/valori_rose.csv"
squadre_correnti = set(df.casa[df.stagione == COD_CORRENTE]) | set(df.trasf[df.stagione == COD_CORRENTE])
if os.path.exists(F_VAL):
    vf = pd.read_csv(F_VAL)
    vf["squadra"] = vf["squadra"].astype(str).str.strip()
    sconosciute_vf = sorted(set(vf.squadra) - squadre_correnti)
    if sconosciute_vf:
        NOTE.append("Nel file valori_rose.csv questi nomi non corrispondono a squadre della Serie A attuale: "
                    + ", ".join(sconosciute_vf) + ". Controlla come sono scritti.")
    vf_s = vf.set_index("squadra")["valore_milioni"].astype(float) * 1e6
    valori_ora = pd.concat([valori_ora, vf_s[[s_ for s_ in vf_s.index if s_ not in valori_ora.index]]])
    agg = pd.to_datetime(vf["aggiornato"], errors="coerce").max()
    if pd.notna(agg) and (oggi - agg).days > 150:
        NOTE.append(f"Il file valori_rose.csv è fermo al {agg.strftime('%d/%m/%Y')}: conviene aggiornarlo.")
    print(f"Valori rose da file ({F_VAL}):", len(vf_s))
manca_val = sorted(squadre_correnti - set(valori_ora.index))
if manca_val:
    NOTE.append("Valore della rosa mancante per: " + ", ".join(manca_val) + ". Le loro partite sono previste senza valori di mercato.")

fm = df.futura
df.loc[fm, "diff_valore"] = (np.log(df.loc[fm, "casa"].map(valori_ora)) - np.log(df.loc[fm, "trasf"].map(valori_ora))).values

# ==== Modello di Poisson (gol e xG)
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
    """Tabella delle probabilità di ogni risultato esatto (righe = gol casa, colonne = gol trasferta)."""
    lc, lt = np.atleast_1d(lc).astype(float), np.atleast_1d(lt).astype(float)
    k = np.arange(MAXG + 1)
    P = poisson.pmf(k[None, :, None], lc[:, None, None]) * poisson.pmf(k[None, None, :], lt[:, None, None])
    P[:, 0, 0] *= 1 - lc * lt * rho; P[:, 0, 1] *= 1 + lc * rho
    P[:, 1, 0] *= 1 + lt * rho;      P[:, 1, 1] *= 1 - rho
    P = np.clip(P, 0, None)
    return P / P.sum(axis=(1, 2), keepdims=True)

def p1x2(P):
    p1 = np.tril(P, -1).sum(axis=(1, 2)); pX = np.trace(P, axis1=1, axis2=2)
    return np.column_stack([p1, 1 - p1 - pX, pX])       # ordine: 1, 2, X

RHO = {}
def probs_poisson(d, col_c, col_t, nome):
    lam = stima_lambda(d, col_c, col_t)
    ok = ~np.isnan(lam[:, 0])
    tr = ok & (d.stagione < PRIMO_TEST).values & d.esito.notna().values
    RHO[nome] = minimize_scalar(lambda r: log_loss(d.esito[tr], np.clip(p1x2(matrice(lam[tr, 0], lam[tr, 1], r)), 1e-9, 1),
                                                   labels=CLASSI), bounds=(-0.3, 0.3), method="bounded").x
    out = np.full((len(d), 3), np.nan)
    out[ok] = p1x2(matrice(lam[ok, 0], lam[ok, 1], RHO[nome]))
    return lam, out

t0 = time.time()
lam, pr = probs_poisson(df, "gol_c", "gol_t", "gol")
df[["lam_c", "lam_t"]] = lam; df[["p_pois_1", "p_pois_2", "p_pois_X"]] = pr
if USA_XG:
    lam, pr = probs_poisson(df, "xg_c", "xg_t", "xg")
    df[["lamxg_c", "lamxg_t"]] = lam; df[["p_xg_1", "p_xg_2", "p_xg_X"]] = pr
for pref in ["pois"] + (["xg"] if USA_XG else []):
    df[f"{pref}_1v2"] = np.log(df[f"p_{pref}_1"] / df[f"p_{pref}_2"])
    df[f"{pref}_X"] = np.log(df[f"p_{pref}_X"])
print(f"Poisson: {time.time() - t0:.0f}s | rho: {RHO}")

# ==== Modello combinato 1-X-2: test su 3 stagioni
BASE = ["elo_diff", "pois_1v2", "pois_X"] + (["xg_1v2", "xg_X", "diff_xg"] if USA_XG else [])
CON_VAL = BASE + ["diff_valore"]
prima_train = codice(PRIMA_XG + 1) if USA_XG else "1314"
ev = df[(df.stagione >= prima_train) & df.esito.notna()].dropna(subset=CON_VAL + ["p_book_1"])

def allena(train, cols):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(train[cols], train.esito)

# le quote del bookmaker entrano nel modello come informazione in più: il modello parte da lì e le corregge
df["book_1v2"] = np.log(df.p_book_1 / df.p_book_2)
df["book_X"] = np.log(df.p_book_X)
BOOK = ["book_1v2", "book_X"]
ev = df[(df.stagione >= prima_train) & df.esito.notna()].dropna(subset=CON_VAL + BOOK)

test_idx, P_comb, P_senza = [], [], []
for ts in TEST_STAGIONI:
    train, test = ev[ev.stagione < ts], ev[ev.stagione == ts]
    P_comb.append(allena(train, CON_VAL + BOOK).predict_proba(test[CON_VAL + BOOK]))
    P_senza.append(allena(train, CON_VAL).predict_proba(test[CON_VAL]))
    test_idx.append(test.index)
test_idx = np.concatenate(test_idx); P_comb = np.vstack(P_comb); P_senza = np.vstack(P_senza)
tt = df.loc[test_idx]
P_book = tt[["p_book_1", "p_book_2", "p_book_X"]].values
LL_MOD, LL_BOOK = log_loss(tt.esito, P_comb, labels=CLASSI), log_loss(tt.esito, P_book, labels=CLASSI)
LL_SENZA = log_loss(tt.esito, P_senza, labels=CLASSI)
print(f"1-X-2 senza quote in ingresso: {LL_SENZA:.4f}")
ACC_MOD = accuracy_score(tt.esito, np.array(CLASSI)[P_comb.argmax(1)])
ACC_BOOK = accuracy_score(tt.esito, np.array(CLASSI)[P_book.argmax(1)])
print(f"1-X-2 su {len(tt)} partite di test — log loss modello: {LL_MOD:.4f} | bookmaker: {LL_BOOK:.4f}")

# ==== Dalla tabella dei risultati ai mercati: taratura sul modello combinato
def tabella_tarata(tot_gol, p1, p2, rho):
    """Tabella dei risultati esatti con: gol totali attesi dal Poisson,
    rapporto vittoria casa / vittoria trasferta dal modello combinato."""
    obiettivo = p1 / (p1 + p2)
    def f(x):
        P = matrice(x, tot_gol - x, rho)[0]
        a, b = np.tril(P, -1).sum(), np.triu(P, 1).sum()
        return a / (a + b) - obiettivo
    lo, hi = 0.02, tot_gol - 0.02
    x = lo if f(lo) > 0 else hi if f(hi) < 0 else brentq(f, lo, hi)
    return matrice(x, tot_gol - x, rho)[0]

def gol_totali(riga, modo):
    tg = riga.lam_c + riga.lam_t
    if not USA_XG or modo == "gol" or np.isnan(riga.lamxg_c):
        return tg
    tx = riga.lamxg_c + riga.lamxg_t
    return tx if modo == "xg" else (tg + tx) / 2

def p_over(P, soglia=2.5):
    i, j = np.indices(P.shape)
    return P[(i + j) > soglia].sum()

# ==== Test Under/Over 2.5 sul passato + simulazione giocate "di valore"
ou = tt.assign(over=(tt.gol_c + tt.gol_t > 2.5).astype(int))
risultati_ou = {}
for modo in ["gol", "xg", "misto"] if USA_XG else ["gol"]:
    risultati_ou[f"Modello tarato ({modo})"] = np.array([
        p_over(tabella_tarata(gol_totali(r, modo), pc[0], pc[1], RHO["gol"]))
        for r, pc in zip(ou.itertuples(), P_comb)])
risultati_ou["Poisson puro (gol)"] = np.array([p_over(matrice(r.lam_c, r.lam_t, RHO["gol"])[0]) for r in ou.itertuples()])
ok_ou = ou.p_book_over.notna().values
risultati_ou["Bookmaker"] = ou.p_book_over.values
tab_ou = pd.DataFrame({k: {"log_loss": log_loss(ou.over[ok_ou], np.clip(v[ok_ou], 1e-6, 1 - 1e-6)),
                           "prob_media_over": v[ok_ou].mean()} for k, v in risultati_ou.items()}).T.round(4)
tab_ou.loc["Frequenza reale", "prob_media_over"] = round(ou.over[ok_ou].mean(), 4)
MODO_OU = min([k for k in risultati_ou if k.startswith("Modello")], key=lambda k: tab_ou.loc[k, "log_loss"])
print(f"Under/Over 2.5 su {ok_ou.sum()} partite di test (più basso = meglio):")
print(tab_ou.to_string())
MODO_GOL = MODO_OU.split("(")[1].rstrip(")")
print("Gol totali stimati con:", MODO_OU)

# probabilità di Over 2.5 del modello per tutte le partite, poi combinata con quella del bookmaker
def totale_vett(d, modo):
    tg = (d.lam_c + d.lam_t).values
    if not USA_XG or modo == "gol":
        return tg
    tx = (d.lamxg_c + d.lamxg_t).values
    tx = np.where(np.isnan(tx), tg, tx)
    return tx if modo == "xg" else (tg + tx) / 2

ok_l = df.lam_c.notna().values
T_all = totale_vett(df, MODO_GOL)
k_all = T_all / (df.lam_c + df.lam_t).values
df["p_over_mod"] = np.nan
if ok_l.any():
    Pm = matrice(df.lam_c.values[ok_l] * k_all[ok_l], df.lam_t.values[ok_l] * k_all[ok_l], RHO["gol"])
    ii, jj = np.indices(Pm.shape[1:])
    df.loc[ok_l, "p_over_mod"] = Pm[:, (ii + jj) > 2.5].sum(axis=1)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
df["ou_mod"], df["ou_book"] = logit(df.p_over_mod), logit(df.p_book_over)
df["over"] = np.where(df.esito.notna(), (df.gol_c + df.gol_t > 2.5).astype(float), np.nan)
OU_COLS = ["ou_mod", "ou_book"]
ev_ou = df[(df.stagione >= prima_train) & df.esito.notna()].dropna(subset=OU_COLS)
def allena_ou(train):
    return LogisticRegression(max_iter=1000).fit(train[OU_COLS], train.over.astype(int))
p_blend = pd.Series(np.nan, index=df.index)
for ts in TEST_STAGIONI:
    tr_, te_ = ev_ou[ev_ou.stagione < ts], ev_ou[ev_ou.stagione == ts]
    if len(te_):
        p_blend[te_.index] = allena_ou(tr_).predict_proba(te_[OU_COLS])[:, 1]
risultati_ou["Modello + quote"] = p_blend.reindex(ou.index).values
tab_ou.loc["Modello + quote", "log_loss"] = round(log_loss(ou.over[ok_ou], np.clip(risultati_ou["Modello + quote"][ok_ou], 1e-6, 1 - 1e-6)), 4)
tab_ou.loc["Modello + quote", "prob_media_over"] = round(np.nanmean(risultati_ou["Modello + quote"][ok_ou]), 4)
MODO_OU = "Modello + quote"
print(tab_ou.to_string())

def simula(prob, quote, vinte, soglie=(0.0, 0.05, 0.10, 0.20), paga=None):
    """Punta 1 unità ogni volta che probabilità × quota − 1 supera la soglia.
    'paga' = quota a cui la giocata viene davvero pagata (es. quota di chiusura); di default la stessa."""
    prob, quote, vinte = np.asarray(prob, float), np.asarray(quote, float), np.asarray(vinte, bool)
    paga = quote if paga is None else np.where(np.isfinite(np.asarray(paga, float)), np.asarray(paga, float), quote)
    righe = []
    for s in soglie:
        ev_ = prob * quote - 1
        punta = (ev_ > s) & np.isfinite(quote)
        n = int(punta.sum())
        guad = np.where(vinte[punta], paga[punta] - 1, -1).sum() if n else 0.0
        righe.append({"soglia_valore": f">{s:.0%}", "giocate": n, "vinte": int(vinte[punta].sum()),
                      "profitto_unita": round(guad, 1), "ROI": f"{guad / n:+.1%}" if n else "—"})
    return pd.DataFrame(righe)

q = tt[["q1", "q2", "qX"]].values
vinte = (tt.esito.to_numpy(dtype=object)[:, None] == np.array(CLASSI, dtype=object)[None, :])
print("\nSimulazione 1-X-2 (tutte le giocate su 1, X o 2 con valore sopra soglia):")
SIM_1X2 = simula(P_comb.ravel(), q.ravel(), vinte.ravel())
print(SIM_1X2.to_string(index=False))
po = risultati_ou[MODO_OU]
qo = np.column_stack([ou.q_over.values, ou.q_under.values])
print("\nSimulazione Under/Over 2.5:")
SIM_OU = simula(np.column_stack([po, 1 - po]).ravel(), qo.ravel(),
                np.column_stack([ou.over == 1, ou.over == 0]).ravel())
print(SIM_OU.to_string(index=False))
print("\nROI negativo = in quelle stagioni si sarebbe perso. Con poche giocate il risultato è molto incerto.")

# ==== Verifica contro le quote di chiusura
qs = tt[["qs1", "qs2", "qsX"]].values          # ordine 1, 2, X
qc = tt[["qc1", "qc2", "qcX"]].values
ok_c = np.isfinite(qs).all(axis=1)
inv_s = 1 / qs
p_chius = inv_s / inv_s.sum(axis=1, keepdims=True)
CHIUS = {"partite": int(ok_c.sum())}
if ok_c.sum() >= 100:
    CHIUS["log_loss_modello"] = round(float(log_loss(tt.esito[ok_c], P_comb[ok_c], labels=CLASSI)), 4)
    CHIUS["log_loss_chiusura"] = round(float(log_loss(tt.esito[ok_c], p_chius[ok_c], labels=CLASSI)), 4)
    # "closing line value": la quota presa batte la quota giusta di chiusura?
    for soglia in (0.0, 0.05):
        ev_ = P_comb * q - 1
        pk = (ev_ > soglia) & np.isfinite(q) & ok_c[:, None]
        if pk.sum():
            CHIUS[f"clv_{int(soglia*100)}"] = round(float((q[pk] * p_chius[pk]).mean() - 1), 4)
            CHIUS[f"giocate_clv_{int(soglia*100)}"] = int(pk.sum())
    SIM_CHIUS = simula(P_comb.ravel(), q.ravel(), vinte.ravel(), paga=qc.ravel())
    print("\nContro la chiusura:", CHIUS)
    print("Simulazione 1-X-2 pagata alla quota di chiusura Bet365:")
    print(SIM_CHIUS.to_string(index=False))
else:
    SIM_CHIUS = pd.DataFrame()

# ==== Piano di budget (criterio di Kelly prudente) sulle stagioni di test
FRAZ_KELLY, MAX_GIOCATA, MAX_GIORNO = 0.25, 0.05, 0.30

def frazioni_kelly(p, quote):
    """Per ogni partita: la giocata 1-X-2 con più vantaggio e la % di budget (Kelly/4, max 5%)."""
    ev_ = p * quote - 1
    ev_ = np.where(np.isfinite(ev_), ev_, -1)
    k = ev_.argmax(axis=1)
    e = ev_[np.arange(len(p)), k]
    qk = quote[np.arange(len(p)), k]
    f = np.where(e > 0, np.minimum(FRAZ_KELLY * e / (qk - 1), MAX_GIOCATA), 0.0)
    return k, f, e

def simula_kelly(paga, budget=100.0):
    k, f, _ = frazioni_kelly(P_comb, q)
    paga = np.where(np.isfinite(paga), paga, q)
    d = pd.DataFrame({"data": tt.data.values, "f": f, "vinta": vinte[np.arange(len(k)), k],
                      "quota": paga[np.arange(len(k)), k]})
    banca, picco, ribasso, minimo, n, vinte_n, storia = budget, budget, 0.0, budget, 0, 0, []
    for giorno, g in d.groupby("data", sort=True):
        g = g[g.f > 0]
        if g.empty:
            continue
        scala = min(1.0, MAX_GIORNO / g.f.sum())
        puntate = banca * g.f.values * scala
        banca += np.where(g.vinta.values, puntate * (g.quota.values - 1), -puntate).sum()
        n += len(g); vinte_n += int(g.vinta.sum())
        picco = max(picco, banca); minimo = min(minimo, banca); ribasso = max(ribasso, 1 - banca / picco)
        storia.append([str(pd.Timestamp(giorno).date()), round(float(banca), 2)])
    return {"budget_iniziale": budget, "finale": round(float(banca), 2), "giocate": n, "vinte": vinte_n,
            "minimo": round(float(minimo), 2), "ribasso_massimo": round(float(ribasso), 3), "andamento": storia}

KELLY = {"quote_anticipo": simula_kelly(q), "quote_chiusura": simula_kelly(qc)}

# ==== Risultati stagione per stagione (per vedere se il vantaggio è costante)
PER_STAGIONE = []
ev5 = P_comb * q - 1
for st_ in TEST_STAGIONI:
    m_ = (tt.stagione == st_).values
    if m_.sum() < 50:
        continue
    riga = {"stagione": f"20{st_[:2]}/{st_[2:]}", "partite": int(m_.sum()),
            "errore_modello": round(float(log_loss(tt.esito[m_], P_comb[m_], labels=CLASSI)), 4),
            "errore_bookmaker": round(float(log_loss(tt.esito[m_], P_book[m_], labels=CLASSI)), 4)}
    mc_ = m_ & ok_c
    if mc_.sum() >= 50:
        riga["errore_chiusura"] = round(float(log_loss(tt.esito[mc_], p_chius[mc_], labels=CLASSI)), 4)
    pk = (ev5 > 0.05) & np.isfinite(q) & m_[:, None]
    n_ = int(pk.sum())
    pay = np.where(np.isfinite(qc), qc, q)
    riga["giocate_valore"] = n_
    riga["resa_valore"] = round(float(np.where(vinte[pk], pay[pk] - 1, -1).sum() / n_), 3) if n_ else None
    PER_STAGIONE.append(riga)
print(pd.DataFrame(PER_STAGIONE).to_string(index=False))
for nome_k, r_k in KELLY.items():
    print(f"Kelly ({nome_k}): 100 → {r_k['finale']} | giocate {r_k['giocate']} | minimo {r_k['minimo']} | ribasso max {r_k['ribasso_massimo']:.0%}")
PIANO_ATTIVO = (KELLY["quote_chiusura"]["finale"] > 100 and CHIUS.get("clv_0", -1) > 0
                and CHIUS.get("log_loss_modello", 9) < CHIUS.get("log_loss_chiusura", 0) + 0.005)
MOTIVO_PIANO = ("Il test sulle stagioni passate è positivo anche contro le quote di chiusura." if PIANO_ATTIVO else
                "Piano disattivato: il test sulle stagioni passate non mostra un vantaggio solido contro le quote di chiusura.")
print(MOTIVO_PIANO)

# ==== Previsioni per la prossima giornata
fin = df[df.esito.notna() & (df.stagione >= prima_train)]
MODELLI_FIN = {}
for usa_v in (True, False):
    for usa_b in (True, False):
        cols = (CON_VAL if usa_v else BASE) + (BOOK if usa_b else [])
        MODELLI_FIN[(usa_v, usa_b)] = (cols, allena(fin.dropna(subset=cols), cols))
mod_ou = allena_ou(ev_ou)

def tabella_tarata_ou(obiettivo_over, p1, p2, rho):
    """Come tabella_tarata, ma sceglie i gol totali in modo che la probabilità di Over 2.5 sia quella voluta."""
    f = lambda T: p_over(tabella_tarata(T, p1, p2, rho)) - obiettivo_over
    lo, hi = 0.8, 6.0
    T = lo if f(lo) > 0 else hi if f(hi) < 0 else brentq(f, lo, hi, xtol=1e-3)
    return tabella_tarata(T, p1, p2, rho)

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
    usa_val = not pd.isna(r.diff_valore)
    usa_book = not pd.isna(r.book_1v2)
    cols, mdl = MODELLI_FIN[(usa_val, usa_book)]
    x = pd.DataFrame([{c: getattr(r, c) for c in cols}])
    if pd.isna(r.lam_c) or x.isna().values.any():
        print(f"{r.casa}-{r.trasf}: dati insufficienti, salto")
        NOTE.append(f"{r.casa} - {r.trasf}: dati insufficienti per la previsione.")
        continue
    pc = mdl.predict_proba(x)[0]      # ordine 1, 2, X
    if pd.notna(r.ou_mod) and pd.notna(r.ou_book):
        obiettivo = mod_ou.predict_proba(pd.DataFrame([{"ou_mod": r.ou_mod, "ou_book": r.ou_book}]))[0, 1]
        P = tabella_tarata_ou(obiettivo, pc[0], pc[1], RHO["gol"])
    else:
        P = tabella_tarata(gol_totali(r, MODO_GOL), pc[0], pc[1], RHO["gol"])
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
    partite.append({"data": str(r.data.date()), "ora": ora,
                    "casa": r.casa, "trasf": r.trasf, "valori_usati": usa_val, "gol_attesi": round(gol_att, 2),
                    "mercati": per_mercato})
    candidati = [(v["valore"], m_, v) for m_, lst in per_mercato.items() for v in lst if v["valore"] is not None]
    miglior = max(candidati, key=lambda t: t[0]) if candidati else None
    righe_storico.append({"stagione": r.stagione, "data": str(r.data.date()), "casa": r.casa, "trasf": r.trasf,
                          "p1": ms[("1X2", "1")], "pX": ms[("1X2", "X")], "p2": ms[("1X2", "2")],
                          "pb1": r.p_book_1, "pbX": r.p_book_X, "pb2": r.p_book_2,
                          "p_over25": ms[("Under/Over", "Over 2.5")], "p_goal": ms[("Goal/No Goal", "Goal")],
                          "pick_mercato": miglior[1] if miglior and miglior[0] > 0.05 else "",
                          "pick_esito": miglior[2]["esito"] if miglior and miglior[0] > 0.05 else "",
                          "pick_quota": miglior[2]["quota_book"] if miglior and miglior[0] > 0.05 else np.nan,
                          "pick_valore": miglior[0] if miglior and miglior[0] > 0.05 else np.nan})
print(f"Previsioni calcolate: {len(partite)}")
tot_piano = sum(g["percentuale"] for g in piano)
for g in piano:
    g["percentuale"] = round(g["percentuale"] * min(1.0, MAX_GIORNO / tot_piano), 4) if tot_piano else 0

# ==== Storico: salva le previsioni e confrontale con i risultati
os.makedirs("docs", exist_ok=True)
F_STORICO = "docs/storico.csv"
st_old = pd.read_csv(F_STORICO, dtype={"stagione": str, "pick_mercato": str, "pick_esito": str}) if os.path.exists(F_STORICO) else pd.DataFrame()
nuovo = pd.DataFrame(righe_storico)
if len(st_old) and len(nuovo):
    chiave = ["stagione", "casa", "trasf"]
    st_old = st_old.merge(nuovo[chiave], on=chiave, how="left", indicator=True)
    st_old = st_old[st_old._merge == "left_only"].drop(columns="_merge")   # previsione più recente vince
storico = pd.concat([st_old, nuovo], ignore_index=True)
if len(storico):
    storico["stagione"] = storico["stagione"].astype(str).str.zfill(4)
    storico = storico.drop(columns=[c for c in ["gol_c", "gol_t"] if c in storico])
    ris = giocate[["stagione", "casa", "trasf", "gol_c", "gol_t"]]
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
        pr = {"1": r.p1, "X": r.pX, "2": r.p2}
        pick = f"{r.pick_mercato} {r.pick_esito}".strip() if str(r.pick_mercato) not in ("", "nan") else ""
        ultime.append({"data": r.data, "casa": r.casa, "trasf": r.trasf, "risultato": f"{int(r.gol_c)}-{int(r.gol_t)}",
                       "esito": str(e), "previsto": max(pr, key=pr.get), "p_esito": round(float(pr[str(e)]), 3),
                       "pick": pick})
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
    return json.loads(t.to_json(orient="records"))

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
        "x12": {"log_loss_modello": round(LL_MOD, 4), "log_loss_bookmaker": round(LL_BOOK, 4),
                "log_loss_senza_quote": round(LL_SENZA, 4),
                "esito_azzeccato_modello": round(ACC_MOD, 3), "esito_azzeccato_bookmaker": round(ACC_BOOK, 3)},
        "ou25": [{"modello": k, "log_loss": float(v.log_loss)} for k, v in tab_ou.dropna(subset=["log_loss"]).iterrows()],
        "ou_scelto": MODO_OU,
        "sim_1x2": tab_json(SIM_1X2), "sim_ou": tab_json(SIM_OU),
        "chiusura": {**CHIUS, "sim": tab_json(SIM_CHIUS) if len(SIM_CHIUS) else []},
        "kelly": KELLY,
        "per_stagione": PER_STAGIONE,
    },
    "storico": {"riepilogo": riep_storico, "ultime": ultime_storico},
}
with open("docs/dati.json", "w", encoding="utf-8") as f:
    json.dump(dati, f, ensure_ascii=False, indent=1, default=str)
pd.DataFrame(righe_csv).to_csv("docs/previsioni_giornata.csv", index=False)
print("Scritto docs/dati.json e docs/previsioni_giornata.csv")
