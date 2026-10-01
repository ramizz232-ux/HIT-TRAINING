# ═══════════════════════════════════════════════════════════════════════════
# ARTEMIS SIGNAL SCANNER v1.0
# Sistema: ARTEMIS Dragon Portfolio
# Data: 2026-05-02
#
# Funzione: Raccoglie ~55 segnali freemium + 3 stack Artemis-specifici
#           e li scrive in ARTEMIS_SIGNALS (Supabase).
#           Produce artemis_status.json per la Skill ARTEMIS.
#
# Stack originali (da ARCHIMEDES, rinominati e adattati):
#   · Volatility Stack       — VIX, VVIX, SKEW, MOVE, OVX, GVZ, VXN, VIX3M
#   · Rates/Credit Stack     — curva, CPI, PCE, NFP, Fed BS, Net Liquidity
#   · FX/Commodity Stack     — DXY, Gold, Silver, Copper, Copper/Gold ratio
#   · Sentiment Stack        — F&G (reale+sintetico), NAAIM, GPR, EPU, UMCSENT
#   · EIA Stack              — inventari petrolio
#   · SOCMINT Lite           — Short Interest + Volume anomalo
#   · Insider Intelligence   — Form4 cluster + Congress trades
#
# Stack NUOVI Artemis-specifici:
#   · Long Volatility Monitor — term structure, realized/implied ratio,
#                               CBOE Long Vol proxy, skew Artemis
#   · CWARP Calculator        — Sortino + Return-to-MaxDD + CWARP score
#                               per asset candidati al Dragon Portfolio
#   · Dragon Health Check     — drift allocazione corrente vs target Dragon
#
# Output:
#   · JSON snapshot → ARTEMIS_SIGNALS (Supabase, tabella principale)
#   · artemis_status.json → file locale (input Skill ARTEMIS)
#   · Report console strutturato
#
# GitHub Secrets richiesti:
#   ARTEMIS_FRED_KEY      — FRED API (fred.stlouisfed.org)
#   ARTEMIS_SUPABASE_URL  — URL progetto Supabase
#   ARTEMIS_SUPABASE_KEY  — Service Role Key Supabase
#   ARTEMIS_EIA_KEY       — EIA API (eia.gov) — opzionale
#
# GitHub Secret opzionale (già usato in ORACLE):
#   FRED_API_KEY          — fallback se ARTEMIS_FRED_KEY non presente
#   ARTEMIS_NASDAQ_KEY    — NON PIÙ USATA (fix 2026-07-14): Nasdaq Data Link
#                           è protetto da anti-bot (Incapsula/Imperva),
#                           bloccato sia da browser che da GitHub Actions.
#                           Il secret può restare configurato, innocuo.
#                           NAAIM usa solo fallback naaim.org o resta None.
#
# Schedule: ogni giorno (cron GitHub Actions)
# Breakpoint: il Gestore legge il report console prima di aprire
#             la sessione con la Skill ARTEMIS.
# ═══════════════════════════════════════════════════════════════════════════

# ── SEZIONE 1: IMPORTS ──────────────────────────────────────────────────────
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from supabase import create_client
from datetime import datetime, timedelta
import requests
import json
import logging
import math
import os
import time
import warnings
from artemis_regime_engine import classify_regime, _fetch_oecd_cli

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ARTEMIS.signal_scanner")


# ── SEZIONE 2: CONFIGURAZIONE ───────────────────────────────────────────────

# Secrets GitHub → env vars
FRED_KEY     = (os.environ.get("ARTEMIS_FRED_KEY") or
                os.environ.get("FRED_API_KEY", ""))
SUPABASE_URL = os.environ.get("ARTEMIS_SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("ARTEMIS_SUPABASE_KEY", "")
EIA_KEY      = os.environ.get("ARTEMIS_EIA_KEY", "")

# ── SOGLIE OPERATIVE ARTEMIS ─────────────────────────────────────────────────
# Soglie volatilità
VIX_CRISIS_THRESHOLD    = 30      # VIX > 30 → stress acuto → Hawk imminente
VIX_ATTENTION_THRESHOLD = 20      # VIX 20-30 → attenzione
SKEW_ATTENTION          = 115     # SKEW > 115 → tail risk asimmetrico
SKEW_PROTECTION_PRICED  = 130     # SKEW > 130 → protezione cara → Volatility at World's End
MOVE_ATTENTION          = 120     # MOVE > 120 → stress bond strutturale
VVIX_ATTENTION          = 40      # VVIX > 40 → vol della vol elevata (stesso valore già usato altrove nel file)
OVX_ATTENTION           = 60      # OVX > 60 → stress energia → FALCO_DX
GVZ_ATTENTION           = 25      # GVZ > 25 → stress oro → segnale FENICE

# Soglie tassi
T10Y2Y_INVERSION        = 0.0    # T10Y2Y < 0 → curva invertita → recessione
T10Y3M_INVERSION        = 0.0    # T10Y3M < 0 → più predittiva

# Soglie Artemis-specifiche
# Oro → FENICE (rivisto 26/09/2026). Prima: soglie di PREZZO fisse
# (2.500 / 3.000 / 4.000 $). Il prezzo nominale dell'oro sale nel tempo anche
# solo per l'inflazione: con l'oro a 4.300 $ la regola dava "FENICE FORTE" in
# permanenza, anche con un rialzo annuo normale (+14,6%) e il motore in SERPENTE.
# Ora si usa la VARIAZIONE annua, letta insieme al tasso reale USA 10 anni:
# un oro che sale forte mentre i tassi reali sono alti (normalmente lo
# frenano) indica domanda da sfiducia verso le valute, non un semplice ciclo.
GOLD_YOY_FENICE_STRONG  = 30.0   # oro +30% in un anno → rialzo eccezionale
GOLD_YOY_FENICE         = 15.0   # oro +15% in un anno …
GOLD_REAL_YIELD_HIGH    = 1.5    # … con tasso reale 10y ≥ 1,5% (dovrebbe frenarlo)

# Copper/Gold ratio (calibrato per oro $3,000-4,500 — 2026)
COPPER_GOLD_BULL_RATIO  = 0.0010  # risk-on se Cu/Au ratio > soglia

# Long Volatility Monitor soglie
VIX_TERM_CONTANGO_MIN   = 0.02    # slope > 2% → contango sano (Serpent)
VIX_TERM_BACKWARDATION  = -0.05   # slope < -5% → backwardation acuta (Hawk)
REALVOL_IMPLIED_RATIO   = 0.8     # realized/implied < 0.8 → vol implicita cara

# CWARP soglie (Cole framework)
CWARP_POSITIVE_MIN      = 0.0     # CWARP > 0 → asset migliora il portafoglio
CWARP_STRONG            = 10.0    # CWARP > 10 → forte diversificatore
CWARP_EXCELLENT         = 20.0    # CWARP > 20 → eccellente (Long Vol, Gold tipici)

# ─────────────────────────────────────────────────────────────────────────────
# CWARP CANDIDATES — Lista configurabile degli asset da valutare ad ogni run
# ─────────────────────────────────────────────────────────────────────────────
# Per aggiungere un nuovo asset: aggiungi un dict con id/name/component.
# component deve essere uno di: equity / fixed_income / long_volatility /
#                                commodity_trend / gold
#
# GitHub Actions calcola il CWARP di TUTTI gli asset in questa lista ad ogni run.
# Il risultato va in ARTEMIS_SIGNALS.full_snapshot["cwarp"]["candidates"]
# e il top scorer va in cwarp_top1_ticker / cwarp_top1_score.
# ─────────────────────────────────────────────────────────────────────────────
def _load_cwarp_candidates_from_supabase() -> list[dict]:
    """Legge ARTEMIS_CWARP_CANDIDATES da Supabase (solo is_active=TRUE).
    Permette di aggiungere/pausare asset senza modificare questo file.
    Fallback: lista vuota con WARNING se Supabase non raggiungibile.

    Legge anche exit_trigger_cwarp/exit_trigger_regime per il check di
    soglia in compute_cwarp() (vedi nota lì per cosa fa e cosa NON fa).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("[cwarp] Credenziali Supabase mancanti — candidati vuoti")
        return []
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = (
            client.table("ARTEMIS_CWARP_CANDIDATES")
            .select("ticker, name, dragon_component, priority, "
                    "exit_trigger_cwarp, exit_trigger_regime, is_in_portfolio")
            .eq("is_active", True)
            .order("priority")
            .execute()
        )
        if resp.data:
            candidates = [
                {"id": r["ticker"], "name": r["name"], "component": r["dragon_component"],
                 "exit_trigger_cwarp": r.get("exit_trigger_cwarp"),
                 "exit_trigger_regime": r.get("exit_trigger_regime"),
                 "is_in_portfolio": r.get("is_in_portfolio", False)}
                for r in resp.data
            ]
            logger.info(f"[cwarp] {len(candidates)} candidati attivi da ARTEMIS_CWARP_CANDIDATES")
            return candidates
        logger.warning("[cwarp] ARTEMIS_CWARP_CANDIDATES: nessun asset is_active=TRUE")
        return []
    except Exception as exc:
        logger.warning(f"[cwarp] Fallback lista vuota — Supabase non raggiungibile: {exc}")
        return []


def _classify_cwarp_trend(cwarp_now, cwarp_30d) -> str:
    """Classifica trend CWARP — 4 stati (SALITA/STABILE/ATTENZIONE/USCITA)."""
    if cwarp_now is None:
        return "NO_DATA"
    if cwarp_30d is None:
        return "STABILE"
    delta = cwarp_now - cwarp_30d
    if delta > 5:    return "SALITA"
    if delta >= -5:  return "STABILE"
    if delta >= -10: return "ATTENZIONE"
    return "USCITA"


def collect_manual_signals() -> dict:
    """
    Legge segnali che NON sono automatizzabili via ticker — dati
    fondamentali che richiedono aggiornamento manuale trimestrale.
    Aggiunto 2026-09-16 per il capex/FCF degli hyperscaler AI (idea dal
    DREAM / Lighthouse Macro): il cuore quantitativo dello Scenario 1
    (collisione tassi-AI), che nessun ticker fornisce.

    Legge dalla tabella Supabase ARTEMIS_MANUAL_SIGNALS (colonne:
    nome, valore, unita, fonte, data_aggiornamento, note). Se un valore
    è più vecchio di 120 giorni, lo segnala come STALE invece di usarlo
    in silenzio — stesso principio anti-silent-failure di tutto il sistema.

    Restituisce dict {nome: {...}}. Vuoto se la tabella non esiste o è
    vuota (nessun errore — è opzionale).
    """
    out = {}
    if not SUPABASE_URL or not SUPABASE_KEY:
        return out
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        rows = client.table("ARTEMIS_MANUAL_SIGNALS").select("*").execute()
        if not rows.data:
            return out
        for r in rows.data:
            nome = r.get("nome")
            if not nome:
                continue
            data_agg = r.get("data_aggiornamento")
            stale = False
            giorni = None
            if data_agg:
                try:
                    d = datetime.fromisoformat(str(data_agg)[:10])
                    giorni = (datetime.now() - d).days
                    stale = giorni > 120
                except Exception:
                    pass
            out[nome] = {
                "valore": r.get("valore"), "unita": r.get("unita"),
                "fonte": r.get("fonte"), "data_aggiornamento": data_agg,
                "giorni_fa": giorni, "stale": stale, "note": r.get("note"),
            }
            if stale:
                logger.warning(f"[manuale] {nome}={r.get('valore')} "
                               f"AGGIORNAMENTO VECCHIO ({giorni}gg) — verificare")
            else:
                logger.info(f"[manuale] {nome}={r.get('valore')} "
                            f"({r.get('unita','')}) fonte={r.get('fonte','?')}")
    except Exception as e:
        logger.info(f"[manuale] ARTEMIS_MANUAL_SIGNALS non letta: {e}")
    return out


def check_scenario_alerts(artemis_status: dict) -> list[dict]:
    """
    Valuta gli scenari di monitoraggio definiti in ARTEMIS_SCENARIOS.

    Gli scenari NON sono hardcoded qui: si leggono da Supabase, così una
    soglia si modifica dal Table Editor senza toccare questo file (stesso
    principio di ARTEMIS_CWARP_CANDIDATES).

    Conteggio run consecutive: molti scenari richiedono che la condizione
    regga su più run, non un singolo tocco. Il contatore è "day-aware":
    più run nello stesso giorno NON lo gonfiano (verificato 16/9/2026,
    3 run nello stesso giorno). Si incrementa solo al cambio di data.

    Restituisce la lista degli scenari scattati (per il report).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    signals = dict((artemis_status or {}).get("signals", {}))  # copia
    # Inietta i segnali manuali (capex AI ecc.) nel dict valutabile dagli
    # scenari, appiattiti a nome->valore. Solo quelli NON stale: un dato
    # vecchio non deve far scattare un alert su un numero superato.
    for nome, meta in (artemis_status or {}).get("manual_signals", {}).items():
        if isinstance(meta, dict) and not meta.get("stale") and meta.get("valore") is not None:
            signals[nome] = meta["valore"]

    if not signals:
        return []

    scattati = []
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        rows = (client.table("ARTEMIS_SCENARIOS").select("*")
                .eq("is_active", True).execute())
        if not rows.data:
            return []

        oggi = datetime.today().date().isoformat()

        for sc in rows.data:
            nome = sc.get("nome", "?")
            campo = sc.get("campo")
            op = (sc.get("operatore") or ">").strip()
            soglia = sc.get("soglia")
            run_richieste = int(sc.get("run_consecutive") or 1)

            valore = signals.get(campo)
            if valore is None or soglia is None:
                continue
            try:
                valore = float(valore)
                soglia = float(soglia)
            except (TypeError, ValueError):
                continue

            # Valutazione condizione
            if   op == ">":  cond = valore >  soglia
            elif op == ">=": cond = valore >= soglia
            elif op == "<":  cond = valore <  soglia
            elif op == "<=": cond = valore <= soglia
            else:
                logger.warning(f"[scenari] {nome}: operatore '{op}' non gestito")
                continue

            conta = int(sc.get("run_consecutive_attuali") or 0)
            ultima = sc.get("ultima_data_valutata")

            if cond:
                # Incrementa solo se è un giorno nuovo (evita gonfiaggio
                # con più run nella stessa giornata)
                if ultima != oggi:
                    conta += 1
            else:
                conta = 0   # condizione rientrata → azzera

            try:
                client.table("ARTEMIS_SCENARIOS").update({
                    "run_consecutive_attuali": conta,
                    "ultima_data_valutata": oggi,
                    "ultimo_valore": round(valore, 4),
                    "updated_at": datetime.now().isoformat(),
                }).eq("nome", nome).execute()
            except Exception as _e_up:
                logger.warning(f"[scenari] {nome}: update fallito: {_e_up}")

            if cond and conta >= run_richieste:
                scattati.append({
                    "nome": nome, "campo": campo, "valore": valore,
                    "operatore": op, "soglia": soglia,
                    "run_consecutive": conta,
                    "descrizione": sc.get("descrizione"),
                    "note_operativa": sc.get("note_operativa"),
                })
                logger.warning(
                    f"[scenari] ⚠️  TRIGGER — {nome}: {campo}={valore} {op} "
                    f"{soglia} da {conta} run consecutive")
            elif cond:
                logger.info(
                    f"[scenari] {nome}: condizione VERA ({campo}={valore} {op} "
                    f"{soglia}) — {conta}/{run_richieste} run, non ancora trigger")
            else:
                # Quanto manca alla soglia (utile per vedere avvicinamenti)
                dist = abs(valore - soglia)
                logger.info(f"[scenari] {nome}: non attivo "
                            f"({campo}={valore}, soglia {op} {soglia}, "
                            f"distanza {dist:.3f})")

    except Exception as exc:
        logger.warning(f"[scenari] Valutazione fallita: {exc}")

    return scattati


def diagnose_price_conversion(cwarp_candidates_meta: list[dict] | None = None) -> None:
    """
    FASE 1 (diagnostica, 2026-09-15) — NON SCRIVE NULLA SU SUPABASE.

    Scarica il prezzo corrente di ogni ticker candidato, applica la conversione
    valutaria verso EUR, e logga il confronto con il current_price_eur già
    presente in ARTEMIS_CWARP_CANDIDATES (che oggi il Gestore aggiorna a mano).

    Scopo: verificare che la conversione sia corretta PRIMA di abilitare la
    scrittura automatica. Un prezzo sbagliato scritto ogni giorno falserebbe
    silenziosamente Dragon Score, drift e raccomandazioni — molto peggio di un
    prezzo semplicemente vecchio, perché sembra giusto.

    Complicazione nota (verificata in produzione 15/9/2026): la valuta di
    quotazione NON è deducibile dal suffisso del ticker. Sulla LSE convivono
    classi in valute diverse — SGLN.L è in GBp (pence), ENCO.L e FWRA.L sono
    in USD pur avendo lo stesso suffisso .L. Dividere per 100 questi ultimi
    produceva -98% di errore. Nemmeno il campo 'currency' di
    ARTEMIS_CWARP_CANDIDATES è affidabile (dice "USD" anche per SGLN.L, che
    è in pence). La valuta viene quindi letta dai metadati yfinance.

    ANOMALIA DTLE.L — RISOLTA 15/9/2026: lo scarto ~19% non era un errore di
    conversione ma una questione di CLASSE DI AZIONI. DTLE.L (IE00BJSFR200) è
    la classe EUR Hedged; su Trade Republic è invece disponibile SXRC/A2JKTZ
    (IE00BFM6TC58), classe USD non coperta dello stesso fondo (stesso indice
    ICE U.S. Treasury 20+ Year). Prezzi diversi perché classi diverse, non
    perché la conversione sbaglia. Entrambe tracciate come candidati separati
    per confrontare il costo della copertura valutaria sul CWARP.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.info("[price_diag] Supabase non configurato — skip diagnostica prezzi")
        return

    logger.info("=" * 70)
    logger.info("[price_diag] AGGIORNAMENTO PREZZI — confronto + scrittura su Supabase")
    logger.info("=" * 70)

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        rows = (client.table("ARTEMIS_CWARP_CANDIDATES")
                .select("ticker,currency,current_price_eur,shares,is_in_portfolio")
                .eq("is_active", True).execute())
        if not rows.data:
            logger.info("[price_diag] Nessun candidato attivo — skip")
            return

        # ── Tassi di cambio necessari ────────────────────────────────────────
        # EURUSD: quanti USD per 1 EUR (es. 1.16) → per convertire USD→EUR divido
        # GBPEUR: quanti EUR per 1 GBP (es. 1.19) → per convertire GBP→EUR moltiplico
        # I ticker .L sono in PENCE: prima /100 per avere GBP, poi × GBPEUR.
        fx_rates = {}
        for yf_sym, nome in [("EURUSD=X", "eurusd"), ("GBPEUR=X", "gbpeur")]:
            try:
                _v = _safe_yf(yf_sym, period="5d")
                if _v:
                    fx_rates[nome] = float(_v)
                    logger.info(f"[price_diag] FX {nome} = {fx_rates[nome]:.4f}")
                else:
                    logger.warning(f"[price_diag] FX {nome}: non ottenuto")
            except Exception as _e_fx:
                logger.warning(f"[price_diag] FX {nome} errore: {_e_fx}")

        n_scritti = 0
        logger.info("-" * 70)
        logger.info(f"{'TICKER':<10} {'YF RAW':>12} {'VALUTA':>8} {'→ EUR CALC':>12} "
                    f"{'DB ATTUALE':>12} {'SCARTO %':>10}")
        logger.info("-" * 70)

        for row in rows.data:
            ticker = row.get("ticker")
            if ticker == "CASH_EUR":
                continue   # saldo reale del Gestore, nessun prezzo di mercato
            db_price = row.get("current_price_eur")
            try:
                raw = _safe_yf(ticker, period="5d")
                if raw is None:
                    logger.info(f"{ticker:<10} {'FETCH FALLITO':>12}")
                    continue
                raw = float(raw)

                # ── Valuta di quotazione: LETTA da Yahoo, non dedotta ────────
                # FIX 2026-09-15: la prima versione deduceva la valuta dal
                # suffisso (.L → pence). SBAGLIATO, verificato in produzione:
                # sulla LSE convivono classi di quotazione in valute diverse.
                #   SGLN.L → GBp (6165 pence = €72)
                #   ENCO.L → USD (20.23 USD = €17.52)   ← .L ma NON pence
                #   FWRA.L → USD (9.43 USD = €8.17)     ← .L ma NON pence
                # Dividere per 100 questi ultimi dava -98% (€0.23 invece di
                # €17.57): avrebbe distrutto Dragon Score e drift.
                # Il campo 'currency' di ARTEMIS_CWARP_CANDIDATES non è
                # affidabile (dice "USD" anche per SGLN.L che è in pence),
                # quindi si legge la valuta reale dai metadati yfinance.
                valuta_quot = None
                try:
                    _info = yf.Ticker(ticker).fast_info
                    valuta_quot = (_info.get("currency")
                                   if hasattr(_info, "get") else None)
                except Exception:
                    pass
                if not valuta_quot:
                    try:
                        valuta_quot = yf.Ticker(ticker).info.get("currency")
                    except Exception:
                        valuta_quot = None

                if not valuta_quot:
                    logger.info(f"{ticker:<10} {raw:>12.4f} {'?':>8} "
                                f"{'VALUTA IGNOTA':>12}  ⚠️")
                    continue

                valuta_quot = str(valuta_quot).strip()
                if valuta_quot == "GBp":          # pence britannici
                    eur_calc = ((raw / 100.0) * fx_rates["gbpeur"]
                                if "gbpeur" in fx_rates else None)
                elif valuta_quot == "GBP":        # sterline
                    eur_calc = (raw * fx_rates["gbpeur"]
                                if "gbpeur" in fx_rates else None)
                elif valuta_quot == "USD":
                    eur_calc = (raw / fx_rates["eurusd"]
                                if fx_rates.get("eurusd") else None)
                elif valuta_quot == "EUR":
                    eur_calc = raw
                else:
                    logger.info(f"{ticker:<10} {raw:>12.4f} {valuta_quot:>8} "
                                f"{'VALUTA NON GESTITA':>12}  ⚠️")
                    continue

                if eur_calc is None:
                    logger.info(f"{ticker:<10} {raw:>12.4f} {valuta_quot:>8} "
                                f"{'FX MANCANTE':>12}")
                    continue

                # ── FASE 2 (attiva dal 16/9/2026): scrittura su Supabase ─────
                # Sanity guard: se il prezzo calcolato devia oltre il 25% da
                # quello in DB, NON scrive e segnala. Protegge da glitch di
                # Yahoo, ticker errati o classi di azioni sbagliate che
                # altrimenti corromperebbero silenziosamente il valore del
                # portafoglio (come sarebbe successo il 15/9 col bug pence).
                scarto = None
                if db_price:
                    scarto = (eur_calc - float(db_price)) / float(db_price) * 100
                    flag = "  ✅" if abs(scarto) < 2.0 else "  ⚠️"
                    logger.info(f"{ticker:<10} {raw:>12.4f} {valuta_quot:>8} "
                                f"{eur_calc:>12.4f} {float(db_price):>12.4f} "
                                f"{scarto:>9.2f}%{flag}")
                else:
                    logger.info(f"{ticker:<10} {raw:>12.4f} {valuta_quot:>8} "
                                f"{eur_calc:>12.4f} {'(vuoto)':>12}")

                if scarto is not None and abs(scarto) > 25.0:
                    logger.warning(f"[price_diag] {ticker}: scrittura BLOCCATA "
                                   f"(scarto {scarto:.1f}% > 25%). Prezzo in DB "
                                   f"lasciato invariato — verificare ticker/classe.")
                    continue

                try:
                    client.table("ARTEMIS_CWARP_CANDIDATES").update({
                        "current_price_eur": round(eur_calc, 4),
                        "updated_at": datetime.now().isoformat(),
                    }).eq("ticker", ticker).execute()
                    n_scritti += 1
                except Exception as _e_w:
                    logger.warning(f"[price_diag] {ticker}: scrittura fallita: {_e_w}")
            except Exception as _e_row:
                logger.warning(f"[price_diag] {ticker}: {_e_row}")

        logger.info("-" * 70)
        logger.info(f"[price_diag] ✅ Prezzi aggiornati su Supabase: {n_scritti} ticker")
        logger.info("[price_diag] CASH_EUR escluso — saldo reale, aggiornamento manuale.")
        logger.info("=" * 70)

    except Exception as exc:
        logger.warning(f"[price_diag] Diagnostica fallita: {exc}")


def update_cwarp_candidates_mirror(snapshot: dict) -> None:
    """Scrive CWARP+trend+prezzo come specchio in ARTEMIS_CWARP_CANDIDATES.
    Itera sui candidati nel JSON (include watchlist con shares=0).
    Storico completo resta in ARTEMIS_SIGNALS.cwarp_candidates."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    cwarp_data = (snapshot.get("cwarp") or {}).get("candidates") or {}
    if not cwarp_data:
        logger.info("[cwarp_mirror] Nessun CWARP da scrivere — skip")
        return
    data_oggi = snapshot.get("meta", {}).get("date")
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        from datetime import date, timedelta
        target = (date.today() - timedelta(days=30)).isoformat()
        hist = (
            client.table("ARTEMIS_SIGNALS")
            .select("cwarp_candidates")
            .lte("date", target).order("date", desc=True).limit(1).execute()
        )
        hist_cwarp = {}
        if hist.data and hist.data[0].get("cwarp_candidates"):
            hc = hist.data[0]["cwarp_candidates"]
            if isinstance(hc, dict):
                hist_cwarp = hc
        aggiornati = 0
        for ticker, cdata in cwarp_data.items():
            cwarp_now = cdata.get("cwarp")
            if cwarp_now is None:
                continue
            cwarp_30d = (hist_cwarp.get(ticker) or {}).get("cwarp")
            trend = _classify_cwarp_trend(cwarp_now, cwarp_30d)
            client.table("ARTEMIS_CWARP_CANDIDATES").update({
                "cwarp_current":      cwarp_now,
                "cwarp_30d_ago":      cwarp_30d,
                "cwarp_trend":        trend,
                "cwarp_last_updated": data_oggi,
                "updated_at":         "now()",
            }).eq("ticker", ticker).execute()
            aggiornati += 1
        logger.info(f"[cwarp_mirror] {aggiornati} candidati aggiornati con CWARP+trend")
    except Exception as exc:
        logger.warning(f"[cwarp_mirror] Aggiornamento fallito: {exc}")


# Caricata dinamicamente da Supabase in main(). Per aggiungere/pausare asset:
# usa il Supabase Table Editor su ARTEMIS_CWARP_CANDIDATES, NON questo file.
CWARP_CANDIDATES: list[dict] = []  # fallback statico (solo se Supabase down)


# ─────────────────────────────────────────────────────────────────────────────
# ALLOCAZIONI TARGET DRAGON
# ─────────────────────────────────────────────────────────────────────────────
# DRAGON_TARGET_COLE — allocazione originale Cole (1928-2019). TENUTA COME
#   RIFERIMENTO: non è più il target operativo, ma lo scanner ne calcola comunque
#   drift e Dragon Score in parallelo (result["comparison_cole"]) per affiancare
#   gli andamenti Growth (attivo) vs Cole puro nel tempo.
DRAGON_TARGET_COLE = {
    "equity":          0.24,   # US Equity
    "fixed_income":    0.18,   # US Treasury Bonds
    "long_volatility": 0.21,   # Active Long Volatility
    "commodity_trend": 0.18,   # CTA / Commodity Trend Following
    "gold":            0.19,   # Physical Gold / Fiat Alternatives
}
# DRAGON_TARGET_GROWTH — "Profilo 2" (Dragon Growth). Scelto da Davide il
#   2026-09-24 dopo backtest 2000-2026 (artemis_strategy_backtest_v3).
#   Razionale: la long vol NON è implementabile bene col retail (il VIXM
#   sanguina); la protezione è affidata a CTA + oro (nel backtest il CTA da
#   solo ha battuto la long vol permanente). Cash/bond breve 7% = zavorra
#   intenzionale. Fixed income 10% = sleeve bond lungo (assicurazione
#   deflazione). Equity portata al 50% (scelta Davide 2026-09-24: punto dolce
#   risk-adjusted, riduce il lag nei tori). Somma 5 componenti = 0.93; residuo 7% = cash.
DRAGON_TARGET_GROWTH = {
    "equity":          0.50,   # Azionario globale (motore di crescita)
    "fixed_income":    0.10,   # Bond lungo (assicurazione deflazione)
    "long_volatility": 0.00,   # DISATTIVA — protezione via CTA + oro
    "commodity_trend": 0.15,   # CTA / trend following (crisi-alpha)
    "gold":            0.18,   # Oro (hedge debasement)
}
# TARGET OPERATIVO ATTIVO.
# Dal 2026-09-24 (punto 2) la FONTE PRIMARIA è la tabella Supabase
# ARTEMIS_TARGET_ALLOCATION (righe con attivo = true): per cambiare profilo si
# modifica la tabella, NON questo file. I pesi qui sotto sono solo la RISERVA,
# usata se la tabella non risponde o contiene pesi non validi (con avviso
# esplicito nell'output: nessun fallback silenzioso).
DRAGON_TARGET = DRAGON_TARGET_GROWTH
ACTIVE_PROFILE_NAME = "Dragon Growth (Profilo 2)"
DRAGON_TARGET_SOURCE  = "CODICE (riserva — tabella non ancora letta)"
DRAGON_TARGET_WARNING = None
_PROFILE_DISPLAY_NAMES = {
    "GROWTH_P2": "Dragon Growth (Profilo 2)",
    "COLE":      "Cole puro (24/18/21/18/19)",
}


def load_active_target_from_supabase() -> None:
    """Carica il target attivo da ARTEMIS_TARGET_ALLOCATION (attivo = true).

    Imposta i globali DRAGON_TARGET, ACTIVE_PROFILE_NAME, DRAGON_TARGET_SOURCE
    e DRAGON_TARGET_WARNING. Il riferimento Cole resta nel codice
    (DRAGON_TARGET_COLE), per scelta di Davide.

    Validazioni (se una fallisce → riserva dal codice + avviso esplicito):
      · esattamente UN profilo attivo;
      · presenti tutte e 5 le gambe Dragon;
      · ogni peso tra 0 e 1;
      · somma 5 gambe ≤ 100%; se c'è la riga 'cash', 5 gambe + cash = 100% (±1%).
    """
    global DRAGON_TARGET, ACTIVE_PROFILE_NAME, DRAGON_TARGET_SOURCE, DRAGON_TARGET_WARNING

    def _fallback(motivo: str) -> None:
        global DRAGON_TARGET, ACTIVE_PROFILE_NAME, DRAGON_TARGET_SOURCE, DRAGON_TARGET_WARNING
        DRAGON_TARGET = DRAGON_TARGET_GROWTH
        ACTIVE_PROFILE_NAME = "Dragon Growth (Profilo 2)"
        DRAGON_TARGET_SOURCE = "CODICE (riserva)"
        DRAGON_TARGET_WARNING = (
            f"⚠️  TARGET DA RISERVA NEL CODICE: {motivo}. Verificare la tabella "
            f"ARTEMIS_TARGET_ALLOCATION — lo scanner sta usando i pesi scritti nel file."
        )
        logger.warning(f"[target] {DRAGON_TARGET_WARNING}")

    if not SUPABASE_URL or not SUPABASE_KEY:
        _fallback("credenziali Supabase non configurate")
        return
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        rows = (client.table("ARTEMIS_TARGET_ALLOCATION")
                .select("profilo, componente, peso_target, attivo")
                .eq("attivo", True)
                .execute()).data or []
    except Exception as e:
        _fallback(f"tabella non leggibile ({e})")
        return

    if not rows:
        _fallback("nessuna riga con attivo = true")
        return

    profili = sorted({r.get("profilo") for r in rows})
    if len(profili) != 1:
        _fallback(f"profili attivi multipli o assenti: {profili}")
        return
    profilo = profili[0]

    pesi, cash = {}, None
    try:
        for r in rows:
            comp = (r.get("componente") or "").strip()
            w = float(r.get("peso_target"))
            if not (0.0 <= w <= 1.0):
                _fallback(f"peso fuori range per {comp}: {w}")
                return
            if comp == "cash":
                cash = w
            elif comp in DRAGON_TARGET_GROWTH:
                pesi[comp] = w
            else:
                logger.info(f"[target] componente ignorata (non Dragon): {comp}")
    except (TypeError, ValueError) as e:
        _fallback(f"peso non numerico ({e})")
        return

    mancanti = [c for c in DRAGON_TARGET_GROWTH if c not in pesi]
    if mancanti:
        _fallback(f"gambe mancanti nel profilo {profilo}: {mancanti}")
        return
    somma = sum(pesi.values())
    if somma > 1.0 + 1e-6:
        _fallback(f"somma 5 gambe = {somma:.2%} (> 100%)")
        return
    if cash is not None and abs(somma + cash - 1.0) > 0.01:
        _fallback(f"5 gambe {somma:.2%} + cash {cash:.2%} ≠ 100%")
        return

    # Ordine delle chiavi identico a quello del codice (stabilità output/log)
    DRAGON_TARGET = {c: round(pesi[c], 4) for c in DRAGON_TARGET_GROWTH}
    ACTIVE_PROFILE_NAME = _PROFILE_DISPLAY_NAMES.get(profilo, profilo)
    DRAGON_TARGET_SOURCE = f"SUPABASE ({profilo})"
    DRAGON_TARGET_WARNING = None
    logger.info(f"[target] ✅ Target da Supabase ({profilo}): "
                + ", ".join(f"{c}={w:.0%}" for c, w in DRAGON_TARGET.items())
                + f" · cash={max(0.0, 1.0 - somma):.0%}")
    if any(abs(DRAGON_TARGET[c] - DRAGON_TARGET_GROWTH[c]) > 1e-6 for c in DRAGON_TARGET):
        logger.info("[target] ℹ️  Il target in Supabase differisce dalla riserva nel "
                    "codice: normale se hai cambiato profilo dalla tabella.")
DRAGON_DRIFT_ALERT      = 0.05    # drift > 5% per componente → rebalancing warning
DRAGON_DRIFT_CRITICAL   = 0.10    # drift > 10% → rebalancing urgente

# SOCMINT watchlist (asset ad alta asimmetria potenziale)
SOCMINT_WATCHLIST = [
    "GME", "AMC", "MSTR", "TSLA", "NVDA",
    "COIN", "HOOD", "RIVN", "PLTR", "SMCI",
]
SHORT_INTEREST_ALERT    = 20.0    # SI > 20% → squeeze candidato
VOLUME_SPIKE_FACTOR     = 5.0     # Volume > 5× media → anomalia


# ── SEZIONE 3: HANDLER ERRORI ───────────────────────────────────────────────

def _fmt(val, decimals: int = 2) -> str:
    """Formatta valore numerico per il report console. None → 'N/A'."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


class ArtemisHardStop(Exception):
    pass


def _safe_yf(ticker: str, period: str = "5d",
             interval: str = "1d") -> float | None:
    """Scarica l'ultimo Close via yfinance. None se fallisce."""
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if not close.empty:
            _note_obs("YAHOO", ticker, close)
            return round(float(close.iloc[-1]), 4)
        return None
    except Exception as e:
        logger.warning(f"[yf] {ticker}: {e}")
        return None


def _safe_fred(series_id: str, fred: Fred,
               lookback_days: int = 180) -> float | None:
    """Recupera l'ultimo valore da FRED. None se fallisce.
    Include throttle anti rate-limit (FRED: max ~120 req/min)."""
    try:
        time.sleep(0.6)  # throttle: ~100 req/min, sotto il limite FRED
        start = datetime.today() - timedelta(days=lookback_days)
        s = _fred_with_retry(series_id, fred, observation_start=start)
        if s is not None and not s.empty:
            return float(s.dropna().iloc[-1])
        return None
    except Exception as e:
        logger.warning(f"[fred] {series_id}: {e}")
        return None


def _fred_with_retry(series_id: str, fred, attempts: int = 4,
                     backoff: float = 3.0, **kwargs):
    """FRED get_series con retry su 500 Internal Server Error E 429 Rate Limit.
    Backoff esponenziale: 429 (Too Many Requests) richiede attese più lunghe."""
    last_err = None
    for i in range(attempts):
        try:
            _res = fred.get_series(series_id, **kwargs)
            _note_obs("FRED", series_id, _res)
            return _res
        except Exception as e:
            last_err = e
            msg = str(e)
            # 429 Rate Limit → backoff esponenziale lungo (3, 6, 12, 24s)
            if "429" in msg or "Too Many Requests" in msg or "Exceeded Rate Limit" in msg:
                wait = backoff * (2 ** i)
                logger.info(f"[fred_retry] {series_id} rate-limited, attendo {wait:.0f}s (tentativo {i+1}/{attempts})")
                time.sleep(wait)
                continue
            # 500 / temporanei → backoff lineare
            if "500" in msg or "Temporarily" in msg:
                time.sleep(backoff * (i + 1))
                continue
            break
    logger.warning(f"[fred_retry] {series_id} failed: {last_err}")
    return None


# ════════════════════════════════════════════════════════════════════════════
# DATA AUDIT AUTOMATICO (audit 2026-09-25)
# Ogni serie scaricata (FRED, Yahoo) e ogni fonte esterna registra QUI la data
# della sua ultima osservazione. A fine run build_data_audit() assegna a ogni
# dato uno stato: OK / VECCHIO / FERMO / MANCANTE / STIMA / NON_DISPONIBILE.
# Nessun dato vecchio o stimato può più passare per "dato di oggi" in silenzio.
# ════════════════════════════════════════════════════════════════════════════
DATA_OBS: dict[str, dict] = {}

# Ritardo massimo accettabile (giorni tra data osservazione e oggi) per classe
# di frequenza. La data FRED di un dato mensile/trimestrale è l'INIZIO del
# periodo, quindi i limiti includono il normale ritardo di pubblicazione.
_AUDIT_MAX_AGE = {"giornaliera": 10, "settimanale": 17, "mensile": 100,
                  "trimestrale": 290, "annuale": 550, "sconosciuta": 45}
# Serie a ritardo strutturale noto (non è un guasto): limite personalizzato.
_AUDIT_AGE_OVERRIDE = {
    "HDTGPDUSQ163N": 560,   # IMF Financial Soundness Indicators: ~15 mesi di ritardo
    "TCMDO": 300,           # Z.1 totale: aggiornato a volte con un trimestre di ritardo
    "CSUSHPISA": 135,       # Case-Shiller: esce l'ultimo martedì del mese per 2 mesi prima
    "CCSA": 21,             # sussidi continuativi: settimana chiusa il sabato, pubblicata il giovedì
                            # 12 giorni dopo → il mercoledì prima del nuovo dato ha 18 giorni (01/10/2026)
}


def _freq_class(spacing_days: float | None) -> str:
    if spacing_days is None:
        return "sconosciuta"
    if spacing_days <= 4:
        return "giornaliera"
    if spacing_days <= 10:
        return "settimanale"
    if spacing_days <= 45:
        return "mensile"
    if spacing_days <= 120:
        return "trimestrale"
    return "annuale"


def _note_obs(source: str, series_id: str, s) -> None:
    """Registra data ultima osservazione + frequenza di una serie scaricata."""
    try:
        if s is None:
            return
        s2 = s.dropna() if hasattr(s, "dropna") else None
        if s2 is None or len(s2) == 0:
            return
        idx = pd.to_datetime(s2.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        spacing = None
        if len(idx) >= 2:
            spacing = float(pd.Series(idx).diff().dt.days.dropna().median())
        prev = DATA_OBS.get(f"{source}:{series_id}")
        info = {"source": source, "series": series_id,
                "obs_date": idx[-1].strftime("%Y-%m-%d"),
                "first_date": idx[0].strftime("%Y-%m-%d"),
                "freq": _freq_class(spacing)}
        # Stessa serie scaricata più volte: tieni la frequenza più informativa
        if prev and prev.get("freq") != "sconosciuta" and info["freq"] == "sconosciuta":
            info["freq"] = prev["freq"]
        DATA_OBS[f"{source}:{series_id}"] = info
    except Exception:
        pass


def _note_ext(name: str, source: str, obs_date: str | None, status: str | None = None,
              note: str | None = None, max_age_days: int | None = None) -> None:
    """Registra una fonte esterna (non FRED/Yahoo) nel registro audit."""
    DATA_OBS[f"EXT:{name}"] = {"source": source, "series": name, "obs_date": obs_date,
                               "freq": "esterna", "status_forzato": status,
                               "note": note, "max_age": max_age_days}


def _yoy_by_date(s, months: int = 12, tol_days: int = 20):
    """Variazione % anno su anno confrontando per DATA (non per posizione).
    Evita l'errore 'iloc[-4] su serie trimestrale = 3 trimestri, non 1 anno'."""
    s = s.dropna()
    if len(s) < 2:
        return None
    last_d = pd.Timestamp(s.index[-1])
    target = last_d - pd.DateOffset(months=months)
    diffs = abs(pd.to_datetime(s.index) - target)
    i = int(diffs.argmin())
    if diffs[i].days > tol_days:
        return None
    base = float(s.iloc[i])
    if base == 0:
        return None
    return (float(s.iloc[-1]) / base - 1) * 100


def _yoy_or_nan(s) -> float:
    """YoY % per DATA (stesso mese dell'anno prima). NaN se il mese base manca.
    AUDIT 2026-09-25 (ter): la versione posizionale iloc[-13] saltava il buco di
    ottobre 2025 (CPI non pubblicato per lo shutdown) → misurava 13 mesi:
    CPI 3.71% contro il 3.4% ufficiale BLS di agosto 2026."""
    v = _yoy_by_date(s, tol_days=5)
    return float("nan") if v is None else v


def _fred_series_throttled(fred, series_id: str, **kwargs):
    """Wrapper per fred.get_series con throttle + retry 429/500.
    Sostituisce le chiamate dirette fred.get_series() che altrimenti
    bypassano la protezione rate-limit."""
    time.sleep(0.6)  # throttle anti rate-limit
    return _fred_with_retry(series_id, fred, **kwargs)


def _stooq_download(ticker: str, period_days: int = 400):
    """Fallback Stooq per yfinance rate-limit."""
    import io as _io
    mapping = {
        "^VIX": "%5Evix", "^VVIX": "%5Evvix", "^SKEW": "%5Eskew",
        "^MOVE": "%5Emove", "^OVX": "%5Eovx", "^GVZ": "%5Egvz",
        "^VXN": "%5Evxn", "^VIX3M": "%5Evix3m",
        "EURUSD=X": "eurusd", "USDJPY=X": "usdjpy", "DX-Y.NYB": "%5Edxy",
        "GC=F": "xauusd", "HG=F": "%5Ecopper", "SI=F": "xagusd",
        "CL=F": "cl.f", "SPY": "spy.us", "QQQ": "qqq.us",
        "GLD": "gld.us", "TLT": "tlt.us",
    }
    sym = mapping.get(ticker)
    if sym is None:
        if ticker.isalpha():
            sym = ticker.lower() + ".us"
        else:
            return pd.DataFrame()
    try:
        r = requests.get(f"https://stooq.com/q/d/l/?s={sym}&i=d",
                         timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text.lstrip().startswith("Date"):
            return pd.DataFrame()
        import io as _io2
        df = pd.read_csv(_io2.StringIO(r.text))
        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        if period_days and len(df) > period_days:
            df = df.iloc[-period_days:]
        return df
    except Exception as e:
        logger.warning(f"[stooq] {ticker}: {e}")
        return pd.DataFrame()


def _compute_market_breadth_200ma() -> tuple[float | None, str]:
    """
    Ampiezza di mercato reale: % di titoli S&P 500 sopra la propria MA200.

    FIX 2026-07 — sostituisce il vecchio metodo (persistenza di SPY sopra
    la SUA media, non vera ampiezza cross-sezionale — vedi audit sessione
    2026-07-13). Due livelli:

    LIVELLO 1 (ex "Livello 2") — calcolo diretto: lista costituenti S&P 500
        da Wikipedia (fonte pubblica, stabile ma non garantita) + download
        batch storico via yfinance + verifica individuale sopra/sotto la
        propria MA200. Pesante (~500 titoli) ma corretto. Nessun limite di
        tempo stretto: lo scanner gira 1 volta/giorno via GitHub Actions.
    LIVELLO 2 (fallback finale) — vecchio metodo (persistenza SPY sopra la
        propria MA200). Impreciso come proxy di ampiezza ma sempre disponibile,
        meglio di un campo vuoto.

    NOTA 2026-07-13: un originario "Livello 1" tentava ticker candidati per
    un indice ufficiale di ampiezza (S5TH, SPXA200R, S5FI) — rimosso dopo
    verifica in produzione: tutti e 4 i simboli confermati inesistenti su
    Yahoo Finance ("possibly delisted; no price data found"). Rimosso per
    pulizia invece di lasciare tentativi destinati a fallire per sempre.

    Restituisce: (valore_percentuale | None, "livello1"/"livello2"/"fallito")
    """
    # ── LIVELLO 1 — calcolo diretto su costituenti S&P 500 ───────────────────
    try:
        _wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # FIX 2026-07-13: pd.read_html(url) fa una richiesta HTTP interna
        # SENZA User-Agent identificabile — Wikipedia risponde 403 Forbidden
        # (verificato in produzione). Fetch manuale con header, poi parsing
        # del contenuto già scaricato.
        _wiki_resp = requests.get(_wiki_url, timeout=20,
                                  headers={"User-Agent": "Mozilla/5.0"})
        _wiki_resp.raise_for_status()
        import io as _io_wiki
        _tables = pd.read_html(_io_wiki.StringIO(_wiki_resp.text))
        _tickers_raw = _tables[0]["Symbol"].tolist()
        # Yahoo Finance usa '-' al posto di '.' per classi azionarie (es. BRK.B -> BRK-B)
        _tickers = [str(t).replace(".", "-").strip() for t in _tickers_raw if t]

        if len(_tickers) < 400:   # sanity check: la lista deve essere plausibile
            raise ValueError(f"Lista costituenti sospetta: solo {len(_tickers)} ticker")

        logger.info(f"[breadth] Livello 2: download batch {len(_tickers)} titoli S&P 500...")
        _batch = yf.download(_tickers, period="1y", progress=False,
                             auto_adjust=True, group_by="ticker", threads=True)

        _sopra = 0
        _validi = 0
        for _tk in _tickers:
            try:
                _closes = _batch[_tk]["Close"].dropna()
                if len(_closes) >= 200:
                    _ma200 = _closes.rolling(200).mean().iloc[-1]
                    _last = _closes.iloc[-1]
                    if pd.notna(_ma200) and pd.notna(_last):
                        _validi += 1
                        if _last > _ma200:
                            _sopra += 1
            except Exception:
                continue   # titolo singolo mancante/delistato — si salta, non blocca il resto

        # Serve una copertura minima ragionevole per fidarsi del risultato
        if _validi >= 300:
            _pct = round((_sopra / _validi) * 100, 1)
            logger.info(f"[breadth] Livello 2 riuscito: {_sopra}/{_validi} titoli "
                        f"validi sopra MA200 ({_pct}%)")
            return _pct, f"livello1 ({_validi} titoli)"
        else:
            logger.warning(f"[breadth] Livello 2: copertura insufficiente "
                           f"({_validi} titoli validi su {len(_tickers)}) — fallback")
    except Exception as e:
        logger.warning(f"[breadth] Livello 2 fallito: {e}")

    # ── LIVELLO 2 — fallback finale: vecchio metodo (persistenza SPY) ───────
    try:
        _spy = yf.download("SPY", period="1y", interval="1d",
                           progress=False, auto_adjust=True)
        if isinstance(_spy.columns, pd.MultiIndex):
            _close = _spy["Close"].iloc[:, 0] if _spy["Close"].ndim > 1 else _spy["Close"]
        else:
            _close = _spy["Close"]
        _close = _close.squeeze().dropna()
        if len(_close) >= 200:
            _ma200 = _close.rolling(200).mean()
            _above = (_close > _ma200).iloc[-20:]
            return round(float(_above.mean() * 100), 1), "livello2 (fallback SPY)"
    except Exception as e:
        logger.warning(f"[breadth] Livello 3 (fallback) fallito: {e}")

    return None, "fallito"


def _robust_yf(ticker: str, period: str = "1y") -> pd.DataFrame:
    """yfinance con fallback Stooq se rate-limited.

    RIPRISTINATA 2026-07-13: cancellata per errore durante l'inserimento
    di _compute_market_breadth_200ma sopra (str_replace ha sostituito
    l'intero blocco invece di solo anteporre codice). Corpo identico
    all'originale, verificato dal diff prima di questo fix.
    """
    try:
        df = yf.download(ticker, period=period, progress=False,
                         auto_adjust=True, multi_level_index=False)
        if df is not None and not df.empty and "Close" in df.columns:
            _note_obs("YAHOO", ticker, df["Close"])
            return df
    except Exception as e:
        logger.info(f"[robust_yf] {ticker}: yf fail ({e}), trying Stooq")
    period_days_map = {"1y": 400, "2y": 800, "6mo": 200, "3mo": 100, "5y": 1800}
    df2 = _stooq_download(ticker, period_days=period_days_map.get(period, 400))
    if df2 is not None and not df2.empty and "Close" in df2.columns:
        _note_obs("STOOQ", ticker, df2["Close"])
    return df2 if df2 is not None else pd.DataFrame()




# ────────────────────────────────────────────────────────────────────────────
# 4.1 VOLATILITY STACK
# ────────────────────────────────────────────────────────────────────────────

def collect_volatility_stack() -> dict:
    """
    Stack volatilità:
    VIX · VVIX · SKEW · MOVE · OVX · GVZ · VXN · VIX3M · WTI
    + derivati: VIX term slope · SKEW/VIX · MOVE/VIX · OVX/WTI
    """
    sig = {}
    tickers = {
        "vix":   "^VIX",
        "vvix":  "^VVIX",
        "skew":  "^SKEW",
        "ovx":   "^OVX",
        "gvz":   "^GVZ",
        "vxn":   "^VXN",
        "move":  "^MOVE",
        "vix3m": "^VIX3M",
        "wti":   "CL=F",
    }
    for key, ticker in tickers.items():
        sig[key] = _safe_yf(ticker)
        time.sleep(0.3)

    # WTI sanity check $20-$300 + fallback Stooq + EIA
    if sig.get("wti") is not None and not (20.0 <= sig["wti"] <= 300.0):
        logger.warning(f"[vol] WTI={sig['wti']} fuori range -> scartato")
        sig["wti"] = None
    if sig.get("wti") is None:
        try:
            _st = _stooq_download("CL=F", period_days=10)
            if not _st.empty and "Close" in _st.columns:
                _v = float(_st["Close"].dropna().iloc[-1])
                if 20.0 <= _v <= 300.0:
                    sig["wti"] = round(_v, 2)
                    logger.info(f"[vol] WTI={sig['wti']} via Stooq")
        except Exception as e:
            logger.warning(f"[vol] WTI Stooq fallback: {e}")
    if sig.get("wti") is None and EIA_KEY:
        try:
            _url = (f"https://api.eia.gov/v2/petroleum/pri/spt/data/"
                    f"?api_key={EIA_KEY}&frequency=weekly&data[0]=value"
                    f"&facets[product][]=EPCWTI&sort[0][column]=period"
                    f"&sort[0][direction]=desc&length=1")
            _r = requests.get(_url, timeout=15)
            if _r.status_code == 200:
                _d = _r.json().get("response", {}).get("data", [])
                if _d:
                    _v = float(_d[0].get("value", 0))
                    if 20.0 <= _v <= 300.0:
                        sig["wti"] = round(_v, 2)
                        logger.info(f"[vol] WTI={sig['wti']} via EIA")
        except Exception as e:
            logger.warning(f"[vol] WTI EIA fallback: {e}")

    # GVZ sanity check (5-100) + FRED fallback
    if sig.get("gvz") is not None and not (5.0 <= sig["gvz"] <= 100.0):
        logger.warning(f"[vol] GVZ={sig['gvz']} fuori range -> scartato")
        sig["gvz"] = None
    if sig.get("gvz") is None and FRED_KEY:
        try:
            _fb = Fred(api_key=FRED_KEY)
            _gvz = _safe_fred("GVZCLS", _fb, lookback_days=30)
            if _gvz is not None and 5.0 <= _gvz <= 100.0:
                sig["gvz"] = round(_gvz, 2)
                logger.info(f"[vol] GVZ={sig['gvz']} via FRED GVZCLS")
        except Exception as e:
            logger.warning(f"[vol] GVZ FRED fallback: {e}")

    # VVIX sanity check (60-200) + CBOE CDN fallback
    if sig.get("vvix") is not None and not (60.0 <= sig["vvix"] <= 200.0):
        logger.warning(f"[vol] VVIX={sig['vvix']} fuori range -> scartato")
        sig["vvix"] = None
    if sig.get("vvix") is None:
        try:
            _r = requests.get(
                "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX.json",
                timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if _r.status_code == 200:
                _data = _r.json().get("data", [])
                for _row in reversed(_data):
                    # FIX C2: formato array CBOE CDN = [date, open, high, low, close, ...]
                    # Indice 1 = Open (sbagliato — distorto da gap-up intraday)
                    # Indice 4 = Close (corretto per prezzi di volatilità Artemis)
                    # Guard: richiede almeno 5 elementi per garantire l'accesso a [4]
                    if not isinstance(_row, (list, tuple)) or len(_row) < 5:
                        continue
                    _val = _row[4]   # Close price
                    if _val and 60.0 <= float(_val) <= 200.0:
                        sig["vvix"] = round(float(_val), 2)
                        logger.info(f"[vol] VVIX={sig['vvix']} via CBOE CDN (Close)")
                        break
        except Exception as e:
            logger.warning(f"[vol] VVIX CBOE CDN: {e}")

    # VIX sanity check (8-90)
    if sig.get("vix") is not None and not (8.0 <= sig["vix"] <= 90.0):
        logger.warning(f"[vol] VIX={sig['vix']} fuori range -> scartato")
        sig["vix"] = None

    # VIX fallback multi-sorgente
    if sig.get("vix") is None and FRED_KEY:
        try:
            _fb = Fred(api_key=FRED_KEY)
            _v = _safe_fred("VIXCLS", _fb)
            if _v is not None:
                sig["vix"] = round(float(_v), 2)
                logger.info(f"[vol] VIX={sig['vix']} via FRED fallback")
        except Exception as e:
            logger.warning(f"[vol] FRED VIX fallback: {e}")
    if sig.get("vix") is None:
        try:
            import io as _io
            _r = requests.get("https://stooq.com/q/d/l/?s=%5Evix&i=d",
                              timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if _r.status_code == 200 and _r.text.lstrip().startswith("Date"):
                _df = pd.read_csv(_io.StringIO(_r.text))
                if not _df.empty and "Close" in _df.columns:
                    sig["vix"] = round(float(_df["Close"].dropna().iloc[-1]), 2)
                    logger.info(f"[vol] VIX={sig['vix']} via Stooq fallback")
        except Exception as e:
            logger.warning(f"[vol] Stooq VIX fallback: {e}")

    # VIX Term Structure slope
    if sig.get("vix") and sig.get("vix3m") and sig["vix3m"] != 0:
        sig["vix_term_slope"] = round((sig["vix3m"] - sig["vix"]) / sig["vix3m"], 4)
    else:
        sig["vix_term_slope"] = None

    # Segnali derivati
    sig["skew_vix_ratio"] = (round(sig["skew"] / sig["vix"], 2)
                             if sig.get("skew") and sig.get("vix") and sig["vix"] > 0
                             else None)
    sig["move_vix_ratio"] = (round(sig["move"] / sig["vix"], 2)
                             if sig.get("move") and sig.get("vix") and sig["vix"] > 0
                             else None)
    sig["ovx_wti_ratio"]  = (round(sig["ovx"] / sig["wti"], 4)
                             if sig.get("ovx") and sig.get("wti") and sig["wti"] > 0
                             else None)

    logger.info(f"[vol] VIX={sig.get('vix')} SKEW={sig.get('skew')} "
                f"MOVE={sig.get('move')} slope={sig.get('vix_term_slope')} "
                f"WTI={sig.get('wti')} OVX={sig.get('ovx')} GVZ={sig.get('gvz')}")
    return sig


# ────────────────────────────────────────────────────────────────────────────
# 4.2 RATES / CREDIT STACK
# ────────────────────────────────────────────────────────────────────────────

def collect_rates_credit_stack(fred: Fred) -> dict:
    """
    Stack tassi e credito:
    T10Y2Y · T10Y3M · T5YIE · SOFR · CPI YoY · PCE · NFP · LEI
    HY/IG OAS spreads · Fed Balance Sheet · Net Liquidity · Margin Debt
    % S&P500 sopra MA200
    """
    sig = {}

    fred_series = {
        "t10y2y":  "T10Y2Y",
        "t10y3m":  "T10Y3M",
        "t5yie":   "T5YIE",
        "sofr":    "SOFR",
        # "lei": USSLIND rimosso (audit 2026-09-25): dismesso dalla Philadelphia
        # Fed nel 2020 → ora = CLI OECD USA, calcolato sotto.
        "fed_bs":  "WALCL",
        "tga":     "WTREGEN",
        "rrp":     "RRPONTSYD",
        # ATTENZIONE: da aprile 2026 FRED pubblica solo 3 anni di storia ICE BofA.
        # Qui si usa solo il LIVELLO di oggi (corretto); gli z-score del motore
        # usano BAA10Y e STLFSI4 che hanno storia completa.
        "hy_oas":  "BAMLH0A0HYM2",
        "ig_oas":  "BAMLC0A0CM",
        "baa10y_spread": "BAA10Y",   # Moody's Baa − Treasury 10y (dal 1986)
        "stlfsi4":       "STLFSI4",  # St. Louis Fed Financial Stress Index (media 0)
        # ── Nuovi segnali (audit conferma regime/transizioni, fonte FRED gratis) ──
        "real_yield_10y": "DFII10",   # TIPS 10y reale — driver primario dell'oro (FENICE)
        "infl_5y5y":      "T5YIFR",   # aspettative inflazione 5y5y forward — anticipa FALCO_DX
        "continued_claims": "CCSA",   # continued jobless claims — con ICSA anticipa FALCO_SX
    }
    for key, series_id in fred_series.items():
        sig[key] = _safe_fred(series_id, fred)
    # AUDIT 2026-09-25 (verifica finale): nella run delle 16:07 T10Y2Y e T5YIE
    # sono tornati vuoti per un errore momentaneo di FRED, mentre su FRED i
    # dati c'erano (0.31 e 2.33). Seconda passata dopo una pausa per i vuoti.
    _vuoti = [k for k, v in sig.items() if v is None and k in fred_series]
    if _vuoti:
        logger.warning(f"[rates] FRED vuoti al primo tentativo: {_vuoti} — riprovo tra 8s")
        time.sleep(8)
        for key in _vuoti:
            sig[key] = _safe_fred(fred_series[key], fred)

    # LEI USA = CLI OECD USA (stessa fonte del regime engine; una sola chiamata
    # OECD per run grazie alla cache interna del motore)
    sig["lei"] = None
    sig["lei_source"] = None
    try:
        _cli_us = _fetch_oecd_cli("USA")
        if _cli_us is not None and len(_cli_us) > 0:
            sig["lei"] = round(float(_cli_us.iloc[-1]), 3)
            sig["lei_source"] = f"OECD_CLI_USA {_cli_us.index[-1]:%Y-%m}"
            _note_ext("lei", "OECD CLI USA", f"{_cli_us.index[-1]:%Y-%m-01}", max_age_days=130)
    except Exception as e:
        logger.warning(f"[rates] LEI (CLI OECD USA): {e}")

    # SOFR sanity check: range 0.5%-6%. Fallback DFF se fuori range o None
    if sig.get("sofr") is not None and not (0.5 <= sig["sofr"] <= 6.0):
        logger.warning(f"[rates] SOFR={sig['sofr']} fuori range → fallback DFF")
        sig["sofr"] = None
    if sig.get("sofr") is None:
        try:
            dff = _safe_fred("DFF", fred, lookback_days=30)
            if dff is not None and 0.5 <= dff <= 6.0:
                sig["sofr"] = round(dff, 4)
                logger.info(f"[rates] SOFR fallback DFF={sig['sofr']}")
        except Exception as e:
            logger.warning(f"[rates] DFF fallback: {e}")

    # NFP — delta mensile jobs aggiunti (PAYEMS in migliaia × 1000)
    sig["nfp"] = None
    try:
        payems = _fred_series_throttled(fred, "PAYEMS",
            observation_start=datetime.today() - timedelta(days=120)).dropna()
        if len(payems) >= 2:
            delta = (float(payems.iloc[-1]) - float(payems.iloc[-2])) * 1000
            sig["nfp"] = round(delta, 0) if -2_000_000 <= delta <= 2_000_000 else None
            logger.info(f"[rates] NFP delta mensile: {sig['nfp']:,.0f} jobs")
    except Exception as e:
        logger.warning(f"[rates] NFP delta: {e}")

    # CPI YoY
    try:
        time.sleep(0.6)
        cpi_hist = _fred_with_retry("CPIAUCSL", fred,
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(cpi_hist) >= 14:
            raw = _yoy_or_nan(cpi_hist)
            sig["cpi_yoy"] = round(raw, 2) if -5.0 <= raw <= 30.0 else None
        else:
            sig["cpi_yoy"] = None
    except Exception as e:
        logger.warning(f"[rates] CPI YoY: {e}")
        sig["cpi_yoy"] = None

    # PCE YoY — da PCEPI con sanity check vs CPI (diff storica ~0.35pp)
    # FIX C4: rimosso l'override forzato "pce = cpi - 0.35" quando diff > 2pp.
    # Quella logica mascherava divergenze strutturali reali (es. shock da sussidi,
    # deflazione settoriale) che il Regime Engine DEVE poter analizzare.
    # Il dato grezzo passa inalterato. Un log di warning segnala la divergenza.
    try:
        pce_hist = _fred_series_throttled(fred, "PCEPI",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(pce_hist) >= 14:
            raw_pce = _yoy_or_nan(pce_hist)
            if -5.0 <= raw_pce <= 20.0:
                sig["pce_yoy"] = round(raw_pce, 2)
                # Log divergenza strutturale senza sovrascrivere
                if sig.get("cpi_yoy") is not None:
                    diff = abs(sig["pce_yoy"] - sig["cpi_yoy"])
                    if diff > 2.0:
                        logger.warning(f"[rates] PCE divergenza strutturale {diff:.1f}pp da CPI "
                                       f"(PCE={sig['pce_yoy']}% CPI={sig['cpi_yoy']}%) — "
                                       f"dato grezzo mantenuto per analisi Regime Engine")
            else:
                sig["pce_yoy"] = None
        else:
            sig["pce_yoy"] = None
    except Exception as e:
        logger.warning(f"[rates] PCE YoY: {e}")
        sig["pce_yoy"] = None   # nessun proxy: meglio None che un valore inventato

    # Net Liquidity Fed = Fed BS - TGA - RRP
    # FIX F5 — Unità: WALCL e WTREGEN sono in MILIONI di $,
    # RRPONTSYD è in MILIARDI di $ → moltiplica × 1000 per allineare.
    # Errore attuale (RRP ~50B): ~0.8%. Era ~49% nel 2023 (RRP a 2,000B).
    if sig.get("fed_bs") and sig.get("tga") and sig.get("rrp"):
        rrp_millions = sig["rrp"] * 1000   # B$ → M$
        sig["net_liquidity"] = round(sig["fed_bs"] - sig["tga"] - rrp_millions, 2)
    else:
        sig["net_liquidity"] = None

    # Net Liquidity delta 4 settimane
    try:
        walcl = _fred_series_throttled(fred, "WALCL",
            observation_start=datetime.today() - timedelta(days=60)).dropna()
        tga_s = _fred_series_throttled(fred, "WTREGEN",
            observation_start=datetime.today() - timedelta(days=60)).dropna()
        rrp_s = _fred_series_throttled(fred, "RRPONTSYD",
            observation_start=datetime.today() - timedelta(days=60)).dropna()
        if len(walcl) >= 5 and len(tga_s) >= 5 and len(rrp_s) >= 5:
            # FIX F5: rrp_s in B$, walcl/tga_s in M$ → converti rrp × 1000
            # AUDIT 2026-09-25: RRPONTSYD è GIORNALIERA → iloc[-5] era 5 giorni
            # fa, non 4 settimane. Ora ogni serie è letta alla data di 28 giorni fa.
            _d4 = pd.Timestamp(walcl.index[-1]) - pd.Timedelta(days=28)
            liq_now = walcl.iloc[-1] - tga_s.iloc[-1] - rrp_s.iloc[-1] * 1000
            liq_4w  = (walcl.asof(_d4) - tga_s.asof(_d4) - rrp_s.asof(_d4) * 1000)
            sig["net_liq_delta_4w"] = round(liq_now - liq_4w, 2)
        else:
            sig["net_liq_delta_4w"] = None
    except Exception as e:
        logger.warning(f"[rates] Net Liq delta: {e}")
        sig["net_liq_delta_4w"] = None

    # FRA/OIS spread
    try:
        dtb3 = _fred_series_throttled(fred, "DTB3",
            observation_start=datetime.today() - timedelta(days=30)).dropna()
        if len(dtb3) >= 1 and sig.get("sofr"):
            sig["fra_ois"] = round(float(sig["sofr"]) - float(dtb3.iloc[-1]), 4)
        else:
            sig["fra_ois"] = None
    except Exception as e:
        logger.warning(f"[rates] FRA/OIS: {e}")
        sig["fra_ois"] = None

    # ── Yield Curve dinamica: trend + tipo di steepening/flattening ──────────
    # Non basta sapere se la curva è invertita (statico). Il MOVIMENTO della
    # curva è molto più informativo per il regime Dragon:
    #   - Bull steepening (parte corta scende più della lunga) → Fed taglia → FALCO_SX/recessione
    #   - Bear steepening (parte lunga sale più della corta) → inflazione/term premium → FALCO_DX
    #   - Bull flattening (lunga scende) → flight to quality / rallentamento
    #   - Bear flattening (corta sale) → Fed alza → fine ciclo
    try:
        # T10Y2Y di ~20 giorni fa per il trend dello spread
        t10y2y_hist = _fred_series_throttled(fred, "T10Y2Y",
            observation_start=datetime.today() - timedelta(days=45)).dropna()
        # DGS10 e DGS2 per capire QUALE estremità si muove
        dgs10 = _fred_series_throttled(fred, "DGS10",
            observation_start=datetime.today() - timedelta(days=45)).dropna()
        dgs2 = _fred_series_throttled(fred, "DGS2",
            observation_start=datetime.today() - timedelta(days=45)).dropna()
        # DGS30 — tratto lunghissimo della curva, dove si muovono i "Bond
        # Vigilantes" sulla sostenibilità fiscale (non catturato da T10Y2Y/
        # curve_trend, che guarda solo 10y-2y). Stessa finestra 45gg riusata.
        dgs30 = _fred_series_throttled(fred, "DGS30",
            observation_start=datetime.today() - timedelta(days=45)).dropna()

        # TERM PREMIUM 10y (THREEFYTP10, modello Kim-Wright della Fed su FRED — ACMTP10 non esiste su FRED, verificato 17/9) —
        # aggiunto 2026-09-16, idea presa dal DREAM Index / Lighthouse Macro.
        # Misura DIRETTAMENTE il compenso extra per detenere scadenze lunghe:
        # è la "sfiducia fiscale" che prima deducevamo solo indirettamente da
        # rate_30y+curve. Finestra più ampia (90gg) perché la serie è meno
        # frequente e può avere buchi.
        sig["term_premium_10y"] = None
        sig["term_premium_signal"] = None
        try:
            acmtp = _fred_series_throttled(fred, "THREEFYTP10",
                observation_start=datetime.today() - timedelta(days=400)).dropna()
            if len(acmtp) >= 1:
                tp = float(acmtp.iloc[-1])
                sig["term_premium_10y"] = round(tp, 3)
                # Soglie: term premium storicamente oscilla ~ -1% a +2%.
                # Sopra 0 = mercato chiede compenso per rischio duration
                # (sfiducia fiscale/inflazione); molto sopra = stress.
                if tp > 1.0:
                    sig["term_premium_signal"] = (
                        f"ELEVATO ({tp:.2f}%) — mercato chiede forte compenso "
                        f"per rischio duration, sfiducia fiscale marcata")
                elif tp > 0.3:
                    sig["term_premium_signal"] = (
                        f"POSITIVO ({tp:.2f}%) — term premium tornato positivo, "
                        f"repricing del rischio fiscale/inflazione in corso")
                elif tp > -0.3:
                    sig["term_premium_signal"] = (
                        f"NEUTRO ({tp:.2f}%) — term premium vicino a zero")
                else:
                    sig["term_premium_signal"] = (
                        f"COMPRESSO ({tp:.2f}%) — term premium negativo, "
                        f"domanda forte di duration (flight to quality o QE)")
        except Exception as _e_tp:
            logger.info(f"[rates] Term premium THREEFYTP10 non disponibile: {_e_tp}")

        sig["curve_trend"] = None
        sig["curve_signal"] = None
        if len(t10y2y_hist) >= 15:
            spread_now = float(t10y2y_hist.iloc[-1])
            spread_20d = float(t10y2y_hist.iloc[-15])
            delta_spread = spread_now - spread_20d  # >0 steepening, <0 flattening
            sig["curve_delta_20d"] = round(delta_spread, 3)

            # Determina quale estremità guida il movimento (bull vs bear)
            tipo = None
            if len(dgs10) >= 15 and len(dgs2) >= 15:
                d10 = float(dgs10.iloc[-1]) - float(dgs10.iloc[-15])
                d2 = float(dgs2.iloc[-1]) - float(dgs2.iloc[-15])
                if delta_spread > 0.05:  # steepening
                    tipo = "BEAR_STEEPENING" if d10 > abs(d2) else "BULL_STEEPENING"
                elif delta_spread < -0.05:  # flattening
                    tipo = "BEAR_FLATTENING" if d2 > abs(d10) else "BULL_FLATTENING"
                else:
                    tipo = "STABILE"
            sig["curve_trend"] = tipo

            # Mappa il tipo al segnale di regime
            curve_map = {
                "BULL_STEEPENING": "FALCO_SX — Fed verso tagli, recessione/deflazione in arrivo",
                "BEAR_STEEPENING": "FALCO_DX — term premium/inflazione, lunga sotto pressione",
                "BULL_FLATTENING": "TRANSIZIONE — flight to quality, rallentamento",
                "BEAR_FLATTENING": "FINE_CICLO — Fed alza, curva si appiattisce",
                "STABILE": "NEUTRO — curva stabile",
            }
            sig["curve_signal"] = curve_map.get(tipo)

            # ── SEGNALE: 2yr yield grezzo + Scenario A/B (report "La Résistance") ──
            # Il livello e trend del 2yr erano già scaricati sopra per capire la
            # curva (BEAR/BULL steepening/flattening) — qui li esponiamo anche come
            # segnale a sé, perché il framework "equity vs tassi a breve" risponde
            # a una domanda DIVERSA dalla forma della curva: un calo dell'equity è
            # un repricing hawkish (Fed non salva, niente panico da recessione) o
            # una vera fuga dal rischio (mercato prezza tagli per paura recessiva)?
            # Soglie larghe (±10bp sul 2y, ±1.5% su SPY in 20gg) per evitare rumore
            # su normali oscillazioni giornaliere.
            sig["rate_2y"] = None
            sig["rate_2y_delta_20d"] = None
            sig["equity_rates_scenario"] = None
            if len(dgs2) >= 15:
                sig["rate_2y"] = round(float(dgs2.iloc[-1]), 3)
                d2_local = float(dgs2.iloc[-1]) - float(dgs2.iloc[-15])
                sig["rate_2y_delta_20d"] = round(d2_local, 3)
                try:
                    _spy_short = _robust_yf("SPY", period="1mo")
                    if not _spy_short.empty and "Close" in _spy_short.columns:
                        _spy_c = _spy_short["Close"].dropna()
                        if len(_spy_c) >= 10:
                            spy_chg_20d = float(_spy_c.iloc[-1]) / float(_spy_c.iloc[0]) - 1
                            if spy_chg_20d < -0.015 and d2_local > 0.10:
                                sig["equity_rates_scenario"] = (
                                    f"SCENARIO_A_HAWKISH_REPRICING — equity {spy_chg_20d:+.1%} "
                                    f"e 2y +{d2_local:.2f}pp: Fed 'higher for longer', non panico "
                                    f"da recessione. Penalizza NASDAQ/AI/crypto/oro/duration")
                            elif spy_chg_20d < -0.015 and d2_local < -0.10:
                                sig["equity_rates_scenario"] = (
                                    f"SCENARIO_B_FLIGHT_TO_SAFETY — equity {spy_chg_20d:+.1%} "
                                    f"e 2y {d2_local:.2f}pp: mercato teme recessione, "
                                    f"prezza tagli futuri")
                            else:
                                sig["equity_rates_scenario"] = "NEUTRO — nessuna combinazione direzionale chiara"
                except Exception as _e_spy:
                    logger.warning(f"[rates] Scenario A/B (SPY fetch): {_e_spy}")

            # ── SEGNALE: 30yr yield grezzo (tratto lunghissimo, Bond Vigilantes) ──
            # Cattura la parte della curva che T10Y2Y/curve_trend non vede.
            # Un 30y che sale mentre G_z scende è la firma "bear steepening
            # asimmetrico" citata nell'analisi del 5/7 (Lops) — sfiducia fiscale
            # sulla sostenibilità del debito, non semplice ciclo economico.
            sig["rate_30y"] = None
            sig["rate_30y_delta_20d"] = None
            if len(dgs30) >= 15:
                sig["rate_30y"] = round(float(dgs30.iloc[-1]), 3)
                sig["rate_30y_delta_20d"] = round(
                    float(dgs30.iloc[-1]) - float(dgs30.iloc[-15]), 3)

            # ── AUDIT 2026-09-25 (quater): 10 anni + rendimenti live + Fed prezzata ──
            # Il 10 anni (il tasso più citato dal mercato) non veniva salvato.
            # I tassi FRED hanno 1-2 giorni di ritardo: si aggiungono i valori
            # live Yahoo (^TNX, ^TYX) per confronto, senza sostituire FRED.
            sig["rate_10y"] = round(float(dgs10.iloc[-1]), 3) if len(dgs10) else None
            # Riserva: spread 10y-2y ricostruito dai due tassi se T10Y2Y resta vuoto
            if sig.get("t10y2y") is None and len(dgs10) and len(dgs2):
                sig["t10y2y"] = round(float(dgs10.iloc[-1]) - float(dgs2.iloc[-1]), 3)
                sig["t10y2y_source"] = "ricostruito DGS10 − DGS2 (T10Y2Y non disponibile)"
            for _k, _tk in (("rate_10y_live", "^TNX"), ("rate_30y_live", "^TYX")):
                _v = _safe_yf(_tk)
                if _v is not None and _v > 20:      # vecchia quotazione Yahoo ×10
                    _v = _v / 10
                sig[_k] = round(_v, 3) if _v is not None and 0.5 <= _v <= 15 else None
            # Rialzi Fed prezzati dal mercato (proxy): 2 anni − Fed funds effettivi.
            # Ogni 0.25 punti ≈ un rialzo da 25bp atteso nei prossimi ~2 anni.
            sig["fed_hikes_priced_2y"] = None
            sig["fed_pricing_signal"] = None
            try:
                _dff = _safe_fred("DFF", fred, lookback_days=15)
                if _dff is not None and len(dgs2):
                    _gap = float(dgs2.iloc[-1]) - float(_dff)
                    sig["fed_funds_effective"] = round(float(_dff), 3)
                    sig["fed_hikes_priced_2y"] = round(_gap / 0.25, 1)
                    if _gap > 0.25:
                        sig["fed_pricing_signal"] = (f"RIALZI PREZZATI — 2y {float(dgs2.iloc[-1]):.2f}% vs Fed funds "
                                                     f"{_dff:.2f}%: ≈{_gap/0.25:.1f} rialzi da 25bp nei prossimi ~2 anni")
                    elif _gap < -0.25:
                        sig["fed_pricing_signal"] = (f"TAGLI PREZZATI — ≈{-_gap/0.25:.1f} tagli da 25bp nei prossimi ~2 anni")
                    else:
                        sig["fed_pricing_signal"] = "NEUTRO — mercato non prezza variazioni rilevanti dei tassi Fed"
            except Exception as _e_fp:
                logger.info(f"[rates] Fed pricing proxy: {_e_fp}")

            # ── SEGNALE: curva 2s30s — tratto lunghissimo (Bond Vigilantes) ──
            # Il curve_trend esistente (10y-2y) può leggere BULL_FLATTENING
            # mentre il tratto 2-30 anni fa l'opposto (bear steepening) — sono
            # parti diverse della curva, non in contraddizione (vedi analisi
            # Lops 5/7/2026). Ricalcola d2/d30 in modo autonomo (non riusa le
            # variabili d10/d2 del blocco 10y-2y sopra, che potrebbero non
            # esistere se quel blocco non è stato eseguito).
            #
            # Deliberatamente NON alimenta compute_regime_flags: il tema
            # "term premium/inflazione" è già pesato lì tramite curve_trend
            # (10y-2y). Aggiungerlo peserebbe due volte lo stesso fenomeno
            # macro senza una calibrazione testata — resta un segnale
            # informativo, letto e sintetizzato in sessione (come gli altri
            # _signal aggiunti in questa sessione: real_yield_signal,
            # infl_5y5y_signal, ecc.), non un input automatico al punteggio.
            sig["curve_2s30s_trend"] = None
            sig["curve_2s30s_signal"] = None
            if len(dgs2) >= 15 and len(dgs30) >= 15:
                spread_2s30s_now = float(dgs30.iloc[-1]) - float(dgs2.iloc[-1])
                spread_2s30s_20d = float(dgs30.iloc[-15]) - float(dgs2.iloc[-15])
                delta_2s30s = spread_2s30s_now - spread_2s30s_20d
                d2_long = float(dgs2.iloc[-1]) - float(dgs2.iloc[-15])
                d30_long = float(dgs30.iloc[-1]) - float(dgs30.iloc[-15])
                tipo_long = None
                if delta_2s30s > 0.05:      # steepening del tratto lungo
                    tipo_long = "BEAR_STEEPENING" if d30_long > abs(d2_long) else "BULL_STEEPENING"
                elif delta_2s30s < -0.05:   # flattening del tratto lungo
                    tipo_long = "BEAR_FLATTENING" if d2_long > abs(d30_long) else "BULL_FLATTENING"
                else:
                    tipo_long = "STABILE"
                sig["curve_2s30s_trend"] = tipo_long
                curve_2s30s_map = {
                    "BEAR_STEEPENING": ("BOND_VIGILANTES — term premium sale sul tratto "
                                        "lunghissimo, sfiducia fiscale (rif. Lops 5/7/2026)"),
                    "BULL_STEEPENING": ("Fed verso tagli aggressivi sul breve, tratto "
                                        "lunghissimo relativamente stabile"),
                    "BEAR_FLATTENING": "Restrizione su tutta la curva, fine ciclo",
                    "BULL_FLATTENING": ("Flight to quality anche sul lunghissimo — "
                                        "timore di recessione seria, non solo rallentamento"),
                    "STABILE": "NEUTRO — tratto 2-30 anni stabile",
                }
                sig["curve_2s30s_signal"] = curve_2s30s_map.get(tipo_long)
    except Exception as e:
        logger.warning(f"[rates] Yield curve dinamica: {e}")
        sig["curve_trend"] = None
        sig["curve_signal"] = None

    # ── SEGNALE: Labor market stress (initial vs continued claims) ──────────
    # Il livello assoluto dei claims è già catturato. Il segnale FORTE è la
    # DIVERGENZA: continued claims che salgono mentre gli initial restano bassi
    # = mercato del lavoro che si irrigidisce → FALCO_SX emergente, mesi prima
    # che lo veda l'NFP. Calcoliamo il rapporto e il suo trend a 4 settimane.
    sig["claims_ratio"] = None
    sig["labor_stress_signal"] = None
    try:
        _icsa = _fred_series_throttled(fred, "ICSA",
            observation_start=datetime.today() - timedelta(days=120)).dropna()
        _ccsa = _fred_series_throttled(fred, "CCSA",
            observation_start=datetime.today() - timedelta(days=120)).dropna()
        if len(_icsa) >= 5 and len(_ccsa) >= 5:
            # Rapporto continued/initial: più alto = più persone restano disoccupate
            ratio_now = float(_ccsa.iloc[-1]) / float(_icsa.iloc[-1])
            ratio_4w  = float(_ccsa.iloc[-5]) / float(_icsa.iloc[-5])
            sig["claims_ratio"] = round(ratio_now, 2)
            delta_ratio = ratio_now - ratio_4w
            # Continued in salita + initial stabili = irrigidimento mercato lavoro
            cc_trend = float(_ccsa.iloc[-1]) / float(_ccsa.iloc[-5]) - 1
            if cc_trend > 0.03 and delta_ratio > 0:
                sig["labor_stress_signal"] = ("FALCO_SX_EMERGENTE — continued claims in salita "
                                              f"(+{cc_trend:.1%} 4w), mercato lavoro si irrigidisce")
            elif cc_trend < -0.03:
                sig["labor_stress_signal"] = "RISK_ON — continued claims in calo, mercato lavoro solido"
            else:
                sig["labor_stress_signal"] = "NEUTRO — mercato lavoro stabile"
    except Exception as e:
        logger.warning(f"[rates] Labor stress (claims ratio): {e}")

    # ── SEGNALE: Real yield 10y interpretato (driver primario dell'oro) ─────
    # DFII10 è già fetchato in fred_series. Qui aggiungiamo l'interpretazione:
    # tassi reali negativi/in calo = ambiente FENICE per l'oro (store of value
    # quando il rendimento reale del cash è negativo). Conferma diretta del
    # rally dell'oro alla fonte, invece di inferirlo da gold+DXY.
    sig["real_yield_signal"] = None
    _ry = sig.get("real_yield_10y")
    if _ry is not None:
        if _ry < 0:
            sig["real_yield_signal"] = f"FENICE FORTE — tasso reale 10y negativo ({_ry:.2f}%), oro favorito"
        elif _ry < 0.5:
            sig["real_yield_signal"] = f"FENICE early — tasso reale 10y basso ({_ry:.2f}%), oro sostenuto"
        elif _ry > 2.0:
            sig["real_yield_signal"] = f"HEADWIND oro — tasso reale 10y alto ({_ry:.2f}%), store of value costoso"
        else:
            sig["real_yield_signal"] = f"NEUTRO — tasso reale 10y {_ry:.2f}%"

    # ── SEGNALE: 5y5y forward inflation interpretato (anticipa FALCO_DX) ────
    # T5YIFR già fetchato. È la metrica Fed per le aspettative di inflazione di
    # lungo periodo, più stabile del breakeven semplice. Distingue inflazione
    # transitoria (TRANSIZIONE) da strutturale (FALCO_DX confermato).
    sig["infl_5y5y_signal"] = None
    _i55 = sig.get("infl_5y5y")
    if _i55 is not None:
        if _i55 > 2.5:
            sig["infl_5y5y_signal"] = (f"FALCO_DX — aspettative inflazione 5y5y disancorate "
                                       f"({_i55:.2f}% > 2.5% target Fed)")
        elif _i55 < 1.8:
            sig["infl_5y5y_signal"] = (f"FALCO_SX — aspettative inflazione 5y5y deboli "
                                       f"({_i55:.2f}% < 1.8%), rischio deflazione")
        else:
            sig["infl_5y5y_signal"] = f"ANCORATE — aspettative 5y5y {_i55:.2f}% in range Fed"

    # Margin Debt YoY
    # AUDIT 2026-09-25: la serie usata prima (BOGZ1FL664090005Q) è il TOTALE
    # delle attività finanziarie dei broker-dealer, NON il margin debt.
    # Serie corretta: crediti dei broker verso i clienti (prestiti a margine),
    # Fed Z.1, trimestrale. Formato invariato: frazione (0.20 = +20%).
    sig["margin_debt_yoy"] = None
    sig["margin_debt_source"] = "FRED BOGZ1FL663067003Q (Z.1 margin loans)"
    try:
        time.sleep(0.6)
        _margin = _fred_with_retry("BOGZ1FL663067003Q", fred,
            observation_start=datetime.today() - timedelta(days=1200))
        _yoy = _yoy_by_date(_margin) if _margin is not None else None
        if _yoy is not None:
            sig["margin_debt_yoy"] = round(_yoy / 100, 4)
    except Exception as e:
        logger.warning(f"[rates] Margin Debt: {e}")

    # % S&P500 sopra MA200
    # NOTA (fix 2026-07): nonostante il nome, NON misura l'ampiezza di
    # NOTA STORICA: nome mantenuto per compatibilità con la colonna Supabase
    # esistente. FIX 2026-07: ora tenta di calcolare l'ampiezza di mercato
    # REALE (Livello 1/2) invece della sola persistenza di SPY (Livello 3,
    # fallback finale). Vedi _compute_market_breadth_200ma() sotto.
    sig["pct_above_ma200"], _breadth_method = _compute_market_breadth_200ma()
    sig["breadth_method"] = _breadth_method
    logger.info(f"[rates] Market breadth 200MA: {sig['pct_above_ma200']}% "
                f"(metodo: {_breadth_method})")

    # ── CICLO MONETARIO — M2 e Bank Credit ──────────────────────────────────
    # M2 YoY%: leading indicator inflazione 6-12 mesi
    sig["m2_yoy"] = None
    try:
        m2 = _fred_series_throttled(fred, "M2SL",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(m2) >= 14:
            raw = _yoy_or_nan(m2)
            sig["m2_yoy"] = round(raw, 2) if -20.0 <= raw <= 50.0 else None
    except Exception as e:
        logger.warning(f"[rates] M2 YoY: {e}")

    # Bank Credit YoY% — espansione/contrazione credito bancario
    sig["bank_credit_yoy"] = None
    try:
        bc = _fred_series_throttled(fred, "TOTBKCR",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(bc) >= 14:
            raw = _yoy_or_nan(bc)
            sig["bank_credit_yoy"] = round(raw, 2) if -30.0 <= raw <= 30.0 else None
    except Exception as e:
        logger.warning(f"[rates] Bank Credit YoY: {e}")

    # Consumer Credit YoY% — ciclo credito al consumo
    sig["consumer_credit_yoy"] = None
    try:
        cc = _fred_series_throttled(fred, "TOTALSL",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(cc) >= 14:
            raw = _yoy_or_nan(cc)
            sig["consumer_credit_yoy"] = round(raw, 2) if -20.0 <= raw <= 30.0 else None
    except Exception as e:
        logger.warning(f"[rates] Consumer Credit YoY: {e}")

    # C&I Loans YoY% — prestiti commerciali e industriali
    sig["ci_loans_yoy"] = None
    try:
        ci = _fred_series_throttled(fred, "BUSLOANS",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(ci) >= 14:
            raw = _yoy_or_nan(ci)
            sig["ci_loans_yoy"] = round(raw, 2) if -30.0 <= raw <= 30.0 else None
    except Exception as e:
        logger.warning(f"[rates] C&I Loans YoY: {e}")

    # PCE Core YoY% — inflazione core preferita dalla Fed
    sig["pce_core_yoy"] = None
    try:
        pce_core = _fred_series_throttled(fred, "PCEPILFE",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(pce_core) >= 14:
            raw = _yoy_or_nan(pce_core)
            sig["pce_core_yoy"] = round(raw, 2) if -5.0 <= raw <= 20.0 else None
    except Exception as e:
        logger.warning(f"[rates] PCE Core YoY: {e}")

    # ── CICLO IMMOBILIARE ────────────────────────────────────────────────────
    # Case-Shiller Home Price YoY% — ciclo housing (lag ~2 mesi)
    sig["case_shiller_yoy"] = None
    try:
        cs = _fred_series_throttled(fred, "CSUSHPISA",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(cs) >= 14:
            raw = _yoy_or_nan(cs)
            sig["case_shiller_yoy"] = round(raw, 2) if -30.0 <= raw <= 30.0 else None
    except Exception as e:
        logger.warning(f"[rates] Case-Shiller YoY: {e}")

    # Housing Starts — leading indicator ciclo costruzioni
    sig["housing_starts"] = None
    try:
        hs = _fred_series_throttled(fred, "HOUST",
            observation_start=datetime.today() - timedelta(days=120)).dropna()
        if len(hs) >= 1:
            sig["housing_starts"] = round(float(hs.iloc[-1]), 0)
    except Exception as e:
        logger.warning(f"[rates] Housing Starts: {e}")

    # 30y Mortgage Rate
    sig["mortgage_30y"] = None
    try:
        mr = _fred_series_throttled(fred, "MORTGAGE30US",
            observation_start=datetime.today() - timedelta(days=60)).dropna()
        if len(mr) >= 1:
            val = float(mr.iloc[-1])
            sig["mortgage_30y"] = round(val, 2) if 2.0 <= val <= 20.0 else None
    except Exception as e:
        logger.warning(f"[rates] Mortgage 30y: {e}")

    # ── CICLO LAVORO AVANZATO ────────────────────────────────────────────────
    # Initial Jobless Claims — leading indicator mercato lavoro (settimanale)
    sig["jobless_claims"] = None
    try:
        ic = _fred_series_throttled(fred, "ICSA",
            observation_start=datetime.today() - timedelta(days=60)).dropna()
        if len(ic) >= 1:
            val = float(ic.iloc[-1])
            sig["jobless_claims"] = round(val, 0) if 100_000 <= val <= 2_000_000 else None
    except Exception as e:
        logger.warning(f"[rates] Jobless Claims: {e}")

    # JOLTS Job Openings — domanda di lavoro
    sig["jolts_openings"] = None
    try:
        jo = _fred_series_throttled(fred, "JTSJOL",
            observation_start=datetime.today() - timedelta(days=120)).dropna()
        if len(jo) >= 1:
            sig["jolts_openings"] = round(float(jo.iloc[-1]), 0)
    except Exception as e:
        logger.warning(f"[rates] JOLTS Openings: {e}")

    # ── SPREADS AGGIUNTIVI ───────────────────────────────────────────────────
    # BBB OAS — stress obbligazioni BBB (una tacca sopra junk)
    sig["bbb_oas"] = None
    try:
        bbb = _fred_series_throttled(fred, "BAMLC0A4CBBB",
            observation_start=datetime.today() - timedelta(days=30)).dropna()
        if len(bbb) >= 1:
            val = float(bbb.iloc[-1])
            sig["bbb_oas"] = round(val, 2) if 0.5 <= val <= 10.0 else None
    except Exception as e:
        logger.warning(f"[rates] BBB OAS: {e}")

    # NFCI — Chicago Fed National Financial Conditions Index
    # Positivo = condizioni più restrittive del normale
    sig["nfci"] = None
    try:
        nf = _fred_series_throttled(fred, "NFCI",
            observation_start=datetime.today() - timedelta(days=60)).dropna()
        if len(nf) >= 1:
            val = float(nf.iloc[-1])
            sig["nfci"] = round(val, 3) if -3.0 <= val <= 3.0 else None
    except Exception as e:
        logger.warning(f"[rates] NFCI: {e}")

    # ── CICLO PRODUTTIVO ─────────────────────────────────────────────────────
    # Industrial Production YoY%
    sig["indpro_yoy"] = None
    try:
        ip = _fred_series_throttled(fred, "INDPRO",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(ip) >= 14:
            raw = _yoy_or_nan(ip)
            sig["indpro_yoy"] = round(raw, 2) if -20.0 <= raw <= 20.0 else None
    except Exception as e:
        logger.warning(f"[rates] IndPro YoY: {e}")

    # Retail Sales YoY%
    sig["retail_sales_yoy"] = None
    try:
        rs = _fred_series_throttled(fred, "RSXFS",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(rs) >= 14:
            raw = _yoy_or_nan(rs)
            sig["retail_sales_yoy"] = round(raw, 2) if -30.0 <= raw <= 30.0 else None
    except Exception as e:
        logger.warning(f"[rates] Retail Sales YoY: {e}")

    # Real Disposable Income YoY%
    sig["real_disp_income_yoy"] = None
    try:
        rdi = _fred_series_throttled(fred, "DSPIC96",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(rdi) >= 14:
            raw = _yoy_or_nan(rdi)
            sig["real_disp_income_yoy"] = round(raw, 2) if -20.0 <= raw <= 20.0 else None
    except Exception as e:
        logger.warning(f"[rates] Real Disp Income YoY: {e}")

    # Segnale ciclo credito composito (per ARGUS)
    # Espansione: M2_yoy>5 AND bank_credit_yoy>5 AND ci_loans_yoy>5
    # Contrazione: M2_yoy<0 OR bank_credit_yoy<-2 OR ci_loans_yoy<-2
    _m2  = sig.get("m2_yoy")
    _bc  = sig.get("bank_credit_yoy")
    _ci  = sig.get("ci_loans_yoy")
    if all(v is not None for v in [_m2, _bc, _ci]):
        if _m2 > 5 and _bc > 5 and _ci > 5:
            sig["credit_cycle_signal"] = "ESPANSIONE"
        elif _m2 < 0 or _bc < -2 or _ci < -2:
            sig["credit_cycle_signal"] = "CONTRAZIONE"
        else:
            sig["credit_cycle_signal"] = "NEUTRO"
    else:
        sig["credit_cycle_signal"] = None

    # ── CICLO DEBITO STRUTTURALE ────────────────────────────────────────────
    # SLOOS Tightening% — % banche che stringono gli standard creditizi (trimestrale)
    # Positivo = banche stringono → credit crunch in arrivo
    # AUDIT 2026-09-25: prima si leggeva DRSDCILM, che è la DOMANDA di prestiti
    # ("banche che vedono domanda più forte"), non la stretta creditizia.
    # Serie corretta: DRTSCILM (banche che INASPRISCONO gli standard C&I).
    # La domanda resta disponibile con il suo nome.
    sig["sloos_tightening"] = None
    sig["sloos_demand"] = None
    try:
        sl = _fred_series_throttled(fred, "DRTSCILM",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(sl) >= 1:
            val = float(sl.iloc[-1])
            sig["sloos_tightening"] = round(val, 1) if -100.0 <= val <= 100.0 else None
    except Exception as e:
        logger.warning(f"[rates] SLOOS: {e}")
    try:
        sd = _fred_series_throttled(fred, "DRSDCILM",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(sd) >= 1:
            val = float(sd.iloc[-1])
            sig["sloos_demand"] = round(val, 1) if -100.0 <= val <= 100.0 else None
    except Exception as e:
        logger.warning(f"[rates] SLOOS demand: {e}")

    # Delinquency Rate Credit Cards — stress famiglie sul debito revolving
    sig["delinquency_cc"] = None
    try:
        dq = _fred_series_throttled(fred, "DRCCLACBS",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(dq) >= 1:
            val = float(dq.iloc[-1])
            sig["delinquency_cc"] = round(val, 2) if 0.5 <= val <= 20.0 else None
    except Exception as e:
        logger.warning(f"[rates] Delinquency CC: {e}")

    # Total Credit Market Debt YoY% — crescita totale del debito USA
    sig["total_credit_yoy"] = None
    try:
        tc = _fred_series_throttled(fred, "TCMDO",
            observation_start=datetime.today() - timedelta(days=800))
        # AUDIT 2026-09-25: iloc[-4] su trimestrale = 3 trimestri, NON un anno.
        # Ora il confronto è per data (stesso trimestre dell'anno prima).
        raw_yoy = _yoy_by_date(tc) if tc is not None else None
        if raw_yoy is not None and -20.0 <= raw_yoy <= 30.0:
            sig["total_credit_yoy"] = round(raw_yoy, 2)
    except Exception as e:
        logger.warning(f"[rates] Total Credit YoY: {e}")

    # Household Debt Service Ratio — % reddito speso per servire il debito
    # Segnale chiave ciclo immobiliare/consumo — sale prima delle recessioni
    sig["debt_service_ratio"] = None
    try:
        dsr = _fred_series_throttled(fred, "TDSP",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(dsr) >= 1:
            val = float(dsr.iloc[-1])
            sig["debt_service_ratio"] = round(val, 2) if 5.0 <= val <= 20.0 else None
    except Exception as e:
        logger.warning(f"[rates] Debt Service Ratio: {e}")

    # Household Debt to GDP% — livello strutturale indebitamento famiglie
    sig["hh_debt_gdp"] = None
    try:
        hd = _fred_series_throttled(fred, "HDTGPDUSQ163N",
            observation_start=datetime.today() - timedelta(days=550)).dropna()
        if len(hd) >= 1:
            val = float(hd.iloc[-1])
            sig["hh_debt_gdp"] = round(val, 1) if 20.0 <= val <= 150.0 else None
    except Exception as e:
        logger.warning(f"[rates] HH Debt/GDP: {e}")

    # Non-Financial Corporate Debt YoY% — ciclo leva aziendale
    sig["corp_debt_yoy"] = None
    try:
        cd = _fred_series_throttled(fred, "BCNSDODNS",
            observation_start=datetime.today() - timedelta(days=800))
        raw = _yoy_by_date(cd) if cd is not None else None   # confronto per data (vedi TCMDO)
        if raw is not None:
            sig["corp_debt_yoy"] = round(raw, 2) if -20.0 <= raw <= 30.0 else None
    except Exception as e:
        logger.warning(f"[rates] Corp Debt YoY: {e}")

    # ── CICLO IMMOBILIARE INTERNAZIONALE ────────────────────────────────────
    # Global House Price Index (BIS) — non disponibile via FRED in tempo reale
    # Proxy: usare OECD House Price Ratio (via OECD SDMX) — aggiunta futura
    # Per ora: Case-Shiller USA come proxy del ciclo globale (correlazione >0.7)
    # Nota: il mercato mortgage USA trasmette stress al credito globale via MBS/CMBS

    # Aggiornaento segnale ciclo credito composito con i nuovi indicatori
    _sloos = sig.get("sloos_tightening")
    _dq    = sig.get("delinquency_cc")
    _dsr   = sig.get("debt_service_ratio")
    # Stress ciclo debito strutturale
    _stress_count = sum([
        1 if _sloos is not None and _sloos > 30 else 0,   # banche stringono
        1 if _dq    is not None and _dq > 4.0 else 0,    # delinquency alta
        1 if _dsr   is not None and _dsr > 12.0 else 0,  # debt service pesante
    ])
    if _stress_count >= 2:
        sig["debt_stress_signal"] = "ALTO"
    elif _stress_count == 1:
        sig["debt_stress_signal"] = "MEDIO"
    else:
        sig["debt_stress_signal"] = "BASSO" if all(
            v is not None for v in [_sloos, _dq, _dsr]) else None

    logger.info(f"[rates] T10Y2Y={sig.get('t10y2y')} CPI={sig.get('cpi_yoy')} "
                f"M2_yoy={sig.get('m2_yoy')} BankCredit={sig.get('bank_credit_yoy')} "
                f"SLOOS={sig.get('sloos_tightening')} DebtService={sig.get('debt_service_ratio')} "
                f"DebtStress={sig.get('debt_stress_signal')} "
                f"CreditCycle={sig.get('credit_cycle_signal')} "
                f"HY_OAS={sig.get('hy_oas')} NFCI={sig.get('nfci')} "
                f"NetLiq={sig.get('net_liquidity')}")
    return sig


# ────────────────────────────────────────────────────────────────────────────
# 4.3 FX / COMMODITY STACK
# ────────────────────────────────────────────────────────────────────────────

def collect_fx_commodity_stack() -> dict:
    """
    Stack FX e commodity — critico per Artemis (Gold e Copper/Gold ratio
    sono segnali diretti del regime FENICE e SERPENTE).
    """
    sig = {}
    tickers = {
        "dxy":    "DX-Y.NYB",
        "eurusd": "EURUSD=X",
        "usdjpy": "USDJPY=X",
        "gold":   "GC=F",
        "silver": "SI=F",
        "copper": "HG=F",
        "bdry":   "BDRY",  # Baltic Dry Index → proxy crescita globale
    }
    for key, ticker in tickers.items():
        sig[key] = _safe_yf(ticker)
        time.sleep(0.3)

    # Gold sanity check $500-$15000 + Stooq fallback
    if sig.get("gold") is not None and not (500.0 <= sig["gold"] <= 15000.0):
        logger.warning(f"[fx] Gold={sig['gold']} fuori range -> scartato")
        sig["gold"] = None
    if sig.get("gold") is None:
        try:
            _st = _stooq_download("GC=F", period_days=10)
            if not _st.empty and "Close" in _st.columns:
                _v = float(_st["Close"].dropna().iloc[-1])
                if 500.0 <= _v <= 15000.0:
                    sig["gold"] = round(_v, 2)
                    logger.info(f"[fx] Gold={sig['gold']} via Gold Stooq fallback")
        except Exception as e:
            logger.warning(f"[fx] Gold Stooq fallback: {e}")

    # Copper/Gold ratio — segnale risk-on/off e crescita globale
    if sig.get("copper") and sig.get("gold") and sig["gold"] > 0:
        sig["copper_gold_ratio"] = round(sig["copper"] / sig["gold"], 6)
        sig["copper_gold_signal"] = (
            "RISK_ON"  if sig["copper_gold_ratio"] > COPPER_GOLD_BULL_RATIO else
            "RISK_OFF"
        )
    else:
        sig["copper_gold_ratio"]  = None
        sig["copper_gold_signal"] = None

    # Gold YoY % (proxy Artemis: Gold outperformance = FENICE segnale)
    try:
        gold_hist = _robust_yf("GC=F", period="2y")
        if not gold_hist.empty and "Close" in gold_hist.columns:
            gold_close = gold_hist["Close"].dropna()
            if len(gold_close) >= 250:
                sig["gold_yoy_pct"] = round(
                    (float(gold_close.iloc[-1]) / float(gold_close.iloc[-252]) - 1) * 100, 2)
            else:
                sig["gold_yoy_pct"] = None
        else:
            sig["gold_yoy_pct"] = None
    except Exception as e:
        logger.warning(f"[fx] Gold YoY: {e}")
        sig["gold_yoy_pct"] = None

    # ── SEGNALE: Gold-Equity rolling correlation (canary risk-off estremo) ──
    # Normalmente gold ed equity sono decorrelati o negativamente correlati —
    # è il cuore della diversificazione Dragon. Nei crash da liquidity crunch
    # (marzo 2020, 2008) la correlazione diventa temporaneamente POSITIVA:
    # tutto scende insieme. Monitorarla dà un allarme precoce del raro regime
    # in cui la diversificazione Dragon temporaneamente non protegge.
    sig["gold_equity_corr_30d"] = None
    sig["correlation_regime_signal"] = None
    try:
        if not gold_hist.empty and "Close" in gold_hist.columns:
            _gold_c = gold_hist["Close"].dropna()
            _spy_hist = _robust_yf("SPY", period="6mo")
            if not _spy_hist.empty and "Close" in _spy_hist.columns:
                _spy_c = _spy_hist["Close"].dropna()
                # Allinea le due serie sulle date comuni, prende rendimenti giornalieri
                _df = pd.DataFrame({"gold": _gold_c, "spy": _spy_c}).dropna()
                if len(_df) >= 31:
                    _ret = _df.pct_change().dropna()
                    _corr30 = _ret["gold"].tail(30).corr(_ret["spy"].tail(30))
                    if _corr30 is not None and not pd.isna(_corr30):
                        sig["gold_equity_corr_30d"] = round(float(_corr30), 3)
                        if _corr30 > 0.5:
                            sig["correlation_regime_signal"] = (
                                f"⚠️ RISK-OFF ESTREMO — corr gold/equity +{_corr30:.2f} "
                                f"anomala: liquidity crunch, diversificazione Dragon sotto stress")
                        elif _corr30 > 0.2:
                            sig["correlation_regime_signal"] = (
                                f"ATTENZIONE — corr gold/equity +{_corr30:.2f} positiva, monitorare")
                        else:
                            sig["correlation_regime_signal"] = (
                                f"NORMALE — corr gold/equity {_corr30:.2f}, diversificazione attiva")
    except Exception as e:
        logger.warning(f"[fx] Gold-equity correlation: {e}")

    logger.info(f"[fx] DXY={sig.get('dxy')} Gold={sig.get('gold')} "
                f"Cu/Au={sig.get('copper_gold_ratio')} Gold_YoY={sig.get('gold_yoy_pct')}%")
    return sig


# ────────────────────────────────────────────────────────────────────────────
# 4.4 SENTIMENT STACK
# ────────────────────────────────────────────────────────────────────────────

def collect_sentiment_stack(fred: Fred) -> dict:
    """
    Stack sentiment:
    Fear & Greed (CNN) · CBOE P/C ratio · UMCSENT · NAAIM · GPR · EPU
    """
    sig = {}

    # ── CBOE Put/Call Ratio ─────────────────────────────────────────────────
    # Fonte ufficiale CBOE: NON disponibile (CSV fermo al 2012, vedi sotto).
    # Stima informativa: SPY opzioni ATM ±5% → sig["spy_pc_proxy"] (non entra nel composito)
    # Sanity check: range valido 0.30-2.50. Fuori range → scartata.
    # Valori storici:
    #   Equity P/C: 0.40-0.80 (tipicamente)
    #   Total  P/C: 0.80-1.30 (tipicamente)
    #   Index  P/C: 0.90-1.50 (tipicamente)
    #   Qualsiasi > 2.50 → anomalia / dato malformato / OI distorto
    # ─────────────────────────────────────────────────────────────────────────
    PC_MIN, PC_MAX = 0.30, 2.50   # range sanity check — fuori da questo → anomalia

    sig["cboe_put_call"] = None
    sig["cboe_put_call_source"] = None   # FIX F4: traccio la fonte per escludere proxy

    # AUDIT 2026-09-25: il CSV CBOE (equitypc.csv) è fermo al 24/1/2012 → non è
    # MAI stato usato (scartato ogni giorno dal controllo di freschezza).
    # Nessuna fonte ufficiale gratuita verificata al momento: "cboe_put_call"
    # resta None (colonna non più ingannevole) e la stima SPY viene salvata con
    # il SUO nome in "spy_pc_proxy" — informativa, esclusa dal composito.
    sig["spy_pc_proxy"] = None
    sig["cboe_put_call_source"] = "NON_DISPONIBILE (CSV CBOE fermo al 2012)"

    # Stima SPY — solo opzioni ATM ±5% (open interest, scadenza più vicina)
    # Limita alle strike ATM per evitare che l'OI delle put OTM accumulate
    # (tipico in periodi di stress geopolitico) distorca il ratio
    if True:
        try:
            _spy_tk  = yf.Ticker("SPY")
            _exps    = _spy_tk.options
            # FIX: fuori orario di mercato lastPrice è None → fallback su
            # previousClose (sempre disponibile, sufficiente per ATM±5%)
            _spot    = (_spy_tk.fast_info.get("lastPrice") or
                        _spy_tk.fast_info.get("previousClose"))
            if _exps and _spot:
                _chain   = _spy_tk.option_chain(_exps[0])
                _atm_lo  = _spot * 0.95
                _atm_hi  = _spot * 1.05
                # Filtra solo opzioni ATM ±5%
                _c_atm   = _chain.calls[
                    (_chain.calls["strike"] >= _atm_lo) &
                    (_chain.calls["strike"] <= _atm_hi)
                ]["openInterest"].sum()
                _p_atm   = _chain.puts[
                    (_chain.puts["strike"] >= _atm_lo) &
                    (_chain.puts["strike"] <= _atm_hi)
                ]["openInterest"].sum()
                if _c_atm > 0:
                    _pc_atm = round(float(_p_atm / _c_atm), 3)
                    if PC_MIN <= _pc_atm <= PC_MAX:
                        sig["spy_pc_proxy"] = _pc_atm
                        logger.info(f"[sent] Stima P/C SPY ATM±5% (non CBOE) = {_pc_atm}")
                    else:
                        logger.info(f"[sent] Stima P/C SPY ATM fuori scala {_pc_atm:.3f} — scartata")
        except Exception as e:
            logger.info(f"[sent] Stima P/C SPY: {e}")

    # UMCSENT (University of Michigan Consumer Sentiment)
    if fred:
        sig["umcsent"] = _safe_fred("UMCSENT", fred)
    else:
        sig["umcsent"] = None

    # CNN Fear & Greed
    # FIX 2026-07: l'URL CNN diretto mancava del parametro data richiesto
    # (endpoint restituisce dati validi solo come /graphdata/{YYYY-MM-DD}),
    # e il parsing cercava una struttura JSON diversa da quella reale
    # (data['fear_and_greed_historical']['data'] = lista di punti {x,y},
    # non un campo 'score' diretto). Aggiunto anche un mirror GitHub
    # mantenuto quotidianamente come secondo tentativo di dato REALE
    # prima di arrendersi al sintetico interno.
    sig["fear_greed_score"] = None
    _fg_start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        fg_urls = [
            "https://fear-and-greed-index.p.rapidapi.com/v1/fgi",   # richiede chiave a pagamento — atteso fallire senza
            f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{_fg_start}",
        ]
        for url in fg_urls:
            try:
                r = requests.get(url, timeout=10,
                                 headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    data = r.json()
                    # Prova prima il formato "corrente" diretto (se mai presente),
                    # poi il formato storico reale confermato (lista di punti x/y —
                    # prende l'ultimo, cioè il più recente).
                    score = data.get("fear_and_greed", {}).get("score")
                    _fg_ts = None
                    if score is not None:
                        _fg_ts = data.get("fear_and_greed", {}).get("timestamp")
                    if score is None:
                        _hist = data.get("fear_and_greed_historical", {}).get("data")
                        if _hist:
                            score = _hist[-1].get("y")
                            _fg_ts = _hist[-1].get("x")
                    if score is None:
                        score = data.get("fgi", {}).get("now", {}).get("value")
                    if score is not None:
                        sig["fear_greed_score"] = round(float(score), 1)
                        _fg_d = None
                        try:
                            if isinstance(_fg_ts, (int, float)):
                                _fg_d = datetime.utcfromtimestamp(_fg_ts / 1000).strftime("%Y-%m-%d")
                            elif _fg_ts:
                                _fg_d = str(_fg_ts)[:10]
                        except Exception:
                            _fg_d = None
                        _note_ext("fear_greed_score", "CNN diretto", _fg_d, max_age_days=5)
                        logger.info(f"[sent] F&G reale ottenuto da CNN diretto: {sig['fear_greed_score']}")
                        break
            except Exception:
                continue

        # Secondo tentativo di dato REALE: mirror GitHub aggiornato quotidianamente,
        # prima di arrendersi al sintetico. Se CNN cambia ancora endpoint in futuro,
        # è un problema di chi mantiene quel repo, non solo nostro.
        if sig["fear_greed_score"] is None:
            try:
                _mirror_url = ("https://raw.githubusercontent.com/whit3rabbit/"
                               "fear-greed-data/main/datasets/cnn_fear_greed.csv")
                _r_mirror = requests.get(_mirror_url, timeout=15,
                                         headers={"User-Agent": "Mozilla/5.0"})
                if _r_mirror.status_code == 200 and len(_r_mirror.text) > 100:
                    import io as _io_fg
                    _df_fg = pd.read_csv(_io_fg.StringIO(_r_mirror.text))
                    _val_col = next((c for c in _df_fg.columns
                                     if "fear" in c.lower() or "greed" in c.lower()
                                     or c.lower() in ("y", "value", "score")), None)
                    if _val_col:
                        # AUDIT 2026-09-25: controllo data — il mirror è di terzi e
                        # potrebbe fermarsi: un valore più vecchio di 5 giorni NON è usato.
                        _date_col = next((c for c in _df_fg.columns if c.lower() == "date"), None)
                        _df_fg = _df_fg.dropna(subset=[_val_col])
                        _last_d = None
                        if _date_col is not None and len(_df_fg) > 0:
                            _last_d = pd.to_datetime(_df_fg[_date_col], errors="coerce").max()
                        _age = (datetime.today() - _last_d).days if _last_d is not None and pd.notna(_last_d) else None
                        if len(_df_fg) > 0 and _age is not None and _age <= 5:
                            _df_fg = _df_fg.sort_values(_date_col)
                            sig["fear_greed_score"] = round(float(_df_fg[_val_col].iloc[-1]), 1)
                            _note_ext("fear_greed_score", "CNN via mirror GitHub whit3rabbit",
                                      _last_d.strftime("%Y-%m-%d"), max_age_days=5)
                            logger.info(f"[sent] F&G reale ottenuto da mirror GitHub: "
                                       f"{sig['fear_greed_score']} (dato del {_last_d:%Y-%m-%d})")
                        else:
                            logger.warning(f"[sent] F&G mirror GitHub: ultimo dato "
                                           f"{_last_d} ({_age} giorni) — troppo vecchio, scartato")
            except Exception as e:
                logger.info(f"[sent] F&G mirror GitHub fallito: {e}")
    except Exception as e:
        logger.warning(f"[sent] F&G: {e}")

    # NAAIM Exposure Index
    # FIX 2026-07: fonte primaria ora Nasdaq Data Link (dataset ufficiale
    # NAAIM/NAAIM, gratuito con API key gratuita in env NASDAQ_API_KEY,
    # stesso pattern di FRED_API_KEY). I due URL diretti naaim.org restano
    # come fallback ma storicamente restituivano sempre None (URL CSV
    # probabilmente cambiato o bloccato lato sito).
    sig["naaim_exposure"] = None
    # FIX 2026-07-14: Nasdaq Data Link RIMOSSO come fonte — verificato in
    # produzione che il dominio è protetto da Incapsula/Imperva (anti-bot),
    # risposta HTTP 403 con pagina-sfida HTML invece di JSON, sia dal
    # browser dell'utente che dal server GitHub Actions. Non è un problema
    # di chiave API né di codice — è un blocco infrastrutturale a monte,
    # non aggirabile con retry/header/User-Agent. La chiave ARTEMIS_NASDAQ_KEY
    # può restare nei secrets GitHub, semplicemente non viene più usata qui.
    # AUDIT 2026-09-25: il dato NAAIM corrente è riservato agli abbonati; la
    # pagina pubblica lo mostra con 3 mesi di ritardo. I due CSV sotto non
    # esistono più: restano come tentativo, ma lo stato è dichiarato.
    sig["naaim_source"] = "NON_DISPONIBILE (NAAIM a pagamento, pubblico con 3 mesi di ritardo)"
    if sig["naaim_exposure"] is None:
      try:
        for url in [
            "https://www.naaim.org/wp-content/uploads/NAAIMExposureIndex.csv",
            "https://naaim.org/wp-content/uploads/NAAIMExposureIndex.csv",
        ]:
            try:
                r = requests.get(url, timeout=15,
                                 headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and len(r.text) > 100:
                    import io as _io
                    df_n = pd.read_csv(_io.StringIO(r.text))
                    for col in df_n.columns:
                        if any(k in col.lower() for k in ["naaim", "exposure", "number"]):
                            vals = df_n[col].dropna()
                            if len(vals) > 0:
                                sig["naaim_exposure"] = round(float(vals.iloc[-1]), 1)
                            break
                    if sig["naaim_exposure"] is not None:
                        logger.info(f"[sent] NAAIM reale da naaim.org: {sig['naaim_exposure']}")
                        break
                else:
                    # FIX 2026-07-14: stesso buco del blocco Nasdaq Data Link
                    # sopra — risposta HTTP non-200 (o corpo troppo corto)
                    # veniva ignorata in silenzio, senza log.
                    logger.info(f"[sent] NAAIM naaim.org ({url}): HTTP {r.status_code}, "
                               f"{len(r.text)} byte")
            except Exception as _e_naaim_url:
                logger.info(f"[sent] NAAIM naaim.org ({url}): eccezione {_e_naaim_url}")
                continue
      except Exception as e:
        logger.warning(f"[sent] NAAIM: {e}")

    # GPR Index (Geopolitical Risk)
    sig["gpr_index"] = None
    GPR_URLS = [
        ("csv",  "https://www.policyuncertainty.com/media/gpr_web_latest.csv"),
        ("xls",  "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"),
        ("xlsx", "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xlsx"),
    ]
    for fmt, url in GPR_URLS:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            import io as _io
            df_g = (pd.read_csv(_io.StringIO(r.text)) if fmt == "csv"
                    else pd.read_excel(_io.BytesIO(r.content)))
            # AUDIT 2026-09-25: colonna scelta per NOME ESATTO "GPR" (indice
            # globale mensile Caldara-Iacoviello), non "la prima che contiene
            # gpr" (che poteva essere GPRT/GPRA/GPRH o un paese). Nessun
            # ripiego su "prima colonna numerica": meglio nessun dato che
            # un dato sbagliato. Data del mese registrata nell'audit.
            _cols = {str(c).strip().lower(): c for c in df_g.columns}
            col = _cols.get("gpr")
            if col is None:
                logger.warning(f"[sent] GPR {fmt}: colonna 'GPR' non trovata "
                               f"(colonne: {list(df_g.columns)[:12]}) — fonte saltata")
                continue
            _mcol = _cols.get("month") or _cols.get("date")
            _dfv = df_g.dropna(subset=[col])
            if len(_dfv):
                _v = float(_dfv[col].iloc[-1])
                if not (20.0 <= _v <= 800.0):
                    logger.warning(f"[sent] GPR {fmt}: valore {_v} fuori scala — scartato")
                    continue
                sig["gpr_index"] = round(_v, 2)
                _gd = None
                if _mcol is not None:
                    try:
                        _gd = pd.to_datetime(_dfv[_mcol].iloc[-1]).strftime("%Y-%m-%d")
                    except Exception:
                        _gd = None
                sig["gpr_month"] = _gd[:7] if _gd else None
                _note_ext("gpr_index", f"Caldara-Iacoviello GPR ({fmt})", _gd, max_age_days=75)
                break
        except Exception as e:
            logger.warning(f"[sent] GPR {fmt}: {e}")

    # EPU Index (Economic Policy Uncertainty)
    # AUDIT 2026-09-25: il valore letto dal file xlsx (ultima colonna numerica)
    # era FERMO a 139.91 da maggio 2026 (130 run identiche). Ora la fonte è
    # FRED, stessi autori (Baker-Bloom-Davis): mensile USEPUINDXM, con riserva
    # sulla media 30 giorni del giornaliero USEPUINDXD. Data registrata.
    sig["epu_index"] = None
    sig["epu_source"] = None
    if fred:
        try:
            _em = _fred_series_throttled(fred, "USEPUINDXM",
                observation_start=datetime.today() - timedelta(days=400))
            _em = _em.dropna() if _em is not None else None
            if _em is not None and len(_em) and (datetime.today() - pd.Timestamp(_em.index[-1])).days <= 100:
                sig["epu_index"] = round(float(_em.iloc[-1]), 2)
                sig["epu_source"] = f"FRED USEPUINDXM {_em.index[-1]:%Y-%m}"
            else:
                _ed = _fred_series_throttled(fred, "USEPUINDXD",
                    observation_start=datetime.today() - timedelta(days=60))
                _ed = _ed.dropna() if _ed is not None else None
                if _ed is not None and len(_ed) >= 10:
                    _ed30 = _ed[_ed.index >= _ed.index[-1] - pd.Timedelta(days=30)]
                    sig["epu_index"] = round(float(_ed30.mean()), 2)
                    sig["epu_source"] = f"FRED USEPUINDXD media 30gg al {_ed.index[-1]:%Y-%m-%d}"
        except Exception as e:
            logger.warning(f"[sent] EPU: {e}")

    logger.info(f"[sent] F&G={sig.get('fear_greed_score')} "
                f"P/C={sig.get('cboe_put_call')} NAAIM={sig.get('naaim_exposure')} "
                f"GPR={sig.get('gpr_index')} EPU={sig.get('epu_index')}")
    return sig


def compute_vix_gpr_divergence(vix: float | None, gpr: float | None) -> dict:
    """
    GAP-1 (stress test 1929-2026): Alert quando VIX è basso ma GPR è alto.
    Pattern verificato in 3 scenari storici: 2001 (11/9 su bear market),
    2008 (Lehman con VIX 17 pre-crash), 2026 (guerra Iran VIX=17 GPR=230).

    Restituisce:
        {
          "level":              str,   # CRITICA / ATTENZIONE / NORMALE
          "divergence_active":  bool,
          "vix":                float,
          "gpr":                float,
          "message":            str
        }
    """
    if vix is None or gpr is None:
        return {"level": "DATI_MANCANTI", "divergence_active": False,
                "vix": vix, "gpr": gpr, "message": "VIX o GPR non disponibili."}

    vix_calm   = vix  < 20.0
    gpr_high   = gpr  > 150.0
    gpr_crisis = gpr  > 200.0

    if vix_calm and gpr_crisis:
        level   = "CRITICA"
        message = (f"VIX={vix:.1f} (calmo) con GPR={gpr:.0f} (crisi). "
                   f"Tail risk geopolitico NON prezzato dall'equity. "
                   f"Pattern storico: 2001 (11/9), 2008 (pre-Lehman), 2026 (Hormuz). "
                   f"Considerare aumento Long Vol — VIX basso = opzioni accessibili.")
    elif vix_calm and gpr_high:
        level   = "ATTENZIONE"
        message = (f"VIX={vix:.1f} con GPR={gpr:.0f}. "
                   f"Mercato azionario ignora il rischio geopolitico. "
                   f"Monitorare escalation nelle prossime 48-72h.")
    else:
        level   = "NORMALE"
        message = "Nessuna divergenza significativa VIX/GPR."

    return {
        "level":             level,
        "divergence_active": vix_calm and gpr_high,
        "vix":               vix,
        "gpr":               gpr,
        "message":           message,
    }


def compute_synthetic_proxies(vol: dict, rates: dict, sentiment: dict) -> dict:
    """
    Proxy sintetici per F&G, GPR, NAAIM quando le fonti reali sono bloccate.
    Qualità: F&G ~0.70 · GPR ~0.60 · NAAIM ~0.55
    Etichettati [PROXY:X] nell'artemis_status.json.
    """
    proxies = {
        "fear_greed_proxy": None,
        "gpr_proxy":        None,
        "naaim_proxy":      None,
        "proxy_quality":    "SYNTHETIC",
    }
    try:
        vix     = vol.get("vix") or 20.0
        skew    = vol.get("skew") or 110.0
        vix_sl  = vol.get("vix_term_slope") or 0.0
        # NOTA (fix 2026-07): "pct_above_ma200" NON è ampiezza di mercato
        # reale (% di titoli S&P 500 sopra la propria MA200) — è solo se
        # SPY (l'indice) è sopra la SUA PROPRIA media a 200gg negli ultimi
        # 20 giorni. In un bull market secolare resta vicino a 100 per mesi,
        # quindi è quasi sempre saturo e contribuisce "greed" massimo anche
        # in giornate di risk-off marcato (verificato 13/7: Nasdaq -1.55%
        # per escalation Iran, eppure pct_ma=100 invariato). Peso ridotto
        # di conseguenza (20%→10%) finché non esiste un sostituto con vera
        # ampiezza cross-sezionale.
        pct_ma  = rates.get("pct_above_ma200") or 50.0
        # FIX 2026-07: stessa esclusione già applicata in compute_smart_sentiment
        # per il P/C con fonte PROXY_YF (range strutturale distorto, vedi FIX F4).
        # Senza questo, un dato che il sistema stesso giudica inaffidabile per
        # UN calcolo entrava comunque, senza filtro, in questo secondo calcolo.
        cboe_pc_raw = sentiment.get("cboe_put_call")
        cboe_pc_src = sentiment.get("cboe_put_call_source")
        cboe_pc = cboe_pc_raw if (cboe_pc_raw is not None and cboe_pc_src == "CBOE_OFFICIAL") else 1.0
        ovx     = vol.get("ovx") or 35.0
        gvz     = vol.get("gvz") or 15.0
        epu     = sentiment.get("epu_index") or 100.0

        # F&G sintetico (0-100) — pesi ribilanciati: bread_s 20%→10% (quasi
        # sempre saturo, poco informativo), skew_s 15%→25% (unico segnale
        # realmente sensibile a un singolo giorno di stress/paura)
        vix_s    = max(0, min(100, 100 - (vix - 10) * (100/70)))
        pc_norm  = min(cboe_pc, 4.0)
        pc_s     = max(0, min(100, 100 - (pc_norm - 0.8) * (100/3.2)))
        bread_s  = max(0, min(100, (pct_ma - 20) * (100/60)))
        skew_s   = max(0, min(100, 100 - (skew - 100) * (100/50)))
        slope_s  = max(0, min(100, 50 + vix_sl * 200))
        # AUDIT 2026-09-25: senza P/C ufficiale il valore era una COSTANTE
        # (1.0 → pc_s≈94) che pesava il 30%: spingeva il F&G sintetico verso
        # "greed" ogni giorno. Ora il P/C entra solo se ufficiale; altrimenti
        # i pesi restanti vengono riproporzionati.
        _pc_ok = cboe_pc_raw is not None and cboe_pc_src == "CBOE_OFFICIAL"
        if _pc_ok:
            proxies["fear_greed_proxy"] = round(
                0.25*vix_s + 0.30*pc_s + 0.10*bread_s + 0.25*skew_s + 0.10*slope_s, 1)
        else:
            proxies["fear_greed_proxy"] = round(
                (0.25*vix_s + 0.10*bread_s + 0.25*skew_s + 0.10*slope_s) / 0.70, 1)

        # GPR proxy (baseline ~100)
        epu_norm  = min(max(epu / 100.0, 0.5), 5.0)
        ovx_norm  = min(max((ovx - 20) / 100.0, 0.0), 2.0)
        gvz_norm  = min(max((gvz - 10) / 50.0,  0.0), 2.0)
        proxies["gpr_proxy"] = round(
            max(50, min(400, 100 * (0.60*epu_norm + 0.25*ovx_norm + 0.15*gvz_norm))), 1)

        # NAAIM proxy (0-100)
        if _pc_ok:
            naaim_raw = (bread_s * 0.50 + (100 - vix_s) * 0.30 +
                         max(0, min(100, (1.5 - pc_norm) / 1.5 * 100)) * 0.20)
        else:
            # NB (audit): (100 - vix_s) = più VIX → più esposizione? Formula
            # originale mantenuta, solo rimossa la componente P/C costante.
            naaim_raw = (bread_s * 0.50 + (100 - vix_s) * 0.30) / 0.80
        proxies["naaim_proxy"] = round(max(0, min(100, naaim_raw)), 1)

    except Exception as e:
        logger.warning(f"[proxies] calc failed: {e}")
    return proxies


def compute_smart_sentiment(sent: dict, rates: dict, vol: dict) -> dict:
    """Score composito sentiment (0-100): Extreme Sell → Extreme Buy."""
    comps, weights = [], []

    fg = sent.get("fear_greed_score") or sent.get("fear_greed_synthetic")
    if fg is not None:
        comps.append(float(fg)); weights.append(0.30)

    pc = sent.get("cboe_put_call")
    pc_source = sent.get("cboe_put_call_source")
    # FIX F4 — Escludi P/C dal composito quando è proxy yfinance.
    # Il proxy ATM±5% vive strutturalmente in 1.5-3.0 (vs range reale 0.4-1.3):
    # con pc=2.38 il componente vale 0 invece di ~100 → sposta lo score di 20 punti
    # e può invertire il segnale da "Buy" a "Neutral". Meglio un input in meno
    # che un input avvelenato. Quando la fonte è CBOE_OFFICIAL il dato è affidabile.
    if pc is not None and pc_source == "CBOE_OFFICIAL":
        comps.append(max(0, min(100, (2.0 - float(pc)) / 1.5 * 100)))
        weights.append(0.20)
    elif pc is not None and pc_source == "PROXY_YF":
        logger.info(f"[smart_sent] P/C={pc} escluso dal composito (fonte PROXY_YF — range strutturale distorto)")

    naaim = sent.get("naaim_exposure") or sent.get("naaim_synthetic")
    if naaim is not None:
        comps.append(min(100, float(naaim) / 2.0)); weights.append(0.20)

    umc = sent.get("umcsent")
    if umc is not None:
        comps.append(min(100, float(umc) / 1.1)); weights.append(0.15)

    pct_ma = rates.get("pct_above_ma200")
    if pct_ma is not None:
        comps.append(float(pct_ma)); weights.append(0.15)

    if not comps:
        return {"score": None, "label": "INDETERMINATO", "inputs_count": 0}

    total_w = sum(weights)
    score   = sum(c * w for c, w in zip(comps, weights)) / total_w
    label   = ("Extreme Sell" if score < 20 else "Sell" if score < 40 else
               "Neutral" if score < 60 else "Buy" if score < 80 else "Extreme Buy")
    return {"score": round(score, 1), "label": label, "inputs_count": len(comps)}


# ────────────────────────────────────────────────────────────────────────────
# 4.5 EIA STACK
# ────────────────────────────────────────────────────────────────────────────

def collect_eia_stack() -> dict:
    """Inventari greggio USA via EIA API v2 (con fallback v1)."""
    sig = {
        "eia_crude_inventories_wk": None,
        "eia_crude_roc_12w":        None,
        "eia_haircut":              False,
    }
    if not EIA_KEY:
        sig["eia_haircut"] = True
        return sig

    eia_urls = [
        (f"https://api.eia.gov/v2/petroleum/sum/sndw/data/"
         f"?api_key={EIA_KEY}&frequency=weekly&data[0]=value"
         f"&facets[series][]=WCESTUS1&sort[0][column]=period"
         f"&sort[0][direction]=desc&length=16"),
        (f"https://api.eia.gov/series/?api_key={EIA_KEY}"
         f"&series_id=WCESTUS1&out=json"),
    ]
    values = []
    for url in eia_urls:
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                body = r.json()
                items = body.get("response", {}).get("data", [])
                if not items:
                    for row in (body.get("series", [{}])[0].get("data", []))[:16]:
                        if row and len(row) >= 2 and row[1] is not None:
                            try:
                                values.append(float(row[1]))
                            except (ValueError, TypeError):
                                pass
                else:
                    if items and items[0].get("period"):
                        sig["eia_period"] = str(items[0].get("period"))[:10]
                        _note_ext("eia_crude_inventories_wk", "EIA API v2 WCESTUS1",
                                  sig["eia_period"], max_age_days=12)
                    for item in items:
                        v = item.get("value") or item.get("VALUE")
                        if v is not None:
                            try:
                                values.append(float(v))
                            except (ValueError, TypeError):
                                pass
                if values:
                    break
        except Exception as e:
            logger.warning(f"[eia] URL failed: {e}")

    if values:
        sig["eia_crude_inventories_wk"] = float(values[0])
        if len(values) >= 13:
            sig["eia_crude_roc_12w"] = round(
                (values[0] - values[12]) / abs(values[12]), 4)
    else:
        sig["eia_haircut"] = True

    logger.info(f"[eia] Crude={sig.get('eia_crude_inventories_wk')} "
                f"ROC12w={sig.get('eia_crude_roc_12w')}")
    return sig


# ────────────────────────────────────────────────────────────────────────────
# 4.6 SOCMINT LITE
# ────────────────────────────────────────────────────────────────────────────

def collect_socmint_lite() -> dict:
    """Short Interest + Volume anomalo su SOCMINT_WATCHLIST."""
    alerts   = []
    raw_data = {}

    for ticker in SOCMINT_WATCHLIST:
        try:
            tk   = yf.Ticker(ticker)
            info = tk.info or {}

            short_pct = info.get("shortPercentOfFloat")
            if short_pct and short_pct < 2:
                short_pct = short_pct * 100

            avg_vol   = info.get("averageVolume")
            cur_vol   = info.get("volume")
            vol_ratio = (round(cur_vol / avg_vol, 2)
                         if avg_vol and cur_vol and avg_vol > 0 else None)

            raw_data[ticker] = {
                "short_pct": round(short_pct, 2) if short_pct else None,
                "vol_ratio": vol_ratio,
                "price":     info.get("currentPrice") or info.get("regularMarketPrice"),
            }

            if (short_pct and short_pct > SHORT_INTEREST_ALERT and
                    vol_ratio and vol_ratio > VOLUME_SPIKE_FACTOR):
                alerts.append({
                    "ticker":    ticker,
                    "short_pct": round(short_pct, 1),
                    "vol_ratio": vol_ratio,
                    "alert":     "ARTEMIS_SQUEEZE_EARLY_WARNING",
                    "note":      "Asset tattico ad alta asimmetria — confermare con CODEX ARGUS",
                })
                logger.warning(f"[socmint] ⚠️ SQUEEZE: {ticker} "
                               f"SI={short_pct:.1f}% Vol={vol_ratio:.1f}×")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"[socmint] {ticker}: {e}")
            raw_data[ticker] = {"error": str(e)}

    return {"alerts": alerts, "raw": raw_data}


# ────────────────────────────────────────────────────────────────────────────
# 4.8 INSIDER INTELLIGENCE
# ────────────────────────────────────────────────────────────────────────────

def collect_insider_intel() -> dict:
    """Form4 cluster (openinsider.com) + Congress trades (quiverquant.com)."""
    sig = {
        "form4_cluster":  None,
        "congress_trade": None,
        "insider_flags":  [],
        "form4_status":   "NON_RAGGIUNTO",
        "congress_status": "OK",
    }

    # Form4 cluster — regex corretto per tabella HTML openinsider
    try:
        url = (
            "http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh="
            "&fd=14&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1"
            "&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999"
            "&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h="
            "&oc=&sortcol=0&cnt=100&action=1&type=P"
        )
        # AUDIT 2026-09-25: openinsider a volte non risponde a GitHub Actions →
        # 3 tentativi con attesa crescente; lo stato registra il motivo.
        r = None
        for _att in range(3):
            try:
                r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and r.text:
                    break
                sig["form4_status"] = f"NON_RAGGIUNTO (HTTP {r.status_code})"
            except Exception as _e_oi:
                sig["form4_status"] = f"NON_RAGGIUNTO ({type(_e_oi).__name__})"
            time.sleep(5 * (_att + 1))
        if r is None:
            raise RuntimeError(sig["form4_status"])
        if r.status_code == 200 and r.text:
            import re
            from collections import Counter
            # Regex primaria: celle tabella openinsider
            # AUDIT 2026-09-25 (ter): HTML reale verificato sulla pagina:
            #   <td><b> <a href="/CMPX" ...>CMPX</a></b></td> ... <a href="/insider/Nome/ID">
            # (spazio dopo <b>: le regex precedenti non trovavano nulla). Si legge
            # riga per riga ticker + insider; il cluster conta gli insider DISTINTI
            # (più acquisti dello stesso insider non sono un cluster).
            _ins_by_tk: dict = {}
            _rows_read = 0
            for _row in r.text.split("<tr")[1:]:
                _mt = re.search(r'<td[^>]*><b>\s*<a href="/([A-Z][A-Z.]{0,5})"', _row)
                if not _mt:
                    continue
                _rows_read += 1
                _mi = re.search(r'href="/insider/[^"]*?/(\d+)"', _row)
                _ins_by_tk.setdefault(_mt.group(1), set()).add(_mi.group(1) if _mi else f"row{_rows_read}")
            sig["form4_rows_read"] = _rows_read
            sig["form4_status"] = ("OK" if _rows_read else
                                   f"ERRORE_PARSING (0 righe lette su {len(r.text)} byte: HTML cambiato o pagina bloccata)")
            cluster = {t: len(v) for t, v in _ins_by_tk.items() if len(v) >= 3}
            if cluster:
                top = max(cluster, key=cluster.get)
                sig["form4_cluster"] = {
                    "signal":   "FORM4_CLUSTER",
                    "ticker":   top,
                    "insiders": cluster[top],
                    "sector":   "UNKNOWN",
                    "window":   "14d",
                    "source":   "openinsider.com",
                }
                sig["insider_flags"].append(
                    f"FORM4_CLUSTER: {top} × {cluster[top]} insider distinti in 14gg"
                )
                logger.info(f"[insider] FORM4_CLUSTER: {top} × {cluster[top]}")
            else:
                logger.info("[insider] FORM4: nessun cluster ≥3 rilevato")
        else:
            logger.warning(f"[insider] openinsider HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"[insider] Form4: {e}")
        if sig.get("form4_status") == "NON_RAGGIUNTO":
            sig["form4_status"] = f"NON_RAGGIUNTO ({type(e).__name__})"

    # Congress trades — endpoint API corretto (JSON strutturato, non HTML)
    try:
        url_cq = "https://api.quiverquant.com/beta/live/congresstrading"
        r_cq = requests.get(
            url_cq, timeout=15,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        )
        if r_cq.status_code == 200:
            trades = r_cq.json() if isinstance(r_cq.json(), list) else []
            # Filtra: solo acquisti · min $15,000
            buys = [
                t for t in trades
                if str(t.get("Transaction", "")).lower() in ("purchase", "buy")
                and float(str(t.get("Amount", "0")).replace(",", "").replace("$", "") or 0)
                >= 15000
            ]
            if buys:
                latest = buys[0]
                sig["congress_trade"] = {
                    "signal":    "CONGRESS_TRADE",
                    "ticker":    latest.get("Ticker", "UNKNOWN"),
                    "committee": latest.get("Committee", "UNKNOWN"),
                    "trade_usd": latest.get("Amount", "N/A"),
                    "date":      latest.get("TransactionDate", "N/A"),
                    "source":    "quiverquant.com",
                }
                sig["insider_flags"].append(
                    f"CONGRESS_TRADE: {latest.get('Ticker')} "
                    f"({latest.get('Committee', 'N/A')})"
                )
                logger.info(f"[insider] CONGRESS_TRADE: {sig['congress_trade']['ticker']}")
            else:
                logger.info("[insider] CONGRESS: nessun acquisto qualificato")
        elif r_cq.status_code in (401, 403):
            sig["congress_status"] = f"NON_DISPONIBILE (quiverquant HTTP {r_cq.status_code}: serve token)"
            logger.info("[insider] quiverquant: endpoint free non più accessibile "
                        f"({r_cq.status_code}) — segnale Congress disabilitato")
        else:
            logger.warning(f"[insider] quiverquant HTTP {r_cq.status_code}")
    except Exception as e:
        logger.warning(f"[insider] Congress: {e}")

    logger.info(f"[insider] Form4={sig['form4_cluster'] is not None} "
                f"Congress={sig['congress_trade'] is not None} "
                f"Flags={len(sig['insider_flags'])}")
    return sig


# ────────────────────────────────────────────────────────────────────────────
# 4.9 LONG VOLATILITY MONITOR  ← NUOVO STACK ARTEMIS
# ────────────────────────────────────────────────────────────────────────────

def collect_long_vol_monitor(vol: dict) -> dict:
    """
    Stack Long Volatility Monitor — specifico per il framework Artemis.

    Misura lo stato attuale delle strategie Long Volatility nel portafoglio:
    - Costo del carry corrente (quanto "sanguina" il portafoglio long vol)
    - Term structure VIX (contango → carry negativo / backwardation → carry positivo)
    - Realized vs Implied vol ratio (vol cara vs economica)
    - CBOE Long Vol proxy performance (via VXZ come approssimazione HTF)

    Ref: Cole "Dennis Rodman" — Long Vol perde in bull market ma è essenziale.
         La perdita del carry è il "costo dell'assicurazione".
    """
    sig = {
        "vix_term_slope":         vol.get("vix_term_slope"),
        "term_structure_signal":  None,
        "realized_implied_ratio": None,
        "long_vol_carry_regime":  None,
        "vxz_price":              None,   # VXZ = iPath S&P 500 VIX Mid-Term Futures
        "vxz_3m_return":          None,
        "artemis_vol_regime":     None,
        "cole_insight":           None,
    }

    # Term Structure Signal
    slope = vol.get("vix_term_slope")
    if slope is not None:
        if slope < VIX_TERM_BACKWARDATION:
            sig["term_structure_signal"] = "BACKWARDATION_ACUTA"
            sig["long_vol_carry_regime"] = "POSITIVO — long vol guadagna da carry"
            sig["cole_insight"] = ("Backwardation: la struttura temporale della "
                                   "volatilità si è invertita. Long Vol in carry "
                                   "positivo — momento ottimale per aumentare "
                                   "l'esposizione.")
        elif slope < VIX_TERM_CONTANGO_MIN:
            sig["term_structure_signal"] = "CONTANGO_PIATTO"
            sig["long_vol_carry_regime"] = "NEUTRO — carry basso"
            sig["cole_insight"] = ("Term structure piatta. Carry cost ridotto. "
                                   "Regime di transizione — mantenere allocazione "
                                   "Dragon target.")
        else:
            sig["term_structure_signal"] = "CONTANGO_NORMALE"
            sig["long_vol_carry_regime"] = "NEGATIVO — carry erode il valore (normale in bull)"
            sig["cole_insight"] = ("Contango normale in bull market. Long Vol "
                                   "sanguina lentamente — conferma perché il Profilo 2 "
                                   "NON tiene long vol permanente (protezione via "
                                   "CTA + oro). Rilevante solo per l'eventuale overlay "
                                   "opzioni tattico. Ref: Cole 'Dennis Rodman'.")

    # Realized vs Implied Vol ratio
    try:
        spy_hist = _robust_yf("SPY", period="3mo")
        if not spy_hist.empty and "Close" in spy_hist.columns:
            returns = spy_hist["Close"].pct_change().dropna()
            if len(returns) >= 21:
                realized_21d = float(returns.iloc[-21:].std() * math.sqrt(252) * 100)
                vix_current  = vol.get("vix") or 20.0
                sig["realized_implied_ratio"] = round(realized_21d / vix_current, 3)
                if sig["realized_implied_ratio"] < REALVOL_IMPLIED_RATIO:
                    sig["artemis_vol_regime"] = (
                        "VOL_CARA — implied > realized. "
                        "Acquisto long vol costoso. "
                        "Considerare overlay su posizioni esistenti.")
                else:
                    sig["artemis_vol_regime"] = (
                        "VOL_ECONOMICA — realized >= implied. "
                        "Momento favorevole per aggiungere long vol.")
    except Exception as e:
        logger.warning(f"[long_vol] Realized vol: {e}")

    # VXZ come proxy HTF long volatility (mid-term futures — meno soggetto a roll)
    try:
        vxz = _safe_yf("VXZ", period="5d")
        sig["vxz_price"] = vxz
        if vxz is not None:
            vxz_hist = _robust_yf("VXZ", period="6mo")
            if not vxz_hist.empty and "Close" in vxz_hist.columns:
                closes = vxz_hist["Close"].dropna()
                if len(closes) >= 63:
                    sig["vxz_3m_return"] = round(
                        (float(closes.iloc[-1]) / float(closes.iloc[-63]) - 1) * 100, 2)
    except Exception as e:
        logger.warning(f"[long_vol] VXZ: {e}")

    logger.info(f"[long_vol] slope={sig.get('vix_term_slope')} "
                f"signal={sig.get('term_structure_signal')} "
                f"rv/iv={sig.get('realized_implied_ratio')} "
                f"carry={sig.get('long_vol_carry_regime')}")
    return sig


# ────────────────────────────────────────────────────────────────────────────
# 4.10 CWARP CALCULATOR  ← NUOVO STACK ARTEMIS
# ────────────────────────────────────────────────────────────────────────────

def compute_cwarp(portfolio_current: dict | None = None,
                  candidates: list | None = None) -> dict:
    """
    CWARP™ — Cole Wins Above Replacement Portfolio.

    Formula (Cole, Moneyball for Modern Portfolio Theory, 2020):
        CWARP = [√(Sn/Sp × RMDDn/RMDDp) - 1] × 100

    dove:
        Sn    = Sortino Ratio del nuovo portafoglio (con candidato al 25%)
        Sp    = Sortino Ratio del portafoglio di sostituzione (replacement)
        RMDDn = Return to Max Drawdown del nuovo portafoglio
        RMDDp = Return to Max Drawdown del portafoglio di sostituzione

    Assunzioni standard Cole:
        - Replacement portfolio = S&P 500 (SPY) o 60/40 Equity-Bond
        - Overlay weight = 25% (asset finanziato a leva)
        - RF Rate = 1-month T-bill (FRED TB3MS)
        - Financing = RF + 30bps
        - Periodicity = mensile

    Qualità CWARP:
        FULL    — dati storici reali disponibili (≥ 36 mesi)
        PROXY   — storico < 36 mesi o proxy ETF usato
        ESTIMATED — solo correlazione + skew stimati

    Ref: https://github.com/jpartemis/cwarp (Python code Cole)
    """
    result = {
        "cwarp_available":   False,
        "cwarp_quality":     "ESTIMATED",
        "replacement_port":  "SPY",     # S&P 500 come baseline (Equity Beta)
        "rf_rate":           None,
        "candidates":        {},
        "ranking":           [],
        "cole_note":         ("CWARP > 0 → asset migliora il portafoglio. "
                              "CWARP < 0 → asset replica esposizioni esistenti "
                              "o aumenta drawdowns. "
                              "Ref: Cole CWARP paper 2020."),
    }

    # Fetch RF rate — AUDIT 2026-09-25: DTB3 giornaliero (T-bill 3 mesi) invece
    # della media mensile TB3MS, che resta indietro di un mese (non vedeva il
    # rialzo Fed del 16/9). TB3MS resta come riserva.
    # AUDIT 2026-09-25 (CWARP): tutti i rendimenti sono ora in EURO (sei un
    # investitore in euro e i candidati quotano in EUR, USD e GBp) → il tasso
    # privo di rischio coerente è quello in euro (€STR). Riserva: T-bill USA.
    rf_annual = 0.05  # default 5% se FRED non disponibile
    result["rf_source"] = "default 5%"
    if FRED_KEY:
        try:
            _fred_tmp = Fred(api_key=FRED_KEY)
            for _sid, _lb in (("ECBESTRVOLWGTTRMDMNRT", 15), ("DTB3", 15), ("TB3MS", 60)):
                tb3 = _safe_fred(_sid, _fred_tmp, lookback_days=_lb)
                if tb3 is not None:
                    rf_annual = float(tb3) / 100.0
                    result["rf_rate"] = round(rf_annual, 4)
                    result["rf_source"] = _sid
                    break
        except Exception as e:
            logger.warning(f"[cwarp] RF rate: {e}")

    # Conversione annuo → mensile geometrica, come nel codice di Cole
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1

    # Asset candidati di default se non specificati
    # Questi sono i proxy ETF/index per le 5 asset class Dragon
    if candidates is None:
        candidates = [
            {"id": "SPY",  "name": "US Equity (SPY)", "component": "equity"},
            {"id": "TLT",  "name": "US Treasury Bond 20y+ (TLT)", "component": "fixed_income"},
            {"id": "GLD",  "name": "Gold ETF (GLD)", "component": "gold"},
            {"id": "VXZ",  "name": "VIX Mid-Term Futures (VXZ)", "component": "long_volatility"},
            {"id": "PDBC", "name": "Commodities Diversified (PDBC)", "component": "commodity_trend"},
            {"id": "DBMF", "name": "Managed Futures (DBMF)", "component": "commodity_trend"},
            {"id": "BTAL", "name": "Anti-Beta (BTAL)", "component": "long_volatility"},
        ]

    # Replacement portfolio = SPY (Equity Beta — standard Cole)
    spy_monthly = _fetch_monthly_returns_eur("SPY", months=60)
    result["valuta_base"] = "EUR"

    if spy_monthly is None or len(spy_monthly) < 12:
        result["cwarp_available"] = False
        result["cwarp_quality"]   = "ESTIMATED"
        result["error"]           = "Dati SPY insufficienti per calcolo CWARP"
        logger.warning("[cwarp] SPY storico insufficiente — CWARP non calcolabile")
        return result

    sp_sortino = _sortino_ratio(spy_monthly.to_numpy(), rf_monthly)
    sp_rmdd    = _return_to_maxdd(spy_monthly.to_numpy(), rf_annual)

    if sp_sortino is None or sp_rmdd is None:
        result["cwarp_available"] = False
        result["error"]           = "Sortino/RMDD replacement non calcolabili"
        return result

    result["cwarp_available"] = True

    # Calcola CWARP per ogni candidato
    overlay_weight = 0.25   # 25% overlay standard Cole
    financing_rate = (1 + rf_annual + 0.0030) ** (1 / 12) - 1  # RF + 30bps, mensile geometrico

    cwarp_scores = []
    for cand in candidates:
        ticker    = cand["id"]
        cand_name = cand.get("name", ticker)
        component = cand.get("component", "unknown")

        cand_monthly = _fetch_monthly_returns_eur(ticker, months=60)
        quality = "PROXY"
        if cand_monthly is None or len(cand_monthly) < 12:
            # Fallback: stima di CATEGORIA (non dato reale del fondo).
            # Fix 2026-09-25: registra i mesi REALMENTE disponibili (non più 0 fisso)
            # per distinguere "ticker non trovato" da "storico troppo corto".
            # Il candidato stimato NON entra nel ranking né nel gate come se
            # fosse misurato (principio ARTEMIS: il CWARP non si inventa).
            cwarp_est = _cwarp_estimated(ticker, component)
            _mesi = _count_monthly_closes(ticker)
            if _mesi == 0:
                _note = ("TICKER NON TROVATO su Yahoo o nessun dato — verificare il "
                         "ticker. Valore = stima di categoria, NON dato reale.")
                logger.warning(f"[cwarp] {ticker}: nessun dato Yahoo — ticker da verificare")
            else:
                _note = (f"CWARP non calcolabile: {_mesi} mesi di storico (< 12). "
                         f"Valore = stima di categoria, NON dato reale del fondo. "
                         f"Diventa calcolato in automatico a 12 mesi di dati.")
                logger.info(f"[cwarp] {ticker}: {_mesi} mesi di storico — CWARP solo stimato")
            cwarp_scores.append({
                "ticker":       ticker,
                "name":         cand_name,
                "component":    component,
                "cwarp":        cwarp_est,
                "quality":      "ESTIMATED",
                "months":       _mesi,
                "ticker_found": _mesi > 0,
                "note":         _note,
            })
            continue

        quality = "FULL" if len(cand_monthly) >= 36 else "PROXY"

        # Portafoglio nuovo = 75% SPY + 25% candidato (a leva)
        # FIX F1 — Window matching: sp calcolato sulla STESSA finestra del candidato
        # (non sull'intera storia SPY). Evita che CWARP misuri la differenza tra
        # due periodi di mercato invece del contributo del candidato.
        # Allineamento per MESE di calendario (non per posizione): le serie
        # hanno date e buchi diversi.
        _al = pd.concat([spy_monthly, cand_monthly], axis=1, join="inner").dropna()
        min_len = len(_al)
        spy_sub  = _al.iloc[:, 0].to_numpy()
        cand_sub = _al.iloc[:, 1].to_numpy()

        # Replacement ricalcolato sulla finestra del candidato
        sp_sortino_w = _sortino_ratio(spy_sub, rf_monthly)
        sp_rmdd_w    = _return_to_maxdd(spy_sub, rf_annual)

        # AUDIT 2026-09-25: formula di Cole = OVERLAY. Il portafoglio di
        # sostituzione resta al 100% e il candidato si AGGIUNGE al 25% finanziato
        # (cwarp_defs.py: new_port = (asset − fin)×0.25 + replace×1.0).
        # Prima era un mix 75% SPY + 25% candidato: toglieva equity e penalizzava
        # sistematicamente i diversificatori a basso rendimento (bond, oro).
        new_returns = spy_sub * 1.0 + overlay_weight * (cand_sub - financing_rate)

        sn = _sortino_ratio(new_returns, rf_monthly)
        rn = _return_to_maxdd(new_returns, rf_annual)

        if (sn is None or rn is None or
                sp_sortino_w is None or sp_rmdd_w is None or
                sp_sortino_w == 0 or sp_rmdd_w == 0):
            cwarp_scores.append({
                "ticker":    ticker,
                "name":      cand_name,
                "component": component,
                "cwarp":     None,
                "quality":   quality,
                "months":    len(cand_monthly),
                "note":      "Calcolo impossibile — denominatore zero o dati insufficienti",
            })
            continue

        try:
            # FIX F3 — Guard segni prima della radice quadrata
            # La formula sqrt((Sn/Sp) * (Rn/Rp)) è definita solo se il prodotto è >= 0.
            # Casi patologici:
            #   segni misti (es. Sn<0, Sp>0) → prodotto negativo → sqrt invalida
            #   doppio negativo (Sn<0, Sp<0) → prodotto positivo ma SENSO INVERTITO
            #     (un portafoglio peggiore appare migliore)
            # Entrambi i casi si verificano in bear market prolungati — esattamente
            # quando il Dragon serve di più. Soluzione: se Sp o Rp <= 0, o se il
            # prodotto risulta negativo, CWARP è non calcolabile in quel contesto.
            prod = (sn / sp_sortino_w) * (rn / sp_rmdd_w)
            if prod < 0:
                cwarp = None
                cwarp_note = ("NON_CALCOLABILE — prodotto Sortino/RMDD negativo "
                              "(bear market: usa ARGUS per valutazione qualitativa)")
            else:
                cwarp = round((math.sqrt(prod) - 1) * 100, 2)
                cwarp_note = None
        except Exception:
            cwarp = None
            cwarp_note = "Eccezione nel calcolo CWARP"

        interpretation = (
            "ECCELLENTE diversificatore Dragon" if cwarp is not None and cwarp >= CWARP_EXCELLENT else
            "FORTE diversificatore Dragon"      if cwarp is not None and cwarp >= CWARP_STRONG    else
            "Migliora il portafoglio"           if cwarp is not None and cwarp > 0               else
            "NEUTRO"                            if cwarp == 0                                    else
            "DANNEGGIA il portafoglio — NON aggiungere"
        )

        # ── Exit trigger CWARP — SOLO WARNING, mai azione automatica ────────
        # Confronto numerico puro: se il CWARP corrente scende sotto la soglia
        # che hai impostato manualmente su questa riga (exit_trigger_cwarp),
        # E la posizione è realmente posseduta (is_in_portfolio=TRUE), il
        # sistema segnala — non vende, non modifica nulla. Decidi sempre tu.
        exit_alert = None
        _trigger_cwarp = cand.get("exit_trigger_cwarp")
        _is_owned = cand.get("is_in_portfolio", False)
        if _is_owned and _trigger_cwarp is not None and cwarp is not None and cwarp < _trigger_cwarp:
            exit_alert = (f"⚠️ EXIT TRIGGER: CWARP={cwarp:.1f} < soglia {_trigger_cwarp:.1f} "
                          f"impostata su questa posizione — valutare uscita")
            logger.warning(f"[cwarp] {ticker}: {exit_alert}")

        # exit_trigger_regime: esposto come campo grezzo, NON automatizzato.
        # Confrontare testo libero (es. 'FALCO_DX_EMERGENTE') con il regime
        # corrente rischia falsi positivi/negativi per piccole differenze di
        # stringa. Il Gestore legge questo campo e valuta lui stesso il match
        # contro regime.primary o i_z_momentum.momentum_signal nel JSON.
        _trigger_regime = cand.get("exit_trigger_regime")

        cwarp_scores.append({
            "ticker":          ticker,
            "name":            cand_name,
            "component":       component,
            "cwarp":           cwarp,
            "note":            cwarp_note,
            "sortino_new":     round(sn, 3) if sn else None,
            "sortino_replace": round(sp_sortino_w, 3),   # finestra del candidato
            "rmdd_new":        round(rn, 3) if rn else None,
            "rmdd_replace":    round(sp_rmdd_w, 3),       # finestra del candidato
            "quality":         quality,
            "months":          min_len,                   # finestra effettiva usata
            "interpretation":  interpretation,
            "exit_trigger_cwarp_threshold": _trigger_cwarp,   # soglia impostata (None se non definita)
            "exit_alert":      exit_alert,                     # warning attivo o None
            "exit_trigger_regime_note": _trigger_regime,       # testo grezzo, valutazione manuale
        })

    # Ranking per CWARP decrescente
    cwarp_scores.sort(
        key=lambda x: (x.get("cwarp") or -999),
        reverse=True
    )

    result["candidates"] = {c["ticker"]: c for c in cwarp_scores}
    # Ranking e qualità complessiva SOLO sui CWARP misurati: le stime di
    # categoria (ESTIMATED) restano visibili in "candidates" e in
    # "estimated_excluded", ma non scalano la classifica (fix 2026-09-25).
    _misurati = [c for c in cwarp_scores if c.get("quality") != "ESTIMATED"]
    result["ranking"]    = [c["ticker"] for c in _misurati if c.get("cwarp") is not None]
    result["estimated_excluded"] = [c["ticker"] for c in cwarp_scores
                                    if c.get("quality") == "ESTIMATED"]
    result["cwarp_quality"] = (
        "ESTIMATED" if not _misurati else
        "FULL"      if all(c.get("quality") == "FULL" for c in _misurati) else
        "PROXY"
    )

    logger.info(f"[cwarp] Calcolato per {len(cwarp_scores)} asset · "
                f"quality={result['cwarp_quality']} · "
                f"top={result['ranking'][0] if result['ranking'] else 'N/A'}")
    return result


def _count_monthly_closes(ticker: str, months: int = 60) -> int:
    """Numero di chiusure mensili disponibili su Yahoo (0 = ticker non trovato
    o nessun dato). Usata solo per i candidati con storico < 12 mesi, per
    distinguere un ticker errato da un fondo semplicemente giovane."""
    try:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=int((months + 6) * 30.5))
        hist = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1mo",
            progress=False, auto_adjust=True
        )
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist.empty or "Close" not in hist.columns:
            return 0
        return int(hist["Close"].dropna().shape[0])
    except Exception as e:
        logger.info(f"[cwarp] conteggio mesi {ticker} fallito: {e}")
        return 0


_CCY_CACHE: dict = {}
_FX_MONTHLY_CACHE: dict = {}


def _ticker_currency(ticker: str) -> str | None:
    """Valuta di quotazione letta da Yahoo (stessa logica verificata in
    diagnose_price_conversion: sulla LSE convivono GBp e USD)."""
    if ticker in _CCY_CACHE:
        return _CCY_CACHE[ticker]
    ccy = None
    try:
        _fi = yf.Ticker(ticker).fast_info
        ccy = _fi.get("currency") if hasattr(_fi, "get") else None
    except Exception:
        pass
    if not ccy:
        try:
            ccy = yf.Ticker(ticker).info.get("currency")
        except Exception:
            ccy = None
    if not ccy:
        # Riserva se Yahoo non risponde: valute verificate in produzione
        # (diagnose_price_conversion, 15/9/2026) + borse euro per suffisso.
        _known = {"SPY": "USD", "SGLN.L": "GBp", "FWRA.L": "USD", "ENCO.L": "USD",
                  "DTLA.L": "USD"}
        ccy = _known.get(ticker) or ("EUR" if ticker.endswith((".DE", ".AS", ".PA", ".MI", ".F")) else None)
        if ccy:
            logger.info(f"[cwarp] {ticker}: valuta da riserva interna ({ccy}) — Yahoo non ha risposto")
    _CCY_CACHE[ticker] = str(ccy).strip() if ccy else None
    return _CCY_CACHE[ticker]


def _monthly_closes(ticker: str, months: int) -> pd.Series | None:
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=int((months + 6) * 30.5))
    hist = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                       end=end_date.strftime("%Y-%m-%d"), interval="1mo",
                       progress=False, auto_adjust=True)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    c = hist["Close"].dropna()
    c.index = pd.to_datetime(c.index).to_period("M")
    return c[~c.index.duplicated(keep="last")]


def _fetch_monthly_returns_eur(ticker: str, months: int = 60) -> pd.Series | None:
    """Rendimenti mensili convertiti in EURO (indice = mese).
    AUDIT 2026-09-25: prima SPY (USD) veniva confrontato con VWCE (EUR), SGLN
    (GBp), DTLA (USD) ognuno nella sua valuta → il CWARP mescolava l'effetto
    cambio con quello dell'asset. Ora tutto è misurato in euro."""
    try:
        closes = _monthly_closes(ticker, months)
        if closes is None or len(closes) < 13:
            return None
        ccy = _ticker_currency(ticker)
        if ccy in ("USD", "GBP", "GBp"):
            fx_tk = "EURUSD=X" if ccy == "USD" else "GBPEUR=X"
            fx = _FX_MONTHLY_CACHE.get(fx_tk)
            if fx is None:
                fx = _monthly_closes(fx_tk, months + 12)
                _FX_MONTHLY_CACHE[fx_tk] = fx
            if fx is None:
                logger.warning(f"[cwarp] {ticker}: cambio {fx_tk} non disponibile — escluso")
                return None
            fx = fx.reindex(closes.index).ffill()
            closes = closes / fx if ccy == "USD" else closes * fx   # GBp: scala costante, rendimenti invariati
        elif ccy != "EUR":
            logger.warning(f"[cwarp] {ticker}: valuta {ccy} non gestita — escluso")
            return None
        rets = closes.dropna().pct_change().dropna()
        return rets.iloc[-months:] if len(rets) > months else rets
    except Exception as e:
        logger.warning(f"[cwarp] {ticker} rendimenti mensili EUR: {e}")
        return None


def _fetch_monthly_returns(ticker: str, months: int = 60) -> np.ndarray | None:
    """Scarica rendimenti mensili per ticker. None se storico insufficiente.
    Usa start/end date invece di period stringa per evitare valori non-standard."""
    try:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=int((months + 6) * 30.5))
        hist = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1mo",
            progress=False, auto_adjust=True
        )
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna()
        if len(closes) < 13:
            return None
        returns = closes.pct_change().dropna().values
        return returns[-months:] if len(returns) > months else returns
    except Exception as e:
        logger.warning(f"[cwarp] {ticker} monthly returns: {e}")
        return None


def _sortino_ratio(returns: np.ndarray, rf_monthly: float) -> float | None:
    """Sortino Ratio mensile annualizzato — fedele a Cole (cwarp_defs.py).

    TDD (Target Downside Deviation) calcolato come root-mean-square su TUTTI
    i periodi con floor a zero — identico a target_downside_deviation() di Cole:
        floored = where(returns < 0, returns, 0)
        tdd = sqrt(mean(floored^2))
    Numeratore: mean(returns) - rf (eccesso di rendimento mensile × 12).
    MAR = 0 (Cole default: include_risk_free_in_vol=False).

    Differenza rispetto alla versione precedente: quella usava std(periodi_negativi,
    ddof=1) — divideva per N_negativi-1 invece che per N_totali, producendo TDD
    sistematicamente troppo alto per asset con pochi mesi negativi (buoni asset).
    """
    try:
        if len(returns) < 6:
            return None
        # Eccesso di rendimento (numeratore Sortino)
        rf_period = rf_monthly          # già mensile
        mean_ex = float(np.nanmean(returns) - rf_period) * 12  # annualizzato

        # TDD Cole: RMS su TUTTI i periodi, floor a zero (MAR=0)
        floored = np.where(returns < 0, returns, 0.0)
        tdd = float(np.sqrt(np.nanmean(floored ** 2))) * math.sqrt(12)  # annualizzato

        if tdd <= 0:
            return None
        return mean_ex / tdd
    except Exception:
        return None


def _return_to_maxdd(returns: np.ndarray, rf_annual: float) -> float | None:
    """Return to Maximum Drawdown annualizzato — fedele a Cole (cwarp_defs.py).

    Usa CAGR geometrico (non aritmetico), identico a annualized_return() + return_maxdd_ratio() di Cole:
        years = len(returns) / 12
        CAGR = cum_product(1+returns)^(1/years) - 1
        RMDD = (CAGR - rf_annual) / abs(maxdd)

    Differenza rispetto alla versione precedente: quella usava mean(returns)*12
    (rendimento aritmetico) che per Jensen's inequality sopravvaluta sempre il
    rendimento effettivo per asset volatili. CAGR è il rendimento reale dell'investitore.

    NOTA: accetta rf_annual (tasso annuo, es. 0.036), NON rf_monthly.
    """
    try:
        if len(returns) < 6:
            return None
        # CAGR geometrico (Cole: annualized_return)
        cum   = np.nancumprod(1.0 + returns)
        years = len(returns) / 12.0
        cagr  = float(cum[-1]) ** (1.0 / years) - 1.0

        # MaxDD — valore assoluto (positivo), identico a max_dd() di Cole
        peak  = np.maximum.accumulate(cum)
        dd    = (cum - peak) / peak
        maxdd = abs(float(np.nanmin(dd)))

        if maxdd == 0:
            return None
        # RMDD = (CAGR - rf_annuale) / MaxDD  — Cole sottr. rf annualizzato
        return (cagr - rf_annual) / maxdd
    except Exception:
        return None


def _cwarp_estimated(ticker: str, component: str) -> float | None:
    """
    CWARP stimato da valori storici Cole (Moneyball paper 2020, tabella pag. 6).
    Usato quando i dati storici reali non sono disponibili.
    Solo per componenti Dragon Portfolio — non per asset generici.
    """
    # Valori Cole (2008-2020, vs S&P 500 Equity Beta)
    cole_reference = {
        "long_volatility": 24.67,   # CBOE Long Vol HF Index — top CWARP Cole
        "gold":            21.87,   # Gold XAU
        "fixed_income":    11.03,   # Bloomberg US Treasury Bond
        "commodity_trend":  8.08,   # HFRX Macro Systematic CTA
        "equity":           None,   # N/A — è il replacement portfolio
    }
    return cole_reference.get(component)


# ────────────────────────────────────────────────────────────────────────────
# 4.11 DRAGON HEALTH CHECK  ← NUOVO STACK ARTEMIS
# ────────────────────────────────────────────────────────────────────────────

def assess_risk_regime(vol_signals: dict) -> dict:
    """
    LIVELLO DI GESTIONE DEL RISCHIO — aggiunto 2026-09-16 su suggerimento
    esterno (analisi comparativa GPT), colma un buco reale: il sistema
    passava direttamente da "drift rispetto al target" a "raccomandazione
    compra/non comprare", senza un livello intermedio che dica QUANTO
    velocemente eseguire un ribilanciamento dato lo stress di mercato
    attuale.

    Architettura a 4 livelli, ora completa:
      Regime strutturale (Regime Engine, G_z/I_z)
        → Segnali tattici (Signal Scanner, ampiezza/sentiment/curva)
          → GESTIONE DEL RISCHIO (questa funzione — nuovo)
            → Ribilanciamento (Dragon Health Check)

    Usa SOLO soglie già nominate altrove nel file (nessun numero nuovo
    inventato): VIX_CRISIS_THRESHOLD, VIX_ATTENTION_THRESHOLD,
    SKEW_PROTECTION_PRICED, VVIX_ATTENTION, MOVE_ATTENTION, OVX_ATTENTION.

    IMPORTANTE — cosa questa funzione NON fa: non cambia mai i pesi target
    del Dragon (restano 24/18/21/18/19 per disciplina Cole), non genera
    raccomandazioni di acquisto/vendita. Il suo unico output è: "se e
    quando decidi di ribilanciare, quanto stress c'è in giro in questo
    momento" — puro contesto di esecuzione, mai una decisione.
    """
    vix  = vol_signals.get("vix")
    vvix = vol_signals.get("vvix")
    skew = vol_signals.get("skew")
    move = vol_signals.get("move")
    ovx  = vol_signals.get("ovx")

    flags = []
    if vix is not None and vix > VIX_CRISIS_THRESHOLD:
        flags.append(f"VIX={vix:.1f}>{VIX_CRISIS_THRESHOLD} (stress acuto)")
    elif vix is not None and vix > VIX_ATTENTION_THRESHOLD:
        flags.append(f"VIX={vix:.1f}>{VIX_ATTENTION_THRESHOLD} (attenzione)")
    if skew is not None and skew > SKEW_PROTECTION_PRICED:
        flags.append(f"SKEW={skew:.0f}>{SKEW_PROTECTION_PRICED} (tail risk caro)")
    if vvix is not None and vvix > VVIX_ATTENTION:
        flags.append(f"VVIX={vvix:.1f}>{VVIX_ATTENTION} (vol-della-vol elevata)")
    if move is not None and move > MOVE_ATTENTION:
        flags.append(f"MOVE={move:.1f}>{MOVE_ATTENTION} (stress bond)")
    if ovx is not None and ovx > OVX_ATTENTION:
        flags.append(f"OVX={ovx:.1f}>{OVX_ATTENTION} (stress energia)")

    n = len(flags)
    vix_crisi = vix is not None and vix > VIX_CRISIS_THRESHOLD

    if vix_crisi or n >= 3:
        livello = "STRESS"
        guida = ("Più segnali di stress attivi insieme. Se un ribilanciamento "
                 "è già deciso, preferire ingressi frazionati su più giorni "
                 "invece di eseguire tutto in un'unica soluzione — lo spread "
                 "bid/ask e la volatilità intraday sono probabilmente più "
                 "ampi del normale.")
    elif n == 2:
        livello = "TESO"
        guida = ("Alcuni segnali di stress attivi. Nessuna urgenza di "
                 "accelerare o rallentare un ribilanciamento pianificato, "
                 "ma vale la pena controllare i segnali specifici scattati "
                 "prima di eseguire.")
    elif n == 1:
        livello = "NORMALE"
        guida = "Un solo segnale di stress isolato — condizioni di mercato ordinarie."
    else:
        livello = "CALMO"
        guida = ("Nessun segnale di stress attivo tra quelli monitorati. "
                 "Condizioni di esecuzione ordinarie.")

    return {
        "livello": livello,
        "n_segnali_stress": n,
        "segnali_attivi": flags,
        "guida_esecuzione": guida,
        "nota": ("Livello di rischio puramente informativo — non modifica mai "
                "i pesi target del Dragon né genera raccomandazioni di "
                "acquisto/vendita. Riguarda solo il RITMO di un eventuale "
                "ribilanciamento già deciso altrove."),
    }


def collect_dragon_health_check(portfolio_current: dict | None = None,
                                cwarp_data: dict | None = None) -> dict:
    """
    Dragon Portfolio Health Check.

    Confronta l'allocazione corrente con il target Dragon ottimale di Cole.
    Target ATTIVO (Profilo 2 Growth): Equity 50% · Fixed Income 10% · Long Vol 0% ·
    CTA 15% · Gold 18% · Cash ~7%. Cole puro (24/18/21/18/19) in comparison_cole.

    FIX C5 — cwarp_data: dict candidati CWARP (da compute_cwarp["candidates"]).
    Se un componente richiede AUMENTA ma il candidato migliore per quella
    asset class ha CWARP < 0, la raccomandazione diventa un warning di tossicità.
    Evita di raccomandare acquisti che distruggono il Sortino del portafoglio.

    Se portfolio_current è None → restituisce solo il target con placeholder.

    Output:
        target_allocation   — allocazione target Cole
        current_allocation  — allocazione corrente (None se non fornita)
        drift               — scostamento per componente
        rebalancing_needed  — True se drift > soglia
        rebalancing_urgency — LOW / MEDIUM / HIGH / CRITICAL
        dragon_score        — 0-100 (100 = perfettamente allineato al target)
        recommendations     — lista di azioni di ribilanciamento
    """
    result = {
        "target_allocation":   DRAGON_TARGET.copy(),
        "current_allocation":  None,
        "drift":               None,
        "rebalancing_needed":  False,
        "rebalancing_urgency": "N/A",
        "dragon_score":        None,
        "recommendations":     [],
        "portfolio_source":    "PLACEHOLDER — fornire portafoglio reale",
        "cole_note":           (f"Profilo ATTIVO: {ACTIVE_PROFILE_NAME} — "
                                + " | ".join(f"{c} {w:.0%}" for c, w in DRAGON_TARGET.items())
                                + f" | cash ~{max(0.0, 1.0 - sum(DRAGON_TARGET.values())):.0%}. "
                                f"Fonte target: {DRAGON_TARGET_SOURCE}. "
                                "Cole puro 24/18/21/18/19 mantenuto come riferimento "
                                "in comparison_cole. "
                                "Ref: Allegory of the Hawk and Serpent, 2020."),
        "active_profile":      ACTIVE_PROFILE_NAME,
        "target_source":       DRAGON_TARGET_SOURCE,
        "target_warning":      DRAGON_TARGET_WARNING,
    }
    if DRAGON_TARGET_WARNING:
        result["recommendations"].append(DRAGON_TARGET_WARNING)

    if portfolio_current is None:
        result["recommendations"].append(
            "Fornire portafoglio corrente per calcolare drift e ribilanciamento. "
            "Formato: {'equity': 0.XX, 'fixed_income': 0.XX, "
            "'long_volatility': 0.XX, 'commodity_trend': 0.XX, 'gold': 0.XX}"
        )
        return result

    result["current_allocation"] = portfolio_current
    result["portfolio_source"]   = "INPUT_GESTORE"

    # Validazione: somma allocazioni deve essere ≈ 1.0
    # Se < 1.0 il residuo è cash non allocato (Dead Cash) — normale in fase di costruzione
    total_alloc  = sum(portfolio_current.get(k, 0) for k in DRAGON_TARGET)
    cash_residuo = max(0.0, round(1.0 - total_alloc, 4))
    target_cash  = max(0.0, round(1.0 - sum(DRAGON_TARGET.values()), 4))
    sum_warning  = None
    # Nel Dragon Growth (Profilo 2) ~12% di cash/bond breve è zavorra INTENZIONALE,
    # non "Dead Cash". Avvisa solo se il cash corrente si scosta molto dal target.
    if abs(cash_residuo - target_cash) > 0.05:
        sum_warning = (
            f"ℹ️  Cash {cash_residuo:.1%} vs target Profilo 2 ~{target_cash:.0%}. "
            f"Nel Dragon Growth ~{target_cash:.0%} di cash/bond breve è zavorra "
            f"INTENZIONALE (non Dead Cash). Drift calcolato sui componenti Dragon."
        )
        logger.info(f"[dragon_health] Cash={cash_residuo:.1%} (target ~{target_cash:.0%})")
    elif abs(total_alloc - 1.0) > 0.02 and cash_residuo == 0:
        sum_warning = (
            f"⚠️  Allocazioni sommate = {total_alloc:.2%} (atteso ≤ 100%)."
        )
        logger.warning(f"[dragon_health] Portfolio sum={total_alloc:.2%} anomalo")

    # Drift per componente
    drift = {}
    for comp, target_w in DRAGON_TARGET.items():
        current_w = portfolio_current.get(comp, 0.0)
        drift[comp] = round(current_w - target_w, 4)

    result["drift"] = drift

    # Urgency
    max_drift      = max(abs(d) for d in drift.values())
    drift_critical = any(abs(d) >= DRAGON_DRIFT_CRITICAL for d in drift.values())
    drift_alert    = any(abs(d) >= DRAGON_DRIFT_ALERT    for d in drift.values())

    urgency = (
        "CRITICAL" if drift_critical else
        "HIGH"     if max_drift >= 0.07 else
        "MEDIUM"   if drift_alert else
        "LOW"
    )
    result["rebalancing_urgency"] = urgency
    result["rebalancing_needed"]  = drift_alert

    # Dragon Score regime-adjusted (GAP-4 stress test)
    # In SERPENTE il drift equity è fisiologico — tolleranza +12%
    # In FALCO_DX i bond sono emergenza — tolleranza ridotta a +3%
    REGIME_TOLERANCE = {
        "SERPENTE":     {"equity": 0.12, "long_volatility": 0.08, "default": 0.08},
        "FALCO_DX":     {"equity": 0.05, "fixed_income": 0.03, "default": 0.05},
        "FALCO_SX":     {"long_volatility": 0.10, "default": 0.05},
        "FENICE":       {"gold": 0.08, "fixed_income": 0.00, "default": 0.05},
        "TRANSIZIONE":  {"default": 0.10},
    }
    # FIX A3: legge "regime_engine_corrente" (regime engine attuale, iniettato
    # dal main prima di questa chiamata). Distinto da "regime_at_snapshot"
    # (nota storica del regime al momento dell'ultimo PAC — invariante).
    regime_key = (portfolio_current or {}).get("regime_engine_corrente", "TRANSIZIONE")
    tol_map    = REGIME_TOLERANCE.get(regime_key, REGIME_TOLERANCE["TRANSIZIONE"])

    adjusted_score = 100.0
    for comp, d in drift.items():
        tol = tol_map.get(comp, tol_map.get("default", 0.08))
        excess = max(0, abs(d) - tol)
        adjusted_score -= excess * 200  # 2 punti per ogni 1% eccedente la tolleranza

    score = max(0, adjusted_score)
    result["dragon_score"] = round(score, 1)
    result["dragon_score_regime"] = regime_key  # traccia quale regime ha calibrato lo score

    # ── CONFRONTO vs COLE PURO (riferimento, NON target operativo) ──────────────
    # Calcola drift e Dragon Score anche sull'allocazione Cole originale, per
    # affiancare gli andamenti Growth (attivo) vs Cole puro nel tempo. Richiesto
    # da Davide 2026-09-24: switch a Profilo 2 senza cancellare Cole.
    cole_drift = {c: round(portfolio_current.get(c, 0.0) - w, 4)
                  for c, w in DRAGON_TARGET_COLE.items()}
    cole_adj = 100.0
    for comp, d in cole_drift.items():
        tol = tol_map.get(comp, tol_map.get("default", 0.08))
        cole_adj -= max(0, abs(d) - tol) * 200
    result["active_profile"] = ACTIVE_PROFILE_NAME
    result["comparison_cole"] = {
        "target_allocation": DRAGON_TARGET_COLE.copy(),
        "drift":             cole_drift,
        "dragon_score":      round(max(0.0, cole_adj), 1),
        "note": ("Riferimento Cole puro (24/18/21/18/19) — NON è il target "
                 "operativo. Affiancato per confrontare gli andamenti nel tempo."),
    }

    # FIX C5 — Mappa componente → miglior CWARP candidato (per il gate)
    best_cwarp_by_component: dict = {}
    best_ticker_by_component: dict = {}
    # Candidati con CWARP solo STIMATO (storico < 12 mesi): tenuti a parte,
    # mai usati come se fossero misurati (fix 2026-09-25).
    estimated_by_component: dict = {}
    if cwarp_data:
        for _ticker, _cdata in cwarp_data.items():
            _comp = _cdata.get("component")
            _cw   = _cdata.get("cwarp")
            if _cdata.get("quality") == "ESTIMATED":
                if _comp and _comp not in estimated_by_component:
                    estimated_by_component[_comp] = (_ticker, _cw, _cdata.get("months"))
                continue
            if _comp and _cw is not None:
                if _comp not in best_cwarp_by_component or _cw > best_cwarp_by_component[_comp]:
                    best_cwarp_by_component[_comp] = _cw
                    best_ticker_by_component[_comp] = _ticker

    # Raccomandazioni
    recs = []
    for comp, d in drift.items():
        target_pct  = DRAGON_TARGET[comp] * 100
        current_pct = portfolio_current.get(comp, 0) * 100
        if abs(d) >= DRAGON_DRIFT_ALERT:
            direction = "AUMENTA" if d < 0 else "RIDUCI"
            # FIX C5 — CWARP gate: se drift richiede acquisto ma candidato è tossico,
            # sostituisci la raccomandazione con un warning invece di un ordine cieco
            best_cw     = best_cwarp_by_component.get(comp)
            best_ticker = best_ticker_by_component.get(comp, "?")
            # L'equity è il portafoglio di sostituzione (benchmark) del CWARP:
            # il suo CWARP è ~0/negativo PER COSTRUZIONE (non diversifica sé stesso),
            # NON è un segnale di tossicità. Escluso dal gate per evitare il falso
            # allarme "equity tossica → tieni in cash" (fix 2026-09-24).
            is_equity = (comp == "equity")
            # Fixed income: gamba STATICA del Profilo 2 costruita per TESI
            # (assicurazione deflazione), non per CWARP. Il CWARP guarda indietro e
            # sconta il bear market dei bond lunghi 2020-2023: negativo è atteso.
            # Resta visibile come informazione, ma non blocca l'acquisto
            # (decisione Davide 2026-09-24).
            is_thesis_leg = (comp == "fixed_income")
            if direction == "AUMENTA" and is_equity:
                recs.append(
                    f"AUMENTA EQUITY: attuale {current_pct:.1f}% → target {target_pct:.1f}% "
                    f"(drift {d*100:+.1f}%) | nota: CWARP equity negativo è strutturale "
                    f"(è il benchmark), NON tossico — comprare regolarmente per il target."
                )
            elif direction == "AUMENTA" and is_thesis_leg:
                _cw_txt = (f"{best_ticker} CWARP={best_cw:.1f}"
                           if best_cw is not None else "CWARP n/d")
                recs.append(
                    f"AUMENTA FIXED INCOME: attuale {current_pct:.1f}% → target {target_pct:.1f}% "
                    f"(drift {d*100:+.1f}%) | {_cw_txt} (informativo) — gamba statica per "
                    f"tesi deflazione: costruire via PAC. Tesi rotta solo se i bond lunghi "
                    f"scendono INSIEME all'azionario in una recessione conclamata."
                )
            elif direction == "AUMENTA" and best_cw is None and comp in estimated_by_component:
                # Solo candidati con storico < 12 mesi: nessun CWARP reale.
                # La nota sparisce da sola quando il fondo arriva a 12 mesi di dati
                # (il CWARP viene calcolato e si passa al ramo standard).
                _et, _ecw, _emesi = estimated_by_component[comp]
                _stima = f"{_ecw:.1f}" if _ecw is not None else "n/d"
                _storico = (f"storico {_emesi} mesi < 12" if _emesi
                            else "NESSUN DATO su Yahoo — verificare ticker")
                recs.append(
                    f"{direction} {comp.upper().replace('_',' ')}: "
                    f"attuale {current_pct:.1f}% → target {target_pct:.1f}% "
                    f"(drift {d*100:+.1f}%) | candidato: {_et} — CWARP non calcolabile "
                    f"({_storico}); stima di categoria {_stima} solo indicativa, NON dato reale."
                )
            elif direction == "AUMENTA" and best_cw is not None and best_cw < 0:
                recs.append(
                    f"⚠️  ATTENZIONE {comp.upper().replace('_',' ')}: "
                    f"drift {d*100:+.1f}% richiede acquisto ma candidato "
                    f"{best_ticker} CWARP={best_cw:.1f} NEGATIVO — distrugge Sortino. "
                    f"Mantenere in Dead Cash o cercare proxy a duration/volatilità inferiore."
                )
            elif direction == "AUMENTA" and best_cw is not None:
                recs.append(
                    f"{direction} {comp.upper().replace('_',' ')}: "
                    f"attuale {current_pct:.1f}% → target {target_pct:.1f}% "
                    f"(drift {d*100:+.1f}%) | candidato: {best_ticker} CWARP={best_cw:.1f} ✅"
                )
            elif direction == "RIDUCI" and comp == "cash":
                recs.append(
                    f"INVESTI CASH via PAC: attuale {current_pct:.1f}% → target {target_pct:.1f}% "
                    f"(drift {d*100:+.1f}%) — la liquidità in eccesso va sulle gambe sottopesate."
                )
            elif direction == "RIDUCI" and d < DRAGON_DRIFT_CRITICAL:
                # Regola ARTEMIS (decisione 1): non si vende per ribilanciare; il
                # sovrappeso si riassorbe comprando le altre gambe (fix 2026-09-26:
                # prima scriveva "RIDUCI", in contrasto con la regola).
                recs.append(
                    f"NON COMPRARE {comp.upper().replace('_',' ')}: sovrappeso "
                    f"{current_pct:.1f}% vs target {target_pct:.1f}% (drift {d*100:+.1f}%) — "
                    f"nessuna vendita: si riassorbe con il PAC sulle gambe sottopesate."
                )
            elif direction == "RIDUCI":
                recs.append(
                    f"SOVRAPPESO CRITICO {comp.upper().replace('_',' ')}: "
                    f"{current_pct:.1f}% vs target {target_pct:.1f}% (drift {d*100:+.1f}%, "
                    f"oltre la soglia critica {DRAGON_DRIFT_CRITICAL*100:.0f} punti) — "
                    f"valutare vendita parziale (decisione 1: si vende solo oltre soglia critica)."
                )
            else:
                recs.append(
                    f"{direction} {comp.upper().replace('_',' ')}: "
                    f"attuale {current_pct:.1f}% → target {target_pct:.1f}% "
                    f"(drift {d*100:+.1f}%)"
                )

    if not recs:
        recs.append("Portafoglio Dragon ben bilanciato — nessun ribilanciamento urgente.")

    # Aggiungi warning somma se presente
    if sum_warning:
        recs.insert(0, sum_warning)
    # Avviso target da riserva: sempre in cima, così finisce in Supabase ben visibile
    if DRAGON_TARGET_WARNING:
        recs.insert(0, DRAGON_TARGET_WARNING)

    result["recommendations"] = recs

    logger.info(f"[dragon_health] score={result['dragon_score']} "
                f"urgency={urgency} max_drift={max_drift:.2%}")
    return result


# ── SEZIONE 5: REGIME FLAGS (adattato per nomenclatura Artemis) ──────────────

def compute_regime_flags(vol: dict, rates: dict,
                          fx: dict = None, sent: dict = None) -> dict:
    """
    Regime flags dal signal scanner — prima sintesi qualitativa.
    Integra con artemis_regime_engine per la classificazione definitiva.
    Usa nomenclatura Artemis: SERPENTE / FALCO_SX / FALCO_DX / FENICE / TRANSIZIONE
    """
    flags = {
        "regime_primary":   "INDETERMINATO",
        "regime_secondary": None,
        "confidence":       0,
        "signals_active":   [],
        "signals_missing":  [],
        "dead_stop_reason": None,
    }

    score = {
        "SERPENTE":  0,
        "FALCO_SX":  0,
        "FALCO_DX":  0,
        "FENICE":    0,
    }
    active_signals = []
    missing        = []
    is_transition_hints = 0   # FIX F8: segnali che indicano ambiguità (es. BULL_FLATTENING)
    fx = fx or {}
    sent = sent or {}

    # ── VIX ─────────────────────────────────────────────────────────────────
    vix = vol.get("vix")
    if vix is None:
        missing.append("VIX")
    elif vix > VIX_CRISIS_THRESHOLD:
        score["FALCO_SX"] += 3
        active_signals.append(f"VIX={vix:.1f}>30 → Hawk crisis acuto")
    elif vix > VIX_ATTENTION_THRESHOLD:
        score["FALCO_SX"] += 1
        active_signals.append(f"VIX={vix:.1f}>20 → Hawk early warning")

    # ── SKEW ─────────────────────────────────────────────────────────────────
    skew = vol.get("skew")
    if skew and skew > SKEW_PROTECTION_PRICED:
        score["FALCO_SX"] += 2
        active_signals.append(f"SKEW={skew:.0f}>130 → tail risk cara (Volatility at World's End)")
    elif skew and skew > SKEW_ATTENTION:
        score["FALCO_SX"] += 1

    # ── MOVE ─────────────────────────────────────────────────────────────────
    move = vol.get("move")
    if move and move > MOVE_ATTENTION:
        score["FALCO_SX"] += 2
        active_signals.append(f"MOVE={move:.0f}>120 → stress bond strutturale")

    # ── Curva dei tassi ──────────────────────────────────────────────────────
    t10y2y = rates.get("t10y2y")
    t10y3m = rates.get("t10y3m")
    if t10y2y is not None and t10y2y < T10Y2Y_INVERSION:
        score["FALCO_SX"] += 2
        active_signals.append(f"T10Y2Y={t10y2y:.2f} invertita → recessione")
    if t10y3m is not None and t10y3m < T10Y3M_INVERSION:
        score["FALCO_SX"] += 2
        active_signals.append(f"T10Y3M={t10y3m:.2f} invertita → segnale più affidabile")

    # ── Yield curve dinamica: il MOVIMENTO della curva pesa sul regime ───────
    curve_trend = rates.get("curve_trend")
    if curve_trend == "BULL_STEEPENING":
        score["FALCO_SX"] += 2
        active_signals.append("Curva BULL STEEPENING → Fed verso tagli, deflazione in arrivo")
    elif curve_trend == "BEAR_STEEPENING":
        score["FALCO_DX"] += 2
        active_signals.append("Curva BEAR STEEPENING → term premium/inflazione, lunga sotto pressione")
    # FIX F8 — BULL_FLATTENING non aggiunge più una quinta chiave "TRANSIZIONE"
    # al dict score (che ha solo 4 chiavi: SERPENTE/FALCO_SX/FALCO_DX/FENICE).
    # Contribuisce invece a is_transition_hints, che abbassa la soglia di
    # attivazione del flag is_transition. È semanticamente corretto: BULL_FLATTENING
    # non è un regime in sé, è un segnale che il regime è ambiguo/in transizione.
    elif curve_trend == "BULL_FLATTENING":
        is_transition_hints += 1
        active_signals.append("Curva BULL FLATTENING → flight to quality, rallentamento")
    elif curve_trend == "BEAR_FLATTENING":
        score["FALCO_DX"] += 1
        active_signals.append("Curva BEAR FLATTENING → Fed alza, fine ciclo")

    # ── CPI / Inflazione → FALCO_DX ─────────────────────────────────────────
    cpi = rates.get("cpi_yoy")
    if cpi and cpi > 4.0:
        score["FALCO_DX"] += 3
        active_signals.append(f"CPI={cpi:.1f}%>4% → Hawk Ala Destra (stagflazione)")
    elif cpi and cpi > 2.5:
        score["FALCO_DX"] += 1

    # T5YIE (breakeven inflazione)
    t5yie = rates.get("t5yie")
    if t5yie and t5yie > 2.5:
        score["FALCO_DX"] += 2
        active_signals.append(f"T5YIE={t5yie:.2f} → aspettative inflazione elevate")

    # ── GOLD → FENICE (variazione annua, non livello di prezzo) ─────────────
    gold_yoy = fx.get("gold_yoy_pct")
    real_y = rates.get("real_yield_10y")
    if gold_yoy is not None and gold_yoy >= GOLD_YOY_FENICE_STRONG:
        score["FENICE"] += 2
        active_signals.append(f"Oro +{gold_yoy:.1f}% in un anno (≥{GOLD_YOY_FENICE_STRONG:.0f}%) "
                              f"→ rialzo eccezionale, possibile sfiducia valutaria → FENICE")
    elif (gold_yoy is not None and gold_yoy >= GOLD_YOY_FENICE
          and real_y is not None and real_y >= GOLD_REAL_YIELD_HIGH):
        score["FENICE"] += 1
        active_signals.append(f"Oro +{gold_yoy:.1f}% in un anno nonostante tasso reale "
                              f"{real_y:.2f}% → domanda oltre i fattori normali → FENICE early")

    # ── DXY → FENICE o SERPENTE ──────────────────────────────────────────────
    dxy = fx.get("dxy")
    if dxy and dxy < 95:
        score["FENICE"] += 2
        active_signals.append(f"DXY={dxy:.1f}<95 → dollaro debole → FENICE")
    elif dxy and dxy < 100:
        score["FENICE"] += 1
        active_signals.append(f"DXY={dxy:.1f}<100 → dollaro in indebolimento → FENICE early")
    elif dxy and dxy > 103:
        score["FALCO_DX"] += 1
        active_signals.append(f"DXY={dxy:.1f}>103 → dollaro forte → FALCO_DX")

    # ── Copper/Gold → SERPENTE ───────────────────────────────────────────────
    cg = fx.get("copper_gold_ratio")
    if cg and cg > COPPER_GOLD_BULL_RATIO:
        score["SERPENTE"] += 2
        active_signals.append(f"Cu/Au={cg:.6f} → risk appetite globale → SERPENTE")

    # ── VIX term slope → SERPENTE ────────────────────────────────────────────
    vix_slope = vol.get("vix_term_slope")
    if vix_slope is not None and vix_slope > VIX_TERM_CONTANGO_MIN:
        score["SERPENTE"] += 1
        active_signals.append(f"VIX slope={vix_slope:.3f} contango → mercato calmo → SERPENTE")

    # ── Net Liquidity → SERPENTE / FENICE ────────────────────────────────────
    net_liq_delta = rates.get("net_liq_delta_4w")
    if net_liq_delta and net_liq_delta > 100_000:
        score["SERPENTE"] += 1
        score["FENICE"]   += 1
        active_signals.append(f"Net Liq +{net_liq_delta/1e6:.1f}T in 4w → QE injection")

    # ── VVIX → FALCO ─────────────────────────────────────────────────────────
    vvix = vol.get("vvix")
    if vvix and vvix > 40:
        score["FALCO_SX"] += 2
        active_signals.append(f"VVIX={vvix:.1f}>40 → vol della vol elevata → Hawk")
    elif vvix and vvix > 30:
        score["FALCO_SX"] += 1

    # ── PCT_MA200 → segnale di ampiezza ──────────────────────────────────────
    # RIVISTO 2026-09-16, AFFINATO su backtest esteso 33 anni (1993-2026, 503
    # titoli reali). SCOPERTA: la vecchia logica "pct_ma<30 → +2 FALCO_SX" era
    # CONTRO-DIREZIONALE. Su 8 crisi INDIPENDENTI e di natura diversa (1998
    # LTCM, 2001-02 dot-com, 2008 subprime, 2011 debito EU, 2018, 2020 COVID,
    # 2022 inflazione, 2025 tariffe) l'ampiezza sotto il 20-25% ha preceduto un
    # rendimento POSITIVO a 60gg (+6.1% vs base +2.7%) — MA con un drawdown
    # medio di -10% PRIMA del rimbalzo (base -4.7%). Pattern netto e confermato
    # su tutte le crisi: "scende ancora, poi finisce più in alto".
    #
    # SOGLIA: fissata a <25% (non <30%). A 30% il pattern diventa "misto" —
    # troppi giorni di debolezza ordinaria si mescolano ai veri fondi. A
    # 20-25% il segnale è pulito su 33 anni.
    #
    # DISCIPLINA: INFORMATIVO — compare nel report ma NON altera lo score.
    # 8 crisi in 33 anni sono un campione ragionevole, ma un fondo di mercato
    # non si azzecca al giorno: meglio segnalare al Gestore (che ha lo stomaco
    # per decidere l'accumulo) che far pesare l'euristica sul punteggio
    # automatico. Coerente con Cole: accumulare mentre fa paura, sapendo che
    # il minimo esatto è imprendibile.
    pct_ma = rates.get("pct_above_ma200")
    if pct_ma is not None:
        if pct_ma < 25:
            # NESSUN punteggio di regime — solo segnale informativo per il report
            active_signals.append(
                f"PCT_MA200={pct_ma:.0f}%<25% → CAPITOLAZIONE — storicamente "
                f"(8 crisi/33y) vicino a un fondo, MA di norma con un'altra "
                f"gamba di ribasso (~-10%) prima del rimbalzo. Opportunità di "
                f"accumulo graduale, non ingresso immediato. Non altera lo score.")
        elif pct_ma < 50:
            # Zona 40-60%: il backtest dava scarto predittivo su correzioni, ma
            # il test di disambiguazione suggerisce sia in parte lo stesso
            # artefatto "già in turbolenza". Peso ridotto e non enfatizzato.
            score["FALCO_SX"] += 1
        elif pct_ma > 70:
            # NON testato nel backtest — lasciato invariato, coerente col resto
            score["SERPENTE"] += 1
            active_signals.append(f"PCT_MA200={pct_ma:.0f}%>70% → breadth forte → Serpent")

    # ── Sentiment ────────────────────────────────────────────────────────────
    epu = sent.get("epu_index")
    if epu and epu > 150:
        score["FALCO_DX"] += 2
        active_signals.append(f"EPU={epu:.0f}>150 → incertezza policy estrema → FALCO_DX")
    elif epu and epu > 100:
        score["FALCO_DX"] += 1

    gpr = sent.get("gpr_index") or sent.get("gpr_synthetic")
    if gpr and gpr > 200:
        score["FALCO_DX"] += 2
        active_signals.append(f"GPR={gpr:.0f}>200 → rischio geopolitico elevato")
    elif gpr and gpr > 130:
        score["FALCO_DX"] += 1

    naaim = sent.get("naaim_exposure") or sent.get("naaim_synthetic")
    if naaim:
        if naaim > 90:
            score["SERPENTE"] += 1
            active_signals.append(f"NAAIM={naaim:.0f}>90 → crowding estremo → Serpent surriscaldato")
        elif naaim < 20:
            score["FALCO_SX"] += 1
            active_signals.append(f"NAAIM={naaim:.0f}<20 → capitulation → Hawk bottom")

    fg = sent.get("fear_greed_score") or sent.get("fear_greed_synthetic")
    if fg is not None:
        if fg < 20:
            score["FALCO_SX"] += 2
            active_signals.append(f"F&G={fg:.0f}<20 → paura estrema → Hawk")
        elif fg < 35:
            score["FALCO_SX"] += 1
        elif fg > 75:
            score["SERPENTE"] += 1
            active_signals.append(f"F&G={fg:.0f}>75 → avidità → Serpent surriscaldato")

    umcs = sent.get("umcsent")
    if umcs:
        if umcs < 55:
            score["FALCO_DX"] += 2
            active_signals.append(f"UMCSENT={umcs:.1f}<55 → fiducia collassata → Hawk")
        elif umcs < 65:
            score["FALCO_DX"] += 1
        elif umcs > 85:
            score["SERPENTE"] += 2
            active_signals.append(f"UMCSENT={umcs:.1f}>85 → fiducia alta → Serpent")

    # ── DEAD STOP se VIX mancante ────────────────────────────────────────────
    if "VIX" in missing:
        flags["dead_stop_reason"] = (
            "VIX non disponibile — impossibile determinare regime Artemis. "
            "Verificare yfinance/FRED e rieseguire.")
        flags["signals_missing"] = missing
        return flags

    # ── Determina regime ─────────────────────────────────────────────────────
    sorted_scores = sorted(score.items(), key=lambda x: x[1], reverse=True)
    top_regime, top_score = sorted_scores[0]
    sec_regime, sec_score = sorted_scores[1]
    total_signals = sum(score.values())
    gap = top_score - sec_score

    # TRANSIZIONE: gap ≤ 2 e almeno 3 regimi attivi → segnali contraddittori
    n_active = sum(1 for v in score.values() if v >= 2)
    # FIX F8: is_transition_hints (segnali come BULL_FLATTENING) abbassano la soglia
    is_transition = (gap <= 2 and n_active >= 3 and top_score >= 2) or \
                    (gap <= 3 and n_active >= 2 and is_transition_hints >= 1 and top_score >= 2)

    TARGET_SIGNALS = 12
    n_active_signals = len(active_signals)
    coverage = min(n_active_signals / TARGET_SIGNALS, 1.0)

    if is_transition:
        flags["regime_primary"]   = "TRANSIZIONE"
        flags["regime_secondary"] = f"{top_regime}/{sec_regime}"
        missing_penalty = min(len(missing) * 4, 20)
        chimera_conf = round(coverage * 88 - missing_penalty, 1)
        flags["confidence"] = round(max(10.0, min(chimera_conf, 88.0)), 1)
    else:
        frac  = (top_score / max(total_signals, 1)) * 45
        gap_c = min((gap / max(top_score, 1)) * 35, 35)
        sig_c = min(n_active_signals * 1.8, 20)
        missing_penalty = min(len(missing) * 3, 15)
        raw_conf = frac + gap_c + sig_c - missing_penalty
        flags["regime_primary"]   = top_regime if top_score > 0 else "INDETERMINATO"
        flags["regime_secondary"] = (sec_regime
                                     if sec_score > 0 and sec_score >= top_score * 0.5
                                     else None)
        flags["confidence"]       = round(max(10.0, min(raw_conf, 95.0)), 1)

    flags["signals_active"]  = active_signals
    flags["signals_missing"] = missing

    logger.info(f"[regime_flags] {flags['regime_primary']} "
                f"conf={flags['confidence']}% signals={n_active_signals}")

    # ── CONFERMA REGIME (isteresi) — aggiunto 2026-09-16 ─────────────────────
    # Ispirato ai meccanismi di conferma di DREAM (Lops) e DTO Cycle Allocator:
    # non dichiarare un cambio di regime al primo segnale, ma solo quando
    # persiste. Risolve le oscillazioni osservate (USA TRANSIZIONE→SERPENTE→
    # FALCO in poche run). PURAMENTE INFORMATIVO: aggiunge campi al report,
    # NON sovrascrive regime_primary. Il Gestore vede sia il regime grezzo di
    # oggi sia se è confermato o ancora in attesa.
    flags["regime_confirmed"] = _confirm_regime(flags["regime_primary"])
    return flags


def _confirm_regime(regime_grezzo: str, giorni_richiesti: int = 3) -> dict:
    """
    Isteresi sul regime qualitativo, basata su GIORNI DI CALENDARIO
    DISTINTI (non su numero di run). Un nuovo regime è "confermato" solo
    se persiste per >= giorni_richiesti giorni diversi.

    RIVISTO 2026-09-16: la prima versione contava "run consecutive", ma
    più run nello stesso giorno (già successo: 3 run il 16/9) gonfiavano
    il conteggio e potevano confermare un cambio in mezza giornata. Ora
    lo storico è DEDUPLICATO per data (una osservazione per giorno, la più
    recente) prima di contare — stesso principio day-aware già usato negli
    scenari. Immune sia alle run multiple/giorno sia alle run saltate.

    Con una run/sera, giorni_richiesti=3 significa: 3 sere diverse con lo
    stesso regime nuovo prima di dichiararlo confermato.

    PURAMENTE INFORMATIVO — non sovrascrive regime_primary.
    Restituisce: grezzo, confermato, stato
    ("STABILE"|"CONFERMATO"|"IN_ATTESA"|"PRIMA_RUN"), giorni_consecutivi.
    """
    default = {"grezzo": regime_grezzo, "confermato": regime_grezzo,
               "stato": "PRIMA_RUN", "giorni_consecutivi": 1}
    if not SUPABASE_URL or not SUPABASE_KEY:
        return default
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Prendo più righe del necessario perché possono esserci più run/giorno
        rows = (client.table("ARTEMIS_SIGNALS")
                .select("date,regime_primary")
                .order("date", desc=True).limit(30).execute())
        if not rows.data:
            return default

        # DEDUPLICA per data: tengo il regime della run più recente di ogni
        # giorno (le righe sono già ordinate date desc, quindi la prima vista
        # per ciascuna data è quella giusta). Escludo il giorno di oggi, che
        # non è ancora scritto in questa run.
        oggi = datetime.today().date().isoformat()
        per_giorno = {}
        for r in rows.data:
            d = str(r.get("date"))[:10]
            reg = r.get("regime_primary")
            if not d or not reg or d == oggi:
                continue
            if d not in per_giorno:      # prima occorrenza = run più recente del giorno
                per_giorno[d] = reg

        if not per_giorno:
            return default

        # Ordino i giorni dal più recente al più vecchio
        giorni_ordinati = sorted(per_giorno.keys(), reverse=True)
        storico = [per_giorno[d] for d in giorni_ordinati]

        precedente_stabile = storico[0]   # regime dell'ultimo GIORNO scritto

        # Conto per quanti giorni distinti consecutivi (incluso oggi) vale
        # il regime grezzo di oggi
        conta = 1  # oggi
        for reg in storico:
            if reg == regime_grezzo:
                conta += 1
            else:
                break

        if regime_grezzo == precedente_stabile:
            stato = "STABILE"
            confermato = regime_grezzo
        elif conta >= giorni_richiesti:
            stato = "CONFERMATO"
            confermato = regime_grezzo
        else:
            stato = "IN_ATTESA"
            confermato = precedente_stabile
            logger.info(f"[regime_conferma] {regime_grezzo} rilevato ma NON "
                        f"confermato ({conta}/{giorni_richiesti} giorni) — "
                        f"regime confermato resta {precedente_stabile}")

        return {"grezzo": regime_grezzo, "confermato": confermato,
                "stato": stato, "giorni_consecutivi": conta}
    except Exception as e:
        logger.info(f"[regime_conferma] fallita: {e}")
        return default


# ── SEZIONE 6: REGIME PER REGIONE (dal regime engine) ───────────────────────

# Da questa data G_z/I_z sono calcolati con fonti vive e verificate (audit
# 2026-09-25: EU/CAN usavano serie ferme o di un altro paese, USA includeva
# USSLIND fermo dal 2020). I valori precedenti NON vanno confrontati con quelli
# nuovi: velocità di regime e fallback "ultimo valore valido" li ignorano.
REGIME_DATA_CLEAN_FROM = "2026-09-26"


def _fetch_prev_regime_scores(region: str, days_ago: int = 28) -> tuple[float | None, float | None, int | None]:
    """
    Legge G_z e I_z dalla riga più vicina a N giorni fa in ARTEMIS_SIGNALS.
    FIX F6 — Restituisce anche i giorni effettivi tra oggi e la riga trovata,
    così il Regime Velocity Flag può normalizzare il delta per la distanza reale.
    Restituisce (prev_gz, prev_iz, actual_days) oppure (None, None, None).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, None, None
    try:
        client   = create_client(SUPABASE_URL, SUPABASE_KEY)
        target   = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        col_map = {
            "USA": ("usa_g_z", "usa_i_z"),
            "EU":  ("eu_g_z",  "eu_i_z"),
            "CAN": ("can_g_z", "can_i_z"),
        }
        col_gz, col_iz = col_map.get(region.upper(), ("usa_g_z", "usa_i_z"))

        resp = (
            client.table("ARTEMIS_SIGNALS")
            .select(f"date,{col_gz},{col_iz}")
            .gte("date", (datetime.now() - timedelta(days=days_ago + 14)).strftime("%Y-%m-%d"))
            .lte("date", (datetime.now() - timedelta(days=days_ago - 7)).strftime("%Y-%m-%d"))
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data if hasattr(resp, "data") else []

        if not rows:
            resp2 = (
                client.table("ARTEMIS_SIGNALS")
                .select(f"date,{col_gz},{col_iz}")
                .lte("date", (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"))
                .order("date", desc=True)
                .limit(1)
                .execute()
            )
            rows = resp2.data if hasattr(resp2, "data") else []
            if rows:
                logger.debug(f"[regime_velocity] {region}: usato fallback riga più vecchia {rows[0].get('date')}")

        if rows and str(rows[0].get("date", "")) < REGIME_DATA_CLEAN_FROM:
            logger.info(f"[regime_velocity] {region}: riga {rows[0].get('date')} anteriore "
                        f"a {REGIME_DATA_CLEAN_FROM} (dati non confrontabili) — ignorata")
            return None, None, None
        if rows:
            prev_gz = rows[0].get(col_gz)
            prev_iz = rows[0].get(col_iz)
            # FIX F6: calcola i giorni effettivi tra oggi e la riga trovata
            try:
                row_date = datetime.strptime(rows[0].get("date", ""), "%Y-%m-%d")
                actual_days = (datetime.now() - row_date).days
            except Exception:
                actual_days = days_ago   # fallback se parsing fallisce
            logger.debug(f"[regime_velocity] {region} storico ~{target}: "
                         f"G_z={prev_gz} I_z={prev_iz} actual_days={actual_days}")
            return (float(prev_gz) if prev_gz is not None else None,
                    float(prev_iz) if prev_iz is not None else None,
                    actual_days)
        else:
            logger.debug(f"[regime_velocity] {region}: nessun dato storico disponibile")
            return None, None, None
    except Exception as e:
        logger.warning(f"[regime_velocity] Fetch storico fallito per {region}: {e}")
        return None, None, None


def collect_regime_by_region() -> dict:
    """Chiama artemis_regime_engine per USA, EU, CAN."""
    fred_local = Fred(api_key=FRED_KEY) if FRED_KEY else None
    out = {}
    for region in ("USA", "EU", "CAN"):
        try:
            # Recupera G_z / I_z di 28 giorni fa per il Regime Velocity Flag
            # FIX F6: unpacking di 3 valori (+ actual_days per normalizzazione delta)
            prev_gz, prev_iz, actual_days = _fetch_prev_regime_scores(region, days_ago=28)
            r = classify_regime(region, fred=fred_local,
                                prev_gz_4w=prev_gz, prev_iz_4w=prev_iz,
                                actual_days_4w=actual_days)
        except Exception as e:
            logger.warning(f"[regime_region] {region}: {e}")
            r = {
                "region": region,
                "motore_primario": "INDETERMINATO",
                "confidence_pct":  0.0,
                "G_z": None, "I_z": None, "L_z": None,
                "generational_season": "INDETERMINATO",
                "asset_bias": {},
            }
        out[region] = {
            "motore":             r.get("motore_primario", "INDETERMINATO"),
            "motore_label":       r.get("motore_label", ""),
            "hawk_wing":          r.get("hawk_wing"),
            "generational_season":r.get("generational_season"),
            "confidence_pct":     r.get("confidence_pct", 0.0),
            "G_z":                r.get("G_z"),
            "I_z":                r.get("I_z"),
            "L_z":                r.get("L_z"),
            "G_z_struct":         r.get("G_z_struct"),
            "I_z_struct":         r.get("I_z_struct"),
            "asset_bias":         r.get("asset_bias", {}),
            "transition_risk":    r.get("transition_risk"),
            "i_z_momentum":       r.get("i_z_momentum", {}),
            "regime_velocity":    r.get("regime_velocity", {}),
            # Diagnostica (fix 2026-09-25): prima scartata, ora salvata nello
            # snapshot → un salto di G_z/I_z si diagnostica leggendo il database.
            "fallback_notes":     r.get("fallback_notes"),
            "data_notes":         r.get("data_notes"),
            "sources":            r.get("sources"),
            "growth_components":  [
                {"name": c.get("name"), "raw": c.get("raw"),
                 "z_op": (round(c["z_op"], 3) if isinstance(c.get("z_op"), (int, float)) else None),
                 "weight": c.get("weight")}
                for c in ((r.get("components") or {}).get("growth") or [])
            ],
            "inflation_components": [
                {"name": c.get("name"), "raw": c.get("raw"),
                 "z_op": (round(c["z_op"], 3) if isinstance(c.get("z_op"), (int, float)) else None),
                 "weight": c.get("weight")}
                for c in ((r.get("components") or {}).get("inflation") or [])
            ],
            "liquidity_components": [
                {"name": c.get("name"), "raw": c.get("raw"),
                 "z_op": (round(c["z_op"], 3) if isinstance(c.get("z_op"), (int, float)) else None),
                 "weight": c.get("weight")}
                for c in ((r.get("components") or {}).get("liquidity") or [])
            ],
        }
        mom = out[region].get("i_z_momentum", {})
        mom_signal = mom.get("momentum_signal", "NEUTRO") if mom else "NEUTRO"
        mom_str = f" | ⚡{mom_signal}" if mom_signal != "NEUTRO" else ""

        vel = out[region].get("regime_velocity", {})
        vel_signal = vel.get("signal", "DATI_INSUFFICIENTI") if vel else "DATI_INSUFFICIENTI"
        vel_str = f" | 🔀{vel_signal}" if vel_signal not in ("DATI_INSUFFICIENTI", "STABILE") else ""

        logger.info(f"[regime_region] {region}: {out[region]['motore']} "
                    f"conf={out[region]['confidence_pct']}%{mom_str}{vel_str}")
    return out


# ── SEZIONE 7: SCRITTURA SUPABASE ────────────────────────────────────────────

def write_to_supabase(snapshot: dict) -> bool:
    """
    Scrive snapshot in ARTEMIS_SIGNALS (Supabase) — Schema v2.
    Struttura: 20 colonne chiave queryabili + full_snapshot JSONB.
    Upsert su date per evitare duplicati nella stessa sessione.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("[supabase] Credenziali mancanti — skip scrittura. "
                       "Aggiungere ARTEMIS_SUPABASE_URL e ARTEMIS_SUPABASE_KEY "
                       "ai GitHub Secrets.")
        return False
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        vol    = snapshot.get("vol", {})
        rates  = snapshot.get("rates", {})
        fx     = snapshot.get("fx", {})
        regime = snapshot.get("regime", {})
        dragon = snapshot.get("dragon_health", {})

        # ── 20 colonne chiave ──────────────────────────────────────────────
        record = {
            "date":               snapshot["date"],
            "session_id":         snapshot["session_id"],
            # ── Volatilità completa ───────────────────────────────────────
            "vix":                vol.get("vix"),
            "vvix":               vol.get("vvix"),
            "skew":               vol.get("skew"),
            "move":               vol.get("move"),
            "ovx":                vol.get("ovx"),
            "gvz":                vol.get("gvz"),
            "vxn":                vol.get("vxn"),
            "vix3m":              vol.get("vix3m"),
            "vix_term_slope":     vol.get("vix_term_slope"),
            "wti":                vol.get("wti"),
            "skew_vix_ratio":     vol.get("skew_vix_ratio"),
            "move_vix_ratio":     vol.get("move_vix_ratio"),
            "ovx_wti_ratio":      vol.get("ovx_wti_ratio"),
            # ── Tassi & Credito completo ──────────────────────────────────
            "t10y2y":             rates.get("t10y2y"),
            "t10y3m":             rates.get("t10y3m"),
            "curve_trend":        rates.get("curve_trend"),
            "curve_signal":       rates.get("curve_signal"),
            "curve_delta_20d":    rates.get("curve_delta_20d"),
            "t5yie":              rates.get("t5yie"),
            "sofr":               rates.get("sofr"),
            "cpi_yoy":            rates.get("cpi_yoy"),
            "pce_yoy":            rates.get("pce_yoy"),
            "nfp":                rates.get("nfp"),
            "lei":                rates.get("lei"),
            "fed_bs":             rates.get("fed_bs"),
            "tga":                rates.get("tga"),
            "rrp":                rates.get("rrp"),
            "net_liquidity":      rates.get("net_liquidity"),
            "net_liq_delta_4w":   rates.get("net_liq_delta_4w"),
            "fra_ois":            rates.get("fra_ois"),
            "fra_ois_flag":       ("CRISI_FUNDING"    if (rates.get("fra_ois") or 0) > 1.00
                                   else "STRESS_MODERATO" if (rates.get("fra_ois") or 0) > 0.50
                                   else "OK"           if rates.get("fra_ois") is not None
                                   else None),
            "hy_oas":             rates.get("hy_oas"),
            "ig_oas":             rates.get("ig_oas"),
            "bbb_oas":            rates.get("bbb_oas"),
            "nfci":               rates.get("nfci"),
            "pct_above_ma200":    rates.get("pct_above_ma200"),
            "margin_debt_yoy":    rates.get("margin_debt_yoy"),

            # ── Ciclo monetario e credito ─────────────────────────────────
            "m2_yoy":             rates.get("m2_yoy"),
            "bank_credit_yoy":    rates.get("bank_credit_yoy"),
            "consumer_credit_yoy":rates.get("consumer_credit_yoy"),
            "ci_loans_yoy":       rates.get("ci_loans_yoy"),
            "pce_core_yoy":       rates.get("pce_core_yoy"),
            "credit_cycle_signal":rates.get("credit_cycle_signal"),
            "sloos_tightening":   rates.get("sloos_tightening"),
            "delinquency_cc":     rates.get("delinquency_cc"),
            "total_credit_yoy":   rates.get("total_credit_yoy"),
            "debt_service_ratio": rates.get("debt_service_ratio"),
            "hh_debt_gdp":        rates.get("hh_debt_gdp"),
            "corp_debt_yoy":      rates.get("corp_debt_yoy"),
            "debt_stress_signal": rates.get("debt_stress_signal"),
            # ── Ciclo immobiliare ─────────────────────────────────────────
            "case_shiller_yoy":   rates.get("case_shiller_yoy"),
            "housing_starts":     rates.get("housing_starts"),
            "mortgage_30y":       rates.get("mortgage_30y"),
            # ── Ciclo lavoro avanzato ─────────────────────────────────────
            "jobless_claims":     rates.get("jobless_claims"),
            "jolts_openings":     rates.get("jolts_openings"),
            # ── Ciclo produttivo ──────────────────────────────────────────
            "indpro_yoy":         rates.get("indpro_yoy"),
            "retail_sales_yoy":   rates.get("retail_sales_yoy"),
            "real_disp_income_yoy":rates.get("real_disp_income_yoy"),
            # ── FX & Commodity completo ───────────────────────────────────
            "dxy":                fx.get("dxy"),
            "eurusd":             fx.get("eurusd"),
            "usdjpy":             fx.get("usdjpy"),
            "bdry":               fx.get("bdry"),
            "gold":               fx.get("gold"),
            "gold_yoy_pct":       fx.get("gold_yoy_pct"),
            "silver":             fx.get("silver"),
            "copper":             fx.get("copper"),
            "copper_gold_ratio":  fx.get("copper_gold_ratio"),
            "copper_gold_signal": fx.get("copper_gold_signal"),
            # ── Sentiment completo ────────────────────────────────────────
            "fear_greed_score":   snapshot.get("sentiment", {}).get("fear_greed_score"),
            "fear_greed_synthetic": snapshot.get("sentiment", {}).get("fear_greed_synthetic"),
            "cboe_put_call":      snapshot.get("sentiment", {}).get("cboe_put_call"),
            "naaim_exposure":     snapshot.get("sentiment", {}).get("naaim_exposure"),
            "naaim_synthetic":    snapshot.get("sentiment", {}).get("naaim_synthetic"),
            "gpr_index":          snapshot.get("sentiment", {}).get("gpr_index"),
            "gpr_synthetic":      snapshot.get("sentiment", {}).get("gpr_synthetic"),
            "epu_index":          snapshot.get("sentiment", {}).get("epu_index"),
            "umcsent":            snapshot.get("sentiment", {}).get("umcsent"),
            "smart_sentiment_score": snapshot.get("sentiment", {}).get("smart_sentiment_score"),
            "smart_sentiment_label": snapshot.get("sentiment", {}).get("smart_sentiment_label"),
            # ── EIA ───────────────────────────────────────────────────────
            "eia_crude_inventories": snapshot.get("eia", {}).get("eia_crude_inventories_wk"),
            "eia_crude_roc_12w":  snapshot.get("eia", {}).get("eia_crude_roc_12w"),
            "eia_haircut":        snapshot.get("eia", {}).get("eia_haircut"),

            "socmint_alerts":     snapshot.get("socmint", {}).get("alerts"),
            "socmint_alert_count":len(snapshot.get("socmint", {}).get("alerts", []) or []),
            # ── Regime qualitativo ────────────────────────────────────────
            "regime_primary":     regime.get("regime_primary"),
            "regime_secondary":   regime.get("regime_secondary"),
            "regime_confidence":  regime.get("confidence"),
            "regime_signals_active": regime.get("signals_active"),
            "vix_gpr_div":        snapshot.get("vix_gpr_div"),
            "artemis_status":     snapshot.get("artemis_status"),

            "signals_missing":    regime.get("signals_missing"),
            # ── Long Volatility Monitor ───────────────────────────────────
            "long_vol_term_structure":  snapshot.get("long_vol", {}).get("term_structure_signal"),
            "long_vol_carry_regime":    snapshot.get("long_vol", {}).get("long_vol_carry_regime"),
            "long_vol_rv_iv_ratio":     snapshot.get("long_vol", {}).get("realized_implied_ratio"),
            "long_vol_artemis_regime":  snapshot.get("long_vol", {}).get("artemis_vol_regime"),
            "long_vol_cole_insight":    snapshot.get("long_vol", {}).get("cole_insight"),
            "vxz_price":          snapshot.get("long_vol", {}).get("vxz_price"),
            "vxz_3m_return":      snapshot.get("long_vol", {}).get("vxz_3m_return"),
            # ── CWARP ─────────────────────────────────────────────────────
            "cwarp_top1_ticker":  (snapshot.get("cwarp", {}).get("ranking") or [None])[0],
            "cwarp_available":    snapshot.get("cwarp", {}).get("cwarp_available"),

            "cwarp_top1_score":   (lambda r, c: c.get(r[0], {}).get("cwarp") if r else None)(
                                      snapshot.get("cwarp", {}).get("ranking") or [],
                                      snapshot.get("cwarp", {}).get("candidates") or {}
                                  ),
            "cwarp_quality":      snapshot.get("cwarp", {}).get("cwarp_quality"),
            "cwarp_rf_rate":      snapshot.get("cwarp", {}).get("rf_rate"),
            "cwarp_candidates":   snapshot.get("cwarp", {}).get("candidates"),
            # ── Dragon Health ─────────────────────────────────────────────
            "dragon_score":       dragon.get("dragon_score"),
            "rebalancing_urgency":dragon.get("rebalancing_urgency"),
            "dragon_rebalancing_urgency": dragon.get("rebalancing_urgency"),
            "dragon_rebalancing_needed":  dragon.get("rebalancing_needed"),
            "dragon_drift":       dragon.get("drift"),
            "dragon_recommendations": dragon.get("recommendations"),
            "dragon_current_alloc":   dragon.get("current_allocation"),

            # ── Insider ───────────────────────────────────────────────────
            # FIX 2026-07: stesso bug di full_snapshot — json.dumps() qui
            # produceva uno scalare jsonb (stringa) invece di un oggetto/None
            # jsonb navigabile. None passa comunque correttamente come NULL.
            "form4_cluster":      snapshot.get("insider", {}).get("form4_cluster"),
            "congress_trade":     snapshot.get("insider", {}).get("congress_trade"),
            "insider_flags":      snapshot.get("insider", {}).get("insider_flags"),

            # ── Regime per regione (USA/EU/CAN) ───────────────────────────
            "usa_motore":         snapshot.get("regime_by_region", {}).get("USA", {}).get("motore"),
            "usa_motore_label":   snapshot.get("regime_by_region", {}).get("USA", {}).get("motore_label"),
            "usa_hawk_wing":      snapshot.get("regime_by_region", {}).get("USA", {}).get("hawk_wing"),
            "usa_confidence_pct": snapshot.get("regime_by_region", {}).get("USA", {}).get("confidence_pct"),
            "usa_transition_risk":snapshot.get("regime_by_region", {}).get("USA", {}).get("transition_risk"),
            "usa_generational_season": snapshot.get("regime_by_region", {}).get("USA", {}).get("generational_season"),
            "usa_g_z":            snapshot.get("regime_by_region", {}).get("USA", {}).get("G_z"),
            "usa_i_z":            snapshot.get("regime_by_region", {}).get("USA", {}).get("I_z"),
            "usa_l_z":            snapshot.get("regime_by_region", {}).get("USA", {}).get("L_z"),
            "usa_g_z_struct":     snapshot.get("regime_by_region", {}).get("USA", {}).get("G_z_struct"),
            "usa_i_z_struct":     snapshot.get("regime_by_region", {}).get("USA", {}).get("I_z_struct"),
            "usa_asset_bias":     snapshot.get("regime_by_region", {}).get("USA", {}).get("asset_bias"),
            "eu_motore":          snapshot.get("regime_by_region", {}).get("EU", {}).get("motore"),
            "eu_confidence_pct":  snapshot.get("regime_by_region", {}).get("EU", {}).get("confidence_pct"),
            "eu_generational_season": snapshot.get("regime_by_region", {}).get("EU", {}).get("generational_season"),
            "eu_g_z":             snapshot.get("regime_by_region", {}).get("EU", {}).get("G_z"),
            "eu_i_z":             snapshot.get("regime_by_region", {}).get("EU", {}).get("I_z"),
            "can_motore":         snapshot.get("regime_by_region", {}).get("CAN", {}).get("motore"),
            "can_confidence_pct": snapshot.get("regime_by_region", {}).get("CAN", {}).get("confidence_pct"),
            "can_generational_season": snapshot.get("regime_by_region", {}).get("CAN", {}).get("generational_season"),
            "can_g_z":            snapshot.get("regime_by_region", {}).get("CAN", {}).get("G_z"),
            "can_i_z":            snapshot.get("regime_by_region", {}).get("CAN", {}).get("I_z"),
            # ── JSONB full snapshot (tutto il resto) ──────────────────────
            # FIX 2026-07: NON avvolgere in json.dumps() — il client
            # supabase-py serializza già correttamente un dict Python come
            # oggetto jsonb navigabile (stesso pattern di cwarp_candidates
            # sopra, che infatti funziona nelle query SQL con -> / ->>).
            # json.dumps() qui produceva una STRINGA contenente JSON, che
            # Postgres salva come scalare jsonb (non un oggetto), rendendo
            # full_snapshot->'rates'->>'campo' sempre NULL nelle query SQL
            # storiche — bug scoperto confrontando query 3/4 su Supabase.
            "full_snapshot": {
                "vol":              vol,
                "rates":            rates,
                "fx":               snapshot.get("fx", {}),
                "sentiment":        snapshot.get("sentiment", {}),
                "long_vol":         snapshot.get("long_vol", {}),
                "cwarp":            snapshot.get("cwarp", {}),
                "dragon_health":    dragon,
                "vix_gpr_div":      snapshot.get("vix_gpr_div", {}),
                "regime":           regime,
                "regime_by_region": snapshot.get("regime_by_region", {}),
                "insider":          snapshot.get("insider", {}),
                "socmint":          snapshot.get("socmint", {}),
                "eia":              snapshot.get("eia", {}),
                "artemis_status":   snapshot.get("artemis_status", {}),
                "data_audit":       snapshot.get("data_audit", {}),
            },
            # Meta
            "scan_duration_sec":  snapshot.get("scan_duration_sec"),
            "updated_at":         datetime.now().isoformat(),
        }

        client.table("ARTEMIS_SIGNALS").upsert(
            record, on_conflict="date"
        ).execute()
        logger.info(f"[supabase] ✅ Scritto → ARTEMIS_SIGNALS v2 ({snapshot['session_id']})")
        return True
    except Exception as e:
        logger.error(f"[supabase] Scrittura fallita: {e} — dato NON persistito")
        return False


# ── SEZIONE 8: REPORT CONSOLE ────────────────────────────────────────────────

def print_report(snapshot: dict) -> None:
    """Report console strutturato per lettura del Gestore prima della Skill ARTEMIS."""
    W  = 76
    vol      = snapshot.get("vol", {})
    rates    = snapshot.get("rates", {})
    fx       = snapshot.get("fx", {})
    sent     = snapshot.get("sentiment", {})
    regime   = snapshot.get("regime", {})
    reg_reg  = snapshot.get("regime_by_region", {})
    long_vol = snapshot.get("long_vol", {})
    cwarp    = snapshot.get("cwarp", {})
    dragon   = snapshot.get("dragon_health", {})
    socmint  = snapshot.get("socmint", {})

    print()
    print("═" * W)
    print("  ARTEMIS SIGNAL SCANNER — REPORT GESTORE")
    print(f"  {snapshot.get('date')} · Session: {snapshot.get('session_id')}")
    print(f"  Durata scan: {snapshot.get('scan_duration_sec', 0):.1f}s")
    print("═" * W)

    # Regime flags
    print()
    print("  🐉 REGIME ARTEMIS (Signal Scanner — qualitativo)")
    print("  " + "─" * (W - 2))
    if regime.get("dead_stop_reason"):
        print(f"  ⛔ DEAD STOP: {regime['dead_stop_reason']}")
    else:
        prim = regime.get("regime_primary", "?")
        sec  = regime.get("regime_secondary")
        conf = regime.get("confidence", 0)
        label = {"SERPENTE": "🐍 Crescita Secolare",
                 "FALCO_SX": "🦅 Deflazione / Hawk Sinistra",
                 "FALCO_DX": "🦅 Stagflazione / Hawk Destra",
                 "FENICE":   "🔥 Svalutazione Fiat / Phoenix",
                 "TRANSIZIONE": "🔀 Regime Misto / Transizione"}.get(prim, prim)
        print(f"  Motore: {label}")
        if sec:
            print(f"  Secondario: {sec}")
        print(f"  Confidence: {conf:.0f}%")
        print()
        for s in regime.get("signals_active", [])[:8]:
            print(f"    ✓ {s}")
        if regime.get("signals_missing"):
            print(f"    ⚠️  Mancanti: {regime['signals_missing']}")

    # Regime per regione (regime engine)
    print()
    print("  📊 REGIME ENGINE (quantitativo) — USA / EU / CAN")
    print("  " + "─" * (W - 2))
    for region, rd in reg_reg.items():
        motore  = rd.get("motore", "?")
        conf    = rd.get("confidence_pct", 0)
        season  = rd.get("generational_season", "?")
        g_z     = rd.get("G_z")
        i_z     = rd.get("I_z")
        t_risk  = rd.get("transition_risk", "?")
        print(f"  {region:<4} → {motore:<14} conf={conf:.0f}%  "
              f"season={season}  G_z={g_z}  I_z={i_z}  risk={t_risk}")

    # Volatilità
    print()
    print("  📈 VOLATILITÀ")
    print("  " + "─" * (W - 2))
    print(f"  VIX={vol.get('vix','N/A'):<8} VVIX={vol.get('vvix','N/A'):<8} "
          f"SKEW={vol.get('skew','N/A'):<8} MOVE={vol.get('move','N/A')}")
    print(f"  OVX={vol.get('ovx','N/A'):<8} GVZ={vol.get('gvz','N/A'):<8} "
          f"WTI={vol.get('wti','N/A'):<8} VIX_slope={vol.get('vix_term_slope','N/A')}")

    # Long Vol Monitor
    print()
    print("  🌊 LONG VOLATILITY MONITOR (Artemis)")
    print("  " + "─" * (W - 2))
    print(f"  Term structure: {long_vol.get('term_structure_signal','N/A')}")
    print(f"  Carry regime  : {long_vol.get('long_vol_carry_regime','N/A')}")
    print(f"  RV/IV ratio   : {long_vol.get('realized_implied_ratio','N/A')}")
    print(f"  Artemis vol   : {long_vol.get('artemis_vol_regime','N/A')}")
    if long_vol.get("cole_insight"):
        print(f"  Cole insight  : {long_vol.get('cole_insight')[:80]}...")

    # Macro tassi
    print()
    print("  💹 TASSI & CREDITO")
    print("  " + "─" * (W - 2))
    print(f"  T10Y2Y={_fmt(rates.get('t10y2y')):<8} T10Y3M={_fmt(rates.get('t10y3m')):<8} "
          f"CPI_YoY={_fmt(rates.get('cpi_yoy'))}% T5YIE={_fmt(rates.get('t5yie'))}")
    if rates.get("curve_trend"):
        print(f"  Curva: {rates.get('curve_trend')} (Δ20g={_fmt(rates.get('curve_delta_20d'),3)}) "
              f"→ {rates.get('curve_signal','')}")
    print(f"  HY_OAS={_fmt(rates.get('hy_oas')):<8} IG_OAS={_fmt(rates.get('ig_oas')):<8} "
          f"PCT_MA200={_fmt(rates.get('pct_above_ma200'), 1)}%")
    print(f"  CreditCycle={rates.get('credit_cycle_signal','N/A')} · "
          f"DebtStress={rates.get('debt_stress_signal','N/A')} · "
          f"SLOOS={_fmt(rates.get('sloos_tightening'))} · NFCI={_fmt(rates.get('nfci'),3)} · "
          f"BBB_OAS={_fmt(rates.get('bbb_oas'))}")
    print(f"  M2_YoY={_fmt(rates.get('m2_yoy'))}% · PCE_core={_fmt(rates.get('pce_core_yoy'))}% · "
          f"IndPro={_fmt(rates.get('indpro_yoy'))}% · Retail={_fmt(rates.get('retail_sales_yoy'))}% · "
          f"Jobless={_fmt(rates.get('jobless_claims'),0)}")
    print(f"  Housing: starts={_fmt(rates.get('housing_starts'),0)} · "
          f"mortgage30y={_fmt(rates.get('mortgage_30y'))}% · "
          f"CaseShiller={_fmt(rates.get('case_shiller_yoy'))}% · NFP={_fmt(rates.get('nfp'),0)}")
    # Nuovi segnali macro (audit 2026-06)
    if rates.get("real_yield_10y") is not None:
        print(f"  RealYield10y={_fmt(rates.get('real_yield_10y'))}% → {rates.get('real_yield_signal','')}")
    if rates.get("infl_5y5y") is not None:
        print(f"  Infl5y5y={_fmt(rates.get('infl_5y5y'))}% → {rates.get('infl_5y5y_signal','')}")
    if rates.get("claims_ratio") is not None:
        print(f"  Claims ratio (cont/init)={_fmt(rates.get('claims_ratio'))} → {rates.get('labor_stress_signal','')}")
    if rates.get("rate_2y") is not None:
        print(f"  2Y={_fmt(rates.get('rate_2y'))}% (Δ20g={_fmt(rates.get('rate_2y_delta_20d'),3)}pp)")
        if rates.get("equity_rates_scenario"):
            print(f"    → {rates.get('equity_rates_scenario')}")
    if rates.get("rate_30y") is not None:
        print(f"  30Y={_fmt(rates.get('rate_30y'))}% (Δ20g={_fmt(rates.get('rate_30y_delta_20d'),3)}pp)")
    if rates.get("term_premium_10y") is not None:
        print(f"  Term premium 10y={_fmt(rates.get('term_premium_10y'),2)}% "
              f"→ {rates.get('term_premium_signal','')}")
    if rates.get("curve_2s30s_trend"):
        print(f"  Curva 2-30y: {rates.get('curve_2s30s_trend')} → {rates.get('curve_2s30s_signal','')}")

    # FX / Commodity
    print()
    print("  💰 FX & COMMODITY")
    print("  " + "─" * (W - 2))
    print(f"  DXY={fx.get('dxy','N/A'):<8} Gold=${fx.get('gold','N/A'):<10} "
          f"Silver=${fx.get('silver','N/A')}")
    print(f"  Cu/Au={fx.get('copper_gold_ratio','N/A'):<10} "
          f"Signal={fx.get('copper_gold_signal','N/A')} "
          f"Gold_YoY={fx.get('gold_yoy_pct','N/A')}%")
    if fx.get("gold_equity_corr_30d") is not None:
        print(f"  Gold/Equity corr 30d={_fmt(fx.get('gold_equity_corr_30d'),3)} "
              f"→ {fx.get('correlation_regime_signal','')}")

    # CWARP
    print()
    print("  📐 CWARP CALCULATOR (Cole)")
    print("  " + "─" * (W - 2))
    if cwarp.get("cwarp_available"):
        print(f"  Quality: {cwarp.get('cwarp_quality')} · "
              f"RF_rate: {cwarp.get('rf_rate','N/A')}")
        for ticker in cwarp.get("ranking", [])[:5]:
            c = cwarp["candidates"].get(ticker, {})
            cw = c.get("cwarp")
            qual = c.get("quality", "?")
            sign = "✅" if cw is not None and cw > 0 else "❌"
            print(f"  {sign} {ticker:<6} CWARP={str(cw):<8} [{qual}] {c.get('name','')[:30]}")
    else:
        print(f"  CWARP non disponibile: {cwarp.get('error','dati insufficienti')}")

    # Livello di gestione del rischio (separato dal Dragon Health Check —
    # non è una raccomandazione di ribilanciamento, solo contesto d'esecuzione)
    rr = snapshot.get("risk_regime", {})
    if rr:
        print()
        print("  🛡️  GESTIONE DEL RISCHIO (ritmo d'esecuzione, non target)")
        print("  " + "─" * (W - 2))
        print(f"  Livello: {rr.get('livello')} "
              f"({rr.get('n_segnali_stress')} segnali di stress attivi)")
        for f in rr.get("segnali_attivi", []):
            print(f"    ⚠️  {f}")
        print(f"  → {rr.get('guida_esecuzione')}")

    # Dragon Health Check
    print()
    print("  🐉 DRAGON PORTFOLIO HEALTH CHECK")
    print("  " + "─" * (W - 2))
    score = dragon.get("dragon_score")
    urgency = dragon.get("rebalancing_urgency", "N/A")
    print(f"  Dragon Score: {score if score is not None else 'N/A'}/100 · "
          f"Urgency: {urgency}")
    print(f"  Profilo: {dragon.get('active_profile', '?')} · "
          f"fonte target: {dragon.get('target_source', '?')}")
    # Confronto affiancato col Cole puro (riferimento)
    _cc = dragon.get("comparison_cole")
    if _cc and _cc.get("dragon_score") is not None:
        _ap = dragon.get("active_profile", "Profilo attivo")
        print(f"  [confronto] {_ap}: {score} · Cole puro (riferimento): "
              f"{_cc.get('dragon_score')}")
    if dragon.get("drift"):
        for comp, d in dragon["drift"].items():
            bar = "⚠️" if abs(d) >= DRAGON_DRIFT_ALERT else "✅"
            target  = DRAGON_TARGET.get(comp, 0) * 100
            current = (dragon.get("current_allocation") or {}).get(comp, 0) * 100
            print(f"  {bar} {comp:<20} target={target:.0f}% "
                  f"attuale={current:.0f}% drift={d*100:+.1f}%")
    for rec in dragon.get("recommendations", []):
        print(f"  → {rec}")

    # Sentiment
    print()
    print("  🧠 SENTIMENT")
    print("  " + "─" * (W - 2))
    smart = sent.get("smart_sentiment_score")
    slabel = sent.get("smart_sentiment_label", "N/A")
    fg = sent.get("fear_greed_score") or f"[PROXY:{sent.get('fear_greed_synthetic','N/A')}]"
    print(f"  Smart Sent: {smart} ({slabel})  F&G: {fg}  "
          f"NAAIM: {sent.get('naaim_exposure','N/A')}  "
          f"GPR: {sent.get('gpr_index','N/A')}")

    # SOCMINT
    alerts = socmint.get("alerts", [])
    print()
    print("  👁️  SOCMINT LITE")
    print("  " + "─" * (W - 2))
    if alerts:
        for a in alerts:
            print(f"  ⚠️  {a['ticker']}: SI={a['short_pct']}% "
                  f"Vol={a['vol_ratio']}× → {a.get('note','')}")
    else:
        print("  Nessun alert squeeze rilevato")

    print()
    print("═" * W)
    print("  ⏸  BREAKPOINT — ARTEMIS")
    print("  Leggere questo report prima di aprire la Skill ARTEMIS.")
    print("  Passare artemis_status.json come contesto iniziale.")
    print("  Il regime del Signal Scanner è ORIENTATIVO —")
    print("  il Regime Engine (quantitativo) è la fonte primaria.")
    print("═" * W)
    print()


# ── SEZIONE 9: GENERA ARTEMIS_STATUS_JSON ────────────────────────────────────

def generate_artemis_status(snapshot: dict) -> dict:
    """
    Genera artemis_status.json — input strutturato per la Skill ARTEMIS.

    Struttura:
        · BLOCCO REGIME    — regime primario + per regione + asset bias
        · BLOCCO SEGNALI   — tutti i segnali chiave etichettati
        · BLOCCO LONG_VOL  — stato long volatility monitor
        · BLOCCO CWARP     — ranking CWARP asset candidati
        · BLOCCO DRAGON    — health check portafoglio Dragon
        · BLOCCO INSIDER   — Form4 cluster + Congress trades
        · BLOCCO EIA       — inventari greggio + ROC
        · BLOCCO SOCMINT   — alert squeeze potenziali
        · META             — date, session, qualità dati
    """
    vol     = snapshot.get("vol", {})
    rates   = snapshot.get("rates", {})
    fx      = snapshot.get("fx", {})
    sent    = snapshot.get("sentiment", {})
    regime  = snapshot.get("regime", {})
    cwarp   = snapshot.get("cwarp", {})
    dragon  = snapshot.get("dragon_health", {})
    lv      = snapshot.get("long_vol", {})
    rreg    = snapshot.get("regime_by_region", {})
    insider = snapshot.get("insider", {})
    eia     = snapshot.get("eia", {})
    socmint = snapshot.get("socmint", {})

    # Regime USA primario (fonte regime engine — più affidabile del signal scanner)
    usa_regime = rreg.get("USA", {})

    status = {
        # ── BLOCCO REGIME ──────────────────────────────────────────────────
        "regime": {
            "primary":            usa_regime.get("motore", regime.get("regime_primary")),
            "label":              usa_regime.get("motore_label", ""),
            "hawk_wing":          usa_regime.get("hawk_wing"),
            "generational_season":usa_regime.get("generational_season"),
            "confidence_pct":     usa_regime.get("confidence_pct", regime.get("confidence")),
            "transition_risk":    usa_regime.get("transition_risk"),
            "asset_bias":         usa_regime.get("asset_bias", {}),
            "G_z":                usa_regime.get("G_z"),
            "I_z":                usa_regime.get("I_z"),
            "L_z":                usa_regime.get("L_z"),
            "G_z_struct":         usa_regime.get("G_z_struct"),
            "I_z_struct":         usa_regime.get("I_z_struct"),
            "signals_scanner":    regime.get("signals_active", []),
            "by_region":          rreg,
            "source_note": ("Regime Engine (quantitativo, 5+15y z-score) "
                            "è la fonte primaria. Signal Scanner (qualitativo) "
                            "è orientativo."),
        },

        # ── BLOCCO SEGNALI ─────────────────────────────────────────────────
        "signals": {
            # Volatilità
            "vix":               vol.get("vix"),
            "vvix":              vol.get("vvix"),
            "skew":              vol.get("skew"),
            "move":              vol.get("move"),
            "ovx":               vol.get("ovx"),
            "gvz":               vol.get("gvz"),
            "vxn":               vol.get("vxn"),
            "vix3m":             vol.get("vix3m"),
            "vix_term_slope":    vol.get("vix_term_slope"),
            # Tassi
            "t10y2y":            rates.get("t10y2y"),
            "t10y3m":            rates.get("t10y3m"),
            "t5yie":             rates.get("t5yie"),
            "sofr":              rates.get("sofr"),
            "cpi_yoy":           rates.get("cpi_yoy"),
            "pce_yoy":           rates.get("pce_yoy"),
            "hy_oas":            rates.get("hy_oas"),
            "ig_oas":            rates.get("ig_oas"),
            "pct_above_ma200":   rates.get("pct_above_ma200"),
            "net_liquidity":     rates.get("net_liquidity"),
            "net_liq_delta_4w":  rates.get("net_liq_delta_4w"),
            "margin_debt_yoy":   rates.get("margin_debt_yoy"),
            "baa10y_spread":     rates.get("baa10y_spread"),
            "stlfsi4":           rates.get("stlfsi4"),
            # FRA/OIS con flag interpretativo (stress interbancario)
            "fra_ois":           rates.get("fra_ois"),
            "fra_ois_flag":      ("CRISI_FUNDING"   if (rates.get("fra_ois") or 0) > 1.00
                                  else "STRESS_MODERATO" if (rates.get("fra_ois") or 0) > 0.50
                                  else "OK"          if rates.get("fra_ois") is not None
                                  else "N/A"),
            # FX/Commodity
            "dxy":               fx.get("dxy"),
            "eurusd":            fx.get("eurusd"),
            "usdjpy":            fx.get("usdjpy"),
            "gold":              fx.get("gold"),
            "silver":            fx.get("silver"),
            "gold_yoy_pct":      fx.get("gold_yoy_pct"),
            "copper_gold_ratio": fx.get("copper_gold_ratio"),
            "copper_gold_signal":fx.get("copper_gold_signal"),
            "bdry":              fx.get("bdry"),
            # Sentiment
            "fear_greed": (sent.get("fear_greed_score")
                           if sent.get("fear_greed_score") is not None
                           else f"[PROXY:{sent.get('fear_greed_synthetic','N/A')}]"),
            "naaim":      (sent.get("naaim_exposure")
                           if sent.get("naaim_exposure") is not None
                           else f"[PROXY:{sent.get('naaim_synthetic','N/A')}]"),
            "gpr":        (sent.get("gpr_index")
                           if sent.get("gpr_index") is not None
                           else f"[PROXY:{sent.get('gpr_synthetic','N/A')}]"),
            "epu":               sent.get("epu_index"),
            "epu_source":        sent.get("epu_source"),
            "gpr_month":         sent.get("gpr_month"),
            "umcsent":           sent.get("umcsent"),
            "smart_sentiment":   {
                "score": sent.get("smart_sentiment_score"),
                "label": sent.get("smart_sentiment_label"),
            },
            "_proxy_note": ("Valori [PROXY:X] sono stime sintetiche. "
                            "CODEX ARGUS può sovrascriverli con web_fetch."),

            # ── Volatilità — ratio derivati ──
            "skew_vix_ratio":    vol.get("skew_vix_ratio"),
            "move_vix_ratio":    vol.get("move_vix_ratio"),
            "ovx_wti_ratio":     vol.get("ovx_wti_ratio"),
            "wti":               vol.get("wti"),

            # ── Yield curve dinamica (trend + tipo steepening/flattening) ──
            "curve_trend":       rates.get("curve_trend"),
            "curve_signal":      rates.get("curve_signal"),
            "curve_delta_20d":   rates.get("curve_delta_20d"),

            # ── NUOVI SEGNALI (audit 2026-06): conferma regime/transizioni ──
            "real_yield_10y":         rates.get("real_yield_10y"),       # TIPS 10y reale
            "real_yield_signal":      rates.get("real_yield_signal"),    # interpretazione oro/FENICE
            "infl_5y5y":              rates.get("infl_5y5y"),            # aspettative 5y5y forward
            "infl_5y5y_signal":       rates.get("infl_5y5y_signal"),     # anticipa FALCO_DX/SX
            "claims_ratio":           rates.get("claims_ratio"),         # continued/initial claims
            "labor_stress_signal":    rates.get("labor_stress_signal"),  # anticipa FALCO_SX
            "gold_equity_corr_30d":   fx.get("gold_equity_corr_30d"),    # canary risk-off estremo
            "correlation_regime_signal": fx.get("correlation_regime_signal"),
            "rate_2y":                rates.get("rate_2y"),              # DGS2 grezzo
            "rate_2y_delta_20d":      rates.get("rate_2y_delta_20d"),    # trend 20gg
            "equity_rates_scenario":  rates.get("equity_rates_scenario"), # Scenario A/B (report "La Résistance" 2026-06-24): equity+tassi breve = hawkish repricing vs flight-to-safety
            "rate_30y":               rates.get("rate_30y"),             # DGS30 grezzo — tratto lunghissimo, Bond Vigilantes
            "rate_10y":               rates.get("rate_10y"),             # DGS10 (FRED, 1-2 giorni di ritardo)
            "rate_10y_live":          rates.get("rate_10y_live"),        # ^TNX Yahoo, tempo reale
            "rate_30y_live":          rates.get("rate_30y_live"),        # ^TYX Yahoo, tempo reale
            "fed_funds_effective":    rates.get("fed_funds_effective"),
            "fed_hikes_priced_2y":    rates.get("fed_hikes_priced_2y"),
            "fed_pricing_signal":     rates.get("fed_pricing_signal"),
            "term_premium_10y":       rates.get("term_premium_10y"),     # THREEFYTP10 (Kim-Wright) — compenso duration, sfiducia fiscale diretta
            "term_premium_signal":    rates.get("term_premium_signal"),
            "rate_30y_delta_20d":     rates.get("rate_30y_delta_20d"),   # trend 20gg
            "curve_2s30s_trend":      rates.get("curve_2s30s_trend"),    # steepening/flattening tratto 2-30y
            "curve_2s30s_signal":     rates.get("curve_2s30s_signal"),   # interpretazione (Bond Vigilantes / Fed / recessione)

            "credit_cycle_signal": rates.get("credit_cycle_signal"),
            "debt_stress_signal":  rates.get("debt_stress_signal"),
            "sloos_tightening":    rates.get("sloos_tightening"),
            "sloos_demand":        rates.get("sloos_demand"),
            "nfci":                rates.get("nfci"),
            "bbb_oas":             rates.get("bbb_oas"),
            "total_credit_yoy":    rates.get("total_credit_yoy"),
            "corp_debt_yoy":       rates.get("corp_debt_yoy"),
            "consumer_credit_yoy": rates.get("consumer_credit_yoy"),
            "bank_credit_yoy":     rates.get("bank_credit_yoy"),
            "ci_loans_yoy":        rates.get("ci_loans_yoy"),
            "delinquency_cc":      rates.get("delinquency_cc"),
            "debt_service_ratio":  rates.get("debt_service_ratio"),
            "hh_debt_gdp":         rates.get("hh_debt_gdp"),

            # ── Ciclo economico (crescita, lavoro, immobiliare, liquidità) ──
            "m2_yoy":              rates.get("m2_yoy"),
            "nfp":                 rates.get("nfp"),
            "pce_core_yoy":        rates.get("pce_core_yoy"),
            "indpro_yoy":          rates.get("indpro_yoy"),
            "retail_sales_yoy":    rates.get("retail_sales_yoy"),
            "real_disp_income_yoy": rates.get("real_disp_income_yoy"),
            "jobless_claims":      rates.get("jobless_claims"),
            "jolts_openings":      rates.get("jolts_openings"),
            "housing_starts":      rates.get("housing_starts"),
            "mortgage_30y":        rates.get("mortgage_30y"),
            "case_shiller_yoy":    rates.get("case_shiller_yoy"),

            # ── Put/Call CBOE (grezzo — può essere proxy yfinance) ──
            "cboe_put_call":       sent.get("cboe_put_call"),
        },

        # ── BLOCCO LONG VOL ────────────────────────────────────────────────
        "long_vol": {
            "term_structure_signal":  lv.get("term_structure_signal"),
            "long_vol_carry_regime":  lv.get("long_vol_carry_regime"),
            "realized_implied_ratio": lv.get("realized_implied_ratio"),
            "artemis_vol_regime":     lv.get("artemis_vol_regime"),
            "vxz_price":              lv.get("vxz_price"),
            "vxz_3m_return":          lv.get("vxz_3m_return"),
            "cole_insight":           lv.get("cole_insight"),
        },

        # ── BLOCCO CWARP ───────────────────────────────────────────────────
        "cwarp": {
            "available":        cwarp.get("cwarp_available"),
            "quality":          cwarp.get("cwarp_quality"),
            "replacement_port": cwarp.get("replacement_port"),
            "rf_rate":          cwarp.get("rf_rate"),
            "ranking":          cwarp.get("ranking", []),
            "estimated_excluded": cwarp.get("estimated_excluded", []),
            "candidates":       cwarp.get("candidates", {}),
            "cole_note":        cwarp.get("cole_note"),
        },

        # ── BLOCCO DRAGON HEALTH ───────────────────────────────────────────
        "dragon_health": {
            "target_allocation":  dragon.get("target_allocation"),
            "current_allocation": dragon.get("current_allocation"),
            "drift":              dragon.get("drift"),
            "dragon_score":       dragon.get("dragon_score"),
            "rebalancing_urgency":dragon.get("rebalancing_urgency"),
            "rebalancing_needed": dragon.get("rebalancing_needed"),
            "recommendations":    dragon.get("recommendations", []),
            "active_profile":     dragon.get("active_profile"),
            "target_source":      dragon.get("target_source"),
            "target_warning":     dragon.get("target_warning"),
            "comparison_cole":    dragon.get("comparison_cole"),
            "cole_note":          dragon.get("cole_note"),
        },

        # ── BLOCCO INSIDER ─────────────────────────────────────────────────
        "insider": {
            "form4_cluster":  insider.get("form4_cluster"),
            "congress_trade": insider.get("congress_trade"),
            "flags":          insider.get("insider_flags", []),
        },

        # ── BLOCCO EIA ─────────────────────────────────────────────────────
        "eia": {
            "crude_inventories_wk": eia.get("eia_crude_inventories_wk"),
            "crude_roc_12w":        eia.get("eia_crude_roc_12w"),
            "haircut":              eia.get("eia_haircut", False),
        },

        # ── BLOCCO SOCMINT ─────────────────────────────────────────────────
        "socmint": {
            "alerts":  socmint.get("alerts", []),
            "n_alerts": len(socmint.get("alerts", [])),
        },


        # ── META ───────────────────────────────────────────────────────────
        "meta": {
            "date":            snapshot.get("date"),
            "session_id":      snapshot.get("session_id"),
            "scan_duration_s": snapshot.get("scan_duration_sec"),
            "system":          "ARTEMIS Dragon Portfolio",
            "version":         "1.0",
            # FIX A2: data_quality esposto nel JSON per SKILL e ARGUS.
            # OK           → tutti i blocchi principali popolati
            # DEGRADED_FRED → blocco FRED completamente null (credenziali/outage)
            # Campo letto da SKILL/ARGUS per attivare il warning di qualità dati.
            "data_quality":    snapshot.get("data_quality", "OK"),
            "instructions":    (
                "Questo JSON è l'input per la Skill ARTEMIS. "
                "Copialo come contesto iniziale della sessione. "
                "La Skill produrrà il report strutturato automatico "
                "(REGIME / DRAGON HEALTH / SEGNALI / ADVERSARIAL) "
                "e poi sarà disponibile per domande libere."
            ),
        },
    }
    return status


# ── SEZIONE 9b: DATA AUDIT ───────────────────────────────────────────────────

def build_data_audit(snapshot: dict) -> dict:
    """Stato di ogni dato usato nel run: fonte, data osservazione, età, stato.

    Stati: OK · VECCHIO (oltre il ritardo normale) · FERMO (oltre il doppio:
    fonte probabilmente dismessa) · MANCANTE (campo vuoto) · STIMA (valore
    sintetico, non dato reale) · NON_DISPONIBILE (fonte nota come inaccessibile).
    """
    today = datetime.today()
    items = []
    for key, o in sorted(DATA_OBS.items()):
        sid = o.get("series")
        if o.get("freq") == "esterna":
            max_age = o.get("max_age") or 45
        else:
            max_age = _AUDIT_AGE_OVERRIDE.get(sid, _AUDIT_MAX_AGE.get(o.get("freq"), 45))
        age = None
        if o.get("obs_date"):
            try:
                age = (today - pd.Timestamp(o["obs_date"])).days
            except Exception:
                age = None
        if o.get("status_forzato"):
            status = o["status_forzato"]
        elif age is None:
            status = "DATA_IGNOTA"
        elif age <= max_age:
            status = "OK"
        elif age <= 2 * max_age:
            status = "VECCHIO"
        else:
            status = "FERMO"
        items.append({"fonte": o.get("source"), "serie": sid, "frequenza": o.get("freq"),
                      "data_dato": o.get("obs_date"), "eta_giorni": age,
                      "limite_giorni": max_age, "stato": status, "nota": o.get("note")})

    sent  = snapshot.get("sentiment", {}) or {}
    rates = snapshot.get("rates", {}) or {}
    stime = []
    for real_k, syn_k in (("fear_greed_score", "fear_greed_synthetic"),
                          ("naaim_exposure", "naaim_synthetic"),
                          ("gpr_index", "gpr_synthetic")):
        if sent.get(real_k) is None and sent.get(syn_k) is not None:
            stime.append(f"{real_k}: usata STIMA interna {sent.get(syn_k)} (fonte reale non disponibile)")
    if sent.get("spy_pc_proxy") is not None:
        stime.append(f"put/call: solo stima SPY ATM {sent.get('spy_pc_proxy')} (informativa, fuori dai calcoli)")
    if rates.get("pct_above_ma200") is not None and "livello2" in str(rates.get("breadth_method", "")):
        stime.append("ampiezza mercato: fallback su SPY (non vera ampiezza)")
    for c in ((snapshot.get("cwarp") or {}).get("estimated_excluded") or []):
        stime.append(f"CWARP {c.get('ticker', c) if isinstance(c, dict) else c}: stimato, escluso dal ranking")

    non_disp = []
    for lbl, v in (("NAAIM", sent.get("naaim_source")),
                   ("Put/Call CBOE", sent.get("cboe_put_call_source")),
                   ("Form 4 insider", (snapshot.get("insider") or {}).get("form4_status")),
                   ("Congress trades", (snapshot.get("insider") or {}).get("congress_status"))):
        if v and not str(v).startswith("OK"):
            non_disp.append(f"{lbl}: {v}")

    mancanti = []
    for blk in ("vol", "rates", "fx", "sentiment", "eia"):
        for k, v in (snapshot.get(blk) or {}).items():
            if v is None and not k.endswith(("_signal", "_source", "_synthetic", "_proxy", "_status",
                                             "_month", "_period", "_reason")):
                mancanti.append(f"{blk}.{k}")

    regime_notes = {}
    for reg, r in (snapshot.get("regime_by_region") or {}).items():
        notes = (r.get("data_notes") or []) + (r.get("fallback_notes") or [])
        if notes:
            regime_notes[reg] = notes

    # Campi che alimentano regime, scenari e report: se vuoti → avviso
    _CRITICI = {"rates": ["t10y2y", "t10y3m", "t5yie", "sofr", "hy_oas", "ig_oas", "cpi_yoy",
                          "pce_core_yoy", "rate_2y", "rate_10y", "rate_30y", "real_yield_10y",
                          "infl_5y5y", "nfci", "baa10y_spread", "stlfsi4", "fed_funds_effective"],
                "vol": ["vix", "vix3m", "move", "wti"],
                "fx": ["dxy", "eurusd", "gold"],
                "sentiment": ["fear_greed_score", "epu_index", "gpr_index"]}
    critici_mancanti = [f"{b}.{k}" for b, ks in _CRITICI.items() for k in ks
                        if (snapshot.get(b) or {}).get(k) is None]
    # Assi del regime (26/09/2026): la sera del 25/9 StatCan non rispondeva, l'asse
    # inflazione CAN è rimasto vuoto e l'audit segnava comunque tutto OK.
    fonti_regime_ko, fonti_regime_riserva = [], []
    for reg, r in (snapshot.get("regime_by_region") or {}).items():
        for ax in ("G_z", "I_z"):
            if r.get(ax) is None:
                critici_mancanti.append(f"regime.{reg}.{ax}")
        if reg == "USA" and r.get("L_z") is None:
            critici_mancanti.append("regime.USA.L_z")
        for src, info in (r.get("sources") or {}).items():
            st = str((info or {}).get("status", ""))
            if st.startswith("NON DISPONIBILE"):
                fonti_regime_ko.append(f"{reg}.{src}: {st[:90]}")
            elif "riserva" in st:
                fonti_regime_riserva.append(f"{reg}.{src} ({(info or {}).get('fonte', 'riserva')})")
    problemi = [i for i in items if i["stato"] in ("VECCHIO", "FERMO", "DATA_IGNOTA")]
    conteggi = {}
    for i in items:
        conteggi[i["stato"]] = conteggi.get(i["stato"], 0) + 1

    warning = None
    parti = []
    if problemi:
        parti.append("fonti vecchie/ferme: " + ", ".join(
            f"{i['serie']} ({i['stato']}, dato del {i['data_dato']})" for i in problemi[:8])
            + (f" +{len(problemi) - 8}" if len(problemi) > 8 else ""))
    if critici_mancanti:
        parti.append("dati chiave VUOTI in questa run: " + ", ".join(critici_mancanti))
    if fonti_regime_ko:
        parti.append("fonti del regime NON raggiunte: " + " | ".join(fonti_regime_ko))
    if fonti_regime_riserva:
        parti.append("fonti del regime da riserva (dati equivalenti): " + ", ".join(fonti_regime_riserva))
    if regime_notes:
        parti.append("regime: " + " | ".join(f"{k}: {'; '.join(v)}" for k, v in regime_notes.items()))
    if parti:
        warning = "⚠️ QUALITÀ DATI — " + " || ".join(parti)
    if fonti_regime_ko:
        conteggi["FONTI_REGIME_KO"] = len(fonti_regime_ko)

    return {"data_run": today.strftime("%Y-%m-%d"), "conteggi": conteggi,
            "fonti_regime_ko": fonti_regime_ko, "fonti_regime_riserva": fonti_regime_riserva,
            "problemi": problemi, "stime_usate": stime, "fonti_non_disponibili": non_disp,
            "campi_mancanti": mancanti, "critici_mancanti": critici_mancanti,
            "note_regime": regime_notes,
            "warning": warning, "dettaglio": items}


# ── SEZIONE 10: ENTRY POINT ──────────────────────────────────────────────────

def main(portfolio_current: dict | None = None,
         cwarp_candidates: list | None = None) -> dict:
    """
    Raccoglie tutti i segnali, computa regime, scrive in Supabase.
    Genera artemis_status.json per la Skill ARTEMIS.

    Args:
        portfolio_current: allocazione corrente Dragon Portfolio
            Formato: {'equity': 0.24, 'fixed_income': 0.18, ...}
            Se None: Dragon Health Check mostra solo target + placeholder.
        cwarp_candidates:  lista asset candidati per CWARP
            Se None: usa CWARP_CANDIDATES (lista configurabile in testa al file)
            Per aggiungere un nuovo asset: modifica CWARP_CANDIDATES sopra.

    ⏸ BREAKPOINT GESTORE:
        Al termine, leggere il report console.
        Copiare il contenuto di artemis_status.json nella sessione
        con la Skill ARTEMIS come contesto iniziale.
    """
    t_start    = time.time()
    session_id = f"ARTEMIS-{datetime.today().strftime('%Y%m%d')}"
    DATA_OBS.clear()

    logger.info("═" * 60)
    logger.info("ARTEMIS SIGNAL SCANNER v1.0 — AVVIO")
    logger.info(f"Session: {session_id}")
    logger.info("═" * 60)

    # Target operativo: prima da Supabase (ARTEMIS_TARGET_ALLOCATION),
    # riserva nel codice con avviso esplicito se la tabella non è valida.
    load_active_target_from_supabase()

    # FRED instance
    if not FRED_KEY:
        logger.warning("[CONFIG] ARTEMIS_FRED_KEY non configurata — "
                       "dati FRED parziali. Aggiungere ai GitHub Secrets.")
    fred = Fred(api_key=FRED_KEY) if FRED_KEY else None

    # Stack 1 — Volatilità
    logger.info("[MAIN] 1/9 — Volatility stack...")
    vol = collect_volatility_stack()

    # Stack 2 — Tassi / Credito
    logger.info("[MAIN] 2/9 — Rates & credit stack...")
    rates = collect_rates_credit_stack(fred) if fred else {}

    # Stack 3 — FX / Commodity
    logger.info("[MAIN] 3/9 — FX & commodity stack...")
    fx = collect_fx_commodity_stack()

    # Stack 4 — Sentiment
    logger.info("[MAIN] 4/9 — Sentiment stack...")
    sentiment = collect_sentiment_stack(fred)

    # Proxy sintetici per fonti bloccate
    proxies = compute_synthetic_proxies(vol, rates, sentiment)
    if sentiment.get("fear_greed_score") is None and proxies.get("fear_greed_proxy"):
        sentiment["fear_greed_synthetic"] = proxies["fear_greed_proxy"]
    if sentiment.get("naaim_exposure") is None and proxies.get("naaim_proxy"):
        sentiment["naaim_synthetic"] = proxies["naaim_proxy"]
    if sentiment.get("gpr_index") is None and proxies.get("gpr_proxy"):
        sentiment["gpr_synthetic"] = proxies["gpr_proxy"]

    # Smart sentiment composito
    smart = compute_smart_sentiment(sentiment, rates, vol)
    sentiment["smart_sentiment_score"] = smart.get("score")
    sentiment["smart_sentiment_label"] = smart.get("label")

    # ── GAP-1: VIX/GPR Divergence Alert ─────────────────────────────────────
    vix_val = vol.get("vix")
    gpr_val = sentiment.get("gpr_index") or sentiment.get("gpr_synthetic")
    vix_gpr_div = compute_vix_gpr_divergence(vix_val, gpr_val)
    if vix_gpr_div.get("divergence_active"):
        logger.warning(f"[vix_gpr_div] \u26a0\ufe0f  {vix_gpr_div['level']}: {vix_gpr_div['message']}")
    else:
        logger.info(f"[vix_gpr_div] {vix_gpr_div['level']}")

    # Stack 5 — EIA
    logger.info("[MAIN] 5/9 — EIA stack...")
    eia = collect_eia_stack()

    # Stack 6 — SOCMINT
    logger.info("[MAIN] 6/9 — SOCMINT lite...")
    socmint = collect_socmint_lite()

    # Stack 8 — Insider Intelligence
    logger.info("[MAIN] 7/9 — Insider intelligence...")
    insider = collect_insider_intel()

    # Stack 9 — Long Vol Monitor (NUOVO Artemis)
    logger.info("[MAIN] 8/9 — Long Volatility Monitor...")
    long_vol = collect_long_vol_monitor(vol)

    # Stack 10 — CWARP + Dragon Health (NUOVI Artemis)
    logger.info("[MAIN] 9/9 — CWARP + Dragon Health Check...")

    # ── Leggi portafoglio reale da Supabase se non passato come argomento ──
    # In GitHub Actions lo script gira senza argomenti CLI → legge da DB
    #
    # CASCATA DI LETTURA (architettura unificata — OPZIONE A):
    #   1. ARTEMIS_CWARP_CANDIDATES dove is_in_portfolio=TRUE — fonte di
    #      verità UNICA e autosufficiente. Ogni riga ha shares + current_price_eur,
    #      il valore di ogni asset è shares×price. Il CASH è una riga come le
    #      altre (ticker='CASH_EUR', shares=1, current_price_eur=valore cash,
    #      dragon_component='cash', is_active=FALSE per non entrare nel ranking
    #      CWARP). Il totale portafoglio è la somma di TUTTE le righe
    #      is_in_portfolio=TRUE, cash incluso — nessuna dipendenza da altre
    #      tabelle. Aggiornare un asset = aggiornare current_price_eur sulla
    #      sua riga, nessun calcolo manuale di percentuali.
    #      Le righe is_in_portfolio=FALSE sono monitorate (CWARP) ma NON
    #      contano nel calcolo del portafoglio.
    #   2. ARTEMIS_PORTFOLIO — fallback finale, storico operazioni immutabile
    #      (comportamento originale, sempre disponibile come rete di sicurezza
    #      se CWARP_CANDIDATES fosse temporaneamente irraggiungibile o vuota).
    # Una sola fonte di verità per il portafoglio corrente (ARTEMIS_CWARP_
    # CANDIDATES); ARTEMIS_PORTFOLIO_CURRENT è stata dismessa per evitare
    # due tabelle parallele che potrebbero disallinearsi senza un proprietario
    # chiaro dell'aggiornamento.
    if portfolio_current is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            _sb = create_client(SUPABASE_URL, SUPABASE_KEY)

            # ── Tentativo 1: calcolo per-asset da ARTEMIS_CWARP_CANDIDATES ──
            _holdings = (
                _sb.table("ARTEMIS_CWARP_CANDIDATES")
                .select("ticker,name,dragon_component,shares,current_price_eur,"
                        "value_eur,updated_at")
                .eq("is_in_portfolio", True)
                .execute()
            )
            _rows = _holdings.data if _holdings.data else []

            if _rows:
                _per_asset = []
                _stale_rows = []
                _skipped_rows = []   # righe scartate per prezzo/quote mancanti
                for _r in _rows:
                    _shares = _r.get("shares") or 0.0
                    _price  = _r.get("current_price_eur")
                    _val_db = _r.get("value_eur")
                    if _price is not None and _shares:
                        _val_calc = _shares * _price
                    else:
                        _val_calc = _val_db or 0.0
                    if _val_calc <= 0:
                        logger.warning(f"[portfolio] {_r.get('ticker')}: valore non calcolabile "
                                       f"(shares={_shares}, price={_price}) — riga ignorata")
                        _skipped_rows.append(_r.get("ticker"))
                        continue
                    _per_asset.append({
                        "ticker": _r.get("ticker"),
                        "component": _r.get("dragon_component"),
                        "value": _val_calc,
                        # 30/09/2026: quote e prezzo per posizione, salvati nello storico
                        # della run → la vista ARTEMIS_CORRELAZIONI ricava i rendimenti
                        # di prezzo di ogni gamba (senza l'effetto dei versamenti PAC).
                        "shares": _shares,
                        "price_eur": _price,
                    })
                    _upd = _r.get("updated_at")
                    if _upd:
                        try:
                            _upd_date = datetime.fromisoformat(_upd.replace("Z", "+00:00"))
                            _age = (datetime.now(_upd_date.tzinfo) - _upd_date).days
                            if _age > 14:
                                _stale_rows.append(f"{_r.get('ticker')} ({_age}g)")
                        except Exception:
                            pass

                if _per_asset:
                    # Totale = somma di TUTTE le posizioni, cash incluso.
                    # Nessuna dipendenza da ARTEMIS_PORTFOLIO_CURRENT per questo numero.
                    _portfolio_value_eur = sum(a["value"] for a in _per_asset)

                    _per_componente = {}
                    for a in _per_asset:
                        _per_componente[a["component"]] = (
                            _per_componente.get(a["component"], 0.0) + a["value"]
                        )
                    # NB: 'cash' non è una delle 5 chiavi Dragon (equity/fixed_income/
                    # long_volatility/commodity_trend/gold) — il suo valore entra nel
                    # denominatore (_portfolio_value_eur) ma in nessun numeratore.
                    # Effetto voluto: abbassa proporzionalmente i 5 pesi, esattamente
                    # come il "Dead Cash non allocato" già gestito da
                    # collect_dragon_health_check.

                    # ── GUARD DATI PARZIALI ─────────────────────────────────
                    # Se una o più righe is_in_portfolio=TRUE sono state scartate
                    # (prezzo o quote mancanti), il portafoglio calcolato è
                    # INCOMPLETO: un asset sparito gonfia artificialmente i pesi
                    # degli altri e produce raccomandazioni di ribilanciamento
                    # SBAGLIATE. Meglio segnalarlo in modo esplicito e visibile
                    # che restituire numeri dall'aspetto valido ma falsi.
                    _data_quality = "OK"
                    _partial_warning = None
                    if _skipped_rows:
                        _data_quality = "PARTIAL"
                        _partial_warning = (
                            f"⚠️ PORTAFOGLIO INCOMPLETO: {len(_skipped_rows)} posizione/i "
                            f"esclusa/e per prezzo/quote mancanti ({', '.join(_skipped_rows)}). "
                            f"I pesi % e il Dragon Score sono CALCOLATI SU DATI PARZIALI e "
                            f"non affidabili finché non aggiorni current_price_eur su quelle righe."
                        )
                        logger.warning(f"[portfolio] {_partial_warning}")

                    portfolio_current = {
                        "equity":          _per_componente.get("equity", 0.0) / _portfolio_value_eur,
                        "fixed_income":    _per_componente.get("fixed_income", 0.0) / _portfolio_value_eur,
                        "long_volatility": _per_componente.get("long_volatility", 0.0) / _portfolio_value_eur,
                        "commodity_trend": _per_componente.get("commodity_trend", 0.0) / _portfolio_value_eur,
                        "gold":            _per_componente.get("gold", 0.0) / _portfolio_value_eur,
                        "portfolio_value_eur": round(_portfolio_value_eur, 2),
                        "mark_to_market_source": "ARTEMIS_CWARP_CANDIDATES",
                        "holdings_detail": _per_asset,
                        "data_quality": _data_quality,
                        "positions_expected": len(_rows),
                        "positions_used": len(_per_asset),
                        "positions_skipped": _skipped_rows,
                        "partial_warning": _partial_warning,
                    }
                    if _stale_rows:
                        logger.warning(f"[portfolio] ⚠️ Prezzi non aggiornati da >14gg: "
                                       f"{', '.join(_stale_rows)}")
                    logger.info(f"[portfolio] ✅ Calcolato da ARTEMIS_CWARP_CANDIDATES "
                                f"({len(_per_asset)}/{len(_rows)} posizioni, cash incluso): "
                                f"€{portfolio_current['portfolio_value_eur']} "
                                f"gold={portfolio_current['gold']:.1%} "
                                f"equity={portfolio_current['equity']:.1%}"
                                f"{' [DATI PARZIALI]' if _skipped_rows else ''}")


            if portfolio_current is None:
                logger.info("[portfolio] Nessuna riga is_in_portfolio=TRUE in CWARP_CANDIDATES "
                            "— fallback su ARTEMIS_PORTFOLIO")

            # ── Tentativo 2: fallback finale su ARTEMIS_PORTFOLIO ───────────
            if portfolio_current is None:
                _pf = _sb.table("ARTEMIS_PORTFOLIO").select(
                    "equity_pct,fixed_income_pct,long_volatility_pct,"
                    "commodity_trend_pct,gold_pct,portfolio_value_eur,"
                    "gold_instrument,cta_instrument,regime_at_snapshot"
                ).eq("is_current", True).limit(1).execute()
                if _pf.data:
                    _row = _pf.data[0]
                    portfolio_current = {
                        "equity":          _row.get("equity_pct", 0.0),
                        "fixed_income":    _row.get("fixed_income_pct", 0.0),
                        "long_volatility": _row.get("long_volatility_pct", 0.0),
                        "commodity_trend": _row.get("commodity_trend_pct", 0.0),
                        "gold":            _row.get("gold_pct", 0.0),
                        "portfolio_value_eur": _row.get("portfolio_value_eur"),
                        "gold_instrument":     _row.get("gold_instrument"),
                        "cta_instrument":      _row.get("cta_instrument"),
                        "regime_at_snapshot":  _row.get("regime_at_snapshot"),
                        "mark_to_market_source": "ARTEMIS_PORTFOLIO",
                    }
                    logger.info(f"[portfolio] ✅ Letto da ARTEMIS_PORTFOLIO (storico operazioni): "
                                f"€{portfolio_current.get('portfolio_value_eur', '?')} "
                                f"gold={portfolio_current['gold']:.1%} "
                                f"cta={portfolio_current['commodity_trend']:.1%}")
                else:
                    logger.warning("[portfolio] Nessuna fonte disponibile in nessuna delle tre tabelle")
        except Exception as e:
            logger.warning(f"[portfolio] Lettura Supabase fallita: {e} — uso placeholder")

    # Regime flags (signal scanner — qualitativo)
    regime = compute_regime_flags(vol, rates, fx, sentiment)

    # DEAD STOP
    if regime.get("dead_stop_reason"):
        logger.error(f"[DEAD_STOP] {regime['dead_stop_reason']}")
        print(f"\n⛔ DEAD STOP: {regime['dead_stop_reason']}\n")
        return {"status": "DEAD_STOP", "reason": regime["dead_stop_reason"]}

    # FIX F2 — Regime engine PRIMA del Dragon Health Check
    # (era dopo: il Dragon Score usava sempre la tolleranza TRANSIZIONE di default
    # perché portfolio_current non aveva mai la chiave "regime_primary")
    logger.info("[MAIN] Extra — Regime engine per regione (USA/EU/CAN)...")
    regime_by_region = collect_regime_by_region()

    # Inietta il regime ENGINE CORRENTE in portfolio_current così Dragon Score
    # usa le tolleranze corrette (SERPENTE/FALCO_DX/FALCO_SX/FENICE).
    # FIX A3: nome esplicito "regime_engine_corrente" per distinguerlo da
    # "regime_at_snapshot" (nota storica del regime al momento dell'ultimo PAC —
    # campo di sola lettura da ARTEMIS_PORTFOLIO, non toccare da qui).
    if portfolio_current is not None:
        portfolio_current["regime_engine_corrente"] = (
            regime_by_region.get("USA", {}).get("motore", "TRANSIZIONE")
        )

    # Usa CWARP_CANDIDATES globale se non passati esplicitamente
    # Candidati CWARP: argomento esplicito NON vuoto, poi Supabase, poi fallback.
    # 'if cwarp_candidates' (non 'is not None') così lista vuota [] → carica Supabase.
    if cwarp_candidates:
        _candidates = cwarp_candidates
    else:
        _candidates = _load_cwarp_candidates_from_supabase()
        if not _candidates:
            _candidates = CWARP_CANDIDATES
    cwarp  = compute_cwarp(portfolio_current=portfolio_current,
                           candidates=_candidates)
    dragon = collect_dragon_health_check(
        portfolio_current=portfolio_current,
        cwarp_data=cwarp.get("candidates") if cwarp else None   # FIX C5 — CWARP gate
    )
    risk_regime = assess_risk_regime(vol)   # livello di gestione del rischio (2026-09-16)
    manual_signals = collect_manual_signals()   # segnali fondamentali manuali (capex AI, ecc.)
    t_end  = time.time()   # misurato dopo tutti i calcoli pesanti

    # Snapshot completo
    snapshot = {
        "date":             datetime.today().strftime("%Y-%m-%d"),
        "session_id":       session_id,
        "vol":              vol,
        "rates":            rates,
        "fx":               fx,
        "risk_regime":      risk_regime,
        "manual_signals":   manual_signals,
        "sentiment":        sentiment,
        "eia":              eia,
        "socmint":          socmint,
        "insider":          insider,
        "long_vol":         long_vol,
        "cwarp":            cwarp,
        "dragon_health":    dragon,
        "regime":           regime,
        "regime_by_region": regime_by_region,
        "vix_gpr_div":      vix_gpr_div,
        "scan_duration_sec":round(t_end - t_start, 1),
        "status":           "OK",
    }

    # ── DATA AUDIT (audit 2026-09-25) ────────────────────────────────────────
    try:
        data_audit = build_data_audit(snapshot)
    except Exception as e:
        logger.warning(f"[data_audit] costruzione fallita: {e}")
        data_audit = {"warning": f"⚠️ QUALITÀ DATI — audit non eseguito ({e})"}
    snapshot["data_audit"] = data_audit
    if data_audit.get("warning"):
        _recs = dragon.setdefault("recommendations", [])
        _pos = 1 if (_recs and DRAGON_TARGET_WARNING and _recs[0] == DRAGON_TARGET_WARNING) else 0
        _recs.insert(_pos, data_audit["warning"])
        logger.warning(f"[data_audit] {data_audit['warning']}")

    # Report console
    print_report(snapshot)
    try:
        print()
        print("═" * 76)
        print("  DATA AUDIT — stato di ogni fonte usata in questo run")
        print("═" * 76)
        print(f"  Conteggi: {data_audit.get('conteggi')}")
        for i in data_audit.get("problemi", []):
            print(f"  🔴 {i['serie']:<22} {i['stato']:<9} dato del {i['data_dato']} "
                  f"({i['eta_giorni']}g, limite {i['limite_giorni']}g) [{i['fonte']}]")
        for s in data_audit.get("stime_usate", []):
            print(f"  🟡 STIMA  {s}")
        for s in data_audit.get("fonti_non_disponibili", []):
            print(f"  ⚪ {s}")
        for reg, notes in (data_audit.get("note_regime") or {}).items():
            for n in notes:
                print(f"  🟠 REGIME {reg}: {n}")
        if data_audit.get("campi_mancanti"):
            print(f"  ⚫ Campi vuoti: {', '.join(data_audit['campi_mancanti'])}")
        print("═" * 76)
    except Exception as e:
        logger.warning(f"[data_audit] stampa: {e}")

    # FIX C6 — Safe-State Fallback: blocca scrittura Supabase se lo snapshot è corrotto.
    # Usa threshold PER CATEGORIA invece di un 30% globale (più preciso e meno rumoroso):
    #   - Blocco FRED (tassi + credito): se TUTTI i 6 indicatori chiave sono None → Hard Stop
    #   - Blocco volatilità: se VIX E VVIX sono entrambi None → Hard Stop (già coperto da DEAD STOP)
    # Rationale: un blocco FRED completo (tutti null) è un segnale di credenziali mancanti
    # o outage FRED — non ha senso salvare 50 campi None che corromperanno lo storico.
    _fred_key_signals = [
        rates.get("t10y2y"), rates.get("cpi_yoy"), rates.get("hy_oas"),
        rates.get("sofr"),   rates.get("t5yie"),   rates.get("net_liquidity"),
    ]
    _fred_all_null = all(v is None for v in _fred_key_signals)

    _vol_key_signals = [vol.get("vix"), vol.get("vvix"), vol.get("skew")]
    _vol_all_null = all(v is None for v in _vol_key_signals)

    if _fred_all_null and _vol_all_null:
        logger.error("[SAFE-STATE] HARD STOP — FRED e Volatilità stack completamente null. "
                     "Snapshot corrotto (>80% segnali macro mancanti). "
                     "Scrittura Supabase bloccata per proteggere lo storico.")
        raise ArtemisHardStop("SAFE-STATE FALLBACK: Inquinamento dati critico — skip Supabase")
    elif _fred_all_null:
        logger.warning("[SAFE-STATE] Blocco FRED completamente null — "
                       "scrittura Supabase con flag data_quality=DEGRADED")
        snapshot["data_quality"] = "DEGRADED_FRED"
    else:
        snapshot["data_quality"] = "OK"

    # Propaga un eventuale stato PARTIAL del portafoglio (asset esclusi per
    # prezzo/quote mancanti) al data_quality del meta, così è visibile nel
    # JSON finale e non solo nel sotto-dict del portafoglio. Non sovrascrive
    # uno stato FRED più grave (DEGRADED_FRED ha priorità).
    try:
        _pf = snapshot.get("dragon_health", {}).get("current_allocation", {})
        if _pf.get("data_quality") == "PARTIAL" and snapshot["data_quality"] == "OK":
            snapshot["data_quality"] = "PARTIAL_PORTFOLIO"
    except Exception:
        pass

    # Genera artemis_status.json — PRIMA della scrittura Supabase, così il
    # campo "artemis_status" nel record/full_snapshot contiene il vero JSON
    # invece di restare sempre vuoto {} (era calcolato dopo, quindi lo
    # snapshot passato a write_to_supabase non lo aveva ancora).
    artemis_status = generate_artemis_status(snapshot)
    artemis_status["manual_signals"] = manual_signals   # per gli scenari (capex AI ecc.)
    _da = snapshot.get("data_audit") or {}
    artemis_status["data_audit"] = {k: _da.get(k) for k in
        ("conteggi", "warning", "problemi", "stime_usate", "fonti_non_disponibili",
         "note_regime", "campi_mancanti", "critici_mancanti")}
    snapshot["artemis_status"] = artemis_status

    # Alert scenari di monitoraggio (definiti in ARTEMIS_SCENARIOS)
    scenari_scattati = check_scenario_alerts(artemis_status)
    if scenari_scattati:
        snapshot["scenari_scattati"] = scenari_scattati
        artemis_status["scenari_scattati"] = scenari_scattati
        print()
        print("═" * 76)
        print("  ⚠️  SCENARI SCATTATI — LEGGERE PRIMA DI OGNI DECISIONE")
        print("═" * 76)
        for sc in scenari_scattati:
            print(f"  🔴 {sc['nome']}")
            print(f"     {sc['campo']}={sc['valore']} {sc['operatore']} "
                  f"{sc['soglia']} · {sc['run_consecutive']} run consecutive")
            if sc.get("descrizione"):
                print(f"     {sc['descrizione']}")
            if sc.get("note_operativa"):
                print(f"     → {sc['note_operativa']}")
        print("═" * 76)

    # Scrittura Supabase — l'esito viene conservato: se fallisce, la run su
    # GitHub deve risultare ROSSA (audit 25/09/2026: prima restava verde).
    snapshot["supabase_written"] = bool(write_to_supabase(snapshot))

    # Stampa JSON per copia-incolla nella Skill ARTEMIS
    print()
    print("═" * 76)
    print("  ARTEMIS_STATUS_JSON")
    print("  Copia questo JSON come contesto iniziale nella Skill ARTEMIS")
    print("═" * 76)
    json_str = json.dumps(artemis_status, indent=2,
                          ensure_ascii=False, default=str)
    print(json_str)
    print("═" * 76)
    print()

    # Salva file locale
    status_path = "artemis_status.json"
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(artemis_status, f, indent=2,
                      ensure_ascii=False, default=str)
        logger.info(f"✅ artemis_status.json salvato → {status_path}")
        print(f"  📁 artemis_status.json salvato ({status_path})")
    except Exception as e:
        logger.warning(f"[status] Salvataggio file fallito: {e} "
                       f"(JSON stampato a console sopra)")

    return snapshot


if __name__ == "__main__":
    import sys

    # Uso: python artemis_signal_scanner.py [portfolio_json]
    # Esempio portafoglio:
    #   python artemis_signal_scanner.py \
    #     '{"equity":0.24,"fixed_income":0.18,"long_volatility":0.21,"commodity_trend":0.18,"gold":0.19}'
    portfolio = None
    if len(sys.argv) > 1:
        try:
            portfolio = json.loads(sys.argv[1])
            logger.info(f"[main] Portafoglio corrente: {portfolio}")
        except json.JSONDecodeError as e:
            logger.warning(f"[main] Portfolio JSON non valido: {e} — uso placeholder")

    # 01/10/2026: i prezzi vanno aggiornati PRIMA di main(), che legge quote e
    # prezzi da ARTEMIS_CWARP_CANDIDATES per valorizzare il portafoglio. Prima
    # l'aggiornamento avveniva dopo, e il valore salvato in ogni run usava i
    # prezzi della run precedente (un giorno di ritardo su valore, pesi e
    # rendimento reale in ARTEMIS_PERFORMANCE).
    diagnose_price_conversion()              # aggiorna current_price_eur (scrive su Supabase)
    # cwarp_candidates=None → main() carica automaticamente da Supabase
    result = main(portfolio_current=portfolio, cwarp_candidates=None)
    update_cwarp_candidates_mirror(result)   # specchio CWARP in ARTEMIS_CWARP_CANDIDATES
    if result.get("status") == "OK":
        print(f"\n✅ ARTEMIS Signal Scanner completato — "
              f"{result['scan_duration_sec']}s")
    else:
        print(f"\n⛔ ARTEMIS Signal Scanner terminato: {result.get('reason')}")
        # DEAD STOP: nessun dato salvato → la run deve risultare fallita
        if SUPABASE_URL and SUPABASE_KEY:
            sys.exit(2)
    # Audit 25/09/2026: se le credenziali ci sono ma il salvataggio su Supabase
    # è fallito, la GitHub Action deve fallire (croce rossa), non restare verde.
    if SUPABASE_URL and SUPABASE_KEY and result.get("status") == "OK" \
            and not result.get("supabase_written"):
        print("\n⛔ SCRITTURA SUPABASE FALLITA — dati di oggi NON salvati. "
              "Run segnata come fallita.")
        sys.exit(1)
