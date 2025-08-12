# -*- coding: utf-8 -*-
import os
import logging
import psycopg2
from dotenv import load_dotenv

# HTTP
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Utils
import datetime as _dt
from statistics import mean
import re

load_dotenv()
logger = logging.getLogger(__name__)


# ================== БАЗА ДАННЫХ ==================

def get_db_connection():
    """Подключение к базе данных."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def get_user_portfolio(user_id: int) -> dict:
    """Возвращает портфель пользователя в виде {TICKER: QUANTITY}."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticker, quantity 
            FROM portfolio 
            WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s)
            """,
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()

        portfolio = {}
        for ticker, quantity in rows:
            if ticker and quantity is not None:
                portfolio[str(ticker).upper()] = int(quantity)

        logger.info(f"Портфель пользователя {user_id}: {portfolio}")
        return portfolio

    except Exception as e:
        logger.error(f"Ошибка получения портфеля для пользователя {user_id}: {str(e)}", exc_info=True)
        return {}


def get_user_costs(user_id: int) -> dict:
    """Возвращает словарь средних цен: {TICKER: avg_cost or None}."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticker, avg_cost
            FROM portfolio
            WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s)
            """,
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()

        out = {}
        for t, avg in rows:
            out[str(t).upper()] = float(avg) if avg is not None else None
        return out
    except Exception as e:
        logger.error(f"Ошибка получения avg_cost: {e}", exc_info=True)
        return {}


# ================== HTTP СЕССИЯ С РЕТРАЯМИ ==================

def _session_with_retries(total=3, backoff=0.5, timeout=(5, 20)) -> requests.Session:
    """Единая сессия с ретраями и таймаутами."""
    s = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.request_timeout = timeout
    s.headers.update({"User-Agent": "moex-bot/1.0"})
    return s


# ================== MOEX: Котировки, Свечи, Названия ==================

def get_moex_data(tickers: list) -> dict:
    """
    Котировки MOEX (TQBR) для списка тикеров.
    Возвращает: {SECID: {"price": float, "prev_price": float|None, "change_pct": float|None, "volume": int}}
    """
    if not tickers:
        return {}
    tickers = [str(t).upper().strip() for t in tickers if t]
    unique = sorted(set(tickers))

    sess = _session_with_retries()
    timeout = getattr(sess, "request_timeout", (5, 20))

    def _extract(row, columns, name_list):
        for col in name_list:
            if col in columns:
                idx = columns.index(col)
                if 0 <= idx < len(row) and row[idx] is not None:
                    return row[idx]
        return None

    moex_data = {}

    try:
        secids = ",".join(unique)
        url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
        params = {"securities": secids, "iss.only": "marketdata,marketdata.columns"}
        r = sess.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            payload = r.json()
            marketdata = payload.get("marketdata", {})
            rows = marketdata.get("data", [])
            cols = marketdata.get("columns", [])
            for row in rows:
                secid = str(_extract(row, cols, ["SECID"]) or "").upper()
                if not secid:
                    continue
                price = _extract(row, cols, ["LAST", "LCURRENTPRICE", "OPEN", "PREVPRICE"])
                prev = _extract(row, cols, ["PREVPRICE"])
                vol  = _extract(row, cols, ["VOLTODAY", "VALUE", "NUMTRADES"]) or 0
                try:
                    price = float(price) if price is not None else None
                    prev  = float(prev) if prev is not None else None
                    vol   = int(float(vol)) if vol is not None else 0
                except Exception:
                    pass
                change_pct = None
                if price is not None and prev not in (None, 0):
                    change_pct = round((price - prev) / prev * 100, 2)
                if price is not None:
                    moex_data[secid] = {
                        "price": price, "prev_price": prev, "change_pct": change_pct, "volume": vol
                    }
        else:
            logger.warning(f"MOEX batch request failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.error(f"MOEX batch request error: {str(e)}", exc_info=True)

    return moex_data


def get_moex_candles(tickers: list, days: int = 60, interval: int = 24) -> dict:
    """
    Дневные свечи (interval=24) по каждому тикеру за N дней.
    Возвращает: {TICKER: [{"date","open","high","low","close","volume","value"}]}
    """
    if not tickers:
        return {}
    sess = _session_with_retries()
    timeout = getattr(sess, "request_timeout", (5, 20))
    till = _dt.date.today()
    frm = till - _dt.timedelta(days=days + 10)  # запас на нерабочие дни

    out = {}
    for t in {str(x).upper().strip() for x in tickers if x}:
        try:
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{t}/candles.json"
            params = {
                "from": frm.strftime("%Y-%m-%d"),
                "till": till.strftime("%Y-%m-%d"),
                "interval": interval,
                "iss.only": "candles,candles.columns"
            }
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code != 200:
                logger.warning(f"Candles request failed for {t}: {r.status_code}")
                continue

            payload = r.json().get("candles", {})
            rows = payload.get("data", [])
            cols = payload.get("columns", [])

            idx = {name: (cols.index(name) if name in cols else None) for name in
                   ["begin", "open", "close", "high", "low", "volume", "value"]}

            series = []
            for row in rows:
                if None in (idx["begin"], idx["close"]):
                    continue
                d = str(row[idx["begin"]])[:10]
                o = row[idx["open"]] if idx["open"] is not None else None
                h = row[idx["high"]] if idx["high"] is not None else None
                l = row[idx["low"]] if idx["low"] is not None else None
                c = row[idx["close"]]
                v = row[idx["volume"]] if idx["volume"] is not None else None
                val = row[idx["value"]] if idx["value"] is not None else None
                if c is None:
                    continue
                series.append({
                    "date": d,
                    "open": float(o) if o is not None else None,
                    "high": float(h) if h is not None else None,
                    "low": float(l) if l is not None else None,
                    "close": float(c),
                    "volume": int(v) if v is not None else 0,
                    "value": float(val) if val is not None else None
                })

            series = sorted(series, key=lambda x: x["date"])[-days:]
            if series:
                out[t] = series
        except Exception as e:
            logger.error(f"Candles error for {t}: {str(e)}", exc_info=True)
    return out


def compute_candle_indicators(candles_by_ticker: dict) -> dict:
    """
    На основе свечей считает индикаторы:
    {TICKER: {
        last_close, day_change_pct, month_change_pct, sma5, sma20, sma50,
        avg_vol20, avg_turnover20, vol_spike_ratio, atr_pct
    }}
    month_change_pct — к закрытию ~20 торговых дней назад.
    """
    res = {}
    for t, series in (candles_by_ticker or {}).items():
        if not series:
            res[t] = {}
            continue

        closes = [x["close"] for x in series]
        highs  = [x["high"] if x["high"] is not None else x["close"] for x in series]
        lows   = [x["low"]  if x["low"]  is not None else x["close"] for x in series]
        vols   = [x["volume"] for x in series]
        vals   = [x.get("value") for x in series]

        last_close = closes[-1]
        prev_close = closes[-2] if len(closes) >= 2 else None
        day_change_pct = round((last_close - prev_close) / prev_close * 100, 2) if prev_close else None

        ref_idx = -21 if len(closes) >= 21 else (0 if closes else None)
        month_change_pct = None
        if ref_idx is not None and closes and closes[ref_idx] not in (None, 0):
            month_change_pct = round((last_close - closes[ref_idx]) / closes[ref_idx] * 100, 2)

        sma5  = round(mean(closes[-5:]), 4) if len(closes) >= 5 else None
        sma20 = round(mean(closes[-20:]), 4) if len(closes) >= 20 else None
        sma50 = round(mean(closes[-50:]), 4) if len(closes) >= 50 else None

        avg_vol20 = round(mean(vols[-20:]), 2) if len(vols) >= 20 else None
        vol_spike_ratio = round(vols[-1] / avg_vol20, 2) if avg_vol20 not in (None, 0) else None

        # оборот из value, если есть; иначе грубо close*volume
        turners = []
        for i, v in enumerate(vals):
            if v is not None:
                turners.append(v)
            else:
                turners.append(closes[i] * vols[i])
        avg_turnover20 = round(mean(turners[-20:]), 2) if len(turners) >= 20 else None

        # ATR(14) в процентах от последнего close
        atr_pct = None
        if len(series) >= 15:
            trs = []
            for i in range(1, len(series)):
                h = highs[i]; l = lows[i]; pc = closes[i-1]
                if h is None or l is None or pc is None:
                    continue
                tr = max(h - l, abs(h - pc), abs(l - pc))
                trs.append(tr)
            if len(trs) >= 14 and last_close:
                atr = mean(trs[-14:])
                atr_pct = round(atr / last_close * 100, 2)

        res[t] = {
            "last_close": last_close,
            "day_change_pct": day_change_pct,
            "month_change_pct": month_change_pct,
            "sma5": sma5,
            "sma20": sma20,
            "sma50": sma50,
            "avg_vol20": avg_vol20,
            "avg_turnover20": avg_turnover20,
            "vol_spike_ratio": vol_spike_ratio,
            "atr_pct": atr_pct
        }
    return res


# ===== Имена компаний (очистка от «АО/ПАО/ОАО») =====

_LEGAL_DROP = re.compile(r'\b(пao|пао|оао|ao|ао|ojsc|pjsc)\b\.?', re.IGNORECASE)

def _clean_display_name(name: str) -> str:
    s = (name or "")
    s = s.replace("—", " ").replace("–", " ")
    s = re.sub(r'[«»“”"]', '', s)
    s = _LEGAL_DROP.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,–—-')
    return s


def get_company_names(tickers: list) -> dict:
    """
    Возвращает {TICKER: {"company": CLEAN_NAME, "short": SHORTNAME, "secname": SECNAME}}
    """
    if not tickers:
        return {}
    tickers = [str(t).upper().strip() for t in tickers if t]
    secids = ",".join(sorted(set(tickers)))

    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    params = {"securities": secids, "iss.only": "securities,securities.columns"}

    sess = _session_with_retries()
    try:
        r = sess.get(url, params=params, timeout=getattr(sess, "request_timeout", (5, 20)))
        if r.status_code != 200:
            logger.warning(f"MOEX securities failed: {r.status_code}")
            return {}
        payload = r.json().get("securities", {})
        rows = payload.get("data", [])
        cols = payload.get("columns", [])
        idx_id = cols.index("SECID") if "SECID" in cols else None
        idx_sn = cols.index("SHORTNAME") if "SHORTNAME" in cols else None
        idx_nm = cols.index("SECNAME") if "SECNAME" in cols else None

        out = {}
        for row in rows:
            secid = str(row[idx_id]).upper() if idx_id is not None else ""
            short = (row[idx_sn] if idx_sn is not None and idx_sn < len(row) else "") or ""
            full  = (row[idx_nm] if idx_nm is not None and idx_nm < len(row) else "") or ""
            company_raw = full or short or secid
            out[secid] = {"company": _clean_display_name(company_raw), "short": str(short), "secname": str(full)}
        return out
    except Exception as e:
        logger.error(f"MOEX securities error: {e}")
        return {t: {"company": t, "short": t, "secname": t} for t in tickers}


# ================== НОВОСТИ ==================

def get_news_annotated(limit: int = 50) -> list:
    """Старый интерфейс — совместимость: новости за 1 день с тональностью."""
    return get_news_multi(limit=limit, days=int(os.getenv("NEWS_DAYS", "1")))


def _rss_collect(max_items: int = 60) -> list:
    """
    Подбор RSS из РБК и Интерфакс, если установлен feedparser. Возвращает список заголовков (title, source, published, age_hours).
    """
    items = []
    try:
        import feedparser  # опционально
    except Exception:
        return items

    now = _dt.datetime.utcnow()

    feeds = [
        ("РБК", "https://rssexport.rbc.ru/rbcnews/news/20/full.rss"),
        ("Интерфакс", "https://www.interfax.ru/rss.asp"),
    ]
    for name, url in feeds:
        try:
            rss = feedparser.parse(url)
            for entry in rss.entries:
                title = (getattr(entry, "title", "") or "").strip()
                if not title:
                    continue
                # published_parsed может отсутствовать
                try:
                    p = entry.published_parsed or entry.updated_parsed
                    dt = _dt.datetime(*p[:6])
                except Exception:
                    dt = now
                age_h = (now - dt).total_seconds() / 3600.0
                items.append({
                    "title": title,
                    "source": name,
                    "published_at": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "age_hours": round(age_h, 2)
                })
        except Exception as e:
            logger.warning(f"RSS parse error {name}: {e}")
    # ограничение
    items.sort(key=lambda x: x.get("published_at",""), reverse=True)
    return items[:max_items]


def get_news_multi(limit: int = 120, days: int = 3) -> list:
    """
    Возвращает новости за N дней с простой тональностью и age_hours:
    [{"title":..., "source":..., "published_at":"...Z", "tone":"positive|negative|neutral", "hits":[...], "age_hours": float}]
    Источники: NewsAPI (+вес по давности), RSS РБК и Интерфакс (если доступен feedparser).
    """
    POS = [w.strip().lower() for w in (os.getenv("NEWS_POSITIVE_KEYWORDS", "рост,улучш,повыш,рекорд,прибыль,buyback,байбек,дивиден,одобрил,расшир,увелич").split(","))]
    NEG = [w.strip().lower() for w in (os.getenv("NEWS_NEGATIVE_KEYWORDS", "паден,снижен,срыв,убыт,штраф,санкц,расслед,делист,отрицат,убав,конфликт").split(","))]

    now = _dt.datetime.utcnow()
    since = now - _dt.timedelta(days=max(1, days))

    items = []

    # ---- Источник 1: NewsAPI ----
    api_key = os.getenv("NEWS_API_KEY")
    if api_key:
        sess = _session_with_retries()
        try:
            params = {
                "q": "экономика OR финансы OR биржа OR акции OR дивиденды OR прибыль OR убыток OR санкции",
                "from": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "language": "ru",
                "sortBy": "publishedAt",
                "pageSize": min(100, max(10, limit)),
                "apiKey": api_key,
            }
            r = sess.get("https://newsapi.org/v2/everything", params=params, timeout=getattr(sess, "request_timeout", (5, 20)))
            if r.status_code == 200:
                for a in r.json().get("articles", []):
                    title = (a.get("title") or "").strip()
                    if not title:
                        continue
                    published_at = (a.get("publishedAt") or "").strip()
                    try:
                        dt = _dt.datetime.strptime(published_at[:19], "%Y-%m-%dT%H:%M:%S")
                    except Exception:
                        dt = now
                    age_h = (now - dt).total_seconds() / 3600.0
                    items.append({
                        "title": title,
                        "source": (a.get("source", {}) or {}).get("name") or "NewsAPI",
                        "published_at": f"{published_at[:19]}Z",
                        "age_hours": round(age_h, 2)
                    })
            else:
                logger.error(f"NewsAPI error: {r.status_code} - {r.text[:200]}")
        except Exception as e:
            logger.error(f"Ошибка получения NewsAPI: {e}", exc_info=True)

    # ---- Источник 2/3: RSS (РБК, Интерфакс) ----
    try:
        items.extend(_rss_collect(max_items=limit))
    except Exception as e:
        logger.warning(f"RSS collecting error: {e}")

    # ---- Сводка + тональность ----
    # дубликаты по title
    seen = set()
    result = []
    for it in sorted(items, key=lambda x: x.get("published_at",""), reverse=True):
        title = (it.get("title") or "").strip()
        low = title.lower()
        if not title or low in seen:
            continue
        seen.add(low)

        hits_pos = [w for w in POS if w and w in low]
        hits_neg = [w for w in NEG if w and w in low]
        tone = "neutral"
        if hits_pos and not hits_neg:
            tone = "positive"
        elif hits_neg and not hits_pos:
            tone = "negative"

        result.append({
            "title": title,
            "source": it.get("source") or "unknown",
            "published_at": it.get("published_at") or now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tone": tone,
            "hits": hits_pos if tone == "positive" else (hits_neg if tone == "negative" else []),
            "age_hours": float(it.get("age_hours") or 24.0)
        })
        if len(result) >= limit:
            break

    return result


# (Необязательная) функция чтения watchlist из .env
def get_watchlist() -> list:
    env = os.getenv("MOEX_WATCHLIST", "")
    if not env:
        return []
    return [s.strip().upper() for s in env.split(",") if s.strip()]
