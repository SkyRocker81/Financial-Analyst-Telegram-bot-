# -*- coding: utf-8 -*-
import os
import re
import json
import time
import uuid
import math
import base64
import logging
import urllib3
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_collector import (
    get_moex_candles,
    compute_candle_indicators,
    get_news_multi,
    get_company_names,
)

# ===== Параметры из .env (с дефолтами) =====
BEAR_DROP_PCT_1 = float(os.getenv("BEAR_DROP_PCT_1", "-3.0"))   # 30д умеренно
BEAR_DROP_PCT_2 = float(os.getenv("BEAR_DROP_PCT_2", "-6.0"))   # 30д сильно
BULL_RISE_PCT   = float(os.getenv("BULL_RISE_PCT", "3.0"))      # 30д рост (новые идеи)

# Быстрые триггеры (дневной нож)
FAST_DROP_PCT          = float(os.getenv("FAST_DROP_PCT", "-3.0"))          # умеренно
FAST_DROP_STRONG_PCT   = float(os.getenv("FAST_DROP_STRONG_PCT", "-5.0"))   # сильно
ATR_SPIKE_MULT         = float(os.getenv("ATR_SPIKE_MULT", "1.5"))          # |day| > ATR% * k

VOL_SPIKE_MIN          = float(os.getenv("VOL_SPIKE_MIN", "1.3"))
LIQ_MIN_TURNOVER_RUB   = float(os.getenv("LIQ_MIN_TURNOVER_RUB", "20000000"))

NEWS_DAYS  = int(os.getenv("NEWS_DAYS", "3"))
W24  = float(os.getenv("NEWS_WEIGHT_24H", "1.0"))
W72  = float(os.getenv("NEWS_WEIGHT_72H", "0.7"))
W7D  = float(os.getenv("NEWS_WEIGHT_7D", "0.4"))

# Порог «перебивания» решением новостей
NEWS_OVERRIDE_RATIO = float(os.getenv("NEWS_OVERRIDE_RATIO", "1.5"))   # neg >= pos*ratio
NEWS_OVERRIDE_DELTA = float(os.getenv("NEWS_OVERRIDE_DELTA", "1.0"))   # или neg >= pos+delta
NEWS_STRONG_DELTA   = float(os.getenv("NEWS_STRONG_DELTA", "2.0"))     # Жёсткий негатив для override роста 30д

# Profit-guard и докупка относительно средней цены
PROFIT_GUARD_PCT     = float(os.getenv("PROFIT_GUARD_PCT", "3.0"))   # защита прибыли: не продавать, если цена ≥ avg*(1+P/100)
REBUY_DISCOUNT_PCT   = float(os.getenv("REBUY_DISCOUNT_PCT", "3.0"))  # докупить, если цена ≤ avg*(1-P/100)

# Антипанический фильтр
ANTI_PANIC_ENABLED     = os.getenv("ANTI_PANIC_ENABLED", "1") == "1"
PANIC_BREADTH_SHARE    = float(os.getenv("PANIC_BREADTH_SHARE", "0.7"))
PANIC_MIN_AVG_DROP     = float(os.getenv("PANIC_MIN_AVG_DROP", "-2.5"))
PANIC_MAX_SELLS        = int(os.getenv("PANIC_MAX_SELLS", "3"))
PANIC_NEWS_RELAX_DELTA = float(os.getenv("PANIC_NEWS_RELAX_DELTA", "0.5"))

SELL_FRACTIONS = {
    "moderate": float(os.getenv("SELL_FRAC_MODERATE", "0.25")),
    "strong":   float(os.getenv("SELL_FRAC_STRONG",   "0.50"))
}
DEFAULT_WATCHLIST = [
    "SBER","GAZP","LKOH","NVTK","GMKN","ROSN","TATN","MTSS","VTBR","YDEX",
    "ALRS","MAGN","CHMF","PHOR","AFKS","SMLT","FIVE","PIKK","MOEX","POLY"
]
MAX_NEW_OPS = int(os.getenv("MAX_NEW_OPS", "5"))
ENABLE_DIVIDENDS = os.getenv("ENABLE_DIVIDENDS", "1") == "1"
USE_GIGACHAT_REPHRASE = os.getenv("USE_GIGACHAT_REPHRASE", "0") == "1"

logger = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

def create_session_with_retries():
    session = requests.Session()
    # ВКЛЮЧЕНА проверка TLS. Можно передать путь к CA в GIGACHAT_CA_BUNDLE.
    session.verify = os.getenv('GIGACHAT_CA_BUNDLE') or True
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'Accept-Charset': 'utf-8',
        'User-Agent': 'moex-bot/1.0',
        'Connection': 'close'
    })
    return session

def get_gigachat_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    rq_uid = str(uuid.uuid4())

    client_id = (os.getenv('GIGACHAT_CLIENT_ID', '') or '').encode('ascii', 'ignore').decode('ascii')
    client_secret = (os.getenv('GIGACHAT_CLIENT_SECRET', '') or '').encode('ascii', 'ignore').decode('ascii')
    if not client_id or not client_secret:
        raise ValueError("Нет GIGACHAT_CLIENT_ID/GIGACHAT_CLIENT_SECRET")

    auth_string = f"{client_id}:{client_secret}"
    encoded_auth = base64.b64encode(auth_string.encode('ascii')).decode('ascii')

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": rq_uid,
        "Authorization": f"Basic {encoded_auth}"
    }
    data = "scope=GIGACHAT_API_PERS"

    for attempt in range(3):
        try:
            session = create_session_with_retries()
            r = session.post(url, headers=headers, data=data, timeout=(10, 30))
            logger.info(f"GigaChat token status: {r.status_code}")
            logger.debug(f"GigaChat token body: {r.text}")
            r.raise_for_status()
            return r.json()["access_token"]
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise e


# ===== Нормализация названий и матчинг новостей =====

_LEGAL = re.compile(r'\b(пao|пао|оао|zao|ао|ao|ooo|нк|гк|пиф|ипиф|pjsc|ojsc|nk)\b', re.IGNORECASE)
_QUOTES = str.maketrans({'«':'','»':'','“':'','”':'','"':'',"'":''})

def _normalize_name(s: str) -> str:
    s = (s or "").lower().translate(_QUOTES)
    s = re.sub(r'[\(\)]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = _LEGAL.sub('', s)
    s = re.sub(r'\b(им\.?|named after)\b.*$', '', s).strip()
    return s

def _build_variants(names_map: dict) -> dict:
    variants = {}
    for t, info in (names_map or {}).items():
        keys = set()
        keys.add(t.lower())
        for k in [info.get("company",""), info.get("short",""), info.get("secname","")]:
            k = _normalize_name(k)
            if k:
                for p in [p.strip() for p in re.split(r'[,–—\-]', k) if p.strip()]:
                    if len(p) >= 3:
                        keys.add(p)
        pats = []
        for x in sorted(keys, key=len, reverse=True):
            pats.append(re.compile(rf'\b{re.escape(x)}\b', re.IGNORECASE))
        variants[t] = pats
    return variants

def _weighted_news_index(news_items: list, name_variants: dict):
    out = {t: {"pos_w": 0.0, "neg_w": 0.0, "matched_titles": []} for t in name_variants.keys()}
    for n in (news_items or []):
        title = n.get("title") or ""
        tone = n.get("tone", "neutral")
        age_h = float(n.get("age_hours", 24.0))
        if age_h <= 24:
            w = W24
        elif age_h <= 72:
            w = W72
        else:
            w = W7D
        low = title.lower()
        for t, regexes in name_variants.items():
            if any(r.search(low) for r in regexes):
                if tone == "positive":
                    out[t]["pos_w"] += w
                elif tone == "negative":
                    out[t]["neg_w"] += w
                out[t]["matched_titles"].append(title)
    return out


# ===== Правила сигналов =====

def _liquidity_ok(ind: dict) -> bool:
    turn = ind.get("avg_turnover20")
    return bool(turn is not None and turn >= LIQ_MIN_TURNOVER_RUB)

def _news_forces_sell(pos_w: float, neg_w: float) -> bool:
    return (neg_w >= pos_w * NEWS_OVERRIDE_RATIO) or (neg_w >= pos_w + NEWS_OVERRIDE_DELTA)

def _news_strong_override(pos_w: float, neg_w: float) -> bool:
    """Жёсткий новнегатив, разрешающий продажу даже при положительной динамике 30д."""
    return neg_w >= pos_w + NEWS_STRONG_DELTA

def _fast_drop_signal(ind: dict, pos_w: float, neg_w: float):
    """
    Мгновенная продажа при резком дневном падении.
    Условие-допуск: month_change_pct <= 0 ИЛИ сильный новнегатив.
    Усилители: vol_spike, SMA5<SMA20, ATR-спайк (|day| > ATR%*k).
    """
    day = ind.get("day_change_pct")
    if day is None:
        return None

    ch30 = ind.get("month_change_pct")
    # Блокируем быстрый триггер на растущих за месяц бумагах, если нет сильных новостей
    if ch30 is not None and ch30 > 0 and not _news_strong_override(pos_w, neg_w):
        return None

    sma5, sma20 = ind.get("sma5"), ind.get("sma20")
    vol_spike = (ind.get("vol_spike_ratio") or 0) >= VOL_SPIKE_MIN
    atr_pct = ind.get("atr_pct")  # в %
    atr_spike = (abs(day) >= (atr_pct or 0) * ATR_SPIKE_MULT) if atr_pct else False
    local_down = (sma5 is not None and sma20 is not None and sma5 < sma20)

    confirm = vol_spike or local_down or atr_spike or _news_forces_sell(pos_w, neg_w)

    if day <= FAST_DROP_STRONG_PCT and confirm:
        return {"sell_fraction": SELL_FRACTIONS["strong"],
                "reason": f"день {day:+.2f}% ; резкое движение (объём/локальный тренд/ATR) ; новости +/− {pos_w:.1f}/{neg_w:.1f}"}
    if day <= FAST_DROP_PCT and confirm:
        return {"sell_fraction": SELL_FRACTIONS["moderate"],
                "reason": f"день {day:+.2f}% ; ускорение вниз (объём/локальный тренд/ATR) ; новости +/− {pos_w:.1f}/{neg_w:.1f}"}
    return None

def _bearish_signal_30d(ind: dict, pos_w: float, neg_w: float):
    """
    Базовый (месячный) сигнал на продажу, с усилением за счёт новостей.
    Всегда требует ch30 <= порога, т.е. растущие за 30д бумаги не попадают сюда.
    """
    ch30 = ind.get("month_change_pct")
    if ch30 is None:
        return None

    sma5, sma20, sma50 = ind.get("sma5"), ind.get("sma20"), ind.get("sma50")
    vol_spike = (ind.get("vol_spike_ratio") or 0) >= VOL_SPIKE_MIN

    sell_frac = None
    if ch30 <= BEAR_DROP_PCT_2:
        sell_frac = SELL_FRACTIONS["strong"]
    elif ch30 <= BEAR_DROP_PCT_1 or _news_forces_sell(pos_w, neg_w):
        sell_frac = SELL_FRACTIONS["moderate"]

    if not sell_frac:
        return None

    reasons = [f"30д {ch30:+.2f}%"]
    if sma20 is not None and sma50 is not None and sma20 < sma50:
        reasons.append("тренд ↓ (SMA20<SMA50)")
    if sma5 is not None and sma20 is not None and sma5 < sma20:
        reasons.append("локально ↓ (SMA5<SMA20)")
    if vol_spike:
        reasons.append("всплеск объёма")
    if pos_w > 0 or neg_w > 0:
        reasons.append(f"новости +/− {pos_w:.1f}/{neg_w:.1f}")

    return {"sell_fraction": sell_frac, "reason": " ; ".join(reasons)}

def _bullish_candidate(ind: dict, pos_w: float, neg_w: float):
    ch30 = ind.get("month_change_pct")
    sma20, sma50 = ind.get("sma20"), ind.get("sma50")
    if ch30 is None or sma20 is None or sma50 is None:
        return None
    if ch30 >= BULL_RISE_PCT and sma20 >= sma50 and pos_w >= neg_w:
        return {"change_pct": round(float(ch30), 2)}
    return None


# ===== Антипанический фильтр =====

def _compute_panic_mode(inds_by_ticker: dict, news_index: dict) -> bool:
    """
    Включает panic-mode, если:
      - доля бумаг с day_change_pct <= FAST_DROP_PCT ≥ PANIC_BREADTH_SHARE, И
      - среднее day_change_pct по портфелю ≤ PANIC_MIN_AVG_DROP, И
      - суммарный негатив в новостях НЕ доминирует (neg_total < pos_total + delta).
    """
    if not ANTI_PANIC_ENABLED or not inds_by_ticker:
        return False

    day_moves = []
    falling = 0
    total = 0
    for t, ind in inds_by_ticker.items():
        day = ind.get("day_change_pct")
        if day is None:
            continue
        total += 1
        day_moves.append(day)
        if day <= FAST_DROP_PCT:
            falling += 1

    if total == 0:
        return False

    breadth = falling / total
    avg_drop = sum(day_moves) / len(day_moves)

    pos_total = sum((news_index.get(t, {}).get("pos_w", 0.0) for t in inds_by_ticker.keys()))
    neg_total = sum((news_index.get(t, {}).get("neg_w", 0.0) for t in inds_by_ticker.keys()))
    news_not_dominant_negative = neg_total < (pos_total + PANIC_NEWS_RELAX_DELTA)

    return (breadth >= PANIC_BREADTH_SHARE) and (avg_drop <= PANIC_MIN_AVG_DROP) and news_not_dominant_negative


def _severity_score(ind: dict) -> float:
    day = ind.get("day_change_pct") or 0.0
    mon = ind.get("month_change_pct") or 0.0
    s20 = ind.get("sma20"); s50 = ind.get("sma50")
    trend_penalty = 1.0 if (s20 is not None and s50 is not None and s20 < s50) else 0.0
    vol_spike = ind.get("vol_spike_ratio") or 0.0
    return abs(min(day, 0.0)) * 1.0 + abs(min(mon, 0.0)) * 0.5 + trend_penalty * 2.0 + max(vol_spike - 1.0, 0.0)


# ===== Основная функция =====

def analyze_with_gigachat(portfolio: dict, _moex_data_unused: dict, news_items: list, avg_costs: dict | None = None):
    """
    Возвращает dict:
    {
      "dividends_good": [{"ticker","company","reason","dividend_yield"?}],
      "speculation":    [{"ticker","company","action","reason","quantity"?,"remaining"?"}],
      "new_opportunities": [{"ticker","company","target_group","reason"}]
    }
    """
    portfolio = {str(k).upper(): int(v) for k, v in (portfolio or {}).items() if v is not None}
    p_tickers = list(portfolio.keys())
    avg_costs = {str(k).upper(): (float(v) if v is not None else None) for k, v in (avg_costs or {}).items()}

    # Индикаторы по портфелю
    candles_p = get_moex_candles(p_tickers, days=60)
    inds_p = compute_candle_indicators(candles_p)

    # Названия и варианты для матчей
    nm_port = get_company_names(p_tickers)
    name_map_port = {t: (nm_port.get(t, {}) or {}).get("company", t) for t in p_tickers}
    variants_port = _build_variants(nm_port)

    # Новости
    news = news_items or get_news_multi(limit=120, days=NEWS_DAYS)
    idx_port = _weighted_news_index(news, variants_port)

    # ---- Антипанический режим? ----
    panic_mode = _compute_panic_mode(inds_p, idx_port)

    # ---- Спекуляция по бумагам портфеля ----
    sells = []
    neutrals_or_buys = []

    for t in p_tickers:
        ind = inds_p.get(t, {}) or {}
        pos_w = idx_port.get(t, {}).get("pos_w", 0.0)
        neg_w = idx_port.get(t, {}).get("neg_w", 0.0)

        # 1) Быстрый дневной триггер (с учётом 30д фильтра)
        fast = _fast_drop_signal(ind, pos_w, neg_w)

        # 2) Базовый месячный медвежий сигнал
        bear = _bearish_signal_30d(ind, pos_w, neg_w)

        sell_candidate = fast or bear

        # ---- Profit-guard от средней цены ----
        last = ind.get("last_close")
        avgc = avg_costs.get(t)
        if sell_candidate and last is not None and avgc not in (None, 0):
            # Если текущая цена достаточно выше средней — блокируем продажу,
            # кроме случая сильного новнегатива.
            guard_level = avgc * (1.0 + PROFIT_GUARD_PCT / 100.0)
            if last >= guard_level and not _news_strong_override(pos_w, neg_w):
                sell_candidate = None

        # В панике не продаём растущие за месяц бумаги без сильного новнегатива
        ch30 = ind.get("month_change_pct")
        if panic_mode and sell_candidate and ch30 is not None and ch30 > 0 and not _news_strong_override(pos_w, neg_w):
            sell_candidate = None

        if sell_candidate:
            qty_port = int(portfolio.get(t, 0))
            frac = float(sell_candidate["sell_fraction"])
            sell_qty = max(1, int(math.ceil(qty_port * frac))) if qty_port else 0
            sells.append({
                "ticker": t,
                "company": name_map_port.get(t, t),
                "action": "продать",
                "quantity": min(sell_qty, qty_port),
                "remaining": max(0, qty_port - min(sell_qty, qty_port)),
                "reason": sell_candidate["reason"],
                "_severity": _severity_score(ind)
            })
            continue

        # ---- Докупить / Удерживать (учёт средней цены) ----
        sma20, sma50 = ind.get("sma20"), ind.get("sma50")
        trend_ok = (sma20 is None or sma50 is None or sma20 >= sma50)
        can_buy_more = False
        if (not panic_mode) and trend_ok and (pos_w >= neg_w) and (last is not None) and (avgc not in (None, 0)):
            # ниже средней на REBUY_DISCOUNT_PCT и более — докупить
            buy_level = avgc * (1.0 - REBUY_DISCOUNT_PCT / 100.0)
            if last <= buy_level:
                can_buy_more = True

        if can_buy_more:
            drop_vs_avg = (last / avgc - 1.0) * 100.0
            neutrals_or_buys.append({
                "ticker": t,
                "company": name_map_port.get(t, t),
                "action": "докупить",
                "reason": f"цена ниже средней на {abs(drop_vs_avg):.2f}% ; тренд не хуже бокового (SMA20≥SMA50) ; новости +/− {pos_w:.1f}/{neg_w:.1f}"
            })
        else:
            # если докупки нет — формируем «удерживать» с прозрачной телеметрией
            trend = "SMA20≥SMA50" if trend_ok else "SMA20<SMA50"
            add_avg = ""
            if last is not None and avgc not in (None, 0):
                diff = (last / avgc - 1.0) * 100.0
                add_avg = f" ; к средней: {diff:+.2f}%"
            neutrals_or_buys.append({
                "ticker": t,
                "company": name_map_port.get(t, t),
                "action": "удерживать",
                "reason": f"сигналов на продажу нет ; 30д {('n/a' if ch30 is None else f'{ch30:+.2f}%')} ; {trend}{add_avg} ; новости +/− {pos_w:.1f}/{neg_w:.1f}"
            })

    # Ограничим число продаж в панике
    if panic_mode and PANIC_MAX_SELLS >= 0 and len(sells) > PANIC_MAX_SELLS:
        sells = sorted(sells, key=lambda x: x.get("_severity", 0.0), reverse=True)[:PANIC_MAX_SELLS]
    for it in sells:
        it.pop("_severity", None)

    spec = sells + neutrals_or_buys

    # ---- Новые возможности (в панике не предлагаем покупки вне портфеля) ----
    new_ops = []
    if not panic_mode:
        watchlist = [t for t in DEFAULT_WATCHLIST if t not in p_tickers]
        candles_w = get_moex_candles(watchlist, days=60)
        inds_w = compute_candle_indicators(candles_w)
        nm_watch = get_company_names(watchlist)
        name_map_watch = {t: (nm_watch.get(t, {}) or {}).get("company", t) for t in watchlist}
        variants_watch = _build_variants(nm_watch)
        idx_watch = _weighted_news_index(news, variants_watch)

        for t in watchlist:
            ind = inds_w.get(t, {}) or {}
            if not _liquidity_ok(ind):
                continue
            pos_w = idx_watch.get(t, {}).get("pos_w", 0.0)
            neg_w = idx_watch.get(t, {}).get("neg_w", 0.0)
            bull = _bullish_candidate(ind, pos_w, neg_w)
            if bull:
                new_ops.append({
                    "ticker": t,
                    "company": name_map_watch.get(t, t),
                    "target_group": "speculation",
                    "reason": f"рост {bull['change_pct']}% за 30д ; тренд не хуже бокового (SMA20≥SMA50) ; новости +/− {pos_w:.1f}/{neg_w:.1f} ; ликвидность ок"
                })
        new_ops = new_ops[:MAX_NEW_OPS]

    # ---- Дивиденды (эвристика по новостям) ----
    dividends = []
    if ENABLE_DIVIDENDS:
        for t in p_tickers:
            titles = idx_port.get(t, {}).get("matched_titles", [])
            if any("дивиден" in (x or "").lower() for x in titles):
                dividends.append({
                    "ticker": t,
                    "company": name_map_port.get(t, t),
                    "reason": "сигналы по дивидендам в новостях"
                })

    result = {
        "dividends_good": dividends,
        "speculation": spec,
        "new_opportunities": new_ops
    }

    if not USE_GIGACHAT_REPHRASE:
        return result

    # Перефраз (опционально; не меняет чисел)
    try:
        token = get_gigachat_token()
    except Exception as e:
        logger.error(f"GigaChat token error: {e}")
        return result

    prompt = f"""
Оформи строго JSON без пояснений. Перепиши тексты "reason" на аккуратный русский, не меняя чисел и фактов.
JSON:
{json.dumps(result, ensure_ascii=True, separators=(',',':'))}
""".strip()

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.encode('ascii','ignore').decode('ascii')}"
    }
    payload = {
        "model": "GigaChat-Max",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(3):
        try:
            s = create_session_with_retries()
            r = s.post(url, headers=headers, json=payload, timeout=(10, 60))
            logger.info(f"GigaChat rephrase status: {r.status_code}")
            logger.debug(f"GigaChat rephrase raw: {r.text}")
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            try:
                out = json.loads(out) if isinstance(out, str) else out
            except Exception:
                out = result
            if isinstance(out, dict):
                out.setdefault("dividends_good", result["dividends_good"])
                out.setdefault("speculation", result["speculation"])
                out.setdefault("new_opportunities", result["new_opportunities"])
                return out
            return result
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise e
    return result
