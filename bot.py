# -*- coding: utf-8 -*-
import sys
import io
import os
import ctypes

# Принудительная настройка UTF-8 для Windows
if sys.platform == "win32":
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        os.environ['PYTHONUTF8'] = '1'
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import logging
import json
import datetime
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from telegram import Update, Document
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.request import HTTPXRequest
import psycopg2
import bcrypt
import re
import html

# ===========================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

load_dotenv()

ACCESS_PASSWORD_HASH = os.getenv("ACCESS_PASSWORD_HASH", "")
ADMIN_IDS = {int(x) for x in (os.getenv("ADMIN_IDS","") or "").split(",") if x.strip().isdigit()}
DEFAULT_TZ = os.getenv("DEFAULT_TZ","Europe/Moscow")


def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    cleaned = ''.join(
        c for c in text
        if (' ' <= c <= '~') or ('\u0400' <= c <= '\u04FF') or c in '\n\r\t•📊✅❌📌⚠️📈📉➖💰'
    )
    return cleaned.encode('utf-8', 'ignore').decode('utf-8')

def _safe(s: str) -> str:
    return html.escape(clean_text(str(s)))

async def post_init(application: Application) -> None:
    logger.info("✅ Бот успешно инициализирован")

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

def verify_password(stored_hash, input_password):
    return bcrypt.checkpw(input_password.encode(), stored_hash.encode())

async def is_authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE telegram_id = %s AND is_blocked = FALSE", (user_id,))
    is_auth = cur.fetchone() is not None
    cur.close()
    conn.close()
    return is_auth

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update, context):
        context.user_data['awaiting_password'] = True
        await update.message.reply_text("🔐 Введите пароль для доступа:")
        return
    if await is_authorized(update, context):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticker, quantity, avg_cost
            FROM portfolio
            WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s)
            ORDER BY ticker
            """,
            (update.effective_user.id,),
        )
        portfolio = cur.fetchall()
        cur.close()
        conn.close()

        if portfolio:
            def _fmt_row(t, q, ac):
                if ac is None:
                    return f"• {t}: {q} акций"
                return f"• {t}: {q} акций, средняя {ac:.2f} ₽"
            portfolio_str = "\n".join([_fmt_row(t, q, a) for t, q, a in portfolio])
            await update.message.reply_text(
                f"✅ Ваш текущий портфель:\n{portfolio_str}\n\n"
                "Чтобы обновить портфель:\n"
                "1) Отправьте CSV-файл (ticker,quantity,avg_cost)\n"
                "2) Или напишите: Купить SBER 10 по 272.5 / Продать GAZP 5"
            )
        else:
            await update.message.reply_text(
                "👋 Добро пожаловать! Я знаю вас, но портфель пуст.\n"
                "Отправьте CSV-файл (ticker,quantity,avg_cost), чтобы загрузить портфель."
            )
    else:
        context.user_data['awaiting_password'] = True
        await update.message.reply_text("🔐 Введите пароль для доступа.")

async def handle_password(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_password"):
        return
    pwd = (update.message.text or "").strip()
    if not ACCESS_PASSWORD_HASH:
        await update.message.reply_text("❌ Бот не настроен (нет ACCESS_PASSWORD_HASH).")
        return
    try:
        ok = bcrypt.checkpw(pwd.encode(), ACCESS_PASSWORD_HASH.encode())
    except Exception:
        ok = False
    if not ok:
        await update.message.reply_text("🔐 Пароль неверный. Попробуйте ещё раз.")
        return

    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO users(telegram_id, is_blocked, timezone)
        VALUES(%s, FALSE, %s)
        ON CONFLICT(telegram_id) DO NOTHING
    """, (update.effective_user.id, DEFAULT_TZ))
    conn.commit(); cur.close(); conn.close()
    context.user_data.pop("awaiting_password", None)
    await update.message.reply_text("✅ Доступ предоставлен. Отправьте CSV портфеля или используйте команды покупки/продажи.")
    await start(update, context)

def _is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

async def admin_block(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_user.id): return
    if not context.args:
        await update.message.reply_text("Формат: /block <telegram_id>"); return
    tgt = int(context.args[0])
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("UPDATE users SET is_blocked=TRUE WHERE telegram_id=%s", (tgt,))
    conn.commit(); cur.close(); conn.close()
    await update.message.reply_text(f"✅ Пользователь {tgt} заблокирован.")

async def admin_unblock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_user.id): return
    if not context.args:
        await update.message.reply_text("Формат: /unblock <telegram_id>"); return
    tgt = int(context.args[0])
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("UPDATE users SET is_blocked=FALSE WHERE telegram_id=%s", (tgt,))
    conn.commit(); cur.close(); conn.close()
    await update.message.reply_text(f"✅ Пользователь {tgt} разблокирован.")

async def admin_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_user.id): return
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT telegram_id, is_blocked, timezone, created_at FROM users ORDER BY created_at DESC NULLS LAST")
    rows = cur.fetchall(); cur.close(); conn.close()
    lines_ = ["Пользователи:"]
    for r in rows or []:
        tid, blocked, tz, ts = r
        lines_.append(f"• {tid} — {'blocked' if blocked else 'active'} — {tz or '-'} — {ts or ''}")
    await update.message.reply_text("\n".join(lines_))

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT is_blocked, timezone FROM users WHERE telegram_id=%s", (update.effective_user.id,))
    row = cur.fetchone(); cur.close(); conn.close()
    await update.message.reply_text(f"ID: {update.effective_user.id}\nСтатус: {'blocked' if row and row[0] else 'active'}\nTZ: {(row and row[1]) or 'n/a'}")

async def handle_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update, context):
        await update.message.reply_text("❌ Сначала авторизуйтесь через /start")
        return

    document: Document = update.message.document
    if document.mime_type != "text/csv":
        await update.message.reply_text("❌ Отправьте файл в формате CSV")
        return

    file = await context.bot.get_file(document.file_id)
    file_stream = BytesIO()
    await file.download_to_memory(file_stream)
    file_stream.seek(0)

    try:
        df = pd.read_csv(file_stream)
        df.columns = [c.strip().lower() for c in df.columns]

        required_columns = {"ticker", "quantity"}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV должен содержать столбцы: ticker, quantity (опционально: avg_cost)")

        has_avg = ("avg_cost" in df.columns) or ("avg_price" in df.columns)
        if "avg_price" in df.columns and "avg_cost" not in df.columns:
            df["avg_cost"] = df["avg_price"]

        conn = get_db_connection()
        cur = conn.cursor()
        user_id = update.effective_user.id

        cur.execute(
            "DELETE FROM portfolio WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s)",
            (user_id,),
        )

        for _, row in df.iterrows():
            ticker = str(row["ticker"]).upper().strip()
            qty = int(row["quantity"])
            avg = None
            if has_avg and pd.notna(row["avg_cost"]):
                try:
                    avg = float(str(row["avg_cost"]).replace(",", "."))
                except Exception:
                    avg = None

            if avg is None:
                cur.execute(
                    """
                    INSERT INTO portfolio (user_id, ticker, quantity)
                    VALUES ((SELECT id FROM users WHERE telegram_id = %s), %s, %s)
                    ON CONFLICT (user_id, ticker)
                    DO UPDATE SET quantity = EXCLUDED.quantity
                    """,
                    (user_id, ticker, qty),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO portfolio (user_id, ticker, quantity, avg_cost)
                    VALUES ((SELECT id FROM users WHERE telegram_id = %s), %s, %s, %s)
                    ON CONFLICT (user_id, ticker)
                    DO UPDATE SET
                        quantity = EXCLUDED.quantity,
                        avg_cost = EXCLUDED.avg_cost
                    """,
                    (user_id, ticker, qty, avg),
                )

        conn.commit()
        cur.close()
        conn.close()

        await update.message.reply_text("✅ Портфель обновлён.")
    except Exception as e:
        logger.error(f"Ошибка обработки CSV: {str(e)}", exc_info=True)
        await update.message.reply_text("❌ Не удалось обработать CSV. Проверьте формат (ticker,quantity,avg_cost).")

async def handle_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update, context):
        await update.message.reply_text("❌ Сначала авторизуйтесь через /start")
        return

    text = update.message.text.strip()

    match = re.match(
        r"(Купить|Продать)\s+([A-Za-zА-Яа-я0-9._-]+)\s+(\d+)(?:\s+по\s+(\d+(?:[.,]\d+)?))?$",
        text,
        re.IGNORECASE
    )

    if not match:
        await update.message.reply_text(
            "Формат: Купить <TICKER> <QTY> по <PRICE>  или  Продать <TICKER> <QTY>\n"
            "Пример: Купить SBER 10 по 272.5"
        )
        return

    action, ticker, quantity, price_str = match.groups()
    action = action.lower()
    ticker = ticker.upper()
    quantity = int(quantity)
    buy_price = float(price_str.replace(',', '.')) if price_str else None

    conn = get_db_connection()
    cur = conn.cursor()
    user_id = update.effective_user.id

    try:
        if action == "продать":
            cur.execute(
                """
                SELECT quantity, avg_cost
                FROM portfolio
                WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s) AND ticker = %s
                """,
                (user_id, ticker),
            )
            row = cur.fetchone()
            if row is None or row[0] < quantity:
                await update.message.reply_text("❌ Недостаточно акций для продажи")
                cur.close()
                conn.close()
                return

            new_qty = row[0] - quantity
            if new_qty == 0:
                cur.execute(
                    """
                    DELETE FROM portfolio
                    WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s) AND ticker = %s
                    """,
                    (user_id, ticker),
                )
            else:
                cur.execute(
                    """
                    UPDATE portfolio
                    SET quantity = %s
                    WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s) AND ticker = %s
                    """,
                    (new_qty, user_id, ticker),
                )

            conn.commit()
            await update.message.reply_text(f"✅ Продажа {quantity} {ticker} учтена. Остаток: {new_qty}")

        else:
            if buy_price is None:
                await update.message.reply_text("❌ Укажите цену покупки: 'Купить TICKER Q по PRICE'")
                cur.close()
                conn.close()
                return

            cur.execute(
                """
                SELECT quantity, avg_cost
                FROM portfolio
                WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s) AND ticker = %s
                """,
                (user_id, ticker),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """
                    INSERT INTO portfolio (user_id, ticker, quantity, avg_cost)
                    VALUES ((SELECT id FROM users WHERE telegram_id = %s), %s, %s, %s)
                    """,
                    (user_id, ticker, quantity, buy_price),
                )
            else:
                old_qty, old_avg = row
                new_qty = old_qty + quantity
                new_avg = (old_avg * old_qty + buy_price * quantity) / new_qty if (old_avg is not None) else buy_price
                cur.execute(
                    """
                    UPDATE portfolio SET quantity=%s, avg_cost=%s
                    WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s) AND ticker=%s
                    """,
                    (new_qty, new_avg, user_id, ticker),
                )

            conn.commit()
            await update.message.reply_text(f"✅ Покупка {quantity} {ticker} по {buy_price:.2f} ₽ учтена.")

    except Exception as e:
        logger.error(f"Ошибка записи трейда: {str(e)}", exc_info=True)
        await update.message.reply_text("❌ Ошибка при учёте сделки.")
    finally:
        cur.close()
        conn.close()

# -----------------------------
# НОВЫЙ: форматирование отчёта
# -----------------------------

def _extract_num(x, default=0.0):
    try:
        if x is None: return default
        if isinstance(x, str):
            x = x.replace('%','').replace(',','.')
        return float(x)
    except Exception:
        return default

def _fmt_pct(x):
    return f"{x:+.2f}%"

def _news_tuple_from_item(it: dict):
    # Пытаемся собрать (pos, neg, neu) из разных возможных ключей
    p = it.get("news_pos") or it.get("news_plus") or it.get("news_p") or 0
    n = it.get("news_neg") or it.get("news_minus") or it.get("news_m") or 0
    u = it.get("news_neu") or it.get("news_neutral") or it.get("news_n") or 0
    try:
        return int(p), int(n), int(u)
    except Exception:
        # если пришла строка вида "+/0/0/0" — отдадим без разбора
        return (p or 0, n or 0, u or 0)

def _sma_relation(it: dict):
    # Если уже есть готовая строка — используем её
    for k in ("sma_relation", "trend", "sma", "sma_note"):
        if it.get(k):
            return str(it.get(k))
    # Иначе строим из отдельных чисел/булей
    s20 = it.get("sma20")
    s50 = it.get("sma50")
    if s20 is not None and s50 is not None:
        try:
            s20 = float(s20); s50 = float(s50)
            return "SMA20>SMA50" if s20 > s50 else ("SMA20<SMA50" if s20 < s50 else "SMA20≈SMA50")
        except Exception:
            pass
    return "—"

def _metrics_line(it: dict):
    ch30 = _extract_num(it.get("change_30d") or it.get("change30d") or it.get("30d_change") or 0.0)
    sma = _sma_relation(it)
    p, n, u = _news_tuple_from_item(it)
    # «новости: 2+/1-/0n»
    news_str = f"{p}+/{n}-/{u}n"
    return f"30д: {_fmt_pct(ch30)} | {sma} | новости: {news_str}"

def _fmt_sell_line(it: dict):
    t = (it.get("ticker") or "").upper()
    nm = it.get("company") or t
    qty = it.get("quantity", "?")
    rem = it.get("remaining", "?")
    reason = it.get("reason") or "сигнал на фиксацию прибыли/снижения риска"
    head = f"📉 {t} — {_safe(nm)}: продать {qty} (останется {rem}), {_safe(reason)}"
    return head + "\n   " + _metrics_line(it)

def _fmt_hold_line(it: dict):
    t = (it.get("ticker") or "").upper()
    nm = it.get("company") or t
    reason = it.get("reason") or "удерживать"
    head = f"➖ {t} — {_safe(nm)}: удерживать, {_safe(reason)}"
    return head + "\n   " + _metrics_line(it)

def _fmt_newop_line(it: dict):
    t = (it.get("ticker") or "").upper()
    nm = it.get("company") or t
    reason = it.get("reason") or "перспективная динамика"
    head = f"📈 {t} — {_safe(nm)}: {_safe(reason)}"
    return head + "\n   " + _metrics_line(it)

def _format_full_report(report_json: dict) -> str:
    # --- ДИВИДЕНДЫ ---
    div_items = []
    for it in report_json.get("dividends_good", []) or []:
        if not isinstance(it, dict): continue
        t = (it.get("ticker") or "").upper()
        nm = it.get("company") or t
        rs = _safe(it.get("reason",""))
        div_items.append(f"• {t} — {nm}: {rs}")
    dividends = "\n".join(div_items) if div_items else "Нет сигнала по дивидендам"

    # --- СПЕКУЛЯЦИЯ ---
    spec_items = []
    for it in report_json.get("speculation", []) or []:
        if not isinstance(it, dict): continue
        action = (it.get("action") or "удерживать").lower()
        if action == "продать":
            spec_items.append(_fmt_sell_line(it))
        else:
            spec_items.append(_fmt_hold_line(it))
    speculation = "\n".join(spec_items) if spec_items else "Сигналов нет"

    # --- НОВЫЕ ВОЗМОЖНОСТИ (сортировка по росту за 30д) ---
    new_ops_list = [it for it in (report_json.get("new_opportunities") or []) if isinstance(it, dict)]
    new_ops_list.sort(key=lambda x: _extract_num(x.get("change_30d") or x.get("change30d") or x.get("30d_change") or 0.0), reverse=True)
    new_items = [_fmt_newop_line(it) for it in new_ops_list]
    new_ops = "\n".join(new_items) if new_items else "Нет новых идей"

    message = f"""
<b>📊 СВОДКА НА {datetime.datetime.now().strftime('%d.%m.%Y')}</b>

<b>💰 ДИВИДЕНДЫ</b>
{dividends}

<b>📉 СПЕКУЛЯЦИЯ ПО ПОРТФЕЛЮ</b>
{speculation}

<b>📈 НОВЫЕ ВОЗМОЖНОСТИ</b>
{new_ops}
""".strip()

    return clean_text(message)

# -----------------------------
# Формирование и отправка отчёта
# -----------------------------

def send_daily_report(bot):
    """Формирует и отправляет сводку с улучшенным форматированием."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT telegram_id FROM users WHERE is_blocked = FALSE")
        user_ids = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()

        if not user_ids:
            logger.info("Нет активных пользователей для отправки сводки")
            return

        try:
            from data_collector import get_moex_data, get_news_annotated, get_user_portfolio, get_user_costs
            from gigachat_analyzer import analyze_with_gigachat
        except ImportError as e:
            logger.error(f"Не удалось импортировать модули: {str(e)}")
            return

        for user_id in user_ids:
            try:
                portfolio = get_user_portfolio(user_id)
                costs = get_user_costs(user_id)
                if not portfolio:
                    logger.info(f"У пользователя {user_id} пустой портфель — пропуск")
                    continue

                tickers = list(portfolio.keys())
                moex_data = get_moex_data(tickers)
                news = get_news_annotated(limit=20)

                report_json = analyze_with_gigachat(portfolio, moex_data, news, avg_costs=costs)
                if not isinstance(report_json, dict):
                    logger.warning(f"Неверный ответ GigaChat для пользователя {user_id}: {type(report_json)}")
                    continue

                message = _format_full_report(report_json)

                try:
                    token = os.getenv("TELEGRAM_TOKEN")
                    if not token:
                        logger.error("TELEGRAM_TOKEN не найден в окружении")
                        continue

                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": user_id,
                        "text": message,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True
                    }
                    headers = {
                        "Content-Type": "application/json; charset=utf-8",
                        "Accept-Charset": "utf-8"
                    }
                    import requests
                    response = requests.post(url, json=payload, headers=headers, timeout=30)

                    if response.status_code == 200:
                        logger.info(f"Сводка успешно отправлена пользователю {user_id}")
                    else:
                        logger.error(
                            f"Ошибка отправки сводки для {user_id}. "
                            f"Статус: {response.status_code}, Тело: {response.text}"
                        )

                except Exception as e:
                    logger.error(f"Критическая ошибка отправки для {user_id}: {str(e)}", exc_info=True)

            except Exception as e:
                logger.error(f"Ошибка для пользователя {user_id}: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"Критическая ошибка в send_daily_report: {str(e)}", exc_info=True)

def start_scheduler(application):
    import pytz
    from datetime import time as _time
    tz = pytz.timezone(DEFAULT_TZ)
    async def daily_job(context: ContextTypes.DEFAULT_TYPE):
        try:
            send_daily_report(context.bot)
        except Exception as e:
            logger.error(f"Ошибка daily_job: {e}", exc_info=True)
    application.job_queue.run_daily(daily_job, time=_time(7,0, tzinfo=tz), name="daily_report_job")
    logger.info("✅ Планировщик JobQueue: ежедневная сводка в 07:00")

async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update, context):
        await update.message.reply_text("❌ Сначала авторизуйтесь через /start")
        return
    user_id = update.effective_user.id
    try:
        from data_collector import get_moex_data, get_news_annotated, get_user_portfolio, get_user_costs
        from gigachat_analyzer import analyze_with_gigachat
        portfolio = get_user_portfolio(user_id)
        costs = get_user_costs(user_id)
        if not portfolio:
            await update.message.reply_text("Ваш портфель пуст. Загрузите CSV (ticker,quantity,avg_cost).")
            return
        tickers = list(portfolio.keys())
        moex_data = get_moex_data(tickers)
        news = get_news_annotated(limit=20)
        report_json = analyze_with_gigachat(portfolio, moex_data, news, avg_costs=costs)
        if not isinstance(report_json, dict):
            await update.message.reply_text("Не удалось сформировать отчёт.")
            return

        message = _format_full_report(report_json)
        await update.message.reply_text(message, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"/report error: {e}", exc_info=True)
        await update.message.reply_text("Ошибка при формировании отчёта.")

def main():
    request = HTTPXRequest(
        read_timeout=30.0,
        write_timeout=30.0,
        connect_timeout=30.0,
        pool_timeout=30.0
    )

    application = Application.builder() \
        .token(os.getenv("TELEGRAM_TOKEN")) \
        .request(request) \
        .post_init(post_init) \
        .build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("report", report))
    application.add_handler(CommandHandler("whoami", whoami))
    application.add_handler(CommandHandler("block", admin_block))
    application.add_handler(CommandHandler("unblock", admin_unblock))
    application.add_handler(CommandHandler("users", admin_users))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_password), group=0)
    application.add_handler(MessageHandler(filters.Document.FileExtension("csv"), handle_csv))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_command))

    start_scheduler(application)

    logger.info("🤖 Бот запущен. Нажмите Ctrl+C для остановки.")
    application.run_polling()

if __name__ == "__main__":
    main()
