# -*- coding: utf-8 -*-
# ===== UTF-8 НАСТРОЙКИ =====
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
            sys.stderr.reconfigure(encoding='utf-8')
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass
# ===========================================================

# ===== ИМПОРТЫ =====
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
from apscheduler.schedulers.background import BackgroundScheduler
import html

# ===========================================================

# ===== НАСТРОЙКА ЛОГИРОВАНИЯ С UTF-8 =====
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding='utf-8'
)
logger = logging.getLogger(__name__)
# ===========================================

# Загружаем переменные окружения
load_dotenv()

# ===== ФУНКЦИЯ ОЧИСТКИ ТЕКСТА =====
def clean_text(text):
    """Улучшенная очистка текста от недопустимых символов"""
    if not isinstance(text, str):
        return str(text)

    # Удаляем control characters кроме разрешенных
    cleaned = ''.join(
        c for c in text
        if (c >= ' ' and c <= '~') or (c >= '\u0400' and c <= '\u04FF')
    )

    # Дополнительная очистка через encode/decode
    return cleaned.encode('utf-8', 'ignore').decode('utf-8')

# ==============================================

# ===== ИНИЦИАЛИЗАЦИЯ БОТА =====
async def post_init(application: Application) -> None:
    """Вызывается после инициализации бота"""
    logger.info("✅ Бот успешно инициализирован")

# Подключение к БД
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

# Проверка пароля
def verify_password(stored_hash, input_password):
    return bcrypt.checkpw(input_password.encode(), stored_hash.encode())

# Проверка авторизации
async def is_authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE telegram_id = %s AND is_blocked = FALSE", (user_id,))
    is_authorized = cur.fetchone() is not None
    cur.close()
    conn.close()
    return is_authorized

# Обработчик /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await is_authorized(update, context):
        # Показываем текущий портфель
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticker, quantity 
            FROM portfolio 
            WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s)
            """,
            (update.effective_user.id,),
        )
        portfolio = cur.fetchall()
        cur.close()
        conn.close()

        if portfolio:
            portfolio_str = "\n".join([f"• {ticker}: {quantity} акций" for ticker, quantity in portfolio])
            await update.message.reply_text(
                f"Ваш текущий портфель:\n{portfolio_str}\n\n"
                "Чтобы обновить портфель:\n"
                "1. Отправьте CSV-файл\n"
                "2. Или напишите: Купить Сбербанк 10 / Продать Газпром 5"
            )
        else:
            await update.message.reply_text(
                "Добро пожаловать! Ваш портфель пуст.\n"
                "Отправьте CSV-файл или используйте команды:\n"
                "Купить [тикер] [количество]\n"
                "Продать [тикер] [количество]"
            )
    else:
        await update.message.reply_text(
            "Введите пароль для доступа к боту.\n"
            "Пароль выдал администратор."
        )
        context.user_data["awaiting_password"] = True


# Обработчик CSV-файлов
async def handle_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update, context):
        await update.message.reply_text("Сначала авторизуйтесь через /start")
        return
    document: Document = update.message.document
    if document.mime_type != "text/csv":
        await update.message.reply_text("Отправьте файл в формате CSV")
        return
    # Скачиваем файл
    file = await context.bot.get_file(document.file_id)
    file_stream = BytesIO()
    await file.download_to_memory(file_stream)
    file_stream.seek(0)
    try:
        # Парсим CSV
        df = pd.read_csv(file_stream)
        required_columns = {"ticker", "quantity"}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV должен содержать столбцы: ticker, quantity")
        # Сохраняем в БД
        conn = get_db_connection()
        cur = conn.cursor()
        user_id = update.effective_user.id
        # Удаляем старые данные
        cur.execute(
            "DELETE FROM portfolio WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s)",
            (user_id,),
        )
        # Добавляем новые
        for _, row in df.iterrows():
            cur.execute(
                """
                INSERT INTO portfolio (user_id, ticker, quantity)
                VALUES (
                    (SELECT id FROM users WHERE telegram_id = %s),
                    %s,
                    %s
                )
                ON CONFLICT (user_id, ticker) 
                DO UPDATE SET quantity = EXCLUDED.quantity
                """,
                (user_id, row["ticker"], row["quantity"]),
            )
        conn.commit()
        cur.close()
        conn.close()
        await update.message.reply_text("Портфель успешно обновлен!")
    except Exception as e:
        await update.message.reply_text(f"Ошибка обработки файла: {str(e)}")

# Обработчик команд покупки/продажи
async def handle_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update, context):
        await update.message.reply_text("Сначала авторизуйтесь через /start")
        return
    text = update.message.text.strip()
    match = re.match(r"(Купить|Продать)\s+(\w+)\s+(\d+)", text, re.IGNORECASE)
    if not match:
        await update.message.reply_text(
            "Неверный формат команды.\n"
            "Используйте:\n"
            "Купить [тикер] [количество]\n"
            "Продать [тикер] [количество]"
        )
        return
    action, ticker, quantity = match.groups()
    quantity = int(quantity)
    # Обновляем БД
    conn = get_db_connection()
    cur = conn.cursor()
    user_id = update.effective_user.id
    try:
        if action.lower() == "продать":
            # Проверяем, достаточно ли акций
            cur.execute(
                """
                SELECT quantity FROM portfolio 
                WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s) 
                AND ticker = %s
                """,
                (user_id, ticker),
            )
            current_quantity = cur.fetchone()
            if not current_quantity or current_quantity[0] < quantity:
                raise ValueError("Недостаточно акций для продажи")
            # Обновляем количество
            new_quantity = current_quantity[0] - quantity
            if new_quantity == 0:
                cur.execute(
                    """
                    DELETE FROM portfolio 
                    WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s) 
                    AND ticker = %s
                    """,
                    (user_id, ticker),
                )
            else:
                cur.execute(
                    """
                    UPDATE portfolio SET quantity = %s
                    WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s) 
                    AND ticker = %s
                    """,
                    (new_quantity, user_id, ticker),
                )
        else:  # Покупка
            cur.execute(
                """
                INSERT INTO portfolio (user_id, ticker, quantity)
                VALUES (
                    (SELECT id FROM users WHERE telegram_id = %s),
                    %s,
                    %s
                )
                ON CONFLICT (user_id, ticker) 
                DO UPDATE SET quantity = portfolio.quantity + EXCLUDED.quantity
                """,
                (user_id, ticker, quantity),
            )
        conn.commit()
        await update.message.reply_text(
            f"Операция выполнена: {action} {quantity} акций {ticker}"
        )
    except Exception as e:
        conn.rollback()
        await update.message.reply_text(f"Ошибка: {str(e)}")
    finally:
        cur.close()
        conn.close()

# === ЕЖЕДНЕВНАЯ СВОДКА ===
def send_daily_report(bot):
    """Формирует и отправляет сводку в указанное время"""
    try:
        # Получаем список всех активных пользователей
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT telegram_id FROM users WHERE is_blocked = FALSE")
        user_ids = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        if not user_ids:
            return
        # Импортируем здесь, чтобы избежать циклических зависимостей
        try:
            from data_collector import get_moex_data, get_news, get_user_portfolio
            from gigachat_analyzer import analyze_with_gigachat
        except ImportError:
            return
        for user_id in user_ids:
            try:
                # Получаем портфель пользователя
                portfolio = get_user_portfolio(user_id)
                if not portfolio:
                    continue
                # Собираем данные
                tickers = list(portfolio.keys())
                moex_data = get_moex_data(tickers)
                news = get_news()
                # Генерируем рекомендации
                report = analyze_with_gigachat(portfolio, moex_data, news)
                # Очищаем ответ от недопустимых символов
                clean_report = clean_text(report)
                try:
                    report_json = json.loads(clean_report)
                except json.JSONDecodeError:
                    continue
                # Формируем рекомендации
                long_term_items = []
                portfolio_tickers = list(portfolio.keys())
                # Создаем словарь для быстрого поиска рекомендаций
                long_term_dict = {}
                for item in report_json.get("long_term", []):
                    if isinstance(item, dict) and 'ticker' in item:
                        long_term_dict[item['ticker'].upper()] = item
                # Для КАЖДОЙ компании из портфеля должна быть рекомендация
                for ticker in portfolio_tickers:
                    item = long_term_dict.get(ticker.upper())
                    if item:
                        action = item.get('action', 'hold')
                        reason = item.get('reason', 'Причина не указана')
                        long_term_items.append(f"• {ticker}: {action} - {html.escape(clean_text(reason))}")
                    else:
                        # Если GigaChat не предоставил рекомендацию, добавляем шаблон
                        long_term_items.append(f"• {ticker}: Удерживать (рекомендация не предоставлена)")
                long_term = "\n".join(long_term_items)
                short_term_items = []
                for item in report_json.get("short_term", []):
                    if not isinstance(item, dict):
                        continue
                    ticker = item.get('ticker', '').upper()
                    reason = item.get('reason', 'Причина не указана')
                    quantity = item.get('quantity', '?')
                    remaining = item.get('remaining', '?')
                    # Проверяем, что тикер есть в портфеле
                    if ticker and ticker in portfolio_tickers:
                        short_term_items.append(
                            f"• {ticker}: {html.escape(clean_text(reason))} (продать {quantity}, оставить {remaining})"
                        )
                short_term = "\n".join(short_term_items)
                new_opportunities_items = []
                for item in report_json.get("new_opportunities", []):
                    if not isinstance(item, dict):
                        continue
                    ticker = item.get('ticker', '').upper()
                    perspective = item.get('perspective', 'не определена')
                    reason = item.get('reason', 'Причина не указана')
                    # Проверяем, что тикер НЕ в портфеле
                    if ticker and ticker not in portfolio_tickers:
                        new_opportunities_items.append(
                            f"• {ticker} ({perspective} перспектива): {html.escape(clean_text(reason))}")
                # Если GigaChat не вернул новых возможностей
                if not new_opportunities_items:
                    new_opportunities_items.append("• Анализ рынка не выявил новых перспективных возможностей")
                new_opportunities = "\n".join(new_opportunities_items)
                # Формируем сообщение в HTML
                message = f"""
<b>СВОДКА НА {datetime.datetime.now().strftime('%d.%m.%Y')}</b>
<b>ДОЛГОСРОЧНЫЕ РЕКОМЕНДАЦИИ:</b>
{long_term or 'Нет изменений'}
<b>КРАТКОСРОЧНЫЕ РЕКОМЕНДАЦИИ:</b>
{short_term or 'Нет изменений'}
<b>НОВЫЕ ВОЗМОЖНОСТИ:</b>
{new_opportunities or 'Нет перспективных акций'}
"""
                # ТРОЙНАЯ ОЧИСТКА ТЕКСТА
                safe_message = clean_text(message)
                safe_message = safe_message.encode('utf-8', 'ignore').decode('utf-8')
                safe_message = safe_message.replace('\x00', '')  # Удаляем нул-байты
                # Прямой вызов Telegram API
                try:
                    token = os.getenv("TELEGRAM_TOKEN")
                    if not token:
                        continue
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": user_id,
                        "text": safe_message,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True
                    }
                    headers = {
                        "Content-Type": "application/json; charset=utf-8",
                        "Accept-Charset": "utf-8"
                    }
                    import requests
                    response = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=30
                    )
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

def start_scheduler(application):
    """Корректная настройка планировщика"""
    scheduler = BackgroundScheduler(timezone="Europe/Moscow")
    scheduler.add_job(
        send_daily_report,
        "cron",
        hour=7,
        minute=0,
        args=[application.bot]
    )
    scheduler.start()
    logger.info("Планировщик запущен: ежедневная сводка в 07:00 МСК")

# === ЗАПУСК БОТА ===
def main():
    # Настройка HTTP-запросов
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
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.FileExtension("csv"), handle_csv))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trade_command))
    # Запускаем планировщик
    start_scheduler(application)
    # Запускаем бота
    logger.info("Бот запущен. Нажмите Ctrl+C для остановки.")
    application.run_polling()

if __name__ == "__main__":
    main()
