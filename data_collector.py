# -*- coding: utf-8 -*-
import psycopg2
import os
import logging
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logger = logging.getLogger(__name__)

def get_db_connection():
    """Подключение к базе данных"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

def get_user_portfolio(user_id):
    """
    Получает портфель пользователя из базы данных
    Args:
        user_id (int): Telegram ID пользователя
    Returns:
        dict: Словарь {тикер: количество} или пустой словарь при ошибке
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Получаем портфель пользователя
        cur.execute(
            """
            SELECT ticker, quantity 
            FROM portfolio 
            WHERE user_id = (SELECT id FROM users WHERE telegram_id = %s)
            """,
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        # Преобразуем в словарь
        portfolio = {}
        for ticker, quantity in rows:
            # Убедимся, что тикер - строка, а количество - целое число
            if ticker and quantity is not None:
                portfolio[str(ticker).upper()] = int(quantity)
        return portfolio
    except Exception:
        return {}

def get_moex_data(tickers):
    """
    Получает текущие котировки с MOEX для списка тикеров
    Args:
        tickers (list): Список тикеров
    Returns:
        dict: Словарь {тикер: {price, volume}} или пустой словарь при ошибке
    """
    import requests
    moex_data = {}
    for ticker in tickers:
        try:
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Извлекаем последнюю цену и объем (индексы могут меняться)
                marketdata = data.get("marketdata", {})
                if marketdata.get("data"):
                    # Берем первую запись (последние данные)
                    row = marketdata["data"][0]
                    columns = marketdata["columns"]
                    # Находим индексы нужных колонок
                    price_idx = columns.index("LAST") if "LAST" in columns else -1
                    volume_idx = columns.index("VOLTODAY") if "VOLTODAY" in columns else -1
                    price = row[price_idx] if price_idx >= 0 and price_idx < len(row) else None
                    volume = row[volume_idx] if volume_idx >= 0 and volume_idx < len(row) else 0
                    if price is not None:
                        moex_data[ticker] = {
                            "price": float(price),
                            "volume": int(volume) if volume else 0
                        }
        except Exception:
            pass
    return moex_data

def get_news():
    """
    Собирает экономические новости за последние 24 часа
    Returns:
        list: Список заголовков новостей или пустой список при ошибке
    """
    import requests
    import datetime
    try:
        today = datetime.datetime.now()
        yesterday = today - datetime.timedelta(days=1)
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            return []
        params = {
            "q": "экономика OR финансы OR биржа OR акции",
            "from": yesterday.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
            "language": "ru",
            "sortBy": "publishedAt",
            "apiKey": api_key
        }
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            articles = response.json().get("articles", [])[:10]  # Берем первые 10 новостей
            return [article["title"] for article in articles if article.get("title")]
    except Exception:
        pass
    return []
