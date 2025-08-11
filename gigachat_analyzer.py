# -*- coding: utf-8 -*-
import requests
import os
import json
import uuid
import base64
import urllib3
import logging
import re
import time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Настройка логирования
logger = logging.getLogger(__name__)

# Отключаем предупреждения SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

def create_session_with_retries():
    """Создает сессию requests с настройками повторных попыток"""
    session = requests.Session()
    session.verify = False  # Отключаем проверку SSL для работы в России
    # Настройка повторных попыток
    retry_strategy = Retry(
        total=3,  # Максимум 3 попытки
        backoff_factor=1,  # Задержка между попытками (1, 2, 4 секунды)
        status_forcelist=[429, 500, 502, 503, 504],  # Статусы для повтора
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]  # Методы для повтора
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # Устанавливаем таймауты
    session.headers.update({
        'Accept-Charset': 'utf-8',
        'User-Agent': 'Python-urllib/3.0',
        'Connection': 'close'  # Закрываем соединение после каждого запроса
    })
    return session

def get_gigachat_token():
    """Получает токен доступа к GigaChat с повторными попытками"""
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    # Генерируем ASCII-совместимый RqUID
    rq_uid = str(uuid.uuid4())
    # Получаем и очищаем учетные данные
    client_id = os.getenv('GIGACHAT_CLIENT_ID', '')
    client_secret = os.getenv('GIGACHAT_CLIENT_SECRET', '')
    # Принудительно кодируем в ASCII, игнорируя не-ASCII символы
    client_id = client_id.encode('ascii', 'ignore').decode('ascii')
    client_secret = client_secret.encode('ascii', 'ignore').decode('ascii')
    if not client_id or not client_secret:
        raise ValueError("GIGACHAT_CLIENT_ID или GIGACHAT_CLIENT_SECRET не заданы в .env")
    # Создаем строку Basic Auth и кодируем в base64
    auth_string = f"{client_id}:{client_secret}"
    try:
        encoded_auth = base64.b64encode(auth_string.encode('ascii')).decode('ascii')
    except Exception:
        raise
    # Формируем заголовки с гарантией ASCII
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": rq_uid,  # Только ASCII символы!
        "Authorization": f"Basic {encoded_auth}"  # Закодировано в base64
    }
    # Передаем данные как строку, а не словарь
    data = "scope=GIGACHAT_API_PERS"
    # Попытки получения токена
    for attempt in range(3):
        try:
            # Создаем сессию с повторными попытками
            session = create_session_with_retries()
            response = session.post(
                url,
                headers=headers,
                data=data,
                timeout=(10, 30)  # (connect_timeout, read_timeout)
            )
            response.raise_for_status()
            token = response.json()["access_token"]
            return token
        except requests.exceptions.Timeout:
            if attempt < 2:  # Если это не последняя попытка
                time.sleep(2 ** attempt)  # Экспоненциальная задержка
                continue
            else:
                raise
        except requests.exceptions.RequestException:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            else:
                raise
        except Exception:
            raise
    raise Exception("Не удалось получить токен GigaChat после 3 попыток")

def clean_json_string(text):
    """
    Очищает ответ от недопустимых символов и исправляет синтаксис JSON
    Args:
        text (str): Сырой ответ от GigaChat
    Returns:
        str: Очищенный JSON-строки
    """
    if not isinstance(text, str):
        return str(text)
    # Удаляем недопустимые управляющие символы
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Заменяем неправильные кавычки на правильные
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('„', '"').replace('“', '"')
    text = text.replace('\'', '"')  # Заменяем одинарные кавычки на двойные
    # Удаляем другие потенциально проблемные символы
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2026', '...')  # Заменяем многоточие
    # Исправляем распространенные синтаксические ошибки
    text = re.sub(r',\s*}', '}', text)  # Убираем запятые перед }
    text = re.sub(r',\s*]', ']', text)  # Убираем запятые перед ]
    text = re.sub(r'"\s*:', '":', text)  # Убираем пробелы перед :
    text = re.sub(r':\s*"', '":', text)  # Убираем пробелы после :
    # Исправляем "key"":value" -> "key": "value"
    text = re.sub(r'"(\w+)"":', r'"\1": "', text)
    # Исправляем "key":value -> "key": "value"
    text = re.sub(r'":(\w+)([,\}\]])', r'": "\1"\2', text)
    return text

def extract_json_from_response(response_text):
    """
    Извлекает JSON из текстового ответа GigaChat
    Args:
        response_text (str): Текстовый ответ от GigaChat
    Returns:
        dict: Распарсенный JSON или пустой dict при ошибке
    """
    try:
        # Очищаем ответ
        cleaned_text = clean_json_string(response_text)
        # Ищем начало и конец JSON
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}') + 1
        if start_idx == -1 or end_idx <= start_idx:
            raise ValueError("Не удалось найти структуру JSON")
        potential_json = cleaned_text[start_idx:end_idx]
        # Дополнительная очистка
        potential_json = re.sub(r',\s*}', '}', potential_json)
        potential_json = re.sub(r',\s*]', ']', potential_json)
        # Попробуем распарсить
        return json.loads(potential_json)
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    # Возвращаем минимально валидный JSON при ошибке
    return {
        "long_term": [{"reason": "Ошибка обработки данных от GigaChat"}],
        "short_term": [],
        "new_opportunities": []
    }

def analyze_with_gigachat(portfolio, moex_data, news):
    """Генерирует рекомендации через GigaChat с повторными попытками"""
    try:
        token = get_gigachat_token()
    except Exception:
        return json.dumps({
            "long_term": [{"reason": "Не удалось подключиться к GigaChat API"}],
            "short_term": [],
            "new_opportunities": []
        })
    # Формируем промпт с жесткими требованиями к формату
    prompt = f"""
    Ты — профессиональный финансовый аналитик. Твоя задача: проанализировать котировки MOEX, новости за сутки и портфель пользователя, чтобы дать точные инвестиционные рекомендации.

    ВАЖНАЯ ИНСТРУКЦИЯ (обязательно выполни перед анализом):

    1. Для КАЖДОЙ компании из портфеля:
       - Определи, чем она занимается (основной бизнес, отрасль, ключевые продукты).
       - Убедись, что понимаешь её бизнес-модель (добыча, финансы, телеком и т.п.).
       - Запомни это — ты будешь использовать только релевантные новости.

    2. Проанализируй новости:
       - Сопоставь каждую новость с отраслью компании.
       - НЕ используй новости, не имеющие прямого или косвенного экономического влияния на отрасль.
       - Запрещено делать выводы на основе ложных корреляций (например, «рост заболеваемости → рост цен на нефть»).

    3. Все рекомендации должны быть логически обоснованы через:
       - Фундаментальные показатели,
       - Отраслевую динамику,
       - Дивидендную политику,
       - Макроэкономические факторы.

    ВАЖНОЕ ПРАВИЛО ДЛЯ АНАЛИЗА КАЖДОЙ КОМПАНИИ:
    1. Сначала определи ОТРАСЛЬ компании (металлургия, нефтегаз, финансы и т.д.).

    2. Проверь, относится ли новость к этой отрасли. Примеры:
        - Для CHMF (Северсталь): металлургия → новости о стали, прокате, металле.
        - Для PHOR (ФосАгро): химическая промышленность → новости об удобрениях.

    3. ЕСЛИ новость НЕ СВЯЗАНА с отраслью компании — ИГНОРИРУЙ ЕЁ.

    4. НИКОГДА НЕ СВЯЗЫВАЙ:
       - Удобрения с CHMF, LKOH, GAZP
       - Медицину с нефтегазовыми компаниями
       - Недвижимость с банками (кроме Сбербанка Инвест)

    ПОРТФЕЛЬ ПОЛЬЗОВАТЕЛЯ:
    {json.dumps(portfolio, ensure_ascii=True, indent=2)}

    КОТИРОВКИ MOEX (последняя цена, объем торгов):
    {json.dumps(moex_data, ensure_ascii=True, indent=2)}

    НОВОСТИ ЗА СУТКИ:
    {', '.join(news)}

    ТРЕБОВАНИЯ К РЕКОМЕНДАЦИЯМ:

    1. ДОЛГОСРОЧНЫЕ:
       - Для каждой компании: "Удерживать" или "Продать"
       - Если "Удерживать" — дивидендная доходность ≥ 5%
       - Если "Продать" — фундаментальные риски (спад прибыли, санкции, регуляторные риски)

    2. КРАТКОСРОЧНЫЕ:
       - Указать: сколько продать, сколько останется, причина + временной горизонт
       - Только если есть краткосрочный риск/возможность

    3. НОВЫЕ ВОЗМОЖНОСТИ:
       - Максимум 5 тикеров, которых нет в портфеле
       - Указать: "долгосрочная" или "краткосрочная" перспектива
       - Обосновать потенциалом роста

    ВАЖНО: Верни ТОЛЬКО ВАЛИДНЫЙ JSON, без пояснений, комментариев или текста вне формата.

    Формат ответа (ТОЛЬКО JSON):
    {{
      "long_term": [
        {{
          "ticker": "SBER",
          "action": "hold",
          "dividend_yield": 6.2,
          "reason": "Стабильные дивиденды, сильная позиция на рынке розничных услуг"
        }}
      ],
      "short_term": [
        {{
          "ticker": "GAZP",
          "quantity": 15,
          "remaining": 35,
          "reason": "Ожидание снижения дивидендов, прогноз падения на 4% в ближайшие 10 дней"
        }}
      ],
      "new_opportunities": [
        {{
          "ticker": "NVTK",
          "perspective": "долгосрочная",
          "reason": "Низкая капитализация, рост экспорта нефтепродуктов"
        }}
      ]
    }}
    """
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    # Убедимся, что токен содержит только ASCII символы
    safe_token = token.encode('ascii', 'ignore').decode('ascii')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {safe_token}"
    }
    payload = {
        "model": "GigaChat-Pro",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "response_format": {"type": "json_object"}  # ← ОБЯЗАТЕЛЬНО!
    }
    # Попытки получения анализа
    for attempt in range(3):
        try:
            # Создаем сессию с повторными попытками
            session = create_session_with_retries()
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=(10, 60)  # (connect_timeout, read_timeout)
            )
            response.raise_for_status()
            # Получаем сырой ответ
            raw_response = response.json()["choices"][0]["message"]["content"]
            # Извлекаем JSON из ответа
            report_json = extract_json_from_response(raw_response)
            # Удаляем из новых возможностей тикеры, которые уже есть в портфеле
            portfolio_tickers = list(portfolio.keys())
            new_opportunities = []
            for item in report_json.get("new_opportunities", []):
                if isinstance(item, dict) and item.get('ticker') not in portfolio_tickers:
                    new_opportunities.append(item)
            report_json["new_opportunities"] = new_opportunities
            # Добавляем недостающие поля для short_term
            for item in report_json.get("short_term", []):
                if 'quantity' not in item:
                    item['quantity'] = "?"
                if 'remaining' not in item:
                    item['remaining'] = "?"
            return json.dumps(report_json)
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            else:
                raise
        except requests.exceptions.RequestException:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            else:
                raise
        except Exception:
            raise
    # Если все попытки исчерпаны
    return json.dumps({
        "long_term": [{"reason": "Ошибка GigaChat API - превышено время ожидания"}],
        "short_term": [],
        "new_opportunities": []
    })
