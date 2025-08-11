import psycopg2
import os
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

def get_db_connection():
    """Создает подключение к базе данных"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

def create_tables():
    """Создает таблицы в базе данных"""
    conn = None
    try:
        # Подключаемся к базе данных
        conn = get_db_connection()
        cur = conn.cursor()    
        # Создаем таблицу пользователей
        logger.info("Создание таблицы users...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                telegram_id BIGINT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_blocked BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                last_login TIMESTAMP
            )
        """)
        # Создаем таблицу портфелей
        logger.info("Создание таблицы portfolio...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                ticker VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                avg_price NUMERIC DEFAULT 0.0,
                PRIMARY KEY (user_id, ticker)
            )
        """)
        # Создаем индексы для улучшения производительности
        logger.info("Создание индексов...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_telegram_id 
            ON users(telegram_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_portfolio_user_id 
            ON portfolio(user_id)
        """)
        # Фиксируем изменения
        conn.commit()
        logger.info("Все таблицы успешно созданы!")
        # Проверяем структуру таблиц
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'users' 
            ORDER BY ordinal_position
        """)
        users_columns = cur.fetchall()
        logger.info("Структура таблицы users:")
        for col_name, col_type in users_columns:
            logger.info(f"  - {col_name}: {col_type}")
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'portfolio' 
            ORDER BY ordinal_position
        """)
        portfolio_columns = cur.fetchall()
        logger.info("Структура таблицы portfolio:")
        for col_name, col_type in portfolio_columns:
            logger.info(f"  - {col_name}: {col_type}")
    except Exception as e:
        logger.error(f"Ошибка при создании таблиц: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def main():
    """Основная функция"""
    logger.info("Начало инициализации базы данных...")
    try:
        # Создаем таблицы
        create_tables()
        logger.info("Инициализация базы данных завершена успешно!")
        logger.info("Теперь вы можете запустить бота командой: python bot.py")
    except Exception as e:
        logger.error(f"Критическая ошибка при инициализации: {str(e)}")
        raise
      
if __name__ == "__main__":
    main()
