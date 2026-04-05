import requests
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.utils.logging_utils import get_logger
from src.utils.retry import retry_api

logger = get_logger(__name__)


class TelegramClient:
    def __init__(self):
        self.base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    @retry_api(max_retries=3, delay=2.0)
    def send_message(self, text: str, parse_mode='HTML'):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials missing. Not sending message.")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.debug("Successfully sent Telegram message.")
        return True
