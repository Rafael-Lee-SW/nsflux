# logger_config.py
import logging
import sys
from colorama import init, Fore, Style

# colorama 초기화
init(autoreset=True)

# 로거 생성
logger = logging.getLogger("RAG_Logger")
logger.setLevel(logging.DEBUG)  # 전체 레벨은 DEBUG 이상 모두 출력

# 콘솔 핸들러 생성
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # 콘솔에는 DEBUG 이상 모두 출력

# 포매터에 색상 적용: 레벨별로 다른 색상
class ColorfulFormatter(logging.Formatter):
    def format(self, record):
        level_color = {
            logging.DEBUG: Fore.CYAN,
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.RED + Style.BRIGHT
        }.get(record.levelno, Fore.WHITE)

        # 로그 메시지
        log_msg = super().format(record)
        return f"{level_color}{log_msg}{Style.RESET_ALL}"

# 포매터 설정
formatter = ColorfulFormatter("[%(levelname)s | %(asctime)s | %(name)s] %(message)s", 
                              datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)

# 로거에 핸들러 추가
logger.addHandler(console_handler)

# 필요하다면 파일 핸들러도 동일하게 추가 가능
# file_handler = logging.FileHandler("app.log", encoding="utf-8")
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# 모듈을 불러오는 쪽에서는
# from logger_config import logger
# 를 통해 logger 객체를 사용하면 됨.
