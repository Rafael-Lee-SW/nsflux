# ray_deploy/ray_setup.py
import ray
from ray import serve
import logging

########## Starting Banner ############
from colorama import init, Fore, Style
init(autoreset=True)

# MetricsCollector 액터 가져오기
from utils.log_system import MetricsCollectorActor

BANNER = Fore.GREEN + r"""
'########:'##:::::::'##::::'##:'##::::'##::::::::::'##::: ##::'######::
 ##.....:: ##::::::: ##:::: ##:. ##::'##::::::::::: ###:: ##:'##... ##:
 ##::::::: ##::::::: ##:::: ##::. ##'##:::::::::::: ####: ##: ##:::..::
 ######::: ##::::::: ##:::: ##:::. ###::::::::::::: ## ## ##:. ######::
 ##...:::: ##::::::: ##:::: ##::: ## ##:::::::::::: ##. ####::..... ##:
 ##::::::: ##::::::: ##:::: ##:: ##:. ##::::::::::: ##:. ###:'##::: ##:
 ##::::::: ########:. #######:: ##:::. ##:'#######: ##::. ##:. ######::
..::::::::........:::.......:::..:::::..::.......::..::::..:::......:::
"""

def init_ray():
    print(BANNER)
    # Ray-Dashboard - GPU 상태, 사용 통계 등을 제공하는 모니터링 툴, host 0.0.0.0로 외부 접속을 허용하고, Default 포트인 8265으로 설정
    ray.init(
        include_dashboard=True,
        dashboard_host="0.0.0.0", # External IP accessable
        # dashboard_port=8265
        ignore_reinit_error=True,
        runtime_env={"py_modules": ["utils", "core"]},
        logging_level=logging.INFO,
    )
    
    # MetricsCollector 액터 시작 (전역 지표 수집용)
    try:
        # 이미 존재하는지 확인
        try:
            metrics_collector = ray.get_actor("MetricsCollector")
            print("기존 MetricsCollector 액터를 재사용합니다.")
        except ValueError:
            # 존재하지 않으면 새로 생성
            metrics_collector = MetricsCollectorActor.options(name="MetricsCollector").remote()
            print("MetricsCollector 액터를 생성했습니다.")
    except Exception as e:
        print(f"MetricsCollector 액터 생성 중 오류 발생: {e}")
        print("전역 지표 수집이 비활성화됩니다.")
    
    print("Ray initialized. DashBoard running at http://192.222.54.254:8265") # New Server(2xH100)
    
    return True