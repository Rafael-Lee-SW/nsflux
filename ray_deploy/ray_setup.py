# ray_deploy/ray_setup.py
import ray
from ray import serve

########## Starting Banner ############
from colorama import init, Fore, Style
init(autoreset=True)

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
        dashboard_host="0.0.0.0" # External IP accessable
        # dashboard_port=8265
    )
    print("Ray initialized. DashBoard running at http://192.222.54.254:8265") # New Server(2xH100)