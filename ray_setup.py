# ray_setup.py

import ray

def init_ray():
    """
    Ray-Dashboard - GPU 상태, 사용 통계 등을 제공하는 모니터링 툴, host 0.0.0.0로 외부 접속을 허용하고, Default 포트인 8265으로 설정
    """
    
    ray.init(
        include_dashboard=True,
        dashboard_host="0.0.0.0" # 외부 IP 접속 가능
        # dashboard_port=8265
    )
    print("Ray initialized. DashBoard running at http://209.20.158.139:6460")