# ray_deploy/sse_manager
import ray
import asyncio

# Ray의 요청을 비동기적으로 관리하기 위해 도입하는 큐-매니저
@ray.remote
class SSEQueueManager:
    def __init__(self):
        self.active_queues = {}
        self.lock = asyncio.Lock()

    async def create_queue(self, request_id):
        async with self.lock:
            self.active_queues[request_id] = asyncio.Queue()
            return True

    async def get_queue(self, request_id):
        return self.active_queues.get(request_id)

    async def get_token(self, request_id, timeout: float):
        queue = self.active_queues.get(request_id)
        if queue:
            try:
                token = await asyncio.wait_for(queue.get(), timeout=timeout)
                return token
            except asyncio.TimeoutError:
                return None
        return None

    async def put_token(self, request_id, token):
        async with self.lock:
            if request_id in self.active_queues:
                await self.active_queues[request_id].put(token)
                return True
            return False

    async def delete_queue(self, request_id):
        async with self.lock:
            if request_id in self.active_queues:
                del self.active_queues[request_id]
                return True
            return False