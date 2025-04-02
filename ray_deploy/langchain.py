# ray_deploy/langchain.py

# 랭체인 도입
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# =============================================================================
# Custom Conversation Memory to store extra metadata (e.g., chunk_ids)
# =============================================================================
class CustomConversationBufferMemory(ConversationBufferMemory):
    """대화 저장 시 추가 메타데이터를 함께 기록"""
    def save_context(self, inputs: dict, outputs: dict) -> None:
        """
        inputs, outputs 예시:
            inputs = {
                "qry_contents": "사용자 질문",
                "qry_id": "...",
                "user_id": "...",
                "auth_class": "...",
                "qry_time": "..."
            }
            outputs = {
                "output": "AI의 최종 답변",
                "chunk_ids": [...참조 chunk_id 리스트...]
            }
        """
        try:
            user_content = inputs.get("qry_contents", "")
            human_msg = HumanMessage(
                content=user_content,
                additional_kwargs={
                    "qry_id": inputs.get("qry_id"),
                    "user_id": inputs.get("user_id"),
                    "auth_class": inputs.get("auth_class"),
                    "qry_time": inputs.get("qry_time")
                }
            )
            ai_content = outputs.get("output", "")
            ai_msg = AIMessage(
                content=ai_content,
                additional_kwargs={
                    "chunk_ids": outputs.get("chunk_ids", []),
                    "qry_id": inputs.get("qry_id"),
                    "user_id": inputs.get("user_id"),
                    "auth_class": inputs.get("auth_class"),
                    "qry_time": inputs.get("qry_time")
                }
            )

            self.chat_memory.messages.append(human_msg)
            self.chat_memory.messages.append(ai_msg)
        except Exception as e:
            print(f"[ERROR in save_context] {e}")
        
    def load_memory_variables(self, inputs: dict) -> dict:
        """
        랭체인 규약에 의해 {"history": [메시지 리스트]} 형태 리턴
        """
        try:
            return {"history": self.chat_memory.messages}
        except Exception as e:
            print(f"[ERROR in load_memory_variables] {e}")
            return {"history": []}

# Serialization function for messages
def serialize_message(msg):
    """
    HumanMessage -> {"role": "human", "content": ...}
    AIMessage    -> {"role": "ai", "content": ..., "references": [...]}
    """
    try:
        if isinstance(msg, HumanMessage):
            return {"role": "human", "content": msg.content}
        elif isinstance(msg, AIMessage):
            refs = msg.additional_kwargs.get("chunk_ids", [])
            # 디버그 출력
            print(f"[DEBUG serialize_message] AI refs: {refs}")
            return {"role": "ai", "content": msg.content, "references": refs}
        else:
            return {
                "role": "unknown",
                "content": getattr(msg, "content", str(msg))
            }
    except Exception as e:
        print(f"[ERROR in serialize_message] {e}")
        return {"role": "error", "content": str(e)}
    
# =============================================================================
# =============================================================================
