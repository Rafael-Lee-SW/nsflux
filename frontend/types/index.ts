// 채팅 메시지 타입
export interface ChatMessage {
    role: "human" | "ai" | "system";
    content: string;
    timestamp: string;
}

// 채팅방 타입
export interface ChatRoom {
    requestId: string;
    title: string;
    messages: ChatMessage[];
    createdAt: string;
}

// 참조 정보 타입
export interface ReferenceResponse {
    type: string;
    content: string;
    source?: string;
}

// 채팅방 데이터 타입
export interface ChatRoomData {
    uploadedImages: string[];
    referenceList: ReferenceResponse[];
    isStreaming: boolean;
}