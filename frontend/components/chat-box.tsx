"use client"

import { useEffect, useRef } from "react"
import { User, MessageSquare, Info, ExternalLink } from "lucide-react"
import type { ChatRoom } from "@/types"
import WelcomeScreen from "./welcome-screen"
import { parseMarkdown } from "@/lib/markdown-parser"

// 디버깅을 위한 로그 함수 추가
const logDebug = (message: string, data?: any) => {
    console.log(`[DEBUG] ${message}`, data ? data : '');
};

interface ChatBoxProps {
    currentRoom: ChatRoom | null
    loadReferenceForMessage: (msgIndex: number) => void
    currentRequestId: string | null
}

export default function ChatBox({ currentRoom, loadReferenceForMessage, currentRequestId }: ChatBoxProps) {
    const chatBoxRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight
            logDebug("채팅 박스 스크롤 위치 업데이트", {
                scrollTop: chatBoxRef.current.scrollTop,
                scrollHeight: chatBoxRef.current.scrollHeight
            });
        }
    }, [currentRoom?.messages])

    useEffect(() => {
        logDebug("채팅방 상태 변경", {
            currentRoom: currentRoom ? {
                requestId: currentRoom.requestId,
                messageCount: currentRoom.messages?.length || 0
            } : null,
            currentRequestId
        });
    }, [currentRoom, currentRequestId]);

    if (!currentRoom || !currentRoom.messages || currentRoom.messages.length === 0) {
        logDebug("채팅방이 비어있어 WelcomeScreen 표시");
        return (
            <div ref={chatBoxRef} className="flex-1 overflow-y-auto p-6 scroll-smooth bg-[#f7f7f8]">
                <WelcomeScreen currentRequestId={currentRequestId} />
            </div>
        )
    }

    return (
        <div ref={chatBoxRef} className="flex-1 overflow-y-auto p-6 scroll-smooth bg-[#f7f7f8]">
            {currentRoom.messages.map((msg, i) => {
                const sender = msg.role === "human" ? "사용자" : msg.role === "ai" ? "AI" : "시스템"
                const isUser = sender === "사용자"
                const isAI = sender === "AI"
                const isSystem = sender === "시스템"

                logDebug("메시지 렌더링", {
                    index: i,
                    role: msg.role,
                    sender,
                    contentLength: msg.content?.length || 0,
                    hasImage: msg.content?.includes("IMAGE_DATA:") || false
                });

                // Check if message contains image data
                const hasImageData = msg.content && msg.content.includes("IMAGE_DATA:")
                let cleanMessage = msg.content
                let imageContent = null

                if (hasImageData) {
                    const parts = msg.content.split("IMAGE_DATA:")
                    cleanMessage = parts[0].trim()
                    const imageBase64 = parts[1].trim()
                    
                    logDebug("이미지 데이터 처리", {
                        messageIndex: i,
                        cleanMessageLength: cleanMessage.length,
                        imageBase64Length: imageBase64.length
                    });

                    imageContent = (
                        <div className="mt-3 rounded-md overflow-hidden border border-[#e5e7eb] bg-white max-w-[300px]">
                            <img
                                src={`data:image/jpeg;base64,${imageBase64}`}
                                alt="Uploaded image"
                                className="w-full max-h-[200px] object-contain block"
                            />
                        </div>
                    )
                }

                return (
                    <div
                        key={i}
                        className={`max-w-[90%] mb-6 leading-relaxed animate-fadeIn ${isUser
                                ? "ml-auto bg-[#dcf8f6] text-[#111827] p-4 rounded-[0.75rem_0.75rem_0_0.75rem] shadow-sm border border-[rgba(16,163,127,0.2)]"
                                : isAI
                                    ? "bg-white text-[#111827] p-4 rounded-[0_0.75rem_0.75rem_0.75rem] shadow-sm border border-[#e5e7eb]"
                                    : "bg-[#fee2e2] text-[#b91c1c] p-4 rounded-[0.75rem] mx-auto text-center max-w-[80%] border border-[#fca5a5]"
                            }`}
                    >
                        <div className="flex items-center mb-2 font-semibold text-sm">
                            {isUser ? (
                                <User className="w-4 h-4 mr-2" />
                            ) : isAI ? (
                                <MessageSquare className="w-4 h-4 mr-2" />
                            ) : (
                                <Info className="w-4 h-4 mr-2" />
                            )}
                            {sender}

                            {isAI && (
                                <button
                                    className="flex items-center gap-1 ml-auto py-1 px-2 bg-transparent border border-[#e5e7eb] rounded-sm text-xs text-[#6b7280] cursor-pointer transition-all duration-300 ease-in-out hover:bg-[#e6f7f4] hover:border-[#10a37f] hover:text-[#10a37f]"
                                    onClick={() => {
                                        logDebug("참조 정보 로드 버튼 클릭", { messageIndex: i });
                                        loadReferenceForMessage(i);
                                    }}
                                >
                                    <ExternalLink className="w-4 h-4 text-[#10a37f]" />
                                    <span className="hidden sm:inline">References</span>
                                </button>
                            )}
                        </div>

                        <div className="text-[0.9375rem] break-words">
                            {isAI ? (
                                <div dangerouslySetInnerHTML={{ __html: parseMarkdown(cleanMessage) }} />
                            ) : (
                                <div>{cleanMessage}</div>
                            )}
                            {imageContent}
                        </div>
                    </div>
                )
            })}
        </div>
    )
}

