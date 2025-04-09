"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import {
    MessageSquare,
    CheckCircle,
    XCircle,
    Info,
    X,
    AlertCircle,
    Send,
    Image as ImageIcon,
    LayoutGrid,
    Square
} from "lucide-react"

// API 도메인 상수
const API_URL = "/api/query_stream" // 프록시 서버 경로

// 인터페이스 정의
interface TokenResponse {
    type: string;
    content: string;
    answer?: string;
}

interface ReferenceResponse {
    type: string;
    content: string;
    source?: string;
}

interface ChatMessage {
    role: "user" | "assistant";
    content: string;
    timestamp: string;
    references?: ReferenceResponse[];
}

interface WelcomeScreenProps {
    currentRequestId: string | null;
    onChatStart?: () => void;
}

export default function WelcomeScreen({ currentRequestId, onChatStart }: WelcomeScreenProps) {
    const [message, setMessage] = useState("");
    const [isProcessing, setIsProcessing] = useState(false);
    const [useRag, setUseRag] = useState(true);
    const [uploadedImage, setUploadedImage] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [showImageDropdown, setShowImageDropdown] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [streamingResponse, setStreamingResponse] = useState<string>("");
    const [isStreaming, setIsStreaming] = useState(false);
    const [references, setReferences] = useState<ReferenceResponse[]>([]);
    const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);
    const isMountedRef = useRef(true); // 컴포넌트 마운트 상태 추적

    // 컴포넌트 마운트/언마운트 관리
    useEffect(() => {
        isMountedRef.current = true;
        return () => {
            isMountedRef.current = false;
            // 진행 중인 요청이 있으면 중단
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);

    // 채팅 컨테이너 스크롤 자동 조정
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory, streamingResponse]);

    // 텍스트 영역 자동 크기 조정
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [message]);

    // 이미지 업로드 처리
    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];

            // 이미지 파일 타입 확인
            if (!file.type.startsWith("image/")) {
                setError("이미지 파일만 업로드 가능합니다.");
                return;
            }

            // 파일 크기 제한 (10MB)
            if (file.size > 10 * 1024 * 1024) {
                setError("이미지 크기는 10MB 이하여야 합니다.");
                return;
            }

            setUploadedImage(file);
            setError(null);

            const reader = new FileReader();
            reader.onload = (evt) => {
                if (evt.target?.result) {
                    setImagePreview(evt.target.result as string);
                }
            };
            reader.onerror = () => {
                setError("이미지를 읽는 중 오류가 발생했습니다.");
            };
            reader.readAsDataURL(file);

            // 드롭다운 숨기기
            setShowImageDropdown(false);
        }
    };

    const removeImage = () => {
        setUploadedImage(null);
        setImagePreview(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const toggleRag = () => {
        setUseRag((prev) => !prev);
    };

    const fillExamplePrompt = (text: string) => {
        setMessage(text);
        if (textareaRef.current) {
            textareaRef.current.focus();
            // 텍스트 영역 크기 조정 트리거
            setTimeout(() => {
                if (textareaRef.current) {
                    textareaRef.current.style.height = "auto";
                    textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
                }
            }, 0);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (message.trim() || uploadedImage) {
                sendWelcomeMessage();
            }
        }
    };

    // 이미지를 Base64로 변환
    const readFileAsBase64 = (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                if (reader.result) {
                    const base64 = (reader.result as string).split(",")[1];
                    resolve(base64);
                } else {
                    reject(new Error("Failed to read file"));
                }
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    };

    // 메시지 전송 및 스트리밍 응답 처리
    const sendWelcomeMessage = async () => {
        if ((!message.trim() && !uploadedImage) || !currentRequestId) return;
        if (isProcessing) return; // 방어적 체크

        // 이전 요청이 있으면 중단
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        // 새 AbortController 생성
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        setIsProcessing(true);
        setError(null);
        setStreamingResponse("");
        setIsStreaming(true);
        setReferences([]);

        // 사용자 메시지를 채팅 기록에 추가
        const userMessage: ChatMessage = {
            role: "user",
            content: message,
            timestamp: new Date().toISOString(),
        };
        setChatHistory((prev) => [...prev, userMessage]);

        try {
            // 이미지 처리
            let imageBase64 = null;
            if (uploadedImage) {
                try {
                    imageBase64 = await readFileAsBase64(uploadedImage);
                } catch (err) {
                    console.error("이미지 처리 오류:", err);
                    if (isMountedRef.current) {
                        setError("이미지 처리 중 오류가 발생했습니다.");
                        setIsProcessing(false);
                        setIsStreaming(false);
                    }
                    return;
                }
            }

            // 요청 데이터 구성
            const payload = {
                qry_id: Date.now().toString() + "-" + Math.floor(Math.random() * 10000),
                user_id: "user123",
                page_id: currentRequestId,
                auth_class: "admin",
                qry_contents: message,
                qry_time: new Date().toISOString(),
                rag: useRag,
                ...(imageBase64 ? { image_data: imageBase64 } : {}),
            };

            // 폼 초기화
            setMessage("");
            if (textareaRef.current) {
                textareaRef.current.style.height = "auto";
            }
            removeImage();

            // API 요청
            console.log("API 요청 시작:", API_URL);
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal, // AbortController 시그널 연결
            });

            if (!response.ok) {
                throw new Error(`서버 응답 오류: ${response.status}`);
            }

            if (!response.body) {
                throw new Error("응답 본문을 읽을 수 없습니다.");
            }

            // 스트리밍 응답 처리
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";
            let partialAnswer = "";
            let currentReferences: ReferenceResponse[] = [];

            while (true) {
                // 컴포넌트가 언마운트되면 읽기 중단
                if (!isMountedRef.current) break;

                try {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;

                    // 라인 단위로 처리
                    const lines = buffer.split("\n");
                    buffer = lines.pop() || "";

                    for (const line of lines) {
                        if (!line) continue;
                        if (line.startsWith("data:")) {
                            const jsonStr = line.slice(5).trim();

                            // 스트림 종료 메시지
                            if (jsonStr === "[[STREAM_DONE]]") {
                                if (isMountedRef.current) {
                                    setIsStreaming(false);
                                    setIsProcessing(false);

                                    // 최종 응답을 채팅 기록에 추가
                                    const assistantMessage: ChatMessage = {
                                        role: "assistant",
                                        content: partialAnswer,
                                        timestamp: new Date().toISOString(),
                                        references: currentReferences.length > 0 ? currentReferences : undefined,
                                    };
                                    setChatHistory((prev) => [...prev, assistantMessage]);

                                    // 채팅 시작 콜백 호출 (있는 경우)
                                    if (onChatStart) {
                                        onChatStart();
                                    }
                                }
                                return;
                            }

                            try {
                                const sseData = JSON.parse(jsonStr);

                                if (isMountedRef.current) {
                                    if (sseData.type === "answer" || sseData.type === "token") {
                                        const content = sseData.answer || sseData.content || "";
                                        partialAnswer += content;
                                        setStreamingResponse(partialAnswer);
                                    } else if (sseData.type === "reference") {
                                        currentReferences.push(sseData);
                                        setReferences((prev) => [...prev, sseData]);
                                    }
                                }
                            } catch (err) {
                                // JSON 파싱 실패시 텍스트로 처리
                                if (isMountedRef.current) {
                                    partialAnswer += jsonStr;
                                    setStreamingResponse(partialAnswer);
                                }
                            }
                        }
                    }
                } catch (readError) {
                    if (signal.aborted) {
                        console.log("스트리밍 응답 읽기가 중단되었습니다.");
                        break;
                    }
                    console.error("스트리밍 읽기 오류:", readError);
                    if (isMountedRef.current) {
                        setError("응답 처리 중 오류가 발생했습니다.");
                    }
                    break;
                }
            }

            // 스트림이 정상적으로 종료되지 않았는데 응답이 있는 경우
            if (isMountedRef.current && isStreaming && partialAnswer) {
                setIsStreaming(false);
                setIsProcessing(false);

                // 최종 응답을 채팅 기록에 추가
                const assistantMessage: ChatMessage = {
                    role: "assistant",
                    content: partialAnswer,
                    timestamp: new Date().toISOString(),
                    references: currentReferences.length > 0 ? currentReferences : undefined,
                };
                setChatHistory((prev) => [...prev, assistantMessage]);

                // 채팅 시작 콜백 호출 (있는 경우)
                if (onChatStart) {
                    onChatStart();
                }
            }
        } catch (error) {
            console.error("메시지 전송 오류:", error);

            // 요청이 취소된 경우 오류 메시지 표시하지 않음
            if (signal.aborted) {
                console.log("사용자에 의해 요청이 취소되었습니다.");
            } else if (isMountedRef.current) {
                setError(
                    `오류가 발생했습니다: ${error instanceof Error ? error.message : "알 수 없는 오류"
                    }`
                );
            }
        } finally {
            if (isMountedRef.current) {
                setIsProcessing(false);
                setIsStreaming(false);
            }
        }
    };

    // stopGeneration: 중단 요청
    const stopGeneration = async () => {
        if (!isProcessing || !currentRequestId) return;

        console.log("중단 요청 시작");
        try {
            // 진행 중인 요청 중단
            if (abortControllerRef.current) {
                console.log("AbortController 중단");
                abortControllerRef.current.abort();
                abortControllerRef.current = null;
            }

            // 서버에 중단 요청 전송 (선택적)
            try {
                const response = await fetch("/api/stop_generation", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ request_id: currentRequestId }),
                });

                if (!response.ok) {
                    console.error("중단 요청 실패:", response.status);
                } else {
                    console.log("중단 요청 성공적으로 전송됨");
                    const responseData = await response.json();
                    console.log("중단 응답:", responseData);
                }
            } catch (fetchError) {
                console.error("중단 요청 전송 오류:", fetchError);
            }

            // 스트리밍 중인 메시지가 있으면 중단 메시지 추가
            if (streamingResponse) {
                const finalResponse =
                    streamingResponse + "\n\n*사용자에 의해 생성이 중단되었습니다.*";
                setStreamingResponse(finalResponse);

                // 채팅 기록에 중단된 응답 추가
                const assistantMessage: ChatMessage = {
                    role: "assistant",
                    content: finalResponse,
                    timestamp: new Date().toISOString(),
                    references: references.length > 0 ? references : undefined,
                };
                setChatHistory((prev) => [...prev, assistantMessage]);

                // 채팅 시작 콜백 호출 (있는 경우)
                if (onChatStart) {
                    onChatStart();
                }
            }

            // 상태 초기화
            if (isMountedRef.current) {
                setIsProcessing(false);
                setIsStreaming(false);
            }
        } catch (error) {
            console.error("중단 처리 중 오류:", error);
            if (isMountedRef.current) {
                setIsProcessing(false);
                setIsStreaming(false);
            }
        }
    };

    // 이미지 드롭다운 토글
    const toggleImageDropdown = () => {
        setShowImageDropdown((prev) => !prev);
    };

    // 클릭 이벤트 처리 - 이미지 드롭다운 외부 클릭 감지
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            const dropdown = document.getElementById("welcomeImageDropdown");
            if (
                dropdown &&
                !dropdown.contains(e.target as Node) &&
                e.target !== document.getElementById("welcomeImageButton")
            ) {
                setShowImageDropdown(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    return (
        <div className="flex flex-col items-center justify-start p-8 mx-auto max-w-[900px] h-full">
            <div className="flex flex-col items-center mb-10 text-center">
                <img
                    src="/NS_LOGO_ONLY.svg?height=80&width=80"
                    alt="FLUX_NS Logo"
                    className="w-20 h-20 mb-4 filter drop-shadow-md"
                />
                <h1 className="text-4xl font-bold text-[#111827] mb-2">
                    Welcome to FLUX_NS
                </h1>
                <p className="text-lg text-[#6b7280] mt-2">
                    Ask me anything about NS information
                </p>
            </div>

            {/* 채팅 내역 표시 영역 */}
            {chatHistory.length > 0 && (
                <div
                    ref={chatContainerRef}
                    className="w-full max-w-[768px] mb-6 p-4 bg-white border border-[#e5e7eb] rounded-md shadow-sm overflow-y-auto max-h-[400px]"
                >
                    <div className="space-y-4">
                        {chatHistory.map((msg, index) => (
                            <div
                                key={index}
                                className={`p-3 rounded-md ${msg.role === "user"
                                        ? "bg-[#dcf8f6] ml-8"
                                        : "bg-white mr-8"
                                    } border border-[#e5e7eb]`}
                            >
                                <div className="font-medium mb-1 flex items-center">
                                    {msg.role === "user" ? (
                                        <>
                                            <span className="inline-block mr-2">
                                                <svg
                                                    xmlns="http://www.w3.org/2000/svg"
                                                    width="16"
                                                    height="16"
                                                    viewBox="0 0 24 24"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    strokeWidth="2"
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                >
                                                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                                                    <circle cx="12" cy="7" r="4"></circle>
                                                </svg>
                                            </span>
                                            사용자
                                        </>
                                    ) : (
                                        <>
                                            <span className="inline-block mr-2">
                                                <svg
                                                    xmlns="http://www.w3.org/2000/svg"
                                                    width="16"
                                                    height="16"
                                                    viewBox="0 0 24 24"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    strokeWidth="2"
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                >
                                                    <rect
                                                        x="3"
                                                        y="3"
                                                        width="18"
                                                        height="18"
                                                        rx="2"
                                                        ry="2"
                                                    ></rect>
                                                    <line x1="3" y1="9" x2="21" y2="9"></line>
                                                    <line x1="9" y1="21" x2="9" y2="9"></line>
                                                </svg>
                                            </span>
                                            AI
                                        </>
                                    )}
                                </div>
                                <div className="text-sm text-[#374151] whitespace-pre-wrap">
                                    {msg.content}
                                </div>

                                {/* 참조 정보 표시 */}
                                {msg.references && msg.references.length > 0 && (
                                    <div className="mt-2 pt-2 border-t border-[#e5e7eb]">
                                        <h4 className="text-xs font-semibold text-[#111827] mb-1">
                                            참조 정보
                                        </h4>
                                        <ul className="space-y-1">
                                            {msg.references.map((ref, refIndex) => (
                                                <li key={refIndex} className="text-xs text-[#6b7280]">
                                                    {ref.source && (
                                                        <span className="font-medium">{ref.source}: </span>
                                                    )}
                                                    {ref.content}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        ))}

                        {/* 스트리밍 응답 표시 */}
                        {isStreaming && streamingResponse && (
                            <div className="p-3 rounded-md bg-white mr-8 border border-[#e5e7eb]">
                                <div className="font-medium mb-1 flex items-center">
                                    <span className="inline-block mr-2">
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            width="16"
                                            height="16"
                                            viewBox="0 0 24 24"
                                            fill="none"
                                            stroke="currentColor"
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                        >
                                            <rect
                                                x="3"
                                                y="3"
                                                width="18"
                                                height="18"
                                                rx="2"
                                                ry="2"
                                            ></rect>
                                            <line x1="3" y1="9" x2="21" y2="9"></line>
                                            <line x1="9" y1="21" x2="9" y2="9"></line>
                                        </svg>
                                    </span>
                                    AI
                                    <div className="typing-indicator ml-2">
                                        <span></span>
                                        <span></span>
                                        <span></span>
                                    </div>
                                </div>
                                <div className="text-sm text-[#374151] whitespace-pre-wrap">
                                    {streamingResponse}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* 오류 메시지 표시 */}
            {error && (
                <div className="w-full max-w-[768px] mb-6 p-4 bg-[#fee2e2] border border-[#fca5a5] rounded-md shadow-sm">
                    <div className="flex items-center text-[#b91c1c] font-medium">
                        <AlertCircle className="w-5 h-5 mr-2" />
                        {error}
                    </div>
                </div>
            )}

            {/* 예시 질문 및 기능 설명 */}
            {chatHistory.length === 0 && (
                <div className="grid grid-cols-1 gap-8 w-full mb-8">
                    <div className="flex flex-col">
                        <h2 className="text-xl font-semibold mb-4 text-[#111827]">
                            Examples
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            <button
                                className="flex items-start gap-3 bg-white border border-[#e5e7eb] rounded-md p-4 cursor-pointer text-left transition-all duration-300 ease-in-out text-[#111827] text-sm shadow-sm hover:border-[#10a37f] hover:shadow-md hover:-translate-y-0.5"
                                onClick={() =>
                                    fillExamplePrompt("디지털 기획팀의 우선과제가 어떻게 돼?")
                                }
                            >
                                <MessageSquare className="w-4 h-4 text-[#10a37f] mt-0.5 flex-shrink-0" />
                                <span>디지털 기획팀의 우선과제가 어떻게 돼?</span>
                            </button>

                            <button
                                className="flex items-start gap-3 bg-white border border-[#e5e7eb] rounded-md p-4 cursor-pointer text-left transition-all duration-300 ease-in-out text-[#111827] text-sm shadow-sm hover:border-[#10a37f] hover:shadow-md hover:-translate-y-0.5"
                                onClick={() =>
                                    fillExamplePrompt(
                                        "UN 클래스 2.1, UNNO 1033인 DG 화물이 부산항에서 고베항에 선적 가능한 지 알려줘."
                                    )
                                }
                            >
                                <MessageSquare className="w-4 h-4 text-[#10a37f] mt-0.5 flex-shrink-0" />
                                <span>
                                    UN 클래스 2.1, UNNO 1033인 DG 화물이
                                    <br />
                                    부산항에서 고베항에 선적 가능한지?
                                </span>
                            </button>

                            <button
                                className="flex items-start gap-3 bg-white border border-[#e5e7eb] rounded-md p-4 cursor-pointer text-left transition-all duration-300 ease-in-out text-[#111827] text-sm shadow-sm hover:border-[#10a37f] hover:shadow-md hover:-translate-y-0.5"
                                onClick={() =>
                                    fillExamplePrompt("IOT 컨테이너 사업 근황에 대해서 알려줘.")
                                }
                            >
                                <MessageSquare className="w-4 h-4 text-[#10a37f] mt-0.5 flex-shrink-0" />
                                <span>IOT 컨테이너 사업 근황에 대해서 알려줘</span>
                            </button>

                            <button
                                className="flex items-start gap-3 bg-white border border-[#e5e7eb] rounded-md p-4 cursor-pointer text-left transition-all duration-300 ease-in-out text-[#111827] text-sm shadow-sm hover:border-[#10a37f] hover:shadow-md hover:-translate-y-0.5"
                                onClick={() =>
                                    fillExamplePrompt(
                                        "최근 남성해운의 운임 동향을 웹 정보와 함께 분석해서 알려줘."
                                    )
                                }
                            >
                                <MessageSquare className="w-4 h-4 text-[#10a37f] mt-0.5 flex-shrink-0" />
                                <span>
                                    최근 남성해운의 운임 동향을
                                    <br />
                                    웹 정보와 함께 분석해서 알려줘
                                </span>
                            </button>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full">
                        <div className="flex flex-col">
                            <h2 className="text-xl font-semibold mb-4 text-[#111827]">
                                Capabilities
                            </h2>
                            <ul className="flex flex-col gap-3 list-none">
                                <li className="flex items-start gap-2 text-sm bg-white p-3 rounded-md border border-[#e5e7eb] shadow-sm">
                                    <CheckCircle className="w-5 h-5 text-[#10a37f] flex-shrink-0 mt-0.5" />
                                    <span>최신 정보와 데이터 기반 응답</span>
                                </li>
                                <li className="flex items-start gap-2 text-sm bg-white p-3 rounded-md border border-[#e5e7eb] shadow-sm">
                                    <CheckCircle className="w-5 h-5 text-[#10a37f] flex-shrink-0 mt-0.5" />
                                    <span>남성해운의 주간회의, 계약서, 인사규범 등에 대한 답변</span>
                                </li>
                                <li className="flex items-start gap-2 text-sm bg-white p-3 rounded-md border border-[#e5e7eb] shadow-sm">
                                    <CheckCircle className="w-5 h-5 text-[#10a37f] flex-shrink-0 mt-0.5" />
                                    <span>상세한 답변 및 해설</span>
                                </li>
                            </ul>
                        </div>

                        <div className="flex flex-col">
                            <h2 className="text-xl font-semibold mb-4 text-[#111827]">
                                Limitations
                            </h2>
                            <ul className="flex flex-col gap-3 list-none">
                                <li className="flex items-start gap-2 text-sm bg-white p-3 rounded-md border border-[#e5e7eb] shadow-sm">
                                    <XCircle className="w-5 h-5 text-[#ef4444] flex-shrink-0 mt-0.5" />
                                    <span>현재 모든 정보를 가지고 있지 아니함</span>
                                </li>
                                <li className="flex items-start gap-2 text-sm bg-white p-3 rounded-md border border-[#e5e7eb] shadow-sm">
                                    <XCircle className="w-5 h-5 text-[#ef4444] flex-shrink-0 mt-0.5" />
                                    <span>가끔 정확하지 않은 정보를 제공할 수 있음</span>
                                </li>
                                <li className="flex items-start gap-2 text-sm bg-white p-3 rounded-md border border-[#e5e7eb] shadow-sm">
                                    <XCircle className="w-5 h-5 text-[#ef4444] flex-shrink-0 mt-0.5" />
                                    <span>민감한 요청에 대한 응답이 제한될 수 있음</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            )}

            {/* 메시지 입력 영역 */}
            <div className="w-full max-w-[768px] mt-auto flex flex-col gap-3">
                <div className="flex items-center gap-2 text-[#6b7280] text-sm mb-2">
                    <Info className="w-5 h-5" />
                    <span>질문을 입력하거나 이미지도 업로드 가능</span>
                </div>

                <div className="flex items-center bg-white border border-[#e5e7eb] rounded-[0.75rem] p-3 shadow-md w-full max-w-[768px] mx-auto">
                    <div className="w-full">
                        <textarea
                            ref={textareaRef}
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            onKeyDown={handleKeyDown}
                            className="w-full flex-1 border-none resize-none text-base p-2 outline-none bg-transparent"
                            placeholder="메시지를 입력하세요..."
                            rows={1}
                            disabled={isProcessing}
                        />

                        <div className="flex items-center">
                            <div className="relative">
                                <input
                                    type="file"
                                    id="welcomeImageUpload"
                                    accept="image/*"
                                    className="hidden"
                                    onChange={handleImageChange}
                                    ref={fileInputRef}
                                    disabled={isProcessing}
                                />
                                <button
                                    id="welcomeImageButton"
                                    className={`bg-transparent border-none rounded-[12%] cursor-pointer mt-2 p-1 flex items-center justify-center ${uploadedImage ? "bg-[#0e8e70] text-white" : ""
                                        } ${isProcessing ? "opacity-50 cursor-not-allowed" : ""}`}
                                    onClick={toggleImageDropdown}
                                    disabled={isProcessing}
                                >
                                    <ImageIcon className="w-5 h-5 text-[#10a37f]" />
                                    <span className="ml-1">이미지</span>
                                </button>

                                {showImageDropdown && (
                                    <div
                                        id="welcomeImageDropdown"
                                        className="absolute top-[-350%] left-0 bg-white border border-[#e5e7eb] shadow-md rounded-md p-2 z-10 min-w-[150px]"
                                    >
                                        <div className="text-sm font-semibold mb-2">
                                            이미지 업로드
                                        </div>
                                        <button
                                            className="w-full py-2 px-2 bg-[#10a37f] text-white border-none rounded-sm cursor-pointer mb-2"
                                            onClick={() => fileInputRef.current?.click()}
                                        >
                                            업로드
                                        </button>
                                        <div className="max-w-[600px] flex-wrap overflow-x-auto flex flex-row border-t border-[#e5e7eb] pt-2">
                                            <p className="text-xs text-[#6b7280] text-center">
                                                업로드된 이미지가 없습니다.
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <button
                                className={`bg-transparent border-none rounded-[12%] cursor-pointer mt-2 p-1 flex items-center justify-center ml-2 ${isProcessing ? "opacity-50 cursor-not-allowed" : ""
                                    }`}
                                onClick={toggleRag}
                                disabled={isProcessing}
                            >
                                <LayoutGrid className="w-5 h-5 text-[#10a37f]" />
                                <span className="ml-1">RAG: {useRag ? "On" : "Off"}</span>
                            </button>
                        </div>
                    </div>

                    {/* If isProcessing === true, show Stop button; otherwise Send button */}
                    {isProcessing ? (
                        <button
                            className="bg-[#f44336] border-none text-white p-2 rounded-full cursor-pointer transition-all duration-300 ease-in-out ml-3 w-10 h-10 flex items-center justify-center hover:bg-[#d32f2f]"
                            onClick={stopGeneration}
                        // Make sure it's never disabled while streaming
                        >
                            <Square className="w-5 h-5" />
                        </button>
                    ) : (
                        <button
                            className={`bg-[#10a37f] border-none text-white p-2 rounded-full cursor-pointer transition-all duration-300 ease-in-out ml-3 w-10 h-10 flex items-center justify-center hover:bg-[#0e8e70] hover:scale-105`}
                            onClick={sendWelcomeMessage}
                            disabled={!message.trim() && !uploadedImage}
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    )}
                </div>

                {/* 이미지 미리보기 */}
                {imagePreview && (
                    <div className="max-w-[600px] mt-2 flex flex-row flex-wrap gap-2 rounded-md overflow-auto bg-white border border-[#e5e7eb] shadow-sm">
                        <div className="relative w-[65px] h-[65px] m-2 border border-[#ddd] overflow-hidden rounded-md">
                            <img
                                src={imagePreview}
                                alt="Preview"
                                className="w-full h-full object-cover"
                            />
                            <button
                                className="absolute top-[5px] right-[5px] bg-[rgba(0,0,0,0.5)] border-none text-white rounded-full cursor-pointer p-[5px] flex items-center justify-center hover:bg-[rgba(0,0,0,0.8)] hover:scale-110"
                                onClick={removeImage}
                                disabled={isProcessing}
                            >
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                        <div className="p-2 text-xs text-[#6b7280] bg-[#f7f7f8] border-t border-[#e5e7eb] whitespace-nowrap overflow-hidden text-ellipsis w-full">
                            {uploadedImage?.name}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
