"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Image, LayoutGrid, Send, X, Square } from "lucide-react"
import type { ChatRoomData, ReferenceResponse } from "@/types"

// Interface definitions (fill in your own or import them)
interface InputContainerProps {
    currentRequestId: string | null;
    isProcessing: boolean;
    setIsProcessing: (isProcessing: boolean) => void;
    storeMessageInRoom: (requestId: string, sender: string, text: string) => void;
    addReferenceData: (refData: any) => void;
    useRag: boolean;
    toggleRag: () => void;
    setChatRoomsData: React.Dispatch<React.SetStateAction<Record<string, ChatRoomData>>>;
}

export default function InputContainer({
    currentRequestId,
    isProcessing,
    setIsProcessing,
    storeMessageInRoom,
    addReferenceData,
    useRag,
    toggleRag,
    setChatRoomsData,
}: InputContainerProps) {
    const [message, setMessage] = useState("");
    const [uploadedImage, setUploadedImage] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [showImageDropdown, setShowImageDropdown] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamingContent, setStreamingContent] = useState("");
    const [isStopClicked, setIsStopClicked] = useState(false);

    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const imageDropdownRef = useRef<HTMLDivElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);
    const isMountedRef = useRef(true);

    useEffect(() => {
        isMountedRef.current = true;
        return () => {
            isMountedRef.current = false;
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);

    useEffect(() => {
        // Auto-resize textarea
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [message]);

    useEffect(() => {
        // Close dropdown when clicking outside
        const handleClickOutside = (event: MouseEvent) => {
            if (
                imageDropdownRef.current &&
                !imageDropdownRef.current.contains(event.target as Node) &&
                event.target !== document.getElementById("imageUploadBtn")
            ) {
                setShowImageDropdown(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];

            // 이미지 파일 타입 확인
            if (!file.type.startsWith("image/")) {
                alert("이미지 파일만 업로드 가능합니다.");
                return;
            }

            // 파일 크기 제한 (10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert("이미지 크기는 10MB 이하여야 합니다.");
                return;
            }

            setUploadedImage(file);
            setShowImageDropdown(false);

            const reader = new FileReader();
            reader.onload = (evt) => {
                if (evt.target?.result) {
                    setImagePreview(evt.target.result as string);
                }
            };
            reader.readAsDataURL(file);
        }
    };

    const removeImage = () => {
        setUploadedImage(null);
        setImagePreview(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (message.trim() || uploadedImage) {
                sendMessage();
            }
        }
    };

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

    const toggleImageDropdown = () => {
        setShowImageDropdown((prev) => !prev);
    };

    const startAIStreamingMessage = () => {
        setIsStreaming(true);
        setStreamingContent("");
    };

    const updateStreamingMessage = (text: string) => {
        setStreamingContent(text);
    };

    const finalizeAIMessage = (finalText: string, requestId: string) => {
        if (requestId && finalText) {
            storeMessageInRoom(requestId, "AI", finalText);
        }
    };

    // 중단 버튼 전용 핸들러
    const handleStopButtonClick = () => {
        console.log("중단 버튼 클릭됨!");
        setIsStopClicked(true);
        stopGeneration();
    };

    const stopGeneration = async () => {
        console.log("stopGeneration 함수 호출됨");
        console.log("isProcessing 상태:", isProcessing);
        console.log("currentRequestId:", currentRequestId);

        if (!currentRequestId) {
            console.log("함수 조기 종료: currentRequestId가 없음");
            return;
        }

        try {
            // 진행 중인 요청 중단
            if (abortControllerRef.current) {
                console.log("AbortController 중단");
                abortControllerRef.current.abort();
                abortControllerRef.current = null;
            }

            // 스트리밍 중인 메시지가 있으면 중단 메시지 추가
            if (streamingContent) {
                const finalContent =
                    streamingContent + "\n\n*메시지 생성이 사용자에 의해 중단되었습니다*";
                finalizeAIMessage(finalContent, currentRequestId);
            }

            try {
                // 서버에 중단 요청 전송 (선택적)
                console.log("서버에 중단 요청 전송 시작");
                const response = await fetch("/api/stop_generation", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ request_id: currentRequestId }),
                });
                console.log("중단 요청 응답 상태:", response.status);

                if (response.ok) {
                    const responseData = await response.json();
                    console.log("중단 응답 데이터:", responseData);
                }
            } catch (fetchError) {
                console.error("중단 요청 전송 오류:", fetchError);
            }

            // 상태 초기화 (항상 실행)
            if (isMountedRef.current) {
                console.log("UI 상태 초기화");
                setIsProcessing(false);
                setIsStreaming(false);
                setStreamingContent("");
                setIsStopClicked(false);

                // 채팅방 데이터 업데이트
                if (setChatRoomsData && currentRequestId) {
                    setChatRoomsData((prev) => ({
                        ...prev,
                        [currentRequestId]: {
                            ...prev[currentRequestId],
                            isStreaming: false,
                        },
                    }));
                }
            }
        } catch (error) {
            console.error("중단 요청 처리 오류:", error);
            // 에러가 발생해도 UI는 초기화
            if (isMountedRef.current) {
                setIsProcessing(false);
                setIsStreaming(false);
                setIsStopClicked(false);

                if (streamingContent && currentRequestId) {
                    const finalContent =
                        streamingContent + "\n\n*메시지 생성 중단 중 오류가 발생했습니다*";
                    finalizeAIMessage(finalContent, currentRequestId);
                }
            }
        }
    };

    const sendMessage = async () => {
        if ((!message.trim() && !uploadedImage) || !currentRequestId) return;
        if (isProcessing) return; // Don't send if already processing

        // 이전 요청이 있으면 중단
        if (abortControllerRef.current) {
            console.log("이전 요청 중단");
            abortControllerRef.current.abort();
        }

        // 새 AbortController 생성
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        setIsProcessing(true);
        setIsStreaming(true);
        setIsStopClicked(false);

        // 채팅방 스트리밍 상태 설정
        setChatRoomsData((prev) => ({
            ...prev,
            [currentRequestId]: {
                ...prev[currentRequestId],
                isStreaming: true,
            },
        }));

        // Process image if present
        let imageBase64 = null;
        try {
            if (uploadedImage) {
                imageBase64 = await readFileAsBase64(uploadedImage);
            }
        } catch (err) {
            console.error("이미지 처리 오류:", err);
            alert("이미지 처리 중 오류가 발생했습니다.");
            setIsProcessing(false);
            setIsStreaming(false);
            return;
        }

        // Store user message
        storeMessageInRoom(
            currentRequestId,
            "사용자",
            message.trim() + (imageBase64 ? " [이미지 첨부됨]" : "")
        );

        // Reset form
        setMessage("");
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }
        removeImage();

        // Construct payload
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

        try {
            startAIStreamingMessage();

            const response = await fetch("/api/query_stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal,
            });

            if (!response.ok) {
                throw new Error("응답 오류: " + response.status);
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
                // 컴포넌트가 언마운트되거나 중단 요청이 된 경우 중단
                if (!isMountedRef.current || isStopClicked) break;

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
                                    finalizeAIMessage(partialAnswer, currentRequestId);
                                    if (currentReferences.length > 0) {
                                        currentReferences.forEach((ref) => {
                                            addReferenceData(ref);
                                        });
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
                                        updateStreamingMessage(partialAnswer);
                                    } else if (sseData.type === "reference") {
                                        currentReferences.push(sseData);
                                        addReferenceData(sseData);
                                    }
                                }
                            } catch (err) {
                                // JSON 파싱 실패시 텍스트로 처리
                                if (isMountedRef.current) {
                                    partialAnswer += jsonStr;
                                    updateStreamingMessage(partialAnswer);
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
                    break;
                }
            }

            // 스트림이 정상적으로 종료되지 않았는데 응답이 있는 경우
            if (isMountedRef.current && isStreaming && partialAnswer) {
                finalizeAIMessage(partialAnswer, currentRequestId);
            }
        } catch (error) {
            console.error("메시지 전송 오류:", error);
            // 요청이 취소된 경우 처리
            if (signal.aborted) {
                console.log("사용자에 의해 요청이 취소되었습니다.");
            } else {
                // 시스템 오류 메시지 추가
                storeMessageInRoom(
                    currentRequestId,
                    "시스템",
                    `Error: ${error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다."
                    }`
                );
            }
        } finally {
            if (isMountedRef.current) {
                setIsProcessing(false);
                setIsStreaming(false);
            }
            // 채팅방 스트리밍 상태 업데이트
            setChatRoomsData((prev) => ({
                ...prev,
                [currentRequestId]: {
                    ...prev[currentRequestId],
                    isStreaming: false,
                },
            }));
        }
    };

    return (
        <div className="bg-white border-t border-[#e5e7eb] p-4 md:p-6 flex gap-3 relative transition-all duration-300 ease-in-out">
            <div className="flex-1 relative max-w-[768px] mx-auto w-full">
                <div className="relative">
                    <textarea
                        id="userMessage"
                        ref={textareaRef}
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={handleKeyDown}
                        className="w-full py-3 px-4 pr-12 text-[0.9375rem] border border-[#e5e7eb] rounded-[0.75rem] resize-none min-h-[2.5rem] max-h-[150px] overflow-y-auto transition-all duration-300 ease-in-out font-inherit shadow-sm focus:outline-none focus:border-[#10a37f] focus:shadow-[0_0_0_2px_rgba(16,163,127,0.2)]"
                        placeholder="메시지를 입력하세요..."
                        rows={1}
                        disabled={isProcessing}
                    />

                    <div className="absolute left-3 bottom-3 flex items-center">
                        <div className="relative" ref={imageDropdownRef}>
                            <input
                                type="file"
                                id="imageUpload"
                                accept="image/*"
                                className="hidden"
                                onChange={handleImageChange}
                                ref={fileInputRef}
                                disabled={isProcessing}
                            />
                            <button
                                id="imageUploadBtn"
                                className={`bg-transparent border-none rounded-[12%] cursor-pointer p-1 flex items-center justify-center ${uploadedImage
                                        ? "bg-[#0e8e70] text-white"
                                        : "hover:bg-[#f3f3f3]"
                                    } ${isProcessing ? "opacity-50 cursor-not-allowed" : ""}`}
                                onClick={toggleImageDropdown}
                                disabled={isProcessing}
                            >
                                <Image className="w-5 h-5 text-[#10a37f]" />
                                <span className="ml-1">이미지</span>
                            </button>

                            {showImageDropdown && (
                                <div className="absolute top-[-350%] left-0 bg-white border border-[#e5e7eb] shadow-md rounded-md p-2 z-10 min-w-[150px]">
                                    <div className="text-sm font-semibold mb-2">이미지 업로드</div>
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
                            className={`bg-transparent border-none rounded-[12%] cursor-pointer p-1 flex items-center justify-center ml-2 hover:bg-[#f3f3f3] ${isProcessing ? "opacity-50 cursor-not-allowed" : ""
                                }`}
                            onClick={toggleRag}
                            disabled={isProcessing}
                        >
                            <LayoutGrid className="w-5 h-5 text-[#10a37f]" />
                            <span className="ml-1">RAG: {useRag ? "On" : "Off"}</span>
                        </button>
                    </div>

                    {isProcessing ? (
                        // 중단 버튼
                        <button
                            id="stopButton"
                            className="absolute right-2 bottom-2 flex items-center justify-center w-8 h-8 border-none rounded-full bg-[#f44336] hover:bg-[#d32f2f] cursor-pointer"
                            onClick={handleStopButtonClick}
                        >
                            <Square className="w-4 h-4 text-white" />
                        </button>
                    ) : (
                        // 전송 버튼
                        <button
                            id="sendButton"
                            className="absolute right-2 bottom-2 flex items-center justify-center w-8 h-8 border-none rounded-full bg-[#10a37f] text-white cursor-pointer transition-all duration-300 ease-in-out hover:bg-[#0e8e70] hover:scale-105"
                            onClick={sendMessage}
                            disabled={!message.trim() && !uploadedImage}
                        >
                            <Send className="w-4 h-4 text-white" />
                        </button>
                    )}
                </div>
            </div>

            {imagePreview && (
                <div className="max-w-[600px] mt-2 flex flex-row flex-wrap gap-2 rounded-md overflow-auto bg-white border border-[#e5e7eb] shadow-sm absolute bottom-20 left-1/2 transform -translate-x-1/2">
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
    );
}
