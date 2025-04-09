"use client"

import type React from "react"

import { useState, useRef } from "react"
import { X, Play, Square, FileText } from "lucide-react"
import { parseMarkdown } from "@/lib/markdown-parser"

interface PromptTestPanelProps {
    isOpen: boolean
    togglePromptTesting: () => void
}

export default function PromptTestPanel({ isOpen, togglePromptTesting }: PromptTestPanelProps) {
    const [systemPrompt, setSystemPrompt] = useState("")
    const [userInput, setUserInput] = useState("")
    const [isTestPromptRunning, setIsTestPromptRunning] = useState(false)
    const [testResult, setTestResult] = useState("")
    const [uploadedFile, setUploadedFile] = useState<File | null>(null)
    const [filePreview, setFilePreview] = useState<string | null>(null)
    const [fileType, setFileType] = useState<"image" | "pdf" | null>(null)

    const fileInputRef = useRef<HTMLInputElement>(null)

    const handleTestFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0]
            setUploadedFile(file)

            // Check file type
            const type = file.type.startsWith("image/") ? "image" : file.type === "application/pdf" ? "pdf" : null

            if (!type) {
                alert("지원되지 않는 파일 형식입니다. 이미지 또는 PDF만 업로드 가능합니다.")
                e.target.value = ""
                return
            }

            setFileType(type)

            if (type === "image") {
                const reader = new FileReader()
                reader.onload = (e) => {
                    if (e.target?.result) {
                        setFilePreview(e.target.result as string)
                    }
                }
                reader.readAsDataURL(file)
            } else {
                // For PDF, we just show an icon
                setFilePreview(null)
            }
        }
    }

    const removeFile = () => {
        setUploadedFile(null)
        setFilePreview(null)
        setFileType(null)
        if (fileInputRef.current) {
            fileInputRef.current.value = ""
        }
    }

    const readFileAsBase64 = (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader()
            reader.onload = () => {
                if (reader.result) {
                    const base64 = (reader.result as string).split(",")[1]
                    resolve(base64)
                } else {
                    reject(new Error("Failed to read file"))
                }
            }
            reader.onerror = reject
            reader.readAsDataURL(file)
        })
    }

    const runPromptTest = async () => {
        if (!systemPrompt.trim()) {
            alert("시스템 프롬프트를 입력해주세요.")
            return
        }

        setIsTestPromptRunning(true)
        setTestResult("")

        // Process file if present
        let fileData = null
        let fileTypeValue = null

        if (uploadedFile && fileType) {
            try {
                fileData = await readFileAsBase64(uploadedFile)
                fileTypeValue = fileType
            } catch (error) {
                console.error("파일 읽기 오류:", error)
                setTestResult(`<p class="text-red-500 text-center my-4">파일 읽기 오류: ${error}</p>`)
                setIsTestPromptRunning(false)
                return
            }
        }

        // Construct payload
        const payload: {
            prompt: string;
            user_input: string;
            file_data?: string;
            file_type?: string;
        } = {
            prompt: systemPrompt,
            user_input: userInput,
        }

        if (fileData && fileTypeValue) {
            payload.file_data = fileData
            payload.file_type = fileTypeValue
        }

        try {
            // In a real implementation, this would be an actual API call
            // For now, we'll simulate a response after a delay

            setTestResult(
                '<div class="flex flex-col items-center justify-center h-[150px]"><div class="flex items-center mb-2"><span class="h-2 w-2 mx-0.5 bg-[#10a37f] rounded-full inline-block opacity-60 animate-pulse"></span><span class="h-2 w-2 mx-0.5 bg-[#10a37f] rounded-full inline-block opacity-60 animate-pulse delay-200"></span><span class="h-2 w-2 mx-0.5 bg-[#10a37f] rounded-full inline-block opacity-60 animate-pulse delay-400"></span></div><p>프롬프트 테스트 중...</p></div>',
            )

            await new Promise((resolve) => setTimeout(resolve, 2000))

            const simulatedResponse =
                "This is a simulated response from the prompt test. In a real implementation, this would be streamed from the server with the results of testing your prompt."

            setTestResult(`<div class="min-h-[100px]">${parseMarkdown(simulatedResponse)}</div>`)
        } catch (error) {
            console.error("Error running prompt test:", error)
            setTestResult(`<p class="text-red-500 text-center my-4">오류: ${error}</p>`)
        } finally {
            setIsTestPromptRunning(false)
        }
    }

    const stopPromptTest = () => {
        setIsTestPromptRunning(false)
        setTestResult((prev) => prev + '<p class="text-red-500 italic mt-4 text-center">*Test stopped by user*</p>')
    }

    if (!isOpen) return null

    return (
        <div className="fixed top-0 right-0 w-[500px] h-screen bg-white border-l border-[#e5e7eb] shadow-lg transition-all duration-300 ease-in-out flex flex-col z-50">
            <div className="flex items-center justify-between p-4 md:p-6 border-b border-[#e5e7eb]">
                <h3 className="text-lg font-semibold text-[#10a37f]">프롬프트 테스트</h3>
                <button
                    className="bg-transparent border-none text-[#6b7280] cursor-pointer flex items-center justify-center p-2 rounded-sm transition-all duration-300 ease-in-out hover:bg-[rgba(0,0,0,0.05)] hover:text-[#10a37f]"
                    onClick={togglePromptTesting}
                >
                    <X className="w-4 h-4" />
                </button>
            </div>

            <div className="flex-1 p-6 overflow-y-auto">
                <div className="mb-6">
                    <label className="block text-sm font-medium mb-2">시스템 프롬프트</label>
                    <textarea
                        id="systemPrompt"
                        value={systemPrompt}
                        onChange={(e) => setSystemPrompt(e.target.value)}
                        className="w-full p-3 text-[0.9375rem] border border-[#e5e7eb] rounded-md resize-vertical transition-all duration-300 ease-in-out font-inherit focus:outline-none focus:border-[#10a37f] focus:shadow-[0_0_0_2px_rgba(16,163,127,0.2)]"
                        placeholder="테스트할 프롬프트를 입력하세요..."
                        rows={6}
                    />
                </div>

                <div className="mb-6">
                    <label className="block text-sm font-medium mb-2">사용자 입력</label>
                    <textarea
                        id="testUserInput"
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        className="w-full p-3 text-[0.9375rem] border border-[#e5e7eb] rounded-md resize-vertical transition-all duration-300 ease-in-out font-inherit focus:outline-none focus:border-[#10a37f] focus:shadow-[0_0_0_2px_rgba(16,163,127,0.2)]"
                        placeholder="사용자 메시지를 입력하세요..."
                        rows={3}
                    />
                </div>

                <div className="mb-6">
                    <label className="block text-sm font-medium mb-2">파일 업로드 (선택사항)</label>
                    <div className="flex items-center gap-3 mb-3">
                        <input
                            type="file"
                            id="testFileUpload"
                            accept="image/*,application/pdf"
                            className="hidden"
                            onChange={handleTestFileUpload}
                            ref={fileInputRef}
                        />
                        <button
                            className="flex items-center gap-2 py-2 px-3 bg-white border border-[#e5e7eb] rounded-sm text-sm text-[#6b7280] cursor-pointer transition-all duration-300 ease-in-out hover:bg-[#f7f7f8] hover:border-[#10a37f] hover:text-[#10a37f]"
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <FileText className="w-4 h-4 text-[#10a37f]" />
                            파일 업로드 (이미지/PDF)
                        </button>
                    </div>

                    {uploadedFile && fileType && (
                        <div className="flex items-center bg-[#f8f9fa] rounded-md p-2.5 mt-2.5 relative">
                            {fileType === "image" && filePreview ? (
                                <div className="max-w-[200px] mb-3 border border-[#e5e7eb] rounded-md overflow-hidden">
                                    <img
                                        src={filePreview || "/placeholder.svg"}
                                        alt="Preview"
                                        className="w-full max-h-[150px] object-contain block"
                                    />
                                </div>
                            ) : (
                                fileType === "pdf" && (
                                    <div className="w-[60px] h-[60px] flex items-center justify-center bg-[#f8f9fa] rounded-md mr-2.5">
                                        <FileText className="w-8 h-8 text-[#e74c3c]" />
                                    </div>
                                )
                            )}

                            <div className="flex flex-col flex-grow">
                                <span className="font-medium mb-0.5">{uploadedFile.name}</span>
                                <span className="text-[0.85em] text-[#6c757d]">
                                    {fileType === "image" ? "이미지 파일" : "PDF 문서"}
                                </span>
                            </div>

                            <button
                                className="absolute top-[5px] right-[5px] bg-[rgba(255,255,255,0.7)] border-none rounded-full p-[5px] cursor-pointer flex items-center justify-center hover:bg-[rgba(255,255,255,0.9)]"
                                onClick={removeFile}
                            >
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                    )}
                </div>

                <div className="flex justify-end gap-4 mt-4">
                    {!isTestPromptRunning ? (
                        <button
                            id="runTestBtn"
                            className="flex items-center gap-2 py-2 px-4 rounded-md text-sm font-medium cursor-pointer transition-all duration-300 ease-in-out bg-[#10a37f] text-white border-none hover:bg-[#0e8e70]"
                            onClick={runPromptTest}
                        >
                            <Play className="w-4 h-4" />
                            실행
                        </button>
                    ) : (
                        <button
                            id="stopTestBtn"
                            className="flex items-center gap-2 py-2 px-4 rounded-md text-sm font-medium cursor-pointer transition-all duration-300 ease-in-out bg-[#f44336] text-white border-none hover:bg-[#d32f2f]"
                            onClick={stopPromptTest}
                        >
                            <Square className="w-4 h-4" />
                            중지
                        </button>
                    )}
                </div>
            </div>

            <div className="p-6 border-t border-[#e5e7eb] max-h-[40vh] overflow-y-auto">
                <h4 className="text-base font-semibold mb-3">결과</h4>
                <div
                    id="testResultArea"
                    className="bg-[#f7f7f8] p-4 rounded-md text-sm min-h-[150px] overflow-y-auto leading-relaxed"
                    dangerouslySetInnerHTML={{
                        __html:
                            testResult || '<p class="text-[#6b7280] italic text-center my-8">테스트 결과가 여기에 표시됩니다.</p>',
                    }}
                />
            </div>
        </div>
    )
}

