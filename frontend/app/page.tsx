"use client"

import { useEffect, useState } from "react"
import Sidebar from "@/components/sidebar"
import ChatHeader from "@/components/chat-header"
import ChatBox from "@/components/chat-box"
import InputContainer from "@/components/input-container"
import ReferenceContainer from "@/components/reference-container"
import PromptTestPanel from "@/components/prompt-test-panel"
import { useMobile } from "@/hooks/use-mobile"
import type { ChatRoom, ChatRoomData } from "@/types"

export default function Home() {
  const [chatRooms, setChatRooms] = useState<ChatRoom[]>([])
  const [chatRoomsData, setChatRoomsData] = useState<Record<string, ChatRoomData>>({})
  const [currentRequestId, setCurrentRequestId] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [showReferences, setShowReferences] = useState(false)
  const [showPromptTest, setShowPromptTest] = useState(false)
  const [useRag, setUseRag] = useState(true)
  const [currentlyViewingMsgIndex, setCurrentlyViewingMsgIndex] = useState<number | null>(null)
  const isMobile = useMobile()
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile)

  useEffect(() => {
    createNewChatRoom()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setSidebarOpen(!isMobile)
  }, [isMobile])

  // isProcessing 상태를 제대로 추적하기 위한 useEffect 추가
  useEffect(() => {
    // 현재 활성화된 채팅방의 스트리밍 상태에 따라 isProcessing 상태 업데이트
    if (currentRequestId && chatRoomsData[currentRequestId]) {
      const isRoomStreaming = chatRoomsData[currentRequestId].isStreaming;
      console.log(`채팅방 ${currentRequestId} 스트리밍 상태 변경: ${isRoomStreaming}`);
      setIsProcessing(isRoomStreaming);
    }
  }, [currentRequestId, chatRoomsData]);

  // 중단 버튼 작동 확인을 위한 디버깅 로그 추가
  const handleProcessingChange = (newState: boolean) => {
    console.log(`처리 상태 변경: ${isProcessing} -> ${newState}`);
    setIsProcessing(newState);
  };

  const createNewChatRoom = () => {
    const newRequestId = Date.now().toString() + "-" + Math.floor(Math.random() * 10000)
    setChatRooms((prev) => [...prev, {
      requestId: newRequestId,
      title: "새로운 대화",
      messages: [],
      createdAt: new Date().toISOString()
    }])
    setChatRoomsData((prev) => ({
      ...prev,
      [newRequestId]: {
        uploadedImages: [],
        referenceList: [],
        isStreaming: false,
      },
    }))
    setCurrentRequestId(newRequestId)
  }

  const setActiveChatRoom = async (requestId: string) => {
    if (isProcessing && currentRequestId !== requestId) {
      alert("메시지 처리 중입니다. 완료 후 채팅방을 변경해주세요.")
      return
    }

    setCurrentRequestId(requestId)

    // Initialize room data if it doesn't exist
    if (!chatRoomsData[requestId]) {
      setChatRoomsData((prev) => ({
        ...prev,
        [requestId]: {
          uploadedImages: [],
          referenceList: [],
          isStreaming: false,
        },
      }))
    }

    const room = chatRooms.find((r) => r.requestId === requestId)
    if (!room) {
      setChatRooms((prev) => [...prev, {
        requestId,
        title: "새로운 대화",
        messages: [],
        createdAt: new Date().toISOString()
      }])
    } else {
      await loadHistory(requestId)
    }

    clearReferenceData()

    if (isMobile) {
      setSidebarOpen(false)
    }
  }

  const loadHistory = async (requestId: string) => {
    try {
      const response = await fetch(`/history?request_id=${requestId}`)
      if (response.ok) {
        const data = await response.json()
        setChatRooms((prev) =>
          prev.map((room) => (room.requestId === requestId ? {
            ...room,
            messages: data.history.map((msg: any) => ({
              ...msg,
              timestamp: msg.timestamp || new Date().toISOString()
            }))
          } : room)),
        )
      }
    } catch (err) {
      console.error("History load error:", err)
    }
  }

  const clearReferenceData = () => {
    setCurrentlyViewingMsgIndex(null)
    setShowReferences(false)
  }

  const toggleReferences = () => {
    setShowReferences((prev) => !prev)
  }

  const togglePromptTesting = () => {
    setShowPromptTest((prev) => !prev)
  }

  const toggleRag = () => {
    setUseRag((prev) => !prev)
  }

  const toggleSidebar = () => {
    setSidebarOpen((prev) => !prev)
  }

  const storeMessageInRoom = (requestId: string, sender: string, text: string) => {
    setChatRooms((prev) =>
      prev.map((room) =>
        room.requestId === requestId
          ? {
            ...room,
            messages: [
              ...room.messages,
              {
                role: sender === "사용자" ? "human" : "ai",
                content: text,
                timestamp: new Date().toISOString()
              },
            ],
          }
          : room,
      ),
    )
  }

  const addReferenceData = (refData: any) => {
    if (currentRequestId) {
      setChatRoomsData((prev) => ({
        ...prev,
        [currentRequestId]: {
          ...prev[currentRequestId],
          referenceList: [...(prev[currentRequestId]?.referenceList || []), refData],
        },
      }))
    }
  }

  const currentRoom = currentRequestId ? chatRooms.find((room) => room.requestId === currentRequestId) || null : null

  const hasMessages = currentRoom?.messages && currentRoom.messages.length > 0

  return (
    <main className="flex h-screen bg-[#f7f7f8] text-[#111827] overflow-hidden">
      <Sidebar
        chatRooms={chatRooms}
        activeChatRoom={currentRoom}
        onNewChat={createNewChatRoom}
        onSelectChatRoom={(room) => setActiveChatRoom(room.requestId)}
        sidebarOpen={sidebarOpen}
        toggleSidebar={toggleSidebar}
        togglePromptTest={togglePromptTesting}
        isPromptTestOpen={showPromptTest}
      />

      <div className="flex-1 flex flex-col overflow-hidden relative">
        <ChatHeader
          currentRequestId={currentRequestId}
          toggleReferences={toggleReferences}
          showReferences={showReferences}
          toggleSidebar={toggleSidebar}
          sidebarOpen={sidebarOpen}
          hasReferences={currentRequestId ? (chatRoomsData[currentRequestId]?.referenceList?.length || 0) > 0 : false}
        />

        {showReferences && (
          <ReferenceContainer
            referenceList={currentRequestId ? chatRoomsData[currentRequestId]?.referenceList || [] : []}
            currentlyViewingMsgIndex={currentlyViewingMsgIndex}
            closeReferences={() => {
              setShowReferences(false)
              setCurrentlyViewingMsgIndex(null)
            }}
          />
        )}

        <ChatBox
          currentRoom={currentRoom}
          loadReferenceForMessage={(msgIndex: number) => {
            if (currentlyViewingMsgIndex === msgIndex && showReferences) {
              setShowReferences(false)
              setCurrentlyViewingMsgIndex(null)
            } else {
              setCurrentlyViewingMsgIndex(msgIndex)
              setShowReferences(true)
            }
          }}
          currentRequestId={currentRequestId}
        />

        {hasMessages && (
          <InputContainer
            currentRequestId={currentRequestId}
            isProcessing={isProcessing}
            setIsProcessing={setIsProcessing}
            storeMessageInRoom={storeMessageInRoom}
            addReferenceData={addReferenceData}
            useRag={useRag}
            toggleRag={toggleRag}
            setChatRoomsData={setChatRoomsData}
          />
        )}

        <PromptTestPanel isOpen={showPromptTest} togglePromptTesting={togglePromptTesting} />
      </div>
    </main>
  )
}

