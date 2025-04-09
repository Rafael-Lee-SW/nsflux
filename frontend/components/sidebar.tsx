"use client"
import React from 'react';
import { ChatRoom } from '@/types';
import { PlusIcon, ChevronLeftIcon, ChevronRightIcon, ChatBubbleLeftIcon, BeakerIcon } from '@heroicons/react/24/outline';

interface SidebarProps {
  chatRooms: ChatRoom[];
  activeChatRoom: ChatRoom | null;
  onNewChat: () => void;
  onSelectChatRoom: (chatRoom: ChatRoom) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  togglePromptTest: () => void;
  isPromptTestOpen: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  chatRooms,
  activeChatRoom,
  onNewChat,
  onSelectChatRoom,
  sidebarOpen,
  toggleSidebar,
  togglePromptTest,
  isPromptTestOpen
}) => {
  return (
    <div className={`sidebar ${!sidebarOpen ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <ChatBubbleLeftIcon className="w-6 h-6 text-white" />
          <h2>채팅방</h2>
        </div>
        <button className="sidebar-toggle" onClick={toggleSidebar}>
          {sidebarOpen ? <ChevronLeftIcon className="w-5 h-5" /> : <ChevronRightIcon className="w-5 h-5" />}
        </button>
      </div>

      <button className="new-chat-btn" onClick={onNewChat}>
        <PlusIcon className="w-5 h-5" />
        새 채팅
      </button>

      <div className="sidebar-tools">
        <button 
          className={`sidebar-tool-btn ${isPromptTestOpen ? 'bg-white/10' : ''}`} 
          onClick={togglePromptTest}
        >
          <BeakerIcon className="w-5 h-5" />
          프롬프트 테스트
        </button>
      </div>

      <ul className="chat-room-list">
        {chatRooms.map((chatRoom) => (
          <li key={chatRoom.requestId}>
            <button
              className={activeChatRoom?.requestId === chatRoom.requestId ? 'active' : ''}
              onClick={() => onSelectChatRoom(chatRoom)}
            >
              <ChatBubbleLeftIcon className="w-5 h-5" />
              {chatRoom.title || '새 채팅'}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;

