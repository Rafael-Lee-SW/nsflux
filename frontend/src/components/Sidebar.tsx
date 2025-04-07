import { useState } from 'react';
import { 
  Bars3Icon, 
  XMarkIcon, 
  PlusIcon,
  ChatBubbleLeftIcon,
  DocumentTextIcon,
  PhotoIcon
} from '@heroicons/react/24/outline';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  currentChatId: string | null;
  onChatSelect: (chatId: string) => void;
}

export default function Sidebar({ isOpen, onToggle, currentChatId, onChatSelect }: SidebarProps) {
  const [chatRooms] = useState([
    { id: '1', title: '일반 대화' },
    { id: '2', title: '코드 리뷰' },
    { id: '3', title: '이미지 분석' },
  ]);

  return (
    <aside className={`${isOpen ? 'w-72' : 'w-0'} bg-sidebar-bg text-text-light transition-all duration-300 ease-in-out overflow-hidden`}>
      <div className="flex flex-col h-full p-4">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <h2 className="text-xl font-bold">Chat Assistant</h2>
          </div>
          <button
            onClick={onToggle}
            className="p-2 hover:bg-sidebar-hover rounded-md"
          >
            {isOpen ? (
              <XMarkIcon className="w-6 h-6" />
            ) : (
              <Bars3Icon className="w-6 h-6" />
            )}
          </button>
        </div>

        <button
          onClick={() => onChatSelect('new')}
          className="flex items-center justify-center gap-2 w-full bg-primary hover:bg-primary-hover text-white py-2 px-4 rounded-md mb-6 transition-colors"
        >
          <PlusIcon className="w-5 h-5" />
          새 대화
        </button>

        <div className="space-y-2">
          {chatRooms.map((room) => (
            <button
              key={room.id}
              onClick={() => onChatSelect(room.id)}
              className={`flex items-center gap-2 w-full p-2 rounded-md transition-colors ${
                currentChatId === room.id
                  ? 'bg-primary text-white'
                  : 'hover:bg-sidebar-hover'
              }`}
            >
              <ChatBubbleLeftIcon className="w-5 h-5" />
              {room.title}
            </button>
          ))}
        </div>

        <div className="mt-auto space-y-2">
          <button className="flex items-center gap-2 w-full p-2 hover:bg-sidebar-hover rounded-md transition-colors">
            <DocumentTextIcon className="w-5 h-5" />
            문서 업로드
          </button>
          <button className="flex items-center gap-2 w-full p-2 hover:bg-sidebar-hover rounded-md transition-colors">
            <PhotoIcon className="w-5 h-5" />
            이미지 업로드
          </button>
        </div>
      </div>
    </aside>
  );
} 