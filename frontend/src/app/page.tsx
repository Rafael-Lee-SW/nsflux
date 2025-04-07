'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatRoom from '@/components/ChatRoom';
import WelcomeScreen from '@/components/WelcomeScreen';

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);

  return (
    <div className="flex h-screen bg-chat-bg">
      <Sidebar 
        isOpen={isSidebarOpen} 
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        currentChatId={currentChatId}
        onChatSelect={setCurrentChatId}
      />
      
      <main className="flex-1 flex flex-col">
        {currentChatId ? (
          <ChatRoom chatId={currentChatId} />
        ) : (
          <WelcomeScreen onNewChat={() => setCurrentChatId('new')} />
        )}
      </main>
    </div>
  );
} 