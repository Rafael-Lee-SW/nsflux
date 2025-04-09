"use client"

import { ExternalLink, Menu } from "lucide-react"

interface ChatHeaderProps {
    currentRequestId: string | null
    toggleReferences: () => void
    showReferences: boolean
    toggleSidebar: () => void
    sidebarOpen: boolean
    hasReferences: boolean
}

export default function ChatHeader({
    currentRequestId,
    toggleReferences,
    showReferences,
    toggleSidebar,
    sidebarOpen,
    hasReferences,
}: ChatHeaderProps) {
    return (
        <div className="bg-white p-4 md:p-6 border-b border-[#e5e7eb] flex items-center justify-between z-10 shadow-sm">
            {!sidebarOpen && (
                <button id="sidebarOpener" className="flex items-center justify-center md:hidden" onClick={toggleSidebar}>
                    <Menu className="w-6 h-6" />
                </button>
            )}
            <h2 className="text-sm font-medium text-[#6b7280]">Chat Room: {currentRequestId || "-"}</h2>
            <div className="flex items-center gap-2">
                {hasReferences && (
                    <button
                        id="toggleRefBtn"
                        className="flex items-center gap-2 py-2 px-3 bg-white border border-[#e5e7eb] rounded-sm text-xs font-medium text-[#6b7280] cursor-pointer transition-all duration-300 ease-in-out hover:bg-[#f7f7f8] hover:border-[#10a37f] hover:text-[#10a37f]"
                        onClick={toggleReferences}
                    >
                        <ExternalLink className="w-4 h-4 text-[#10a37f]" />
                        <span>{showReferences ? "Hide All References" : "Show All References"}</span>
                    </button>
                )}
            </div>
        </div>
    )
}

