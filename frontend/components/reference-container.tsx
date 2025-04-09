"use client"

import { X } from "lucide-react"

interface ReferenceContainerProps {
    referenceList: any[]
    currentlyViewingMsgIndex: number | null
    closeReferences: () => void
}

export default function ReferenceContainer({
    referenceList,
    currentlyViewingMsgIndex,
    closeReferences,
}: ReferenceContainerProps) {
    if (referenceList.length === 0) {
        return (
            <div className="block m-0 bg-white border-b border-[#e5e7eb] p-4 md:p-6 max-h-[300px] overflow-y-auto">
                <p className="text-center text-[#6b7280] italic p-4">참조 데이터가 없습니다.</p>
            </div>
        )
    }

    return (
        <div className="block m-0 bg-white border-b border-[#e5e7eb] p-4 md:p-6 max-h-[300px] overflow-y-auto">
            {currentlyViewingMsgIndex !== null && (
                <div className="flex items-center justify-between mb-4 pb-2 border-b border-[#e5e7eb]">
                    <h3 className="text-base text-[#10a37f] font-semibold">
                        References for Message #{currentlyViewingMsgIndex + 1}
                    </h3>
                    <button
                        className="bg-transparent border-none text-[#6b7280] cursor-pointer flex items-center justify-center p-2 rounded-sm transition-all duration-300 ease-in-out hover:bg-[rgba(0,0,0,0.05)] hover:text-[#10a37f]"
                        onClick={closeReferences}
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>
            )}

            {referenceList.map((refObj, i) => (
                <div key={i} className="mb-4 p-4 bg-[#f7f7f8] rounded-md border border-[#e5e7eb] shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                        <h4 className="text-sm text-[#10a37f] font-semibold">Reference #{i + 1}</h4>
                    </div>
                    <pre className="text-xs whitespace-pre-wrap bg-white p-3 rounded-sm border border-[#e5e7eb] overflow-x-auto">
                        {JSON.stringify(refObj, null, 2)}
                    </pre>
                </div>
            ))}
        </div>
    )
}

