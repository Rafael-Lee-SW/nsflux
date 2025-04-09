import { useState, useEffect } from 'react';

/**
 * 모바일 기기 여부를 감지하는 커스텀 훅
 * @param mobileWidth 모바일로 간주할 최대 너비 (픽셀)
 * @returns 모바일 기기 여부 (boolean)
 */
export function useMobile(mobileWidth = 768) {
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        // 초기 상태 설정
        const checkMobile = () => {
            setIsMobile(window.innerWidth <= mobileWidth);
        };

        // 브라우저에서만 실행 (SSR 대응)
        if (typeof window !== 'undefined') {
            checkMobile();

            // 리사이즈 이벤트에 반응
            window.addEventListener('resize', checkMobile);

            // 클린업 함수
            return () => {
                window.removeEventListener('resize', checkMobile);
            };
        }
    }, [mobileWidth]);

    return isMobile;
}