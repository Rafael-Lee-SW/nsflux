import { NextRequest, NextResponse } from 'next/server';

// 백엔드 API URL (실제 서버 URL로 변경 필요)
const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:5000';

export async function GET(request: NextRequest) {
  try {
    // URL에서 request_id 파라미터 추출
    const { searchParams } = new URL(request.url);
    const requestId = searchParams.get('request_id');
    
    if (!requestId) {
      return NextResponse.json(
        { error: 'request_id 파라미터가 필요합니다' },
        { status: 400 }
      );
    }
    
    console.log(`채팅 기록 요청: ${requestId}`);

    // 백엔드 서버로 기록 요청 전달
    const backendResponse = await fetch(`${BACKEND_API_URL}/history?request_id=${requestId}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    // 백엔드 응답이 성공적이지 않은 경우 에러 처리
    if (!backendResponse.ok) {
      console.error(`백엔드 서버 응답 오류: ${backendResponse.status}`);
      
      // 기록이 없는 경우 빈 기록 반환 (404 에러를 클라이언트에게 전달하지 않음)
      if (backendResponse.status === 404) {
        return NextResponse.json({ history: [] });
      }
      
      return NextResponse.json(
        { error: `백엔드 서버 오류: ${backendResponse.status}` },
        { status: backendResponse.status }
      );
    }

    // 응답 데이터 처리
    const data = await backendResponse.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('API 처리 중 오류:', error);
    
    // 오류 발생 시 빈 기록 반환 (에러를 클라이언트에게 전달하지 않음)
    return NextResponse.json({ history: [] });
  }
}