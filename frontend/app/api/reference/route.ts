import { NextRequest, NextResponse } from 'next/server';

// 백엔드 API URL (실제 서버 URL로 변경 필요)
const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:5000';

export async function GET(request: NextRequest) {
  try {
    // URL에서 파라미터 추출
    const { searchParams } = new URL(request.url);
    const requestId = searchParams.get('request_id');
    const msgIndex = searchParams.get('msg_index');
    
    if (!requestId) {
      return NextResponse.json(
        { error: 'request_id 파라미터가 필요합니다' },
        { status: 400 }
      );
    }
    
    // 메시지 인덱스가 없으면 모든 참조 정보 반환
    const endpoint = msgIndex 
      ? `${BACKEND_API_URL}/reference?request_id=${requestId}&msg_index=${msgIndex}`
      : `${BACKEND_API_URL}/reference?request_id=${requestId}`;
    
    console.log(`참조 정보 요청: ${endpoint}`);

    // 백엔드 서버로 요청 전달
    const backendResponse = await fetch(endpoint, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    // 백엔드 응답이 성공적이지 않은 경우 에러 처리
    if (!backendResponse.ok) {
      console.error(`백엔드 서버 응답 오류: ${backendResponse.status}`);
      
      // 참조 정보가 없는 경우 빈 배열 반환 (404 에러를 클라이언트에게 전달하지 않음)
      if (backendResponse.status === 404) {
        return NextResponse.json({ references: [] });
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
    
    // 오류 발생 시 빈 참조 정보 반환
    return NextResponse.json({ references: [] });
  }
}