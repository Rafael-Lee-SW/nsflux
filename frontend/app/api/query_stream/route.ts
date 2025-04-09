import { NextRequest, NextResponse } from 'next/server';
import { ReadableStream } from 'stream/web';

// 백엔드 API URL (실제 서버 URL로 변경 필요)
const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
  try {
    // 클라이언트로부터 JSON 요청 파싱
    const requestData = await request.json();
    
    console.log(`API 요청: ${BACKEND_API_URL}/query_stream`);
    console.log('요청 데이터:', JSON.stringify(requestData));

    // 백엔드 서버로 요청 전달
    const backendResponse = await fetch(`${BACKEND_API_URL}/query_stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    // 백엔드 응답이 성공적이지 않은 경우 에러 처리
    if (!backendResponse.ok) {
      console.error(`백엔드 서버 응답 오류: ${backendResponse.status}`);
      const errorText = await backendResponse.text();
      console.error('응답 내용:', errorText);
      
      return NextResponse.json(
        { error: `백엔드 서버 오류: ${backendResponse.status}` },
        { status: backendResponse.status }
      );
    }

    // 스트리밍 응답인지 확인
    const contentType = backendResponse.headers.get('content-type');
    
    // 스트리밍 응답인 경우 스트림 처리
    if (contentType && contentType.includes('text/event-stream')) {
      // 백엔드에서 받은 스트림을 클라이언트로 전달
      const backendStream = backendResponse.body;
      
      if (!backendStream) {
        throw new Error('백엔드 응답에 본문이 없습니다.');
      }

      // 스트림 변환 함수: 백엔드에서 받은 스트림을 클라이언트로 전달
      const stream = new ReadableStream({
        async start(controller) {
          const reader = backendStream.getReader();
          
          try {
            while (true) {
              const { done, value } = await reader.read();
              
              if (done) {
                // 스트림 종료 처리
                controller.close();
                break;
              }
              
              // 받은 청크를 그대로 클라이언트로 전달
              controller.enqueue(value);
            }
          } catch (error) {
            console.error('스트림 처리 중 오류:', error);
            controller.error(error);
          }
        }
      });

      // 스트림 응답 헤더 설정 및 반환
      return new Response(stream as unknown as BodyInit, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    } else {
      // 일반 JSON 응답인 경우
      const data = await backendResponse.json();
      return NextResponse.json(data);
    }
  } catch (error) {
    console.error('API 처리 중 오류:', error);
    return NextResponse.json(
      { error: `요청 처리 중 오류: ${error instanceof Error ? error.message : '알 수 없는 오류'}` },
      { status: 500 }
    );
  }
}