import { PaperAirplaneIcon } from '@heroicons/react/24/solid';

interface WelcomeScreenProps {
  onNewChat: () => void;
}

export default function WelcomeScreen({ onNewChat }: WelcomeScreenProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-text-primary mb-4">
          Chat Assistant에 오신 것을 환영합니다
        </h1>
        <p className="text-text-secondary text-lg">
          AI와 대화를 시작하거나 문서를 업로드하여 분석을 시작하세요.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl w-full mb-8">
        <div className="p-6 bg-white rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-primary mb-4">
            일반 대화
          </h2>
          <p className="text-text-secondary mb-4">
            AI와 자유롭게 대화하고 질문하세요.
          </p>
          <button
            onClick={onNewChat}
            className="flex items-center gap-2 text-primary hover:text-primary-hover"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
            대화 시작하기
          </button>
        </div>

        <div className="p-6 bg-white rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-primary mb-4">
            문서 분석
          </h2>
          <p className="text-text-secondary mb-4">
            PDF나 이미지를 업로드하여 분석을 시작하세요.
          </p>
          <button
            onClick={onNewChat}
            className="flex items-center gap-2 text-primary hover:text-primary-hover"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
            문서 업로드
          </button>
        </div>
      </div>

      <div className="max-w-2xl w-full">
        <h3 className="text-lg font-semibold text-text-primary mb-4">
          예시 질문
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={onNewChat}
            className="p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow text-left"
          >
            "Python으로 웹 크롤링하는 방법을 알려주세요"
          </button>
          <button
            onClick={onNewChat}
            className="p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow text-left"
          >
            "React와 Next.js의 차이점은 무엇인가요?"
          </button>
          <button
            onClick={onNewChat}
            className="p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow text-left"
          >
            "Docker 컨테이너와 이미지의 차이를 설명해주세요"
          </button>
          <button
            onClick={onNewChat}
            className="p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow text-left"
          >
            "머신러닝 모델을 배포하는 best practice는 무엇인가요?"
          </button>
        </div>
      </div>
    </div>
  );
} 