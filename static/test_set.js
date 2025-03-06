// test_set.js
// 테스트 세트 구성을 반환하는 buildTestQueries() 함수 (전역 함수)

function buildTestQueries() {
    const testSet = document.getElementById('testSetSelect').value;
    let queries = [];


    if (testSet === 'set_3_ask') {
        const differentThreeQueries = [
            { id: "only3-1", text: "남성해운 중국 시장 근황" },
            { id: "only3-2", text: "공 컨테이너 수송 전략" },
            { id: "only3-3", text: "남성해운의 새로운 전략" }
        ];
        queries = differentThreeQueries;

    } else if (testSet === 'set_5_ask') {
        const differentFiveQueries = [
            { id: "only5-1", text: "남성해운 중국 시장 근황" },
            { id: "only5-2", text: "공 컨테이너 수송 전략" },
            { id: "only5-3", text: "남성해운의 새로운 전략" },
            { id: "only5-4", text: "신입사원이 알아야 할 필수 인사규범" },
            { id: "only5-5", text: "남성해운이 가장 최근에 체결한 계약서 내용" }
        ];
        queries = differentFiveQueries;

    } else if (testSet === 'set_10_ask') {
        const differentTenQueries = [
            { id: "only10-1", text: "남성해운 중국 시장 근황" },
            { id: "only10-2", text: "공 컨테이너 수송 전략" },
            { id: "only10-3", text: "남성해운의 새로운 전략" },
            { id: "only10-4", text: "신입사원이 알아야 할 필수 인사규범" },
            { id: "only10-5", text: "남성해운이 가장 최근에 체결한 계약서 내용" },
            { id: "only10-6", text: "일본 관련한 계약서 중 남성해운 agent에 지급할 수수료에 대해서 상세히" },
            { id: "only10-7", text: "남성해운의 계약서 특징" },
            { id: "only10-8", text: "디지털 전략 및 가장 유망한 부분" },
            { id: "only10-9", text: "남성해운의 신사업" },
            { id: "only10-10", text: "타운사의 협업 관계" }
        ];
        queries = differentTenQueries;

    } else if (testSet === 'set_15_ask') {
        const differentFifteenQueries = [
            { id: "only15-1", text: "남성해운 중국 시장 근황" },
            { id: "only15-2", text: "공 컨테이너 수송 전략" },
            { id: "only15-3", text: "남성해운의 새로운 전략" },
            { id: "only15-4", text: "신입사원이 알아야 할 필수 인사규범" },
            { id: "only15-5", text: "남성해운이 가장 최근에 체결한 계약서 내용" },
            { id: "only15-6", text: "일본 관련한 계약서 중 남성해운 agent에 지급할 수수료에 대해서 상세히" },
            { id: "only15-7", text: "남성해운의 계약서 특징" },
            { id: "only15-8", text: "디지털 전략 및 가장 유망한 부분" },
            { id: "only15-9", text: "남성해운의 신사업" },
            { id: "only15-10", text: "타운사의 협업 관계" },
            { id: "only15-11", text: "주간회의에서 가장 중요하게 언급되는 것" },
            { id: "only15-12", text: "회의 중에서 나온 영업팀 유의 사항" },
            { id: "only15-13", text: "가장 오래된 계약서 및 해당 계약서의 유효성" },
            { id: "only15-14", text: "남성해운의 컨테이너 운용 계획" },
            { id: "only15-15", text: "남성해운의 선복 계획" }
        ];
        queries = differentFifteenQueries;

    } else if (testSet === 'set_20_ask_1') {
        const differentTwentyteenQueries1 = [
            { id: "ask1", text: "남성해운의 중국 시장 동향" },
            { id: "ask2", text: "남성해운의 일본 시장 영업 전략" },
            { id: "ask3", text: "남성해운의 동남아 시장 발전 가능성" },
            { id: "ask4", text: "남성해운의 중국 시장 동향" },
            { id: "ask5", text: "신입사원 인사 필수로 알아야 하는 것" },
            { id: "ask6", text: "디지털화 근황" },
            { id: "ask7", text: "IOT 컨테이너 사업에 대해서" },
            { id: "ask8", text: "남성해운 운임 동향" },
            { id: "ask9", text: "주간회의 특징 및 주요 말씀" },
            { id: "ask10", text: "최근 해운업계 동향" },
            { id: "ask11", text: "디지털화 근황" },
            { id: "ask12", text: "IOT 컨테이너 사업에 대해서" },
            { id: "ask13", text: "남성해운 운임 동향" },
            { id: "ask14", text: "주간회의 특징 및 주요 말씀" },
            { id: "ask15", text: "최근 해운업계 동향" },
            { id: "ask16", text: "디지털화 근황" },
            { id: "ask17", text: "IOT 컨테이너 사업에 대해서" },
            { id: "ask18", text: "남성해운 운임 동향" },
            { id: "ask19", text: "주간회의 특징 및 주요 말씀" },
            { id: "ask20", text: "최근 해운업계 동향" }
        ];
        queries = differentTwentyteenQueries1;

    } else if (testSet === 'sset_20_ask_2') {
        const differentTwentyteenQueries2 = [
            { id: "ask1", text: "타운사의 전략" },
            { id: "ask2", text: "남성해운과 타운사의 차별점" },
            { id: "ask3", text: "남성해운의 수익 구조" },
            { id: "ask4", text: "해운사의 특징과 남성해운이 가진 고유의 특징" },
            { id: "ask5", text: "지난해 매출과 앞으로의 전망" },
            { id: "ask6", text: "남성해운의 AI 추진 과제 현황" },
            { id: "ask7", text: "동영해운과 남성해운의 공통점과 차이점" },
            { id: "ask8", text: "남성해운의 새로운 전략과 먹거리" },
            { id: "ask9", text: "신입사원 채용 계획 및 교육 일정" },
            { id: "ask10", text: "해운업계의 큰 흐름과 현재 남성해운의 판단" },
            { id: "ask11", text: "디지털화 근황" },
            { id: "ask12", text: "IOT 컨테이너 사업에 대해서" },
            { id: "ask13", text: "남성해운 운임 동향" },
            { id: "ask14", text: "주간회의 특징 및 주요 말씀" },
            { id: "ask15", text: "최근 해운업계 동향" },
            { id: "ask16", text: "디지털화 근황" },
            { id: "ask17", text: "IOT 컨테이너 사업에 대해서" },
            { id: "ask18", text: "남성해운 운임 동향" },
            { id: "ask19", text: "주간회의 특징 및 주요 말씀" },
            { id: "ask20", text: "최근 해운업계 동향" }
        ];
        queries = differentTwentyteenQueries2;

    } 
    else if (testSet === 'same100') {
        for (let i = 1; i <= 100; i++) {
            queries.push({ id: `same100-${i}`, text: "디지털 전략" });
        }

    } else if (testSet === 'circulate5x20') {
        const baseQueries = [
            { id: "q1", text: "남성해운 영업 전략" },
            { id: "q2", text: "디지털화 근황" },
            { id: "q3", text: "IOT 컨테이너 사업에 대해서" },
            { id: "q4", text: "주간회의 특징 및 주요 말씀" },
            { id: "q5", text: "해운업계의 경쟁 구도" }
        ];
        for (let k = 1; k <= 20; k++) {
            baseQueries.forEach((q) => {
                queries.push({ id: `circulate${k}-${q.id}`, text: q.text });
            });
        }

    } else if (testSet === 'mixed100') {
        const mixedQueries = [
            { id: "mix-ask1", text: "타운사의 전략" },
            { id: "mix-ask2", text: "남성해운의 중국 시장 동향" },
            { id: "mix-ask3", text: "해운업계의 경쟁 구도" },
            { id: "mix-ask4", text: "디지털화의 최신 트렌드" },
            { id: "mix-ask5", text: "IOT 컨테이너 사업의 전망" },
            { id: "mix-ask6", text: "타운사의 시장 점유율 분석" },
            { id: "mix-ask7", text: "남성해운의 일본 시장 영업 전략" },
            { id: "mix-ask8", text: "해운업계의 글로벌 시장 동향" },
            { id: "mix-ask9", text: "디지털 전환의 성공 사례" },
            { id: "mix-ask10", text: "IOT 기술을 활용한 물류 혁신" },
            { id: "mix-ask11", text: "타운사의 경쟁 우위" },
            { id: "mix-ask12", text: "남성해운의 동남아 시장 발전 가능성" },
            { id: "mix-ask13", text: "해운업계의 기술 혁신 현황" },
            { id: "mix-ask14", text: "디지털 기술이 기업에 미치는 영향" },
            { id: "mix-ask15", text: "IOT와 빅데이터의 결합 사례" },
            { id: "mix-ask16", text: "타운사의 성장 동력" },
            { id: "mix-ask17", text: "남성해운의 미국 시장 진출 전략" },
            { id: "mix-ask18", text: "해운업계의 운임 변동 요인" },
            { id: "mix-ask19", text: "디지털화 도입의 비용 효율성" },
            { id: "mix-ask20", text: "IOT 기술 도입의 장단점" },
            { id: "mix-ask21", text: "타운사의 혁신 사례" },
            { id: "mix-ask22", text: "남성해운의 수익 구조 분석" },
            { id: "mix-ask23", text: "해운업계의 환경 규제 대응" },
            { id: "mix-ask24", text: "디지털 전환 전략 수립 방법" },
            { id: "mix-ask25", text: "IOT 컨테이너의 운영 효율성" },
            { id: "mix-ask26", text: "타운사의 고객 만족도" },
            { id: "mix-ask27", text: "남성해운의 비용 절감 전략" },
            { id: "mix-ask28", text: "해운업계의 디지털 전환" },
            { id: "mix-ask29", text: "디지털화와 빅데이터 활용" },
            { id: "mix-ask30", text: "IOT 기반 실시간 모니터링 시스템" },
            { id: "mix-ask31", text: "타운사의 재무 구조" },
            { id: "mix-ask32", text: "남성해운의 신규 서비스 도입" },
            { id: "mix-ask33", text: "해운업계의 물류 효율성 개선" },
            { id: "mix-ask34", text: "디지털화의 보안 이슈" },
            { id: "mix-ask35", text: "IOT와 인공지능의 결합 효과" },
            { id: "mix-ask36", text: "타운사의 리스크 관리" },
            { id: "mix-ask37", text: "남성해운의 고객 만족도 조사" },
            { id: "mix-ask38", text: "해운업계의 비용 구조" },
            { id: "mix-ask39", text: "디지털 트랜스포메이션의 장단점" },
            { id: "mix-ask40", text: "IOT 기술을 활용한 비용 절감" },
            { id: "mix-ask41", text: "타운사의 해외 진출 전략" },
            { id: "mix-ask42", text: "남성해운의 운임 변동 분석" },
            { id: "mix-ask43", text: "해운업계의 신규 시장 개척" },
            { id: "mix-ask44", text: "디지털화 추진을 위한 조직 문화" },
            { id: "mix-ask45", text: "IOT 컨테이너 사업의 경쟁력" },
            { id: "mix-ask46", text: "타운사의 신규 사업 계획" },
            { id: "mix-ask47", text: "남성해운의 미래 성장 전략" },
            { id: "mix-ask48", text: "해운업계의 고객 서비스 혁신" },
            { id: "mix-ask49", text: "디지털 전환과 고객 경험 혁신" },
            { id: "mix-ask50", text: "IOT 기술 도입을 위한 투자 전략" },
            { id: "mix-ask51", text: "타운사의 브랜드 가치" },
            { id: "mix-ask52", text: "남성해운의 경쟁사 비교" },
            { id: "mix-ask53", text: "해운업계의 글로벌 네트워크" },
            { id: "mix-ask54", text: "디지털화와 인공지능의 결합" },
            { id: "mix-ask55", text: "IOT와 클라우드 컴퓨팅의 연계" },
            { id: "mix-ask56", text: "타운사의 공급망 관리" },
            { id: "mix-ask57", text: "남성해운의 글로벌 네트워크" },
            { id: "mix-ask58", text: "해운업계의 재무 안정성" },
            { id: "mix-ask59", text: "디지털 전환의 글로벌 사례" },
            { id: "mix-ask60", text: "IOT 기반 물류 자동화 사례" },
            { id: "mix-ask61", text: "타운사의 인재 육성 전략" },
            { id: "mix-ask62", text: "남성해운의 혁신 기술 도입" },
            { id: "mix-ask63", text: "해운업계의 기술 투자" },
            { id: "mix-ask64", text: "디지털화 추진 시 장애 요인" },
            { id: "mix-ask65", text: "IOT 기술의 보안 이슈" },
            { id: "mix-ask66", text: "타운사의 사회적 책임 활동" },
            { id: "mix-ask67", text: "남성해운의 운영 효율성" },
            { id: "mix-ask68", text: "해운업계의 공급망 관리" },
            { id: "mix-ask69", text: "디지털화에 따른 비용 절감 효과" },
            { id: "mix-ask70", text: "IOT 컨테이너 사업의 글로벌 동향" },
            { id: "mix-ask71", text: "타운사의 기술 혁신" },
            { id: "mix-ask72", text: "남성해운의 재무 건전성" },
            { id: "mix-ask73", text: "해운업계의 미래 전망" },
            { id: "mix-ask74", text: "디지털 트랜스포메이션의 미래 전망" },
            { id: "mix-ask75", text: "IOT와 데이터 분석의 시너지 효과" },
            { id: "mix-ask76", text: "타운사의 마케팅 전략" },
            { id: "mix-ask77", text: "남성해운의 리스크 관리 전략" },
            { id: "mix-ask78", text: "해운업계의 혁신 사례" },
            { id: "mix-ask79", text: "디지털화 전략 수립 시 고려사항" },
            { id: "mix-ask80", text: "IOT 기술 도입 시 장애 요인" },
            { id: "mix-ask81", text: "타운사의 고객 서비스 개선" },
            { id: "mix-ask82", text: "남성해운의 신규 투자 계획" },
            { id: "mix-ask83", text: "해운업계의 위험 요인" },
            { id: "mix-ask84", text: "디지털 전환이 기업 경쟁력에 미치는 영향" },
            { id: "mix-ask85", text: "IOT 컨테이너의 미래 전략" },
            { id: "mix-ask86", text: "타운사의 경쟁사 비교" },
            { id: "mix-ask87", text: "남성해운의 시장 점유율 변화" },
            { id: "mix-ask88", text: "해운업계의 인재 육성" },
            { id: "mix-ask89", text: "디지털화와 클라우드 컴퓨팅" },
            { id: "mix-ask90", text: "IOT 기술을 활용한 혁신 사례" },
            { id: "mix-ask91", text: "타운사의 시장 성장 전망" },
            { id: "mix-ask92", text: "남성해운의 고객 서비스 전략" },
            { id: "mix-ask93", text: "해운업계의 시장 성장 동력" },
            { id: "mix-ask94", text: "디지털 전환의 성공 요인" },
            { id: "mix-ask95", text: "IOT 기반 스마트 물류 솔루션" },
            { id: "mix-ask96", text: "타운사의 미래 비전" },
            { id: "mix-ask97", text: "남성해운의 미래 비전" },
            { id: "mix-ask98", text: "해운업계의 신기술 도입 현황" },
            { id: "mix-ask99", text: "디지털 트랜스포메이션이 해운업계에 미치는 영향" },
            { id: "mix-ask100", text: "IOT 기술이 해운업계에 미치는 영향" }
        ];
        queries = mixedQueries;
    }

    return queries;
}
