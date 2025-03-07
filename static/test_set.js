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
    } else if (testSet === 'chatting') {
        const chatQueries = [
            { id: "chat1", text: "남성해운의 디지털 전략 전반적으로 설명해줘." },
            { id: "chat2", text: "그 디지털 전략의 첫 번째 핵심 사항이 뭔지 궁금해." },
            { id: "chat3", text: "그 첫 번째 사항을 좀 더 세부적으로 설명해줄래?" },
            { id: "chat4", text: "두 번째 핵심 사항은 어떤 거야?" },
            { id: "chat5", text: "두 번째 사항도 구체적으로 듣고 싶어." },
            { id: "chat6", text: "남성해운은 디지털 전환을 위해 어떤 시스템을 쓰나요?" },
            { id: "chat7", text: "그 시스템이 도입된 배경은 무엇인지 알려줘." },
            { id: "chat8", text: "시스템의 장점과 단점은 각각 뭐야?" },
            { id: "chat9", text: "구체적으로 어떤 기능들을 제공하나요?" },
            { id: "chat10", text: "디지털 플랫폼 파트너사도 있나요? 있다면 어디인가요?" },
            { id: "chat11", text: "그 파트너사와 협력 시 남성해운이 얻는 이점은?" },
            { id: "chat12", text: "화주 입장에서도 이점이 있나요? 구체적인 사례도 궁금해요." },
            { id: "chat13", text: "화주 사례에서 배운 점이나 개선점이 있으면 말해줘." },
            { id: "chat14", text: "향후 추가로 확장하려는 디지털 영역이 있는지 궁금해." },
            { id: "chat15", text: "AI나 머신러닝 관련 프로젝트도 추진 중인가요?" },
            { id: "chat16", text: "그렇다면 AI 프로젝트 방향은 어떤 식으로 진행되나요?" },
            { id: "chat17", text: "AI 프로젝트의 현재 진행 상태가 어느 정도인지 알고 싶어." },
            { id: "chat18", text: "이전 프로젝트(디지털)와 시너지를 낼 수 있는 부분 있나요?" },
            { id: "chat19", text: "구체적인 출시 일정이나 로드맵도 잡혀있나요?" },
            { id: "chat20", text: "조직 구조가 궁금해요. 디지털 전담 조직 같은 게 있나요?" },
            { id: "chat21", text: "CIO, CTO 등 임원 체계가 어떻게 구성되어 있어?" },
            { id: "chat22", text: "부서 간 협업은 어떤 방식으로 진행되나요?" },
            { id: "chat23", text: "디지털 전환 과정에서 가장 어려웠던 점은 뭐야?" },
            { id: "chat24", text: "그 어려움들을 어떻게 해결했는지 듣고 싶어요." },
            { id: "chat25", text: "그 과정에서 얻은 교훈이 있다면 알려줄래?" },
            { id: "chat26", text: "경영진의 지원이나 의사결정 방식은 어땠나요?" },
            { id: "chat27", text: "직원들의 반응이나 참여도는 어느 정도인가요?" },
            { id: "chat28", text: "사내 교육 프로그램도 운영하고 있나요?" },
            { id: "chat29", text: "교육 프로그램의 구체적인 내용이나 목표가 있으면 알려줘." },
            { id: "chat30", text: "교육 후 실제 현장 적용 사례가 있나요?" },
            { id: "chat31", text: "디지털 전략 실행에 필요한 예산 규모는 어느 정도인지?" },
            { id: "chat32", text: "예산 대비 효과가 충분히 나오고 있는지 궁금해." },
            { id: "chat33", text: "ROI 측정이나 성과 평가 방식은 어떻게 해요?" },
            { id: "chat34", text: "중장기적으로 기대하는 효과는 무엇인가요?" },
            { id: "chat35", text: "디지털화가 전체 해운업계에 어떤 변화를 가져올까?" },
            { id: "chat36", text: "해외 경쟁사 대비 남성해운의 디지털 수준은 어떤 편이야?" },
            { id: "chat37", text: "국내 다른 해운사와 협력하거나 공동 프로젝트도 있나요?" },
            { id: "chat38", text: "공동 플랫폼 같은 건 구축 계획이 없나요?" },
            { id: "chat39", text: "정부 지원이나 규제 문제는 어떻게 대응하고 있나요?" },
            { id: "chat40", text: "정부 과제나 지원 프로그램에 참여한 사례가 있나요?" },
            { id: "chat41", text: "해운 관련 규제들이 디지털화에 어떤 영향 미치는지 궁금해요." },
            { id: "chat42", text: "블록체인 기술도 검토 중인가요?" },
            { id: "chat43", text: "그 외 주목하는 신기술이 있다면?" },
            { id: "chat44", text: "IoT 센서나 트래킹 시스템은 도입했나요?" },
            { id: "chat45", text: "물류 추적 정확도를 높이기 위해 어떤 시도를?" },
            { id: "chat46", text: "선박 운영 효율화를 위한 디지털 혁신은 어떻게?" },
            { id: "chat47", text: "선박 정비나 유지보수에도 데이터 활용하나요?" },
            { id: "chat48", text: "빅데이터 분석팀이나 데이터 사이언티스트가 있나요?" },
            { id: "chat49", text: "데이터 웨어하우스나 레이크를 구축했는지도 궁금해." },
            { id: "chat50", text: "해상 통신 인프라 문제는 없나요? 위성통신 쓰는지?" },
            { id: "chat51", text: "통신 비용 대비 효율은 어떤 편인지 알려줘." },
            { id: "chat52", text: "사이버 보안이나 해킹 위험은 없나요?" },
            { id: "chat53", text: "보안 솔루션이나 정책이 어떻게 적용되는지 궁금해." },
            { id: "chat54", text: "기술 파트너는 주로 어떤 기업들과 협업해요?" },
            { id: "chat55", text: "ERP나 SCM 솔루션은 어떤 걸 사용하나요?" },
            { id: "chat56", text: "외주 개발과 내부 개발 비중은 어느 정도인지?" },
            { id: "chat57", text: "개발자 채용 상황은 어떤가요?" },
            { id: "chat58", text: "인력 유출 문제는 없나요?" },
            { id: "chat59", text: "해운업 특화된 기술 역량이 필요할 텐데, 어떻게 확보하나요?" },
            { id: "chat60", text: "사내 기술 스택(프로그래밍 언어, 프레임워크 등)은 뭔지 궁금." },
            { id: "chat61", text: "클라우드 인프라도 사용하나요? (AWS, Azure, GCP 등)" },
            { id: "chat62", text: "온프레미스 서버랑 혼용 중인지, 전환 중인지?" },
            { id: "chat63", text: "컨테이너 오케스트레이션(K8s) 같은 것도 쓰나요?" },
            { id: "chat64", text: "CI/CD 파이프라인 구축 여부가 궁금해." },
            { id: "chat65", text: "테스트 자동화나 QA 방식은 어떻게 진행되나요?" },
            { id: "chat66", text: "사용자(화주) 피드백 수집 채널은 뭐가 있나요?" },
            { id: "chat67", text: "UI/UX 개선 사례가 있으면 알려줘." },
            { id: "chat68", text: "모바일 앱이나 웹 포털도 제공하나요?" },
            { id: "chat69", text: "e-Service라는 걸 운영한다던데, 어떤 기능이 있지?" },
            { id: "chat70", text: "디지털 포워더와 연계도 하나요?" },
            { id: "chat71", text: "해운 물동량 예측에 AI 활용 계획이 있나요?" },
            { id: "chat72", text: "선박 스케줄이나 정시성 관리도 디지털화했는지?" },
            { id: "chat73", text: "화주가 실시간으로 컨테이너 위치 확인 가능해?" },
            { id: "chat74", text: "문서 업무(서류)도 자동화나 전산화가 잘 되어 있는지?" },
            { id: "chat75", text: "블록체인 기반 스마트 컨트랙트 적용 사례 있나요?" },
            { id: "chat76", text: "비용 절감 효과는 어느 정도로 추정하나요?" },
            { id: "chat77", text: "디지털 전환으로 매출 확대도 기대하나요?" },
            { id: "chat78", text: "내부 구성원의 만족도 조사는 해봤나요?" },
            { id: "chat79", text: "그 조사 결과 중 재밌는 내용이 있나요?" },
            { id: "chat80", text: "차후 개선할 부분이 있다면 뭐가 있을까요?" },
            { id: "chat81", text: "인사이트를 얻기 위해 다른 업계(항공,물류) 벤치마킹도 하나요?" },
            { id: "chat82", text: "벤치마킹 사례 중 인상 깊었던 게 있으면 알려줘." },
            { id: "chat83", text: "RPA(로봇 프로세스 자동화)도 도입했나요?" },
            { id: "chat84", text: "RPA 적용 영역이 궁금해요." },
            { id: "chat85", text: "디지털 전환 전/후로 업무량이나 인력 변화가 있었나요?" },
            { id: "chat86", text: "현장에서 반발은 없었는지도 궁금해." },
            { id: "chat87", text: "클레임 대응이나 사고 처리도 디지털화했나요?" },
            { id: "chat88", text: "이전 대비 처리 속도가 빨라졌는지?" },
            { id: "chat89", text: "선원 관리나 HR 쪽도 디지털로 관리해요?" },
            { id: "chat90", text: "나중에 자율운항선박 같은 것도 염두에 두고 있나요?" },
            { id: "chat91", text: "자율운항 관련 규제나 기술 수준은 어떻다고 보나요?" },
            { id: "chat92", text: "궁극적으로 남성해운은 어떤 디지털 비전을 갖고 있나요?" },
            { id: "chat93", text: "그 비전을 달성하기 위한 로드맵이 궁금해." },
            { id: "chat94", text: "지금까지 달성한 주요 마일스톤이 있다면?" },
            { id: "chat95", text: "앞으로 1~2년 내 달성하고 싶은 목표는?" },
            { id: "chat96", text: "중장기적으로는 5년 후 어떤 모습을 기대하나요?" },
            { id: "chat97", text: "디지털 경쟁력 강화를 위해 추가 투자 계획이 있나요?" },
            { id: "chat98", text: "마지막으로, 전체 전략에서 가장 중요한 포인트는 뭐라 보나요?" },
            { id: "chat99", text: "추가로 참고할만한 내부 자료나 문서가 있으면 알려줘." },
            { id: "chat100", text: "전체적으로 남성해운 디지털 전략을 요약해줄래?" }
        ];
        queries = chatQueries;
    }

    return queries;
}
