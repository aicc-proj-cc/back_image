## 이미지 생성 서버 프로토타입 시작하는 방법

- 1. RabbitMQ 서버 실행 ( 메인 백엔드 서버 )
메세지 큐 사용을 위해 메인 백엔드 서버 ( 프로토 타입 실행 기준 loacl 서버 )가 위치한 PC 에서 RabbitMQ 서버를 실행해야함.  


- 2. Docker 실행
메인 서버 PC 에서 Docker Desktop 실행하여 도커 실행 

- 3. 도커에서 rabbitmq 실행
docker run -d --hostname image-rabbit --name rabbitmq -p 5672:5672 -p 15672:15672 -e RABBITMQ_LOOPBACK_USERS="none" rabbitmq:management  
실 서비스 적용할때는 유저, 비밀번호 새로 만들어서 관리할 때 사용해야함.  
관리 대시보드에 접근 방법(URL) : http://localhost:15672  

- 3-1. 기본 로그인 정보:
사용자 이름: guest  
비밀번호: guest  

- 4. PORT 설정 ( window 기준 방화벽 설정 )
Windows 방화벽에서 RabbitMQ의 5672 포트와 관리 콘솔의 15672 포트를 허용해야 함.  

Windows 방화벽 설정:  
제어판 > 시스템 및 보안 > Windows Defender 방화벽으로 이동.  
고급 설정 > 인바운드 규칙 > 새 규칙을 선택.  
포트를 선택하고 다음을 클릭.  
TCP를 선택하고 특정 로컬 포트에 5672, 15672를 입력.  
연결 허용을 선택하고 다음.  
규칙 이름에 "RabbitMQ"를 입력하고 저장.  

- 5. 외부에서 접근 가능한 IP 확인
외부에서 서버에 연결하려면 서버의 퍼블릭 IP를 확인해야 함.
명령어 : curl ifconfig.me

- 6. 공유기에서도 포트포워딩 진행  
브라우저에서 192.168.0.1 주소로 접속 이후 ID/PW 입력후 포트 포워딩 진행  
이 부분은 자세히 적을수가 없어서 처음 진행해보시는 분은 실행하실 때 저 불러주세요.  

- 7. 구글 코랩에서 백엔드 서버 세팅
코랩 노트북 파일 생성 후 폴더 구조 아래와 같이 만들기 (코랩 local에 파일들 옮기기)
코랩 런타임은 T4 로 설정
back_api.ipynb
requirements.txt
app/main

- 8. 서버 실행
back_api.ipynb 파일 쭉 실행

- 9. 프론트 서버에서 요청 날리고 테스트