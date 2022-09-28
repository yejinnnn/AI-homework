# AI-homework

## 연합학습
![image](https://user-images.githubusercontent.com/48613073/192774714-60dcfb52-406a-4d04-9b5d-16298793a011.png)

* 데이터를 개개인의 로컬 클라이언트에 두고 그 로컬 클라이언트에서 학습을 수행한 후, 업데이트된 모델의 파라미터들을 중앙 서버로 보내 취합해서 하나의 모델을 업데이트 하는 것
* 데이터가 아닌 학습을 통해 도출된 가중치들만 중앙 서버로 전송이 됨

  * 데이터 유출 위험이 줄고 커뮤니케이션 효율성이 올라감
***
## Femnist 데이터를 이용하여 연합학습 진행
* Femnist 필기체 데이터(숫자, 소문자, 대문자)
* 데이터 개수: 637,877개
![image](https://user-images.githubusercontent.com/48613073/192770739-20e9a75a-570e-4e7a-8170-6d372ce8f429.png)


## 훈련 모델
![image](https://user-images.githubusercontent.com/48613073/192770583-1e1a263c-4197-457e-ab0e-b84dea0217e0.png)


## server.py
![image](https://user-images.githubusercontent.com/48613073/192772045-0b7522a7-60ff-4016-9f7e-7bc202c87409.png)

## client.py 실행
    INFO flower 2022-09-28 20:54:21,138 | connection.py:102 | Opened insecure gRPC connection (no certificates were passed)
    INFO flower 2022-09-28 20:54:21,138 | connection.py:102 | Opened insecure gRPC connection (no certificates were passed)
    DEBUG flower 2022-09-28 20:54:21,139 | connection.py:39 | ChannelConnectivity.IDLE
    DEBUG flower 2022-09-28 20:54:21,139 | connection.py:39 | ChannelConnectivity.IDLE
    DEBUG flower 2022-09-28 20:54:21,139 | connection.py:39 | ChannelConnectivity.CONNECTING
    DEBUG flower 2022-09-28 20:54:21,139 | connection.py:39 | ChannelConnectivity.READY
    DEBUG flower 2022-09-28 20:54:21,139 | connection.py:39 | ChannelConnectivity.READY
    INFO flower 2022-09-28 20:54:21,140 | connection.py:102 | Opened insecure gRPC connection (no certificates were passed)
    DEBUG flower 2022-09-28 20:54:21,141 | connection.py:39 | ChannelConnectivity.IDLE
    DEBUG flower 2022-09-28 20:54:21,141 | connection.py:39 | ChannelConnectivity.CONNECTING
    DEBUG flower 2022-09-28 20:54:21,142 | connection.py:39 | ChannelConnectivity.READY
    
    number : 30, subject number : 60
    number : 10, subject number : 57
    number : 22, subject number : 27
    number : 7, subject number : 83
    number : 9, subject number : 8
    number : 11, subject number : 84
    number : 32, subject number : 73
    number : 20, subject number : 65
    .
    .
    .
    

