1. conda install pytorch
2. conda install transformers
이렇게 했더니, import가 안되어서 아래와 같이 변경함
conda install conda-forge::transformers
참고 링크 -> https://pypi.org/project/transformers/

3. conda install -c anaconda pillow

4. torch version이 안맞게 설치했는지 이런 오류가 나왔따
AssertionError: Torch not compiled with CUDA enabled

그래서 아래와 같이 버전에 맞게 재 설치를 진행했다 
그전에 python 버전을 3.8이상이어야 한다는 안내가 있어서 그것을 먼저 바꾸어 주었다
https://pytorch.org/get-started/previous-versions/
위 링크를 확인해서 cuda 11.4에 맞는게 없으니 11.3으로 맞춰서 설치했다
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

그렇게 하니까 일단 blip1 은돌아가서 결과가 나온다

---- 
# blip2
위와 같이 하니까 blip2도 결과가 잘 나왔다.
하지만 데이터 셋 생성을 하려고 opencv 를 설치하기 시작하니까
바로 cuda버전이 꼬이는 현상이 발생했음 !
