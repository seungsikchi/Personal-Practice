Personal - Practice 개인적으로 만들어본 프로그램을 올리는 Repository 입니다

animalclassfication.py

첫번째 커밋
제가 스스로 모은 데이터 520장을 X데이터로 분류하고 동물별로 숫자를 붙여서 Y데이터에 넣어서 동물을 분류해내는 프로그램을 만들었습니다
이미지 불러오는 라이브러리는 Opencv2를 사용하였고 이미지를 PIL라이브러리로 자르고 GRAYSCALE로 저장해서 X데이터에 집어넣었습니다 
이미지 데이터에 따로 Labeling은 하지 않았습니다.

두번쨰 커밋
여러가지 기능을 함수로 지정을 해서 정리했고 Opencv2로 불러왔던 이미지를 PIL, OS, glob를 활용하여 한번에 불러왔습니다 
이미지를 불러오는 과정에 같이 y데이터에 이전 커밋에 했던 방식대로 숫자를 지정해서 하는방법으로 설정하여 y데이터를 설정했습니다
Model의 레이어는 동일합니다.

세번째 커밋
Tensorflow 2.0에서 배운 K-평균 클러스터링을 활용하여 분류했던 이미지를 동물 별로 같은 동물을 나타내는지 확인해봤습니다.
데이터의 수도 많지 않고 Labelling도 따로 하지 않아서 정확도가 그렇게 좋지 않았습니다.

네번째 커밋
Tensorflow 2.0에서 배운 TSNE 알고리즘을 활용해서 이미지가 분류되는 것을 좀 더 편하게 확인을 하기 위해 TSNE 알고리즘을 추가해봤습니다.


Delete Background
이미지에서 배경만 삭제하기 위해서 여러가지 필터에 넣어보고 물체와 배경의 경계선을 검출해내고 물체 바깥의 프로그램을 검은색으로 지정되어 있는 마스크를 위에 씌워서 물체를 제외한 나머지 배경을 검은색으로 만들어내는 프로그램입니다. =>마스크를 안 씌울 좌표를 지정해야되서 수정필요 => 경계선을 자동으로 찾아내서 그 밖에 색깔은 0으로 지정하고 물체 안에는 색깔을 그대로 유지하는 코드 

CNN 신경망 정리
CNN의 여러가지신경망들을 나중에 공부하기 편하게 정리하였습니다.

파일이름 camera
폰에 내장되어있는 카메라를 불러오는 컴퓨터로 불러오는 파일입니다.
