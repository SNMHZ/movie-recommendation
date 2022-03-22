# Movie Recommendation
## Movie Recommendation Baseline Code

### Installation

```
pip install -r requirements.txt
```

### How to run

1. Pretraining
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py
      ```
3. Inference
   ```
   python inference.py
   ```
   
## 협업 규칙✔️
- ~~프로덕션 코드에 대한 작업은 2주차 수요일(?)부터 진행합니다.~~
- 커밋 메시지 컨벤션은 [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/#specification)을 따릅니다.
   - 커밋 메시지의 첫 문자는 소문자로 통일합니다.
   - ~~커밋 태그에 대한 설명은 다음 [글](https://overcome-the-limits.tistory.com/entry/%ED%98%91%EC%97%85-%ED%98%91%EC%97%85%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EB%B3%B8%EC%A0%81%EC%9D%B8-git-%EC%BB%A4%EB%B0%8B%EC%BB%A8%EB%B2%A4%EC%85%98-%EC%84%A4%EC%A0%95%ED%95%98%EA%B8%B0)을 참고해주세요~~
- 작업할 내용은 미리 Issue를 생성하여 모든 참여자에게 알립니다.
   - 작업은 가설에 대한 실험 결과를 바탕으로 진행됩니다.
   - Issue 제목에는 작업할 내용을 간단히 요약하여 작성합니다
   - Issue에는 작업 내용을 상세히 적어주세요.
      - 예시) 
         - Issue제목 '**BST 모델 추가**'
         - 현재 상황: Linear Regreesion만으로는 Sequential한 데이터의 특성을 모두 학습할 수 없음
         - 가설: 추천시스템에 Transformer 모델을 적용한 BST 모델을 통해 Sequential한 데이터의 특성을 반영하여 추천 성능을 향상시킬 수 있습니다.
         - 실험(가설)에 대한 결과: 기존모델 Recall@10: 0.034 -> BST모델 Recall@10: 0.0422로 성능 향상  
                (이미지)
- 작업은 기본적으로 별도의 브랜치를 생성하여 작업합니다. 작업이 완료되면 PR로 리뷰 받습니다.
   - PR 제목은 되도록 conventinal commit 형태로 만들어주세요.
   - PR에는 `Issue`를 참조해주세요.
   - PR리뷰 후 머지 방식은 Squash & Merge를 따릅니다.
- 머지 이후에는 다음번 실험을 위해 READMD.md의 `How to run`을 수정해주세요
   - 단, 최종 제출 결과가 상승한 경우에만 수정합니다.

## Contributors😎
|김원섭(T3044)|김진수(T3058)|민태원(T3080)|이상목(T3146)|조민재(T3204)|
|:--:|:--:|:--:|:--:|:--:|
|[![](https://avatars.githubusercontent.com/u/83912849?v=4)](https://github.com/whattSUPkim)|[![](https://avatars.githubusercontent.com/u/70852156?v=4)](https://github.com/KimJinSuPKNU)|[![](https://avatars.githubusercontent.com/u/62104797?v=4)](https://github.com/mintaewon)|[![](https://avatars.githubusercontent.com/u/62589993?v=4)](https://github.com/SNMHZ)|[![](https://avatars.githubusercontent.com/u/77037041?v=4)](https://github.com/binyf)|
|[Github](https://github.com/whattSUPkim)|[Github](https://github.com/KimJinSuPKNU)|[Github](https://github.com/mintaewon)|[Github](https://github.com/SNMHZ)|[Github](https://github.com/binyf)|
