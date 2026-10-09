# 담배꽁초 후보 자동 추출 및 역학습 파이프라인

고정 CCTV 영상에서 담배꽁초 후보를 자동으로 모으고, 사람이 검수한 결과를 YOLOv8 재학습 데이터로 바꾸는 도구입니다.
캡스톤디자인 프로젝트 「CCTV 센서화 기반 환경 오염 및 화재 분석 시스템(환경지기)」의 일부입니다.

## 동작 방식

```
고정 CCTV 영상 → 배경 모델링(픽셀별 시간 중앙값) → 배경 차이·움직임 분석
→ 4단계 상태 머신 추적 → 후보 자동 저장(패널·JSON) → 사람 검수 → 학습셋 변환 → YOLOv8 재학습
```

| 단계 | 색 | 의미 |
|---|---|---|
| 1 | 노랑 | 정지 후보 (배경과 다르고 움직임 없음) |
| 2 | 파랑 | 동적 객체 (즉시 제외) |
| 3 | 보라 | 가려짐 (타이머 정지, 유예 후 재개 또는 폐기) |
| 4 | 빨강 | 지속 후보 (설정 시간 이상 정지, 최종 저장) |

오탐 억제: 사람 영역 제외(`--exclude-person`), 주변 밀도 필터(`--neighbor-ratio`), 수동 제외 영역(`--ignore-region`).

## 파일

| 파일 | 역할 |
|---|---|
| `butt_miner.py` | 후보 추출(`mine`)과 검수 결과의 학습셋 변환(`export`) |
| `make_synthetic.py` | 꽁초 합성 테스트 영상 생성, 정답 기록, 채점(`cutout`, `crop`, `make`, `evaluate`) |
| `size_sweep.py` | 꽁초 크기를 줄여 가며 검출 한계를 자동 측정 |

## 설치

```
pip install -r requirements.txt
```

`--exclude-person`을 쓸 때만 `ultralytics`가 필요하고, 첫 실행 때 `yolov8n.pt`(COCO 사전학습)를 내려받습니다.

## 사용법

**1. 후보 추출** (영상은 카메라가 고정되어 있어야 합니다)

```
python butt_miner.py mine --video 영상.mp4 --out runs/결과 --debug-video runs/결과/debug.mp4 --exclude-person --diff-thr 20 --min-area 6 --persist-sec 10
```

결과는 `runs/결과/candidates/ID****/`에 `panel.jpg`(검수용 3x2 패널), `frame.jpg`, `frame.txt`(YOLO 라벨 초안), `meta.json`으로 저장됩니다.
`debug.mp4`를 보면서 노이즈가 반복되는 구역은 `--ignore-region x,y,w,h`로 제외합니다. 이 좌표는 카메라마다 다릅니다.

**2. 검수** : `panel.jpg`를 보고 후보 폴더를 `reviewed/good/`(진짜 꽁초) 또는 `reviewed/noise/`(오탐)로 옮깁니다.

**3. 학습셋 변환** : good은 전체 프레임과 bbox 라벨로, noise는 빈 라벨(Hard Negative)로 저장합니다.

```
python butt_miner.py export --reviewed reviewed --dataset datasets/butt --prefix 영상이름
```

**4. 재학습**

```
yolo train model=best.pt data=data.yaml epochs=50 imgsz=1280
```

## 합성 평가

실제 꽁초 영상이 없어도 검출 성능을 측정할 수 있도록, 고정 카메라 영상 위에 꽁초 사진을 합성하고 정답과 비교합니다.

```
python make_synthetic.py cutout --image 꽁초사진.jpg --out butt.png
python make_synthetic.py make --video 영상.mp4 --butt butt.png --out synth/synth.mp4 --auto 5 --seed 1 --region 0,650,1920,400 --auto-start 25 --auto-spread 60 --duration 25
python butt_miner.py mine --video synth/synth.mp4 --out runs/synth --exclude-person --diff-thr 20 --min-area 6 --persist-sec 10
python make_synthetic.py evaluate --gt synth/synth_gt.json --candidates runs/synth/candidates
python size_sweep.py --video 영상.mp4 --butt butt.png --gt synth/synth_gt.json --scales 1.0 0.6 0.4 --out sweep --exclude-person --diff-thr 20 --min-area 6 --persist-sec 10
```

## 측정 결과 요약 (합성 꽁초, 설정 `--diff-thr 20 --min-area 6 --persist-sec 10`)

| 꽁초 너비 | 영상 1 (단순한 바닥) | 영상 2 (자갈·잡초 바닥) |
|---|---|---|
| 약 23px | - | 5/5 |
| 약 18px | - | 5/5 |
| 약 16px | - | 5/5 |
| 약 14px | 5/5 | 3/5 |
| 약 9px | 2/5 | 2/5 |
| 약 5px | 0/5 | - |

`-`는 미측정입니다. 합성 평가의 오탐은 0개였고, 꽁초가 없는 영상의 오탐은 제외 영역 적용 후 놀이터 영상 19개 → 0개, 1-1 영상 3개 → 0개였습니다.

## 한계

- 모든 검출 평가는 **합성 꽁초 기준**이며, 실제 꽁초 영상으로는 아직 검증하지 못했습니다. 배율마다 꽁초 5개, 영상 2개라 경향을 보는 수준입니다.
- 안정적으로 잡히는 크기는 약 16px 이상입니다(1080p, 5cm 꽁초 가정 시 바닥 폭 약 6m).
- 제외 영역과 임계값은 카메라마다 다시 정해야 합니다.
- 재학습 후 YOLO 성능 변화, 처리 속도와 메모리 사용량은 측정하지 않았습니다.

## 데이터 안내

이 저장소에는 영상, 데이터셋, 학습 가중치를 포함하지 않습니다. 평가에 쓴 영상은 외부 CCTV 데이터이며 이용 조건에 따라 재배포할 수 없습니다.
