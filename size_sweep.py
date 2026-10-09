"""
size_sweep.py - 꽁초 크기를 줄여 가며 검출 한계 찾기

이전에 만든 정답 파일(synth_gt.json)의 위치/시각은 그대로 두고, 꽁초 크기만 배율로 바꿔서
[합성 -> butt_miner 실행 -> 채점]을 반복한다.

사용 예 (--video는 합성 전 '원본' 영상):
  python size_sweep.py --video vape_video.mp4 --butt butt.png --gt synth\\synth_gt.json --scales 1.0 0.6 0.4 0.25 \\
      --out sweep --exclude-person --ignore-region 0,0,1920,400 --persist-sec 10

위에서 정의하지 않은 옵션(--exclude-person 등)은 모두 butt_miner.py mine 에 그대로 전달된다.
"""
import argparse
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_synthetic as ms  # noqa: E402


def run(cmd):
    print("\n>", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"명령 실패(코드 {r.returncode}). 위 오류 메시지를 확인하세요.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="합성 전 원본 영상")
    p.add_argument("--butt", required=True, help="cutout으로 만든 꽁초 PNG")
    p.add_argument("--gt", required=True, help="이전 합성의 정답 json (위치/시각 재사용)")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.6, 0.4, 0.25])
    p.add_argument("--out", default="sweep")
    p.add_argument("--tol", type=float, default=20)
    a, miner_args = p.parse_known_args()

    rows = []
    for sc in a.scales:
        d = Path(a.out) / f"scale_{sc}"
        video, gt = d / "synth.mp4", d / "synth_gt.json"
        run([sys.executable, HERE / "make_synthetic.py", "make", "--video", a.video, "--butt", a.butt,
             "--out", video, "--from-gt", a.gt, "--scale", sc])
        run([sys.executable, HERE / "butt_miner.py", "mine", "--video", video,
             "--out", d / "runs", *miner_args])
        print(f"\n=== 채점: 배율 {sc} ===")
        ms.evaluate(Namespace(gt=str(gt), candidates=str(d / "runs" / "candidates"), tol=a.tol))
        hits, total, extra = ms.evaluate.last
        widths = [b["bbox_xywh"][2] for b in json.loads(gt.read_text())["butts"]]
        rows.append((sc, sum(widths) / len(widths), hits, total, extra))

    print("\n" + "=" * 52)
    print(f"{'배율':>6} {'평균 너비(px)':>14} {'검출':>8} {'정답 외 후보':>12}")
    for sc, w, h, t, e in rows:
        print(f"{sc:>6} {w:>14.1f} {f'{h}/{t}':>8} {e:>12}")
    print("=" * 52)
    print("검출이 떨어지기 시작하는 너비가 현재 설정의 검출 한계입니다.")


if __name__ == "__main__":
    main()
