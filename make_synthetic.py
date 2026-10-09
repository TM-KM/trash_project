"""
make_synthetic.py - 꽁초 합성 테스트 영상 생성 + butt_miner 결과 채점

[전체 흐름]
0) 흰 배경 꽁초 사진(스튜디오 사진)이 있을 때: 배경 제거해서 PNG 만들기
   python make_synthetic.py cutout --image aaa.jpg --out butt.png

1) (선택) 데이터셋 이미지에서 꽁초만 잘라 합성용 PNG 만들기
   python make_synthetic.py crop --image datasets/butt/images/train/a.jpg --label datasets/butt/labels/train/a.txt --out butt.png

2) 영상 위에 꽁초 합성 (좌표는 원본 영상 해상도 기준, 직접 지정)
   python make_synthetic.py make --video vape_video.mp4 --butt butt.png --out synth/synth.mp4 \
       --place 900,700,24,30,60 --place 500,800,24,45,75
   (x,y,너비px,시작초,끝초)   또는 랜덤 배치:  --auto 5 --seed 1
   -> synth/synth.mp4, synth/synth_gt.json (정답)

3) 합성 영상에 butt_miner 실행
   python butt_miner.py mine --video synth/synth.mp4 --out runs/synth --debug-video runs/synth/debug.mp4 --exclude-person

4) 채점
   python make_synthetic.py evaluate --gt synth/synth_gt.json --candidates runs/synth/candidates
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------ io helpers
def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    """Windows 한글 경로에서도 읽히는 imread. 실패 원인을 구분해서 알려준다."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"파일이 없습니다: {p.resolve()}\n"
                         f"  -> 경로/파일명을 확인하세요. 현재 폴더: {Path.cwd()}")
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise SystemExit(f"파일은 있지만 이미지로 읽을 수 없습니다(손상되었거나 이미지 형식이 아님): {p.resolve()}")
    return img


def imwrite_u(path, img):
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise SystemExit(f"이미지를 저장할 수 없습니다: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


# ------------------------------------------------------------ butt image
def feather_alpha(h, w):
    """가장자리가 부드러워지는 알파 마스크 (배경이 포함된 크롭용)"""
    m = np.ones((h, w), np.float32)
    f = min(h, w) // 6
    if f < 1:
        return m
    m = np.zeros((h, w), np.float32)
    m[f:h - f, f:w - f] = 1.0
    k = f * 2 + 1
    return cv2.GaussianBlur(m, (k, k), f / 2)


def load_butt(path):
    img = imread_u(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:      # 투명 배경 PNG
        return img[:, :, :3], img[:, :, 3].astype(np.float32) / 255.0
    return img, feather_alpha(*img.shape[:2])


def prep_patch(bgr, alpha, width, blur):
    h0, w0 = bgr.shape[:2]
    width = max(2, int(width))
    height = max(2, int(round(h0 * width / w0)))
    b = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    al = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    if blur > 0:               # 영상 화질(압축/번짐)에 맞추기
        k = int(blur * 3) * 2 + 1
        b = cv2.GaussianBlur(b, (k, k), blur)
    return b, al[:, :, None]


# ------------------------------------------------------------ cutout
def cutout(a):
    """흰 배경 사진에서 꽁초만 오려 투명 배경 PNG로 저장 (GrabCut)"""
    img = imread_u(a.image, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    white = (img.min(axis=2) >= a.white_thr).astype(np.uint8)
    _, lab = cv2.connectedComponents(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    mask = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    mask[~bg] = cv2.GC_PR_FGD
    mask[:8, :] = mask[-8:, :] = cv2.GC_BGD
    mask[:, :8] = mask[:, -8:] = cv2.GC_BGD
    cv2.grabCut(img, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab2, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        raise SystemExit("물체를 찾지 못했습니다. 배경이 흰색인 사진인지 확인하세요.")
    fg = (lab2 == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ratio = fg.mean()
    if ratio < 0.01 or ratio > 0.6:
        print(f"[경고] 오려낸 영역이 화면의 {ratio * 100:.0f}%입니다. 결과 PNG를 직접 확인하세요.")
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8))          # 흰 테두리 제거
    alpha = cv2.GaussianBlur(fg * 255, (3, 3), 0.8)
    ys, xs = np.where(fg > 0)
    y0, y1, x0, x1 = max(0, ys.min() - 2), min(H, ys.max() + 3), max(0, xs.min() - 2), min(W, xs.max() + 3)
    out = np.dstack([img[y0:y1, x0:x1], alpha[y0:y1, x0:x1]])
    imwrite_u(a.out, out)
    print(f"저장: {Path(a.out).resolve()}  크기={out.shape[1]}x{out.shape[0]}px")


# ------------------------------------------------------------ crop
def crop(a):
    img = imread_u(a.image, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    lines = [l.split() for l in Path(a.label).read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit("라벨 파일이 비어 있습니다.")
    _, xc, yc, w, h = (float(v) for v in lines[a.index][:5])
    bw, bh = w * W, h * H
    m = max(bw, bh) * a.margin
    x0, y0 = max(0, int(xc * W - bw / 2 - m)), max(0, int(yc * H - bh / 2 - m))
    x1, y1 = min(W, int(xc * W + bw / 2 + m)), min(H, int(yc * H + bh / 2 + m))
    c = img[y0:y1, x0:x1]
    alpha = (feather_alpha(*c.shape[:2]) * 255).astype(np.uint8)
    imwrite_u(a.out, np.dstack([c, alpha]))
    print(f"저장: {Path(a.out).resolve()}  크기={c.shape[1]}x{c.shape[0]}px")


# ------------------------------------------------------------ make
def make(a):
    bgr, alpha = load_butt(a.butt)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {Path(a.video).resolve()}\n"
                         f"  -> 파일 위치/이름 확인. 경로에 한글이 있으면 영문 폴더로 옮겨서 시도하세요.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    specs = []
    for s in a.place or []:
        x, y, w, st, en = (float(v) for v in s.split(","))
        specs.append((int(x), int(y), w, st, en))

    if a.from_gt:   # 이전 정답 파일의 위치/시각을 그대로 쓰고 크기만 배율(scale)로 변경
        for g in json.loads(Path(a.from_gt).read_text())["butts"]:
            gx, gy, gw, gh = g["bbox_xywh"]
            nw = max(2.0, gw * a.scale)
            nh = nw * gh / gw
            specs.append((int(gx + gw / 2 - nw / 2), int(gy + gh / 2 - nh / 2), nw,
                          g["start_sec"], g["end_sec"]))

    if a.auto:
        rnd = random.Random(a.seed)
        rx, ry, rw, rh = ((int(v) for v in a.region.split(",")) if a.region
                          else (0, int(H * 0.4), W, int(H * 0.6)))
        for _ in range(a.auto):
            w = a.width * rnd.uniform(0.8, 1.3)
            st = a.auto_start + rnd.uniform(0, a.auto_spread)
            ph = max(2, int(round(bgr.shape[0] * w / bgr.shape[1])))
            x = rnd.randint(rx, max(rx, rx + rw - int(w) - 1))
            y = rnd.randint(ry, max(ry, ry + rh - ph - 1))
            specs.append((x, y, w, st, st + a.duration))
    if not specs:
        raise SystemExit("--place 또는 --auto 중 하나는 지정해야 합니다.")

    items, gt = [], []
    for i, (x, y, w, st, en) in enumerate(specs, 1):
        b, al = prep_patch(bgr, alpha, w, a.blur)
        ph, pw = b.shape[:2]
        if x < 0 or y < 0 or x + pw > W or y + ph > H:
            raise SystemExit(f"#{i} 위치가 화면({W}x{H})을 벗어납니다: x={x}, y={y}, {pw}x{ph}")
        if st < 25:
            print(f"[경고] #{i} 시작 {st:.0f}초: 배경 수집 구간(기본 약 20초)과 겹치면 배경에 흡수되어 검출되지 않을 수 있습니다.")
        if en - st > 28:
            print(f"[경고] #{i} 유지 {en - st:.0f}초: 기본 배경 설정에서는 30초쯤 지나면 배경에 흡수됩니다.")
        items.append((x, y, b, al, st, en))
        gt.append({"id": i, "bbox_xywh": [x, y, pw, ph],
                   "start_sec": round(st, 2), "end_sec": round(en, 2)})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    idx = 0
    max_frames = int(a.max_sec * fps) if a.max_sec else None
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        t = idx / fps
        for x, y, b, al, st, en in items:
            if st <= t < en:
                ph, pw = b.shape[:2]
                roi = frame[y:y + ph, x:x + pw]
                roi[:] = (al * b + (1 - al) * roi).astype(np.uint8)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()

    gt_path = out.with_name(out.stem + "_gt.json")
    gt_path.write_text(json.dumps(
        {"video": str(out), "fps": fps, "resolution": [W, H], "butts": gt},
        ensure_ascii=False, indent=2))
    print(f"합성 영상: {out}  ({idx}프레임, {idx / fps:.0f}초)")
    print(f"정답 파일: {gt_path}  (꽁초 {len(gt)}개)")


# ------------------------------------------------------------ evaluate
def evaluate(a):
    gt = json.loads(Path(a.gt).read_text())["butts"]
    metas = [json.loads(p.read_text()) for p in sorted(Path(a.candidates).glob("ID*/meta.json"))]

    def near(g, c):
        gx, gy, gw, gh = g["bbox_xywh"]
        cx, cy, cw, ch = c["bbox_xywh"]
        dist = float(np.hypot(cx + cw / 2 - gx - gw / 2, cy + ch / 2 - gy - gh / 2))
        in_time = g["start_sec"] <= c["detected_at_sec"] <= g["end_sec"] + 2
        return dist <= max(a.tol, max(gw, gh)) and in_time

    primary, near_any = set(), set()
    hits = 0
    print(f"정답 {len(gt)}개 / 후보 {len(metas)}개\n")
    for g in gt:
        found = None
        for c in metas:
            if near(g, c):
                near_any.add(c["id"])
                if found is None:
                    found = c
        if found:
            primary.add(found["id"])
            hits += 1
            lat = found["detected_at_sec"] - g["start_sec"]
            print(f"  [검출] 꽁초 #{g['id']}  ID{found['id']:04d}  지연 {lat:.1f}초")
        else:
            print(f"  [놓침] 꽁초 #{g['id']}  bbox={g['bbox_xywh']}  {g['start_sec']}~{g['end_sec']}초")
    dups = [c for c in metas if c["id"] in near_any and c["id"] not in primary]
    extra = [c for c in metas if c["id"] not in near_any]
    print(f"\n검출률(recall): {hits}/{len(gt)} = {hits / len(gt) * 100:.0f}%")
    print(f"같은 꽁초 근처의 중복 후보: {len(dups)}개 (한 꽁초가 조각나서 생긴 후보로 추정, 오탐으로 세지 않음)")
    if dups:
        print("  " + ", ".join(f"ID{c['id']:04d}" for c in dups[:20]))
    print(f"정답 외 후보: {len(extra)}개 (오탐이거나 원본 영상에 원래 있던 정지 물체, 사라진 꽁초 자리)")
    if extra:
        print("  " + ", ".join(f"ID{c['id']:04d}" for c in extra[:20]))
    evaluate.last = (hits, len(gt), len(extra))
    evaluate.last_dup = len(dups)
    if hits < len(gt):
        print("\n놓친 경우 점검: 영상 시작 시점(>=25초), 유지 시간, --min-area, --diff-thr, 사람 제외 영역과 겹침 여부")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crop", help="라벨 bbox로 꽁초만 잘라 PNG(알파) 생성")
    c.add_argument("--image", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--index", type=int, default=0, help="라벨 파일에서 몇 번째 박스인지")
    c.add_argument("--margin", type=float, default=0.3, help="박스 주변 여백 비율")
    c.set_defaults(fn=crop)

    o = sub.add_parser("cutout", help="흰 배경 꽁초 사진 -> 투명 배경 PNG")
    o.add_argument("--image", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--white-thr", type=int, default=235, help="흰 배경으로 볼 밝기 기준")
    o.set_defaults(fn=cutout)

    m = sub.add_parser("make", help="영상에 꽁초 합성")
    m.add_argument("--video", required=True)
    m.add_argument("--butt", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--place", action="append", metavar="x,y,너비,시작초,끝초")
    m.add_argument("--from-gt", default=None, help="이전 정답 json의 위치/시각 재사용")
    m.add_argument("--scale", type=float, default=1.0, help="--from-gt 사용 시 꽁초 크기 배율")
    m.add_argument("--auto", type=int, default=0, help="랜덤 배치 개수")
    m.add_argument("--region", default=None, help="랜덤 배치 영역 x,y,w,h (기본: 화면 아래 60%%)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--width", type=float, default=24, help="랜덤 배치 꽁초 너비(px)")
    m.add_argument("--duration", type=float, default=25, help="랜덤 배치 유지 시간(초)")
    m.add_argument("--auto-start", type=float, default=30)
    m.add_argument("--auto-spread", type=float, default=40)
    m.add_argument("--blur", type=float, default=0.8, help="합성 꽁초 번짐(영상 화질에 맞춤)")
    m.add_argument("--max-sec", type=float, default=0, help="앞부분 N초만 사용 (0이면 전체)")
    m.set_defaults(fn=make)

    e = sub.add_parser("evaluate", help="butt_miner 결과 채점")
    e.add_argument("--gt", required=True)
    e.add_argument("--candidates", required=True)
    e.add_argument("--tol", type=float, default=20, help="위치 허용 오차(px)")
    e.set_defaults(fn=evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()