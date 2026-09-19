from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

MODELS = {
    "5":  ("yolov5{s}u", "yolov5{s}u.pt"),
    "8":  ("yolov8{s}",  "yolov8{s}.pt"),
    "11": ("yolo11{s}",  "yolo11{s}.pt"),
    "12": ("yolo12{s}",  "yolo12{s}.pt"),
    "26": ("yolo26{s}",  "yolo26{s}.pt"),
}
ORDER = ["5", "8", "11", "12", "26"]

DATA_DIR = Path("mar20_yolo")
DATA_YAML = DATA_DIR / "data.yaml"
PROJECT = Path("runs_mar20").resolve()
METRICS_JSON = Path("mar20_test_metrics.json")


def find_dir(root: Path, *parts: str):
    want = [p.lower() for p in parts]
    best = None
    for d in root.rglob("*"):
        if d.is_dir() and all(w in d.name.lower() for w in want):
            if best is None or len(d.parts) < len(best.parts):
                best = d
    return best


def discover(root: Path) -> dict:
    images = find_dir(root, "jpegimages") or find_dir(root, "images")
    if images is None:
        sys.exit("JPEGImages klasoru bulunamadi: %s" % root)

    hbb = find_dir(root, "horizontal")
    if hbb is None:
        ann = find_dir(root, "annotation")
        if ann is None:
            sys.exit("Anotasyon klasoru bulunamadi: %s" % root)
        subs = [d for d in ann.iterdir() if d.is_dir() and "orient" not in d.name.lower()]
        hbb = subs[0] if subs else ann

    main = find_dir(root, "main") or find_dir(root, "imagesets")
    sets = {}
    if main is not None:
        for split in ("train", "test", "val"):
            f = main / ("%s.txt" % split)
            if f.exists():
                sets[split] = [ln.strip() for ln in
                               f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {"images": images, "hbb": hbb, "sets": sets}


def parse_voc(xml_path: Path, img_path: Path | None = None):
    r = ET.parse(xml_path).getroot()
    size = r.find("size")
    w = int(float(size.findtext("width"))) if size is not None else 0
    h = int(float(size.findtext("height"))) if size is not None else 0
    if (w <= 0 or h <= 0) and img_path is not None:
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
    out = []
    for obj in r.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if not name or bb is None:
            continue
        out.append((name,
                    float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                    float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
    return w, h, out


def to_yolo(box, w, h):
    _, x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1e-6 or bh <= 1e-6:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw / w, bh / h)


def class_sort_key(n: str):
    if n[:1].upper() == "A" and n[1:].isdigit():
        return (0, int(n[1:]), "")
    return (1, 0, n)


def stratified_split(stems, per_img, ratios, seed):
    rng = random.Random(seed)
    buckets = collections.defaultdict(list)
    for stem in stems:
        boxes = per_img[stem][2]
        major = collections.Counter(b[0] for b in boxes).most_common(1)
        buckets[major[0][0] if major else "__bos__"].append(stem)

    r_tr, r_va, r_te = ratios
    total = r_tr + r_va + r_te
    out = {"train": [], "valid": [], "test": []}
    for _, group in sorted(buckets.items()):
        rng.shuffle(group)
        n = len(group)
        n_tr = round(n * r_tr / total)
        n_va = round(n * r_va / total)
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(1, min(n_va, n - n_tr - 1))
        out["train"] += group[:n_tr]
        out["valid"] += group[n_tr:n_tr + n_va]
        out["test"] += group[n_tr + n_va:]
    for k in out:
        rng.shuffle(out[k])
    return out


def dhash(path: Path, size: int = 8) -> int:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        px = list(im.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def measure_leakage(splits, imgs, max_dist: int = 5) -> None:
    print("\nBolmeler arasi yakin-kopya taramasi (dHash, Hamming <= %d)..." % max_dist)
    hashes = {sp: [(s, dhash(imgs[s])) for s in stems] for sp, stems in splits.items()}

    pairs = [("train", "test"), ("train", "valid"), ("valid", "test")]
    total_flagged = 0
    for a, b in pairs:
        hit, seen_b = 0, set()
        for _, ha in hashes[a]:
            for sb, hb in hashes[b]:
                if bin(ha ^ hb).count("1") <= max_dist:
                    hit += 1
                    seen_b.add(sb)
                    break
        pct = 100 * len(seen_b) / max(1, len(hashes[b]))
        total_flagged += len(seen_b)
        print("  %-6s -> %-6s : %s bolmesindeki %d/%d goruntu (%.1f%%) diger bolmede "
              "yakin-kopyaya sahip" % (a, b, b, len(seen_b), len(hashes[b]), pct))

    if total_flagged:
        print("  UYARI: yakin-kopya sizintisi var; test skorlari iyimser okunmali.")
    else:
        print("  Temiz: esik dahilinde bolmeler arasi yakin-kopya bulunamadi.")


def prepare(root: Path, out: Path, val_frac: float, seed: int,
            split_mode: str = "7:2:1") -> None:
    print("\n" + "=" * 70)
    print("1) VERI HAZIRLIGI")
    print("=" * 70)

    d = discover(root)
    print("Goruntuler : %s" % d["images"])
    print("HBB anotasy: %s" % d["hbb"])
    print("ImageSets  : %s" % {k: len(v) for k, v in d["sets"].items()})

    xmls = {p.stem: p for p in d["hbb"].glob("*.xml")}
    imgs = {p.stem: p for p in d["images"].iterdir() if p.suffix.lower() in IMG_EXT}

    per_img, counts = {}, collections.Counter()
    n_fixed = 0
    for stem, xp in xmls.items():
        if stem not in imgs:
            continue
        w, h, boxes = parse_voc(xp, imgs[stem])
        if ET.parse(xp).getroot().find("size").findtext("width") in ("0", "0.0"):
            n_fixed += 1
        per_img[stem] = (w, h, boxes)
        for b in boxes:
            counts[b[0]] += 1
    if n_fixed:
        print("UYARI: %d XML'de <size> 0x0 idi; boyut goruntuden okundu." % n_fixed)

    names = sorted(counts, key=class_sort_key)
    cls_id = {n: i for i, n in enumerate(names)}
    print("\nEslesen goruntu: %d | sinif: %d | toplam nesne: %d"
          % (len(per_img), len(names), sum(counts.values())))

    official_train = [s for s in d["sets"].get("train", []) if s in per_img]
    official_test = [s for s in d["sets"].get("test", []) if s in per_img]
    if official_train and official_test:
        print("Resmi bolme : train %d / test %d (referans)"
              % (len(official_train), len(official_test)))

    if split_mode == "official":
        if not official_train or not official_test:
            sys.exit("Resmi listeler okunamadi (ImageSets/Main).")
        sub = stratified_split(official_train, per_img,
                               (1.0 - val_frac, val_frac, 0.0), seed)
        splits = {"train": sub["train"], "valid": sub["valid"], "test": official_test}
        print("Bolme modu  : official (resmi test korundu)")
    else:
        ratios = tuple(float(x) for x in split_mode.split(":"))
        if len(ratios) != 3 or sum(ratios) <= 0:
            sys.exit("--split 'train:val:test' seklinde olmali, ornek 7:2:1")
        splits = stratified_split(sorted(per_img, key=lambda s: (len(s), s)),
                                  per_img, ratios, seed)
        print("Bolme modu  : %s (tum havuz yeniden bolundu, sinif-dengeli, seed %d)"
              % (split_mode, seed))

    src_resolved = str(root.resolve()).lower()
    out_resolved = str(out.resolve()).lower()
    if out_resolved == src_resolved or src_resolved.startswith(out_resolved + "\\") \
            or src_resolved.startswith(out_resolved + "/"):
        sys.exit("Cikti klasoru (%s) kaynak MAR20 klasorunu (%s) kapsiyor; "
                 "--out ile farkli bir ad ver." % (out, root))

    if out.exists():
        shutil.rmtree(out)
    for sp in splits:
        (out / sp / "images").mkdir(parents=True)
        (out / sp / "labels").mkdir(parents=True)

    stats = {}
    for sp, stems in splits.items():
        n_obj, seen = 0, collections.Counter()
        for stem in stems:
            w, h, boxes = per_img[stem]
            src = imgs[stem]
            shutil.copy2(src, out / sp / "images" / src.name)
            lines = []
            for b in boxes:
                yb = to_yolo(b, w, h)
                if yb is None:
                    continue
                lines.append("%d %.6f %.6f %.6f %.6f" % (cls_id[b[0]], *yb))
                seen[b[0]] += 1
            (out / sp / "labels" / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_obj += len(lines)
        stats[sp] = (len(stems), n_obj, len(seen))

    DATA_YAML.write_text(
        "# MAR20 - split: %s (seed %d)\n" % (split_mode, seed)
        + "path: %s\n" % out.resolve().as_posix()
        + "train: train/images\nval: valid/images\ntest: test/images\n\n"
        + "nc: %d\n" % len(names)
        + "names: [%s]\n" % ", ".join("'%s'" % n for n in names),
        encoding="utf-8")

    total_img = sum(v[0] for v in stats.values())
    print("\n%-7s %9s %9s %10s %16s" % ("Bolme", "Goruntu", "Oran", "Nesne", "Destekli sinif"))
    print("-" * 56)
    for sp in ("train", "valid", "test"):
        n_img, n_obj, n_cls = stats[sp]
        print("%-7s %9d %8.1f%% %10d %16s"
              % (sp, n_img, 100 * n_img / total_img, n_obj, "%d/%d" % (n_cls, len(names))))
    print("-" * 56)
    print("data.yaml -> %s" % DATA_YAML.resolve())

    measure_leakage(splits, imgs)

    print("\nSinif dagilimi (tum set):")
    for n in names:
        print("  %-5s %6d" % (n, counts[n]))


def train_all(keys, scale, epochs, batch, imgsz, seed, device, workers):
    from ultralytics import YOLO

    shared = dict(
        data=str(DATA_YAML), epochs=epochs, imgsz=imgsz, batch=batch,
        seed=seed, deterministic=True, pretrained=True, patience=100,
        workers=workers, amp=True, plots=True, verbose=True,
        project=str(PROJECT), device=device,
    )

    durations = {}
    for i, key in enumerate(keys, 1):
        name = MODELS[key][0].format(s=scale)
        weights = MODELS[key][1].format(s=scale)
        print("\n" + "=" * 70)
        print("2.%d) EGITIM  %s   (agirlik: %s)   [%d/%d]" % (i, name, weights, i, len(keys)))
        print("=" * 70)

        t0 = time.time()
        YOLO(weights).train(name=name, exist_ok=True, **shared)
        durations[name] = time.time() - t0
        print("[BITTI] %s  %.1f dk  ->  %s"
              % (name, durations[name] / 60, PROJECT / name / "weights" / "best.pt"))

    if durations:
        print("\nEgitim sureleri:")
        for n, s in durations.items():
            print("  %-12s %7.2f dk" % (n, s / 60))
    return durations


def test_all(keys, scale, imgsz, batch, device):
    from ultralytics import YOLO

    results = {}
    for i, key in enumerate(keys, 1):
        name = MODELS[key][0].format(s=scale)
        w = PROJECT / name / "weights" / "best.pt"
        if not w.exists():
            print("[ATLANDI] %s: %s yok" % (name, w))
            continue

        print("\n" + "=" * 70)
        print("3.%d) TEST  %s" % (i, name))
        print("=" * 70)

        r = YOLO(str(w)).val(
            data=str(DATA_YAML), split="test", imgsz=imgsz, batch=batch,
            device=device, workers=0,
            plots=True, verbose=True,
            project=str(PROJECT), name="test_%s" % name, exist_ok=True,
        )

        d = {k: float(v) for k, v in r.results_dict.items()}
        d["classes_with_support"] = int(len(r.box.ap_class_index))
        d["per_class_ap50"] = {r.names[c]: float(r.box.ap50[j])
                               for j, c in enumerate(r.box.ap_class_index)}
        d["per_class_ap50_95"] = {r.names[c]: float(r.box.ap[j])
                                  for j, c in enumerate(r.box.ap_class_index)}
        results[name] = d

    return results


def report(results, durations):
    if not results:
        print("\nDegerlendirilecek model yok.")
        return

    merged = {}
    if METRICS_JSON.exists():
        try:
            merged = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        except ValueError:
            merged = {}
    for name, d in results.items():
        d["train_minutes"] = round(durations.get(name, float("nan")) / 60, 2) \
            if name in durations else merged.get(name, {}).get("train_minutes")
        merged[name] = d
    METRICS_JSON.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print("\n\n" + "=" * 78)
    print("MAR20 TEST BOLMESI SONUCLARI")
    print("=" * 78)
    print("%-12s%12s%11s%11s%15s%12s"
          % ("Model", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Egitim(dk)"))
    print("-" * 78)
    for name, d in results.items():
        tm = d.get("train_minutes")
        print("%-12s%11.2f%%%10.2f%%%10.2f%%%14.2f%%%12s" % (
            name,
            d["metrics/precision(B)"] * 100,
            d["metrics/recall(B)"] * 100,
            d["metrics/mAP50(B)"] * 100,
            d["metrics/mAP50-95(B)"] * 100,
            ("%.2f" % tm) if tm else "-",
        ))
    print("=" * 78)
    print("Tam hassasiyet + sinif bazli AP -> %s" % METRICS_JSON.resolve())


def main():
    ap = argparse.ArgumentParser(description="MAR20 YOLO baseline karsilastirmasi")
    ap.add_argument("--mar20-root", type=Path, default=Path("MAR20"),
                    help="acilmis resmi MAR20 klasoru")
    ap.add_argument("--models", nargs="+", default=ORDER, choices=ORDER,
                    help="egitilecek YOLO surumleri (varsayilan: hepsi, bu sirayla)")
    ap.add_argument("--scale", default="n", choices=list("nsmlx"),
                    help="model olcegi: n=nano (varsayilan), s=small, ...")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="7:2:1",
                    help="train:val:test orani (varsayilan 7:2:1), veya "
                         "'official' -> resmi MAR20 test listesini koru")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="yalnizca --split official icin: resmi train'den val orani")
    ap.add_argument("--only-prepare", action="store_true")
    ap.add_argument("--force-prepare", action="store_true",
                    help="mar20/ zaten varsa bile yeniden uret")
    ap.add_argument("--skip-train", action="store_true", help="egitimi atla, sadece test et")
    a = ap.parse_args()

    keys = [k for k in ORDER if k in set(a.models)]

    if a.force_prepare or not DATA_YAML.exists():
        if not a.mar20_root.exists():
            sys.exit("MAR20 klasoru yok: %s  (--mar20-root ile yolu ver)" % a.mar20_root)
        prepare(a.mar20_root, DATA_DIR, a.val_frac, a.seed, a.split)
    else:
        print("%s zaten var, hazirlik atlandi (--force-prepare ile yeniden uret)" % DATA_YAML)

    if a.only_prepare:
        return

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print("\nCihaz: %s" % ("GPU cuda:0" if device == 0 else "CPU (cok yavas olacak)"))
    print("Sira : %s" % " -> ".join(MODELS[k][0].format(s=a.scale) for k in keys))

    durations = {}
    if not a.skip_train:
        durations = train_all(keys, a.scale, a.epochs, a.batch, a.imgsz,
                              a.seed, device, a.workers)

    results = test_all(keys, a.scale, a.imgsz, a.batch, device)
    report(results, durations)


if __name__ == "__main__":
    main()
