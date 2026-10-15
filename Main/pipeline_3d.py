"""Skördepipeline för äpplen med djupdata (RGB-D-video från en Orbbec Gemini 2).

Steg:
  1. Detektera äpplen i varje bildruta (fine-tuned YOLO) och ta bort dubbla rutor på samma äpple.
  2. Avstånd till varje äpple från djupbilden. Äpplen längre bort än MAX_DEPTH_MM (3,0 m) är bakgrund (fat, hinkar,
     krattor, nästa träd) och räknas inte. Detektioner utan djupvärde behålls.
  3. Kamerarörelsen mellan bildrutor mäts med hjälp av äpplena själva (den förflyttning flest äppelpar är överens om).
     När kameran åker i sidled rör sig nära äpplen snabbare över bilden än äpplen långt bort. Uppmätt på dessa
     inspelningar: äpplets rörelse = 1 + beta x (trädets avstånd / äpplets avstånd - 1), beta ungefär 0,8.
  4. Räkning med strip sampling: var K:e bildruta räknar bara äpplena i sin egen remsa av scenen, och varje äpples remsa
     skalas med dess avstånd. Remsorna täcker scenen, så varje äpple räknas ungefär en gång (0-1 dubbelräkningar per
     passage). Inom de handmarkerade vyerna stämmer antalet med handräkningen inom 0-7 %. Antalet är medelvärdet över
     flera K och startbildrutor; spridningen ger det rimliga intervallet.
  5. Storlek: äpplen som är helt synliga i flera bildrutor (hög confidence, runda, jämn storlek, inget framför dem i
     djupbilden) mäts i mm från rutans storlek och avståndet.
  6. Kalibrering: ett handvägt referensäpple (109 g, 7,0 x 6,5 x 6,0 cm, medelstort på träd 1) bestämmer
     storleksskalan. Skalan anpassas på de träd 1-inspelningar som inte används för test (--calibrate) och används
     sedan på alla inspelningar. Skörd = antal äpplen x medelvikten för inspelningens helt synliga äpplen.

Användning (från mappen Main):
    python pipeline_3d.py --calibrate      # en gång: anpassar storleksskalan -> calibration/calibration.json
    python pipeline_3d.py ../Archive/test_images ../Archive/Tree2 ../Archive/1dceb275-6e92-4368-9df5-1526972df094

Skriver ../Results/pipeline_3d/<inspelning>/: annotated_3d.mp4 (slutar med sammanfattningen), audit_numbered.mp4, final.png,
measured_apples.png, apples_measured.csv, audit_apples.csv, counts.csv, detections.csv och summary.json. Lägg till
--reuse för att hoppa över detektionen när detections.csv redan finns (bara ritningen och siffrorna görs om).

Namn som används genomgående:
  detections   tabell med en rad per detekterad äppelruta: frame, conf, x1, y1, x2, y2, cx, cy, size (px), z_mm
               (avstånd), z_source ("apple" = eget djup, "around" = omgivningen, "none"), occluded, diameter_raw_mm,
               far (bortom MAX_DEPTH_MM) och m (rörelsefaktor)
  shift        (n_frames, 2)-array: kamerans sammanlagda förflyttning i px; position i scenen = position i bilden + shift
  spacing      K: var K:e bildruta är en utvald bildruta
  offset       index för den första utvalda bildrutan
  result       dict från analyse(): allt som är känt om en inspelning
  summary      dict från summarize(): siffrorna som skrivs till summary.json
  calibration  dict från calibration/calibration.json (storleksskala och dess intervall)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import visualize
from visualize import num

WEIGHTS = "models/apple_run1.pt"  # run 1: fine-tuned YOLO11m på 7 bildrutor från träd 1 (Orbbec)
APPLE_CLASS = 47     # "apple" bland COCO-klasserna; de fine-tunade modellerna behåller numret
IMGSZ = 1920
CONF_LOW = 0.2       # detektioner som används för att följa ett äpple mellan bildrutor
CONF_KEEP = 0.45     # detektioner som räknas (fine-tuned modell)
NESTED_IOMIN = 0.5   # en ruta som har mer än denna andel av sin yta inuti en starkare ruta är en dubblett på samma äpple
MAX_DEPTH_MM = 3000  # längre bort än så är bakgrund
DEPTH_READ_MM = 10000  # djup bortom detta litar vi inte alls på
MAX_SHIFT = 100      # px: största kamerarörelse mellan två bildrutor
SHIFT_BIN = 4        # px: upplösning i röstningen om kamerarörelsen
COUNT_ERROR = 0.10   # räknefel per bildruta på de handmarkerade testbildrutorna (7 % med avståndsgränsen), avrundat uppåt
SPACINGS = (3, 5, 8)  # var K:e bildruta; 4 startbildrutor vardera
OUT = Path("../Results/pipeline_3d")  # resultat (videor, bilder, tabeller)
CALIBRATION_DIR = Path("calibration")  # storlekskalibreringen som pipelinen läser
ARCHIVE = Path("../Archive")

# Referensäpple, handmätt på träd 1: 7,0 x 6,5 x 6,0 cm, 109 g, "medelstorlek för äpplena på trädet"
REF_DIAMETER_MM = 65.0
REF_MASS_G = 109.0
DENSITY = REF_MASS_G / ((np.pi / 6) * (REF_DIAMETER_MM / 10) ** 3)  # g/cm3 för en sfär med medeldiametern
# Träd 1-inspelningar som storleksskalan anpassas på. test_images (träd 1) och Tree2 + 1dceb275 (båda träd 2) är test.
CALIBRATION_RECORDINGS = ["1aa65ece-d32f-4254-bcd3-80b26d2db7b9", "341f04ca-054a-4e52-ae96-139378821ea8",
                          "5fa2faba-e6d8-410e-803e-a52e8a0d9089", "849d204a-348a-4494-b9c9-a9e0d3903359",
                          "Annotation_tetst", "d4d77803-a829-42b5-8eb2-a4c86f31e99a"]
TREE_OF = {"test_images": "träd 1", "Tree2": "träd 2", "1dceb275-6e92-4368-9df5-1526972df094": "träd 2"}

# "Helt synligt": följt över flera bildrutor, hög confidence, runt, jämn storlek, inget framför det i djupbilden
VIS_WINDOW = 6           # bildrutor före och efter den räknade bildrutan
VIS_MIN_VIEWS = 7
VIS_MIN_CONF = 0.6
VIS_MAX_ASPECT = 1.2
VIS_MAX_SIZE_CV = 0.10   # rutans storlek får variera högst 10 % mellan vyerna (ett löv som rör sig framför ändrar storleken)
VIS_MIN_SIZE_PX = 16
VIS_MIN_DEPTH_VIEWS = 4
VIS_MAX_DEPTH_SPREAD = 0.06  # avståndet får variera högst 6 % mellan vyerna
OCCLUDER_MM = 60         # djuppixlar så här mycket framför äpplets yta tillhör något som skymmer det
VIS_MAX_OCCLUDED = 0.15

MARKED_FRAMES = [("test_images", 15, "annotations/test_images_000015_points.csv"),
                 ("Tree2", 11, "annotations/Tree2_000011_points.csv"),
                 ("1aa65ece-d32f-4254-bcd3-80b26d2db7b9", 23, "annotations/1aa65e_000023_points.csv")]


def sphere_mass_g(diameter_mm):
    """Vikten för ett äpple med denna diameter, som en sfär med referensäpplets densitet."""
    return DENSITY * (np.pi / 6) * (diameter_mm / 10) ** 3


# ---------------------------------------------------------------- djup

def align_depth_to_color(depth_mm, meta, z_min=300, z_max=DEPTH_READ_MM):
    """Reprojection av en Orbbec-djupbild till färgbilden (djupkameran och färgkameran sitter inte på samma ställe).

    Parametrar:
        depth_mm: rå djupbild, uint16 i mm, i djupkamerans pixlar.
        meta: inspelningens meta.json (intrinsics för djup och färg, extrinsics djup -> färg).
        z_min, z_max: djupvärden utanför detta intervall (mm) ignoreras.

    Returnerar:
        float32-array lika stor som färgbilden, avstånd i mm; NaN där ingen djuppunkt hamnade.
    """
    depth_intr, color_intr = meta["depth_intrinsics"], meta["color_intrinsics"]
    extrinsics = meta["depth_to_color_extrinsics"]
    rotation = np.array(extrinsics["rotation"], float).reshape(3, 3)
    translation = np.array(extrinsics["translation_mm"], float)
    v, u = np.nonzero((depth_mm > z_min) & (depth_mm < z_max))
    z = depth_mm[v, u].astype(np.float32)
    points = np.stack([(u - depth_intr["cx"]) * z / depth_intr["fx"],
                       (v - depth_intr["cy"]) * z / depth_intr["fy"], z], axis=1) @ rotation.T + translation
    u_color = np.round(points[:, 0] * color_intr["fx"] / points[:, 2] + color_intr["cx"]).astype(int)
    v_color = np.round(points[:, 1] * color_intr["fy"] / points[:, 2] + color_intr["cy"]).astype(int)
    inside = (u_color >= 0) & (u_color < color_intr["width"]) & (v_color >= 0) & (v_color < color_intr["height"])
    out = np.full((color_intr["height"], color_intr["width"]), np.inf, np.float32)
    np.minimum.at(out, (v_color[inside], u_color[inside]), points[inside, 2])  # närmaste yta vinner
    out[np.isinf(out)] = np.nan
    return out


def fill_holes(depth, k=5):
    """Fyll de små luckor som reprojection lämnar efter sig (ett rutnät av tomma pixlar): varje lucka får närmaste yta.

    Parametrar:
        depth: avståndsbild i mm, NaN där okänt.
        k: grannskapets storlek i px.

    Returnerar:
        en kopia med luckorna fyllda (stora hål förblir NaN).
    """
    valid = np.isfinite(depth)
    big = np.where(valid, depth, 1e9).astype(np.float32)
    nearest = cv2.erode(big, np.ones((k, k), np.uint8))  # minimumfilter = närmaste yta
    out = depth.copy()
    gap = ~valid & (nearest < 1e8)
    out[gap] = nearest[gap]
    return out


def load_depth(recording, frame_path, meta):
    """Djupet för en färgbild, efter reprojection på färgbilden.

    Returnerar:
        aligned (med luckor, används för kontrollen av skymda äpplen) och filled (luckor fyllda, används för avstånd).
    """
    raw = cv2.imread(str(recording / "depth" / (frame_path.stem + ".png")), cv2.IMREAD_UNCHANGED)
    aligned = align_depth_to_color(raw, meta)
    return aligned, fill_holes(aligned)


def box_depth(aligned, filled, x1, y1, x2, y2):
    """Avståndet till äpplet i en ruta, och hur mycket av det som skyms av något närmare.

    Parametrar:
        aligned, filled: djupbilder från load_depth.
        x1, y1, x2, y2: äpplets ruta i px.

    Returnerar:
        z_mm: medianavstånd över rutans mitt, eller från omgivningen när äpplet självt saknar djup.
        source: "apple" (eget djup), "around" (omgivningen) eller "none".
        occluded: andel av äpplets pixlar med något mer än OCCLUDER_MM framför (NaN om source inte är "apple").
    """
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    core = filled[int(cy - h / 4): int(cy + h / 4) + 1, int(cx - w / 4): int(cx + w / 4) + 1]
    core_z = core[np.isfinite(core)]
    if len(core_z) >= 4:
        z = float(np.median(core_z))
        yy, xx = np.mgrid[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
        in_ellipse = ((xx - cx) / max(w / 2, 1)) ** 2 + ((yy - cy) / max(h / 2, 1)) ** 2 <= 1
        box = aligned[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
        box_z = box[in_ellipse[: box.shape[0], : box.shape[1]] & np.isfinite(box)]
        occluded = float((box_z < z - OCCLUDER_MM).mean()) if len(box_z) >= 4 else np.nan
        return z, "apple", occluded
    around = filled[max(0, int(cy - 1.5 * h)): int(cy + 1.5 * h) + 1, max(0, int(cx - 1.5 * w)): int(cx + 1.5 * w) + 1]
    around_z = around[np.isfinite(around)]
    return (float(np.median(around_z)), "around", np.nan) if len(around_z) >= 4 else (np.nan, "none", np.nan)


# ---------------------------------------------------------------- detektion och rörelse

def suppress_nested(boxes, scores, iomin=NESTED_IOMIN):
    """Ta bort en ruta när största delen av den ligger inuti en ruta med högre confidence. Detektorn lägger ofta 2-5 rutor på samma
    äpple, och vanlig NMS behåller dem när storlekarna skiljer sig. Riktiga grannäpplen överlappar mycket mindre än så.

    Parametrar:
        boxes: (n, 4)-array med x1, y1, x2, y2.
        scores: (n,) detektorns confidence-värden.
        iomin: överlapp (snittyta / den mindre rutans yta) över vilket en ruta räknas som dubblett.

    Returnerar:
        Index för rutorna som behålls, högst confidence först.
    """
    keep = []
    for i in np.argsort(-scores):
        x1, y1, x2, y2 = boxes[i]
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        duplicate = False
        for j in keep:
            a1, b1, a2, b2 = boxes[j]
            inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
            if inter / min(area, max((a2 - a1) * (b2 - b1), 1e-6)) > iomin:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return np.array(keep, dtype=int)


def detect(recording, meta, model, frame_paths):
    """Kör detektorn på varje bildruta.

    Parametrar:
        recording: inspelningsmappen (har color/ och depth/).
        meta: inspelningens meta.json.
        model: YOLO-modell.
        frame_paths: färgbilderna, i ordning.

    Returnerar:
        detections-tabell: en rad per äppelruta, med avstånd, skymd andel och rå storlek i mm.
    """
    fx = meta["color_intrinsics"]["fx"]
    rows = []
    for frame, frame_path in enumerate(frame_paths):
        color = cv2.imread(str(frame_path))
        H, W = color.shape[:2]
        aligned, filled = load_depth(recording, frame_path, meta)
        result = model.predict(color, classes=[APPLE_CLASS], conf=CONF_LOW, imgsz=IMGSZ, verbose=False)[0]
        boxes, scores = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
        keep = suppress_nested(boxes, scores)
        for (x1, y1, x2, y2), conf in zip(boxes[keep], scores[keep]):
            w, h = x2 - x1, y2 - y1
            size = (w + h) / 2
            z, source, occluded = box_depth(aligned, filled, x1, y1, x2, y2)
            # avstånd till framsidan + radien = avstånd till mitten:  D = size * (z + D/2) / fx
            diameter = size * z / (fx - size / 2) if source == "apple" else np.nan
            rows.append(dict(frame=frame, conf=float(conf), x1=x1, y1=y1, x2=x2, y2=y2, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                             size=size, aspect=max(w, h) / max(min(w, h), 1),
                             at_edge=bool(x1 < 5 or y1 < 5 or x2 > W - 5 or y2 > H - 5),
                             z_mm=z, z_source=source, occluded=occluded, diameter_raw_mm=diameter,
                             far=bool(np.isfinite(z) and z > MAX_DEPTH_MM)))
    return pd.DataFrame(rows)


def vote_shift(prev_points, next_points, prior):
    """Kamerarörelsen mellan två bildrutor: den förflyttning som flest äppelpar är överens om.

    Parametrar:
        prev_points, next_points: (n, 2) äpplenas mittpunkter i de två bildrutorna.
        prior: förra bildrutans förflyttning (rörelsen är jämn), eller None.

    Returnerar:
        (dx, dy) förflyttning i bilden, eller None om ingen tydlig vinnare finns.
    """
    if len(prev_points) < 4 or len(next_points) < 4:
        return None
    moves = (next_points[None, :, :] - prev_points[:, None, :]).reshape(-1, 2)  # alla par av äpplen
    if prior is not None:
        moves = moves[np.hypot(*(moves - prior).T) < 40]  # rörelsen är jämn: håll dig nära förra uppskattningen
    moves = moves[np.hypot(*moves.T) < MAX_SHIFT]
    if len(moves) < 5:
        return None
    edges = np.arange(-MAX_SHIFT, MAX_SHIFT + SHIFT_BIN, SHIFT_BIN)
    votes, _, _ = np.histogram2d(moves[:, 0], moves[:, 1], bins=[edges, edges])
    votes = cv2.GaussianBlur(votes.astype(np.float32), (0, 0), 1.0) * (2 * np.pi)
    ix, iy = np.unravel_index(np.argmax(votes), votes.shape)
    if votes[ix, iy] < 5:
        return None
    peak = np.array([edges[ix] + SHIFT_BIN / 2, edges[iy] + SHIFT_BIN / 2])
    return moves[np.hypot(*(moves - peak).T) < 2 * SHIFT_BIN].mean(axis=0)


def camera_positions(detections, n_frames):
    """Kamerans sammanlagda förflyttning per bildruta (position i scenen = position i bilden + shift).

    Parametrar:
        detections: detections-tabell (bara detektioner med minst CONF_KEEP röstar).
        n_frames: antal bildrutor i inspelningen.

    Returnerar:
        shift: (n_frames, 2)-array i px.
        lost: antal bildrutepar utan tydlig röstning (då återanvänds förra rörelsen).
    """
    points = {f: g[g.conf >= CONF_KEEP][["cx", "cy"]].to_numpy() for f, g in detections.groupby("frame")}
    empty = np.zeros((0, 2))
    shift, last, lost = [np.zeros(2)], None, 0
    for frame in range(1, n_frames):
        move = vote_shift(points.get(frame - 1, empty), points.get(frame, empty), last)
        if move is None:
            move = last if last is not None else np.zeros(2)  # ingen tydlig röstning: fortsätt åt samma håll
            lost += 1
        else:
            last = move
        shift.append(shift[-1] - move)
    return np.array(shift), lost


def depth_motion(detections, shift):
    """Hur stor del av kamerarörelsen som är förflyttning i sidled (nära äpplen rör sig snabbare) och inte vridning.
    Anpassar  äpplets rörelse / dominerande rörelse = 1 + beta x (z_ref / z - 1);  beta 1 = bara sidled, 0 = bara vridning.

    Parametrar:
        detections: detections-tabell.
        shift: kamerans förflyttning per bildruta (camera_positions).

    Returnerar:
        z_ref: medianavstånd till äpplena i mm (trädets avstånd).
        beta: anpassad andel sidled, begränsad till 0-1,2 (0 om färre än 30 matchade äpplen).
        n_moves: antal äppelrörelser som anpassningen bygger på.
    """
    good = detections[(detections.conf >= CONF_KEEP) & (detections.z_source == "apple") & ~detections.far]
    z_ref = float(good.z_mm.median())
    by_frame = {f: g for f, g in good.groupby("frame")}
    xs, ys = [], []
    for frame in range(len(shift) - 1):
        now, after = by_frame.get(frame), by_frame.get(frame + 1)
        move = shift[frame] - shift[frame + 1]  # dominerande rörelse i bilden från denna bildruta till nästa
        if now is None or after is None or abs(move[0]) < 4:
            continue
        after_x, after_y = after.cx.to_numpy(), after.cy.to_numpy()
        for apple in now.itertuples():
            dist = np.hypot(after_x - (apple.cx + move[0]), after_y - (apple.cy + move[1]))
            j = int(dist.argmin())
            if dist[j] < 0.6 * apple.size:  # samma äpple i nästa bildruta
                xs.append(z_ref / apple.z_mm - 1)
                ys.append((after_x[j] - apple.cx) / move[0] - 1)
    xs, ys = np.array(xs), np.array(ys)
    ok = np.abs(ys) < 1.5
    if ok.sum() < 30 or np.sum(xs[ok] ** 2) == 0:
        return z_ref, 0.0, int(ok.sum())
    beta = float(np.sum(xs[ok] * ys[ok]) / np.sum(xs[ok] ** 2))  # minstakvadratmetoden genom origo
    return z_ref, float(np.clip(beta, 0, 1.2)), int(ok.sum())


def motion_factor(detections, z_ref, beta):
    """Per detektion: hur snabbt den rör sig över bilden jämfört med den dominerande rörelsen (1 när avståndet är okänt).

    Returnerar:
        Series m, i samma ordning som detections, begränsad till 0,4-2,5.
    """
    m = 1 + beta * (z_ref / detections.z_mm - 1)
    m = m.where((detections.z_source == "apple") & ~detections.far, 1.0)
    return m.clip(0.4, 2.5).fillna(1.0)


# ---------------------------------------------------------------- räkning

def strip_count(detections, shift, W, spacing, offset, use_depth=True, use_limit=True):
    """Äpplen som räknas av var `spacing`:e bildruta (med start i `offset`): varje äpple räknas bara i den utvalda
    bildruta vars mittpunkt ligger närmast det i scenen.

    Parametrar:
        detections: detections-tabell (behöver kolumnen m när use_depth).
        shift: kamerans förflyttning per bildruta.
        W: bildens bredd i px.
        spacing, offset: vilka bildrutor som väljs ut.
        use_depth: mät "närmast" med hänsyn till äpplets avstånd (skala dess rörelse med m).
        use_limit: ta bort detektioner bortom MAX_DEPTH_MM.

    Returnerar:
        De räknade detektionerna (en rad per räknat äpple).
    """
    frames = np.arange(offset, len(shift), spacing)
    confident = detections[detections.conf >= CONF_KEEP]
    if use_limit:
        confident = confident[~confident.far]
    own_frame = confident.frame.to_numpy().astype(int)
    m = confident.m.to_numpy() if use_depth else np.ones(len(confident))
    # avståndet från varje detektion till varje utvald bildrutas mittpunkt, sett från detektionens egen bildruta
    dist = np.abs((confident.cx.to_numpy() - W / 2)[:, None]
                  + m[:, None] * (shift[own_frame, 0][:, None] - shift[frames, 0][None, :]))
    nearest = dist.argmin(axis=1)
    return confident[frames[nearest] == own_frame]


def count_variants(detections, shift, W):
    """Antal för varje metod (2d / 2d_limit / 3d), avstånd K och startbildruta.

    Returnerar:
        tabell med kolumnerna method, k, offset, count.
    """
    rows = []
    for method, use_depth, use_limit in (("2d", False, False), ("2d_limit", False, True), ("3d", True, True)):
        for spacing in SPACINGS:
            for offset in range(min(spacing, 4)):
                count = len(strip_count(detections, shift, W, spacing, offset, use_depth, use_limit))
                rows.append(dict(method=method, k=spacing, offset=offset, count=count))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helt synliga äpplen

def follow(start, by_frame, shift, n_frames):
    """Vyerna av ett räknat äpple i bildrutorna runt det, genom att följa dess egen (djupskalade) rörelse.

    Parametrar:
        start: den räknade detektionen (en rad i detections-tabellen).
        by_frame: {bildruta: detektioner i den bildrutan} att söka i.
        shift: kamerans förflyttning per bildruta.
        n_frames: antal bildrutor.

    Returnerar:
        tabell med äpplets vyer (start inräknad), upp till VIS_WINDOW bildrutor åt varje håll; slutar efter 2 missar.
    """
    views = [start]
    for step in (1, -1):
        current, frame, misses = start, int(start.frame), 0
        while misses < 2 and abs(frame + step - start.frame) <= VIS_WINDOW and 0 <= frame + step < n_frames:
            frame += step
            # där äpplet borde vara i denna bildruta
            px = current.cx + current.m * (shift[int(current.frame), 0] - shift[frame, 0])
            py = current.cy + current.m * (shift[int(current.frame), 1] - shift[frame, 1])
            candidates = by_frame.get(frame)
            hit = None
            if candidates is not None:
                dist = np.hypot(candidates.cx.to_numpy() - px, candidates.cy.to_numpy() - py)
                j = int(dist.argmin())
                closest = candidates.iloc[j]
                if dist[j] < max(5.0, 0.5 * current["size"]) and 0.7 < closest["size"] / current["size"] < 1.4:
                    hit = closest
            if hit is None:
                misses += 1
            else:
                views.append(hit)
                current, misses = hit, 0
    return pd.DataFrame(views)


def median_or_nan(values):
    """Medianen, eller NaN om det inte finns några värden."""
    return float(values.median()) if len(values) else np.nan


def spread(values):
    """Relativ spridning (standardavvikelse / medelvärde), eller NaN med färre än 2 värden."""
    return float(values.std() / values.mean()) if len(values) > 1 else np.nan


def is_fully_visible(apple):
    """Är äpplet tillräckligt synligt för att mätas? Alla kontroller måste vara uppfyllda.
    Ett saknat värde (NaN) gör kontrollen falsk, så då mäts äpplet inte (men det räknas fortfarande)."""
    checks = [
        apple["views"] >= VIS_MIN_VIEWS,                # sett i tillräckligt många bildrutor
        apple["min_conf"] >= VIS_MIN_CONF,              # hög confidence i alla vyer
        apple["max_aspect"] <= VIS_MAX_ASPECT,          # runt, inte avlångt
        apple["size_cv"] <= VIS_MAX_SIZE_CV,            # jämn storlek (inget löv som rör sig framför)
        not apple["at_edge"],                           # inte avskuret av bildkanten
        apple["size_px"] >= VIS_MIN_SIZE_PX,            # inte för litet i bilden
        apple["depth_views"] >= VIS_MIN_DEPTH_VIEWS,    # djup på själva äpplet i flera vyer
        apple["depth_spread"] <= VIS_MAX_DEPTH_SPREAD,  # jämnt avstånd mellan vyerna
        apple["occluded"] <= VIS_MAX_OCCLUDED,          # nästan inget framför äpplet
    ]
    return bool(all(checks))


def measure_visible(counted, detections, shift):
    """För varje räknat äpple: följ det, avgör om det är helt synligt och mät det (rå mm, före kalibrering).

    Parametrar:
        counted: de räknade detektionerna (strip_count).
        detections: alla detektioner (följs ner till CONF_LOW).
        shift: kamerans förflyttning per bildruta.

    Returnerar:
        apples: en rad per räknat äpple med statistik över vyerna och flaggan fully_visible.
        views: alla följda vyer, med en kolumn `apple` som kopplar dem till apples.
    """
    pool = detections[(detections.conf >= CONF_LOW) & ~detections.far]
    by_frame = {f: g for f, g in pool.groupby("frame")}
    n_frames = len(shift)
    rows, all_views = [], []
    for apple_id, start in counted.iterrows():
        views = follow(start, by_frame, shift, n_frames)
        with_depth = views[views.z_source == "apple"]  # vyerna där själva äpplet har ett djupvärde
        # en rad per räknat äpple, sammanfattad över alla dess vyer
        row = dict(
            apple=apple_id,
            frame=int(start.frame),                                     # bildrutan där äpplet räknades
            views=len(views),                                           # antal bildrutor där det hittades
            min_conf=float(views.conf.min()),                           # lägsta confidence
            max_aspect=float(views.aspect.max()),                       # mest avlånga rutan (1 = rund)
            size_px=float(views["size"].median()),                      # typisk storlek i px
            size_cv=spread(views["size"]),                              # hur mycket storleken varierar
            at_edge=bool(views.at_edge.any()),                          # rör bildkanten någon gång
            depth_views=len(with_depth),                                # antal vyer med djup på äpplet
            z_mm=median_or_nan(with_depth.z_mm),                        # avstånd till äpplet
            depth_spread=spread(with_depth.z_mm),                       # hur mycket avståndet varierar
            occluded=median_or_nan(views.occluded.dropna()),            # andel av äpplet som skyms
            diameter_raw_mm=median_or_nan(with_depth.diameter_raw_mm),  # storlek i mm, före kalibreringen
        )
        row["fully_visible"] = is_fully_visible(row)
        rows.append(row)
        all_views.append(views.assign(apple=apple_id))
    return pd.DataFrame(rows), pd.concat(all_views) if all_views else pd.DataFrame()


# ---------------------------------------------------------------- en inspelning

def analyse(recording, model, out, reuse=False):
    """Detektion, kamerarörelse, antal och helt synliga äpplen för en inspelning (kräver ingen kalibrering).

    Parametrar:
        recording: inspelningsmappen.
        model: YOLO-modell.
        out: mapp för resultaten (detections.csv skrivs här).
        reuse: läs detections.csv från en tidigare körning i stället för att detektera igen.

    Returnerar:
        result-dict: rec, meta, frames (sökvägar), W, det (detections), shift, lost, z_ref, beta, n_moves,
        table (räknevarianter), counted, apples, views.
    """
    meta = json.loads((recording / "meta.json").read_text())
    frame_paths = sorted((recording / "color").glob("*.jpg"))
    W = cv2.imread(str(frame_paths[0])).shape[1]
    cache = out / "detections.csv"
    if reuse and cache.exists():
        detections = pd.read_csv(cache)
    else:
        detections = detect(recording, meta, model, frame_paths)
    shift, lost = camera_positions(detections[~detections.far], len(frame_paths))
    z_ref, beta, n_moves = depth_motion(detections, shift)
    detections["m"] = motion_factor(detections, z_ref, beta)
    out.mkdir(parents=True, exist_ok=True)
    detections.to_csv(cache, index=False)

    table = count_variants(detections, shift, W)
    table_3d = table[table.method == "3d"]
    typical = table_3d.iloc[(table_3d["count"] - table_3d["count"].mean()).abs().argmin()]  # varianten som visas i videon
    counted = strip_count(detections, shift, W, int(typical.k), int(typical.offset))
    apples, views = measure_visible(counted, detections, shift)
    return dict(rec=recording, meta=meta, frames=frame_paths, W=W, det=detections, shift=shift, lost=lost,
                z_ref=z_ref, beta=beta, n_moves=n_moves, table=table, counted=counted, apples=apples, views=views)


# ---------------------------------------------------------------- kalibrering

def calibrate(model, reuse=False):
    """Anpassa storleksskalan på träd 1-inspelningarna för kalibrering. Skriver calibration.json och
    fully_visible_apples.csv till CALIBRATION_DIR, och bilder per inspelning till OUT/calibration/.

    skala = referensdiametern / medianen av de helt synliga äpplenas råa diameter, med ett bootstrap-intervall.

    Returnerar:
        calibration-dict (sparas också som calibration.json).
    """
    out = OUT / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    CALIBRATION_DIR.mkdir(exist_ok=True)
    measured, per_recording = [], []
    for name in CALIBRATION_RECORDINGS:
        result = analyse(ARCHIVE / name, model, out / name, reuse)
        visible = result["apples"][result["apples"].fully_visible].assign(recording=name)
        measured.append(visible)
        visualize.crop_sheet(result, visible, lambda a: f"{a.diameter_raw_mm:.0f} mm {num(a.z_mm / 1000)} m", out / name / "fully_visible_apples.png")
        per_recording.append(dict(recording=name, counted=len(result["counted"]), fully_visible=len(visible),
                                  median_raw_diameter_mm=float(visible.diameter_raw_mm.median()) if len(visible) else None))
        print(f"  {name[:12]:12s} räknade {len(result['counted']):3d}  helt synliga {len(visible):3d}  "
              f"median rå diameter {num(visible.diameter_raw_mm.median())} mm  beta {num(result['beta'], 2)}")
    all_measured = pd.concat(measured, ignore_index=True)
    diameters = all_measured.diameter_raw_mm.to_numpy()
    scale = REF_DIAMETER_MM / float(np.median(diameters))
    rng = np.random.default_rng(0)
    boot = REF_DIAMETER_MM / np.array([np.median(rng.choice(diameters, len(diameters))) for _ in range(2000)])
    calibration = dict(
        scale=scale, scale_low=float(np.percentile(boot, 5)), scale_high=float(np.percentile(boot, 95)),
        scale_bootstrap=boot.round(5).tolist(), apples=len(diameters), median_raw_diameter_mm=float(np.median(diameters)),
        reference=dict(diameter_mm=REF_DIAMETER_MM, mass_g=REF_MASS_G, axes_cm=[7.0, 6.5, 6.0],
                       note="handvägt äpple, beskrivet som medelstort på träd 1"),
        density_g_cm3=DENSITY, per_recording=per_recording, recordings=CALIBRATION_RECORDINGS,
        note="skala = referensdiameter / medianen av uppmätt diameter för de helt synliga träd 1-äpplena. "
             "Samma fysiska äpplen syns i flera inspelningar, så bootstrap-intervallet är optimistiskt.")
    (CALIBRATION_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2))
    all_measured.to_csv(CALIBRATION_DIR / "fully_visible_apples.csv", index=False)
    visualize.depth_check_plot(all_measured, scale, out / "depth_check.png", REF_DIAMETER_MM)
    print(f"Storleksskala {num(scale, 3)} (90 % {num(calibration['scale_low'], 3)}-{num(calibration['scale_high'], 3)}) från "
          f"{len(diameters)} helt synliga träd 1-äpplen; uppmätt median {num(np.median(diameters))} mm -> referens {REF_DIAMETER_MM:.0f} mm")
    return calibration


# ---------------------------------------------------------------- resultat för en testinspelning

def summarize(result, calibration):
    """Antal, vikt och skörd (med rimliga intervall) för en inspelning.

    Parametrar:
        result: från analyse().
        calibration: storlekskalibreringen.

    Returnerar:
        summary-dict, och de helt synliga äpplena med kalibrerad diameter_mm och mass_g.
    """
    table, apples = result["table"], result["apples"]
    counts = {method: g["count"] for method, g in table.groupby("method")}
    n = int(round(counts["3d"].mean()))
    count_low = int(round(counts["3d"].min() * (1 - COUNT_ERROR)))
    count_high = int(round(counts["3d"].max() * (1 + COUNT_ERROR)))

    scale = calibration["scale"]
    visible = apples[apples.fully_visible].copy()
    visible["diameter_mm"] = visible.diameter_raw_mm * scale
    visible["mass_g"] = sphere_mass_g(visible.diameter_mm)
    if len(visible) >= 5:
        masses, source = visible.mass_g.to_numpy(), f"mätt med djup på {len(visible)} helt synliga äpplen"
    else:  # för få rena äpplen i denna inspelning: använd kalibreringsäpplena (träd 1)
        calibration_apples = pd.read_csv(CALIBRATION_DIR / "fully_visible_apples.csv")
        masses = sphere_mass_g(calibration_apples.diameter_raw_mm * scale).to_numpy()
        source = f"bara {len(visible)} helt synliga äpplen här; kalibreringsäpplena från träd 1 används"
    # viktintervall: bootstrap, dra äpplena och storleksskalan med återläggning
    rng = np.random.default_rng(1)
    scale_boot = np.array(calibration["scale_bootstrap"])
    boot = np.array([rng.choice(masses, len(masses)).mean() * (rng.choice(scale_boot) / scale) ** 3 for _ in range(4000)])
    mean_g = float(masses.mean())
    g_low, g_high = (float(x) for x in np.percentile(boot, [5, 95]))

    confident = result["det"][result["det"].conf >= CONF_KEEP]
    name = result["rec"].name
    summary = dict(
        # vilken inspelning
        recording=name, tree=TREE_OF.get(name, "?"), frames=len(result["frames"]),
        # antalet (medel av de 12 räknevarianterna) och dess rimliga intervall
        apples_counted=n, count_low=count_low, count_high=count_high,
        # samma räkning utan djup, med bara avståndsgränsen, och med djup
        count_2d_no_limit=float(counts["2d"].mean()), count_2d_with_limit=float(counts["2d_limit"].mean()),
        count_3d=float(counts["3d"].mean()),
        background_removed=float(counts["2d"].mean() - counts["2d_limit"].mean()),
        # kontroller av djupet
        detections_beyond_limit=int(confident.far.sum()), detections_without_depth=int((confident.z_source == "none").sum()),
        max_depth_m=MAX_DEPTH_MM / 1000, tree_distance_m=result["z_ref"] / 1000, sideways_travel_share_beta=result["beta"],
        camera_pan_px=float(abs(result["shift"][-1, 0] - result["shift"][0, 0])),
        # äpplenas vikt
        fully_visible_apples=int(len(visible)), median_diameter_mm=float(visible.diameter_mm.median()) if len(visible) else None,
        mean_weight_g=mean_g, weight_low_g=g_low, weight_high_g=g_high, weight_source=source,
        # skörd = antal x medelvikt
        estimate_kg=n * mean_g / 1000, kg_low=count_low * g_low / 1000, kg_high=count_high * g_high / 1000,
        # kalibreringen som användes
        size_scale=scale, size_scale_range=[calibration["scale_low"], calibration["scale_high"]],
        note=f"rimligt intervall, inte ett statistiskt konfidensintervall: antal = lägsta/högsta av {len(counts['3d'])} "
             f"varianter av strip sampling +/-{COUNT_ERROR:.0%} räknefel; vikt = 90 % bootstrap över de helt synliga äpplena "
             "och storleksskalan. Storleksskalan bygger på ett referensäpple (109 g) som antas vara medelstort för träd 1.",
    )
    return summary, visible


def print_summary(s):
    bar = "=" * 64
    print(bar)
    print(f" {s['recording']}  ({s['tree']})")
    print(f"   Räknade äpplen  : {s['apples_counted']}   (rimligt intervall {s['count_low']} - {s['count_high']})")
    print(f"      2D utan gräns {s['count_2d_no_limit']:.0f} | 2D + 3 m-gräns {s['count_2d_with_limit']:.0f} | 3D {s['count_3d']:.0f}")
    print(f"   Medelvikt       : {s['mean_weight_g']:.0f} g ({s['weight_low_g']:.0f}-{s['weight_high_g']:.0f}), "
          f"median {s['median_diameter_mm'] or 0:.0f} mm, {s['weight_source']}")
    print(f"   Uppskattad skörd: {num(s['estimate_kg'])} kg   (rimligt intervall {num(s['kg_low'])} - {num(s['kg_high'])} kg)")
    print(f"   Trädet på {num(s['tree_distance_m'], 2)} m, andel sidledsrörelse {num(s['sideways_travel_share_beta'], 2)}, "
          f"{s['detections_beyond_limit']} detektioner bortom gränsen")
    print(bar)


def process(recording, model, calibration, reuse=False):
    """Kör hela pipelinen på en inspelning och skriv resultaten till OUT/<inspelningens namn>/.

    Returnerar:
        summary-dict.
    """
    out = OUT / recording.name
    result = analyse(recording, model, out, reuse)
    summary, visible = summarize(result, calibration)
    result["table"].to_csv(out / "counts.csv", index=False)
    result["apples"].merge(visible[["apple", "diameter_mm", "mass_g"]], on="apple", how="left").to_csv(out / "apples_measured.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    # bilder och videor (påverkar inga siffror)
    visualize.crop_sheet(result, visible, lambda a: f"{a.diameter_mm:.0f} mm {a.mass_g:.0f} g", out / "measured_apples.png")
    visualize.render_audit(result, out)
    depth_frames = (load_depth(recording, frame_path, result["meta"])[1] for frame_path in result["frames"])
    visualize.render(result, summary, visible, out, depth_frames, CONF_KEEP, MAX_DEPTH_MM)
    return summary


# ---------------------------------------------------------------- kontroll på de handmarkerade bildrutorna

def evaluate_marked(model):
    """Träffsäkerhet (recall och precision) på de handmarkerade bildrutorna, med och utan avståndsgränsen.

    En markerad punkt räknas som hittad när den ligger inuti en detekterad ruta (8 px marginal); varje ruta och varje
    punkt används en gång.

    Returnerar:
        tabell med en rad per bildruta och gränsinställning (sparas också som marked_frames_eval.csv).
    """
    rows = []
    for name, frame, points_csv in MARKED_FRAMES:
        recording = ARCHIVE / name
        meta = json.loads((recording / "meta.json").read_text())
        frame_path = recording / "color" / f"{frame:06d}.jpg"
        detections = detect(recording, meta, model, [frame_path])
        detections = detections[detections.conf >= CONF_KEEP].reset_index(drop=True)
        points = pd.read_csv(points_csv)[["x", "y"]].to_numpy()
        for limit in (False, True):
            boxes = detections[~detections.far] if limit else detections
            boxes_hit, points_hit = set(), set()
            for i, box in boxes.iterrows():
                inside = [j for j, (px, py) in enumerate(points)
                          if j not in points_hit and box.x1 - 8 <= px <= box.x2 + 8 and box.y1 - 8 <= py <= box.y2 + 8]
                if inside:
                    j = min(inside, key=lambda j: (points[j][0] - box.cx) ** 2 + (points[j][1] - box.cy) ** 2)
                    boxes_hit.add(i)
                    points_hit.add(j)
            rows.append(dict(frame=f"{name[:12]}:{frame}", depth_limit=f"{num(MAX_DEPTH_MM / 1000)} m" if limit else "ingen",
                             marked=len(points), detections=len(boxes), recall=len(points_hit) / len(points),
                             precision_lower_bound=len(boxes_hit) / max(len(boxes), 1), count_error=len(boxes) / len(points) - 1))
    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT / "marked_frames_eval.csv", index=False, encoding="utf-8-sig")
    return table


def main():
    ap = argparse.ArgumentParser(description="Räkna äpplen och uppskatta skörden från RGB-D-inspelningar.")
    ap.add_argument("recordings", nargs="*", type=Path, help="inspelningsmappar")
    ap.add_argument("--calibrate", action="store_true", help="anpassa storleksskalan på träd 1-inspelningarna för kalibrering")
    ap.add_argument("--evaluate", action="store_true", help="kontrollera detektorn på de handmarkerade bildrutorna")
    ap.add_argument("--reuse", action="store_true", help="återanvänd detections.csv från en tidigare körning")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)
    calibration_path = CALIBRATION_DIR / "calibration.json"
    if args.calibrate or not calibration_path.exists():
        calibration = calibrate(model, args.reuse)
    else:
        calibration = json.loads(calibration_path.read_text())
    if args.evaluate:
        print(evaluate_marked(model).to_string(index=False))
    summaries = []
    for recording in args.recordings:
        summary = process(recording, model, calibration, args.reuse)
        print_summary(summary)
        summaries.append(summary)
    if summaries:
        cols = ["recording", "tree", "count_2d_no_limit", "count_2d_with_limit", "count_3d", "count_low", "count_high",
                "fully_visible_apples", "median_diameter_mm", "mean_weight_g", "estimate_kg", "kg_low", "kg_high"]
        # utf-8-sig så att Excel visar å, ä, ö rätt
        pd.DataFrame(summaries)[cols].round(1).to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
