"""Skördepipeline för äpplen från RGB-D-video (Orbbec Gemini 2).

  1. Detektera äpplen i varje bildruta (fine-tuned YOLO).
  2. Avstånd till varje äpple från djupbilden. Längre bort än 3 m är bakgrund och räknas inte.
  3. Mät kamerarörelsen med hjälp av äpplena själva.
  4. Räkna med strip sampling: varje äpple räknas i en enda bildruta.
  5. Mät storleken på de äpplen som är helt synliga och räkna om den till vikt.
  6. Skörd = antal äpplen x medelvikt.

Användning (från mappen Main):
    python pipeline_3d.py ../Archive/test_images ../Archive/Tree2
    python pipeline_3d.py --evaluate      # kontrollera detektorn mot de handmarkerade äpplena
    python pipeline_3d.py --calibrate     # anpassa storleksskalan på nytt (behövs sällan)

Resultaten hamnar i ../Results/pipeline_3d/<inspelning>/. Med --reuse hoppas detektionen över när
detections.csv redan finns.
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

# Detektor
WEIGHTS = "models/apple_run1.pt"  # run 1: fine-tuned YOLO11m (Orbbec, träd 1)
APPLE_CLASS = 47                  # "apple" bland COCO-klasserna
IMGSZ = 1920
CONF_LOW = 0.2                    # för att följa ett äpple mellan bildrutor
CONF_KEEP = 0.45                  # för att räkna det
NESTED_IOMIN = 0.5                # en ruta som till så stor del ligger i en annan är en dubblett

# Djup och kamerarörelse
MAX_DEPTH_MM = 3000               # längre bort är bakgrund
DEPTH_READ_MM = 10000             # djup längre bort än så ignoreras
MAX_SHIFT = 100                   # px: största kamerarörelse mellan två bildrutor
SHIFT_BIN = 4                     # px: upplösning när kamerarörelsen mäts

# Räkning
SPACINGS = (3, 5, 8)              # var K:e bildruta, 4 startbildrutor vardera -> 12 räkningar
COUNT_ERROR = 0.10                # räknefel per bildruta (uppmätt ca 7 %), används för intervallet

# Referensäpple: 7,0 x 6,5 x 6,0 cm, 109 g, medelstort på träd 1
# ! all vikt bygger på detta enda äpple
# TODO: väg ca 10 äpplen och kalibrera om (--calibrate)
REF_DIAMETER_MM = 65.0
REF_MASS_G = 109.0
DENSITY = REF_MASS_G / ((np.pi / 6) * (REF_DIAMETER_MM / 10) ** 3)  # g/cm3

# "Helt synligt": bara sådana äpplen mäts
VIS_WINDOW = 6                    # bildrutor före och efter den räknade
VIS_MIN_VIEWS = 7
VIS_MIN_CONF = 0.6
VIS_MAX_ASPECT = 1.2
VIS_MAX_SIZE_CV = 0.10            # storleken får variera högst 10 % mellan vyerna
VIS_MIN_SIZE_PX = 16
VIS_MIN_DEPTH_VIEWS = 4
VIS_MAX_DEPTH_SPREAD = 0.06       # avståndet får variera högst 6 % mellan vyerna
VIS_MAX_OCCLUDED = 0.15
OCCLUDER_MM = 60                  # så här mycket närmare än äpplet = något som skymmer det

# Filer och mappar
OUT = Path("../Results/pipeline_3d")
CALIBRATION_DIR = Path("calibration")
ARCHIVE = Path("../Archive")

# Träd 1-inspelningar som storleksskalan anpassas på
# ! får aldrig användas för test (då testar vi på kalibreringsdatan)
CALIBRATION_RECORDINGS = [
    "1aa65ece-d32f-4254-bcd3-80b26d2db7b9",
    "341f04ca-054a-4e52-ae96-139378821ea8",
    "5fa2faba-e6d8-410e-803e-a52e8a0d9089",
    "849d204a-348a-4494-b9c9-a9e0d3903359",
    "Annotation_tetst",
    "d4d77803-a829-42b5-8eb2-a4c86f31e99a",
]
TREE_OF = {"test_images": "träd 1", "Tree2": "träd 2", "1dceb275-6e92-4368-9df5-1526972df094": "träd 2"}

# Handmarkerade bildrutor för --evaluate: (inspelning, bildruta, fil med markerade äpplen)
MARKED_FRAMES = [
    ("test_images", 15, "annotations/test_images_000015_points.csv"),
    ("Tree2", 11, "annotations/Tree2_000011_points.csv"),
    ("1aa65ece-d32f-4254-bcd3-80b26d2db7b9", 23, "annotations/1aa65e_000023_points.csv"),
]


def sphere_mass_g(diameter_mm):
    """Vikten för ett äpple med denna diameter, som en sfär med referensäpplets densitet."""
    return DENSITY * (np.pi / 6) * (diameter_mm / 10) ** 3


# Djup

def align_depth_to_color(depth_mm, meta, z_min=300, z_max=DEPTH_READ_MM):
    """Reprojection av djupbilden till färgbilden (djupkameran sitter bredvid färgkameran).
    Returnerar avståndet i mm per färgpixel, NaN där ingen djuppunkt hamnade."""
    depth_intr = meta["depth_intrinsics"]
    color_intr = meta["color_intrinsics"]
    rotation = np.array(meta["depth_to_color_extrinsics"]["rotation"], float).reshape(3, 3)
    translation = np.array(meta["depth_to_color_extrinsics"]["translation_mm"], float)

    # 1. djuppixlar -> 3D-punkter i djupkameran -> färgkamerans koordinater
    v, u = np.nonzero((depth_mm > z_min) & (depth_mm < z_max))
    z = depth_mm[v, u].astype(np.float32)
    x = (u - depth_intr["cx"]) * z / depth_intr["fx"]
    y = (v - depth_intr["cy"]) * z / depth_intr["fy"]
    points = np.stack([x, y, z], axis=1) @ rotation.T + translation

    # 2. 3D-punkter -> färgbildens pixlar
    u_color = np.round(points[:, 0] * color_intr["fx"] / points[:, 2] + color_intr["cx"]).astype(int)
    v_color = np.round(points[:, 1] * color_intr["fy"] / points[:, 2] + color_intr["cy"]).astype(int)
    inside = (u_color >= 0) & (u_color < color_intr["width"]) & (v_color >= 0) & (v_color < color_intr["height"])

    # 3. skriv in avstånden
    # ? flera punkter kan hamna på samma pixel; den närmaste ytan vinner (den skymmer de andra)
    out = np.full((color_intr["height"], color_intr["width"]), np.inf, np.float32)
    np.minimum.at(out, (v_color[inside], u_color[inside]), points[inside, 2])
    out[np.isinf(out)] = np.nan
    return out


def fill_holes(depth, k=5):
    """Fyll de små luckor som reprojection lämnar: varje lucka får närmaste yta inom k px."""
    # ? erode = minimumfilter, alltså närmaste yta i grannskapet
    valid = np.isfinite(depth)
    nearest = cv2.erode(np.where(valid, depth, 1e9).astype(np.float32), np.ones((k, k), np.uint8))

    out = depth.copy()
    gap = ~valid & (nearest < 1e8)
    out[gap] = nearest[gap]
    return out


def load_depth(recording, frame_path, meta):
    """Djupet för en bildruta. Returnerar aligned (med luckor) och filled (luckor fyllda)."""
    # ! cv2.imread klarar inte absoluta sökvägar med Ä: håll sökvägarna relativa
    raw = cv2.imread(str(recording / "depth" / (frame_path.stem + ".png")), cv2.IMREAD_UNCHANGED)
    aligned = align_depth_to_color(raw, meta)
    return aligned, fill_holes(aligned)


def box_depth(aligned, filled, x1, y1, x2, y2):
    """Avståndet till äpplet i en ruta.

    Returnerar (z_mm, source, occluded):
        source är "apple" (äpplets eget djup), "around" (omgivningens) eller "none".
        occluded är andelen av äpplet som skyms av något närmare (bara när source är "apple").
    """
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # 1. avståndet: median över rutans mittersta del
    # ? mitten träffar äpplet; kanterna träffar ofta löv eller bakgrund
    core = filled[int(cy - h / 4): int(cy + h / 4) + 1, int(cx - w / 4): int(cx + w / 4) + 1]
    core_z = core[np.isfinite(core)]

    if len(core_z) >= 4:
        z = float(np.median(core_z))

        # 2. skymt: andel pixlar inom äpplets ellips som ligger klart framför äpplet
        yy, xx = np.mgrid[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
        in_ellipse = ((xx - cx) / max(w / 2, 1)) ** 2 + ((yy - cy) / max(h / 2, 1)) ** 2 <= 1
        box = aligned[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
        box_z = box[in_ellipse[: box.shape[0], : box.shape[1]] & np.isfinite(box)]
        occluded = float((box_z < z - OCCLUDER_MM).mean()) if len(box_z) >= 4 else np.nan

        return z, "apple", occluded

    # ! inget djup på själva äpplet: omgivningens avstånd räcker för 3 m-gränsen, men inte för storlek
    around = filled[max(0, int(cy - 1.5 * h)): int(cy + 1.5 * h) + 1, max(0, int(cx - 1.5 * w)): int(cx + 1.5 * w) + 1]
    around_z = around[np.isfinite(around)]
    if len(around_z) >= 4:
        return float(np.median(around_z)), "around", np.nan
    return np.nan, "none", np.nan


# Detektion

def suppress_nested(boxes, scores, iomin=NESTED_IOMIN):
    """Ta bort rutor som till största delen ligger inuti en ruta med högre confidence.
    Detektorn lägger ofta flera rutor på samma äpple. Returnerar index för rutorna som behålls."""
    # ? vanlig NMS behåller rutor av olika storlek på samma äpple; därför jämförs mot den mindre rutans yta
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
    """Kör detektorn på varje bildruta. Returnerar en tabell med en rad per äppelruta."""
    fx = meta["color_intrinsics"]["fx"]
    rows = []

    for frame, frame_path in enumerate(frame_paths):
        color = cv2.imread(str(frame_path))
        H, W = color.shape[:2]
        aligned, filled = load_depth(recording, frame_path, meta)

        # 1. detektera och ta bort dubbletter
        result = model.predict(color, classes=[APPLE_CLASS], conf=CONF_LOW, imgsz=IMGSZ, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keep = suppress_nested(boxes, scores)

        # 2. avstånd och storlek per äpple
        for (x1, y1, x2, y2), conf in zip(boxes[keep], scores[keep]):
            w, h = x2 - x1, y2 - y1
            size = (w + h) / 2
            z, source, occluded = box_depth(aligned, filled, x1, y1, x2, y2)

            # diameter i mm från storlek i px och avstånd (pinhole-modellen)
            # ? djupet mäts till äpplets framsida; + radien ger mitten:  D = size * (z + D/2) / fx
            diameter = size * z / (fx - size / 2) if source == "apple" else np.nan

            rows.append(dict(
                frame=frame, conf=float(conf),
                x1=x1, y1=y1, x2=x2, y2=y2, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                size=size, aspect=max(w, h) / max(min(w, h), 1),
                at_edge=bool(x1 < 5 or y1 < 5 or x2 > W - 5 or y2 > H - 5),
                z_mm=z, z_source=source, occluded=occluded, diameter_raw_mm=diameter,
                far=bool(np.isfinite(z) and z > MAX_DEPTH_MM),
            ))

    return pd.DataFrame(rows)


# Kamerarörelse

def vote_shift(prev_points, next_points, prior):
    """Kamerarörelsen mellan två bildrutor: den förflyttning som flest äppelpar är överens om.
    Returnerar (dx, dy), eller None om ingen tydlig vinnare finns."""
    # * Vi vet inte vilket äpple som är vilket, så vi tar förflyttningen för VARJE äppelpar. Par av samma
    # * äpple pekar alla åt samma håll (kamerans rörelse); fel par sprids ut. Den vanligaste vinner.

    # 1. tillräckligt många äpplen för att rösta?
    if len(prev_points) < 4 or len(next_points) < 4:
        return None

    # 2. förflyttningen för alla par
    moves = (next_points[None, :, :] - prev_points[:, None, :]).reshape(-1, 2)

    # 3. plausibility: släng förflyttningar som inte kan vara kamerans
    # ? kameran rör sig jämnt (nära förra rörelsen) och aldrig mer än MAX_SHIFT px per bildruta
    if prior is not None:
        moves = moves[np.hypot(*(moves - prior).T) < 40]
    moves = moves[np.hypot(*moves.T) < MAX_SHIFT]
    if len(moves) < 5:
        return None

    # 4. vote: alla förflyttningar i ett 2D-histogram; högsta toppen = kamerarörelsen
    # ? utjämningen låter nästan lika förflyttningar stödja varandra; under ca 5 röster = ingen vinnare
    edges = np.arange(-MAX_SHIFT, MAX_SHIFT + SHIFT_BIN, SHIFT_BIN)
    votes, _, _ = np.histogram2d(moves[:, 0], moves[:, 1], bins=[edges, edges])
    votes = cv2.GaussianBlur(votes.astype(np.float32), (0, 0), 1.0) * (2 * np.pi)
    ix, iy = np.unravel_index(np.argmax(votes), votes.shape)
    if votes[ix, iy] < 5:
        return None

    # 5. förfina: medelvärdet av förflyttningarna nära toppen (exaktare än histogrammets rutor)
    peak = np.array([edges[ix] + SHIFT_BIN / 2, edges[iy] + SHIFT_BIN / 2])
    return moves[np.hypot(*(moves - peak).T) < 2 * SHIFT_BIN].mean(axis=0)


def camera_positions(detections, n_frames):
    """Kamerans sammanlagda förflyttning per bildruta: position i scenen = position i bilden + shift."""
    # 1. äpplenas positioner per bildruta
    # ? bara säkra detektioner röstar; osäkra är ofta löv och ger brus
    points = {f: g[g.conf >= CONF_KEEP][["cx", "cy"]].to_numpy() for f, g in detections.groupby("frame")}
    empty = np.zeros((0, 2))

    # 2. summera rörelsen bildruta för bildruta
    # ! minustecknet: position i bilden + shift ska bli densamma för ett äpple i alla bildrutor
    shift, last = [np.zeros(2)], None
    for frame in range(1, n_frames):
        move = vote_shift(points.get(frame - 1, empty), points.get(frame, empty), last)
        if move is None:
            # ingen tydlig röstning (t.ex. inga äpplen i bild): anta samma rörelse som förut
            move = last if last is not None else np.zeros(2)
        else:
            last = move
        shift.append(shift[-1] - move)

    return np.array(shift)


def depth_motion(detections, shift):
    """Hur mycket snabbare nära äpplen rör sig över bilden än äpplen långt bort.
    Anpassar  rörelse = 1 + beta x (z_ref / z - 1). Returnerar (z_ref, beta).

    När kameran åker i sidled rör sig nära äpplen snabbare än äpplen långt bort (som utanför ett tågfönster);
    när den bara vrider sig rör sig allt lika fort. beta = 1 betyder bara sidled, 0 bara vridning."""
    # 1. säkra äpplen med eget djup; z_ref = trädets typiska avstånd
    good = detections[(detections.conf >= CONF_KEEP) & (detections.z_source == "apple") & ~detections.far]
    z_ref = float(good.z_mm.median())
    by_frame = {f: g for f, g in good.groupby("frame")}

    # 2. för varje äpple i två bildrutor i rad:
    #    xs = hur mycket närmare än trädet det är, ys = hur mycket snabbare än kameran det rörde sig
    xs, ys = [], []
    for frame in range(len(shift) - 1):
        now, after = by_frame.get(frame), by_frame.get(frame + 1)
        move = shift[frame] - shift[frame + 1]
        if now is None or after is None or abs(move[0]) < 4:  # för liten rörelse för att mäta
            continue

        after_x, after_y = after.cx.to_numpy(), after.cy.to_numpy()
        for apple in now.itertuples():
            # samma äpple = detektionen närmast där kamerarörelsen säger att det ska vara
            dist = np.hypot(after_x - (apple.cx + move[0]), after_y - (apple.cy + move[1]))
            j = int(dist.argmin())
            if dist[j] < 0.6 * apple.size:
                xs.append(z_ref / apple.z_mm - 1)
                ys.append((after_x[j] - apple.cx) / move[0] - 1)

    # 3. rät linje genom origo (ys = beta x xs) med minstakvadratmetoden, utan uppenbara felmatchningar
    # ! med färre än 30 mätningar antas ren vridning (beta 0)
    xs, ys = np.array(xs), np.array(ys)
    ok = np.abs(ys) < 1.5
    if ok.sum() < 30 or np.sum(xs[ok] ** 2) == 0:
        return z_ref, 0.0

    beta = float(np.sum(xs[ok] * ys[ok]) / np.sum(xs[ok] ** 2))
    return z_ref, float(np.clip(beta, 0, 1.2))


def motion_factor(detections, z_ref, beta):
    """Per detektion: hur snabbt den rör sig jämfört med kamerarörelsen (1 när avståndet är okänt)."""
    # ? m = 2: äpplet rör sig dubbelt så fort som kameran, alltså sitter det nära
    m = 1 + beta * (z_ref / detections.z_mm - 1)

    # ! utan eget djup antas äpplet sitta på trädets avstånd (m = 1)
    m = m.where((detections.z_source == "apple") & ~detections.far, 1.0)

    # begränsa extremvärden från felaktiga djupvärden
    return m.clip(0.4, 2.5).fillna(1.0)


# Räkning

def strip_count(detections, shift, W, spacing, offset):
    """Äpplen som räknas av var `spacing`:e bildruta: varje äpple räknas bara i den utvalda bildruta
    vars mittpunkt ligger närmast det i scenen, med hänsyn till äpplets avstånd."""
    # * Tänk på passagen som en lång panoramabild. Varje utvald bildruta "äger" remsan närmast sin mittpunkt,
    # * och ett äpple räknas bara av den bildruta som äger dess plats: varje äpple räknas en gång.

    # 1. utvalda bildrutor och säkra detektioner inom avståndsgränsen
    frames = np.arange(offset, len(shift), spacing)
    candidates = detections[(detections.conf >= CONF_KEEP) & ~detections.far]
    own_frame = candidates.frame.to_numpy().astype(int)
    m = candidates.m.to_numpy()

    # 2. sidledsavstånd från varje detektion till varje utvald bildrutas mittpunkt
    # ? (cx - W/2) = avstånd till egna bildrutans mitt; resten = kamerans färd dit, skalad med m
    dist = np.abs((candidates.cx.to_numpy() - W / 2)[:, None]
                  + m[:, None] * (shift[own_frame, 0][:, None] - shift[frames, 0][None, :]))
    nearest = dist.argmin(axis=1)

    # 3. räkna bara detektioner från den bildruta som äger platsen
    return candidates[frames[nearest] == own_frame]


def count_variants(detections, shift, W):
    """Antalet för var och en av de 12 räkningarna (avstånd K och startbildruta)."""
    # ? olika remsor ger lite olika antal: medelvärdet är stabilt, spridningen visar osäkerheten
    rows = [dict(k=k, offset=offset, count=len(strip_count(detections, shift, W, k, offset)))
            for k in SPACINGS for offset in range(min(k, 4))]
    return pd.DataFrame(rows)


# Helt synliga äpplen

def follow(start, by_frame, shift, n_frames):
    """Vyerna av ett räknat äpple i bildrutorna runt det (upp till VIS_WINDOW åt varje håll)."""
    views = [start]

    # följ äpplet framåt och sedan bakåt; sluta efter 2 missar i rad
    for step in (1, -1):
        current, frame, misses = start, int(start.frame), 0
        while misses < 2 and abs(frame + step - start.frame) <= VIS_WINDOW and 0 <= frame + step < n_frames:
            frame += step

            # 1. förutsäg platsen: senaste position + kamerarörelsen, skalad med äpplets m
            px = current.cx + current.m * (shift[int(current.frame), 0] - shift[frame, 0])
            py = current.cy + current.m * (shift[int(current.frame), 1] - shift[frame, 1])

            # 2. träff = närmaste detektion, tillräckligt nära och ungefär lika stor
            hit = None
            candidates = by_frame.get(frame)
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
    return float(values.median()) if len(values) else np.nan


def spread(values):
    """Relativ spridning (standardavvikelse / medelvärde)."""
    return float(values.std() / values.mean()) if len(values) > 1 else np.nan


def is_fully_visible(apple):
    """Är äpplet tillräckligt synligt för att mätas? Alla kontroller måste vara uppfyllda."""
    # ! ett saknat värde (NaN) gör kontrollen falsk: äpplet räknas, men mäts inte
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
    """Följ varje räknat äpple, avgör om det är helt synligt och mät det.
    Returnerar (apples, views): en rad per räknat äpple, och alla vyer som hittades."""
    # ? följ ner till CONF_LOW: ett äpple kan vara osäkert i några bildrutor utan att försvinna
    pool = detections[(detections.conf >= CONF_LOW) & ~detections.far]
    by_frame = {f: g for f, g in pool.groupby("frame")}

    rows, all_views = [], []
    for apple_id, start in counted.iterrows():
        views = follow(start, by_frame, shift, len(shift))
        with_depth = views[views.z_source == "apple"]

        row = dict(
            apple=apple_id,
            frame=int(start.frame),
            views=len(views),
            min_conf=float(views.conf.min()),
            max_aspect=float(views.aspect.max()),
            size_px=float(views["size"].median()),
            size_cv=spread(views["size"]),
            at_edge=bool(views.at_edge.any()),
            depth_views=len(with_depth),
            z_mm=median_or_nan(with_depth.z_mm),
            depth_spread=spread(with_depth.z_mm),
            occluded=median_or_nan(views.occluded.dropna()),
            diameter_raw_mm=median_or_nan(with_depth.diameter_raw_mm),  # före kalibreringen
        )
        row["fully_visible"] = is_fully_visible(row)

        rows.append(row)
        all_views.append(views.assign(apple=apple_id))

    return pd.DataFrame(rows), pd.concat(all_views) if all_views else pd.DataFrame()


# En inspelning

def analyse(recording, model, out, reuse=False):
    """Detektion, kamerarörelse, räkning och mätning för en inspelning. Returnerar en dict med allt."""
    meta = json.loads((recording / "meta.json").read_text())
    frame_paths = sorted((recording / "color").glob("*.jpg"))
    W = cv2.imread(str(frame_paths[0])).shape[1]

    # 1. detektion (eller sparade detektioner med --reuse)
    cache = out / "detections.csv"
    if reuse and cache.exists():
        detections = pd.read_csv(cache)
    else:
        detections = detect(recording, meta, model, frame_paths)

    # 2. kamerarörelse och hur fort varje äpple rör sig
    # ? äpplen bortom 3 m röstar inte: bakgrunden rör sig långsammare än trädet
    shift = camera_positions(detections[~detections.far], len(frame_paths))
    z_ref, beta = depth_motion(detections, shift)
    detections["m"] = motion_factor(detections, z_ref, beta)

    out.mkdir(parents=True, exist_ok=True)
    detections.to_csv(cache, index=False)

    # 3. räkna 12 gånger; räkningen närmast medelvärdet visas i videon och dess äpplen mäts
    counts = count_variants(detections, shift, W)
    typical = counts.iloc[(counts["count"] - counts["count"].mean()).abs().argmin()]
    counted = strip_count(detections, shift, W, int(typical.k), int(typical.offset))
    apples, views = measure_visible(counted, detections, shift)

    return dict(rec=recording, meta=meta, frames=frame_paths, det=detections,
                counts=counts, counted=counted, apples=apples, views=views)


def summarize(result, calibration):
    """Antal, medelvikt och skörd med rimliga intervall.
    Returnerar (summary, visible): sammanfattningen och de uppmätta äpplena med diameter och vikt."""
    # 1. antal: medel av de 12 räkningarna; intervallet = lägsta/högsta +- räknefelet
    # ! rimligt intervall, inte ett statistiskt konfidensintervall
    # TODO: lägg till en andra siffra för bara tydligt synliga äpplen (regeln med 55 % av bredden)
    counts = result["counts"]["count"]
    n = int(round(counts.mean()))
    count_low = int(round(counts.min() * (1 - COUNT_ERROR)))
    count_high = int(round(counts.max() * (1 + COUNT_ERROR)))

    # 2. vikt för de helt synliga äpplena
    scale = calibration["scale"]
    visible = result["apples"][result["apples"].fully_visible].copy()
    visible["diameter_mm"] = visible.diameter_raw_mm * scale
    visible["mass_g"] = sphere_mass_g(visible.diameter_mm)

    if len(visible) >= 5:
        masses = visible.mass_g.to_numpy()
        source = f"mätt med djup på {len(visible)} helt synliga äpplen"
    else:
        # ! för få uppmätta äpplen här: kalibreringsäpplena från träd 1 används i stället
        calibration_apples = pd.read_csv(CALIBRATION_DIR / "fully_visible_apples.csv")
        masses = sphere_mass_g(calibration_apples.diameter_raw_mm * scale).to_numpy()
        source = f"bara {len(visible)} helt synliga äpplen här; kalibreringsäpplena från träd 1 används"

    # 3. viktintervall: bootstrap över äpplena och storleksskalan
    rng = np.random.default_rng(1)
    scale_boot = np.array(calibration["scale_bootstrap"])
    boot = np.array([rng.choice(masses, len(masses)).mean() * (rng.choice(scale_boot) / scale) ** 3
                     for _ in range(4000)])
    mean_g = float(masses.mean())
    g_low, g_high = (float(x) for x in np.percentile(boot, [5, 95]))

    # 4. skörd = antal x medelvikt
    name = result["rec"].name
    summary = dict(
        recording=name,
        tree=TREE_OF.get(name, "?"),
        apples_counted=n,
        count_low=count_low,
        count_high=count_high,
        fully_visible_apples=len(visible),
        median_diameter_mm=float(visible.diameter_mm.median()) if len(visible) else None,
        mean_weight_g=mean_g,
        weight_low_g=g_low,
        weight_high_g=g_high,
        weight_source=source,
        estimate_kg=n * mean_g / 1000,
        kg_low=count_low * g_low / 1000,
        kg_high=count_high * g_high / 1000,
        size_scale=scale,
    )
    return summary, visible


def print_summary(s):
    print("=" * 64)
    print(f" {s['recording']}  ({s['tree']})")
    print(f"   Räknade äpplen  : {s['apples_counted']}   (rimligt intervall {s['count_low']} - {s['count_high']})")
    print(f"   Medelvikt       : {s['mean_weight_g']:.0f} g ({s['weight_low_g']:.0f}-{s['weight_high_g']:.0f}), "
          f"{s['weight_source']}")
    print(f"   Uppskattad skörd: {num(s['estimate_kg'])} kg   (rimligt intervall {num(s['kg_low'])} - {num(s['kg_high'])} kg)")
    print("=" * 64)


def process(recording, model, calibration, reuse=False):
    """Kör hela pipelinen på en inspelning och skriv resultaten. Returnerar sammanfattningen."""
    out = OUT / recording.name
    result = analyse(recording, model, out, reuse)
    summary, visible = summarize(result, calibration)

    # 1. siffror
    result["counts"].to_csv(out / "counts.csv", index=False)
    apples = result["apples"].merge(visible[["apple", "diameter_mm", "mass_g"]], on="apple", how="left")
    apples.to_csv(out / "apples_measured.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # 2. bilder och videor (påverkar inga siffror)
    visualize.crop_sheet(result, visible, lambda a: f"{a.diameter_mm:.0f} mm {a.mass_g:.0f} g", out / "measured_apples.png")
    visualize.render_audit(result, out)
    depth_frames = (load_depth(recording, frame_path, result["meta"])[1] for frame_path in result["frames"])
    visualize.render(result, summary, visible, out, depth_frames, CONF_KEEP, MAX_DEPTH_MM)

    return summary


# Kalibrering och kontroll

def calibrate(model, reuse=False):
    """Anpassa storleksskalan så att medianäpplet på träd 1 blir lika stort som referensäpplet."""
    # ! skriver över calibration/calibration.json och ändrar alla vikter
    # 1. mät de helt synliga äpplena i kalibreringsinspelningarna
    measured = []
    for name in CALIBRATION_RECORDINGS:
        result = analyse(ARCHIVE / name, model, OUT / "calibration" / name, reuse)
        visible = result["apples"][result["apples"].fully_visible]
        measured.append(visible.assign(recording=name))
        print(f"  {name[:12]:12s} helt synliga äpplen: {len(visible)}")

    # 2. skala = referensdiameter / uppmätt mediandiameter
    measured = pd.concat(measured, ignore_index=True)
    diameters = measured.diameter_raw_mm.to_numpy()
    scale = REF_DIAMETER_MM / float(np.median(diameters))

    # 3. osäkerheten i skalan (bootstrap), används för viktintervallet
    # ? samma fysiska äpplen syns i flera inspelningar, så osäkerheten blir något optimistisk
    rng = np.random.default_rng(0)
    boot = REF_DIAMETER_MM / np.array([np.median(rng.choice(diameters, len(diameters))) for _ in range(2000)])

    calibration = dict(
        scale=scale,
        scale_bootstrap=boot.round(5).tolist(),
        apples=len(diameters),
        median_raw_diameter_mm=float(np.median(diameters)),
    )

    # 4. spara
    CALIBRATION_DIR.mkdir(exist_ok=True)
    (CALIBRATION_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2))
    measured.to_csv(CALIBRATION_DIR / "fully_visible_apples.csv", index=False)
    visualize.depth_check_plot(measured, scale, OUT / "calibration" / "depth_check.png", REF_DIAMETER_MM)

    print(f"Storleksskala {num(scale, 3)} från {len(diameters)} helt synliga äpplen")
    return calibration


def evaluate(model):
    """Jämför detektorn med de handmarkerade äpplena: recall och precision per bildruta."""
    # ? precision är en nedre gräns: detektorn kan hitta äpplen som inte markerades
    rows = []
    for name, frame, points_csv in MARKED_FRAMES:
        recording = ARCHIVE / name
        meta = json.loads((recording / "meta.json").read_text())
        detections = detect(recording, meta, model, [recording / "color" / f"{frame:06d}.jpg"])
        boxes = detections[(detections.conf >= CONF_KEEP) & ~detections.far]
        points = pd.read_csv(points_csv)[["x", "y"]].to_numpy()

        # en markering är hittad när den ligger i en ruta (8 px marginal); varje ruta och markering används en gång
        boxes_hit, points_hit = set(), set()
        for i, box in boxes.iterrows():
            inside = [j for j, (px, py) in enumerate(points)
                      if j not in points_hit and box.x1 - 8 <= px <= box.x2 + 8 and box.y1 - 8 <= py <= box.y2 + 8]
            if inside:
                j = min(inside, key=lambda j: (points[j][0] - box.cx) ** 2 + (points[j][1] - box.cy) ** 2)
                boxes_hit.add(i)
                points_hit.add(j)

        rows.append(dict(
            frame=f"{name[:12]}:{frame}",
            marked=len(points),
            detections=len(boxes),
            recall=len(points_hit) / len(points),
            precision=len(boxes_hit) / max(len(boxes), 1),
        ))

    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT / "marked_frames_eval.csv", index=False)
    return table


def main():
    ap = argparse.ArgumentParser(description="Räkna äpplen och uppskatta skörden från RGB-D-inspelningar.")
    ap.add_argument("recordings", nargs="*", type=Path, help="inspelningsmappar")
    ap.add_argument("--reuse", action="store_true", help="återanvänd detections.csv från en tidigare körning")
    ap.add_argument("--evaluate", action="store_true", help="kontrollera detektorn mot de handmarkerade äpplena")
    ap.add_argument("--calibrate", action="store_true", help="anpassa storleksskalan på nytt")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)

    # ! saknas kalibreringen körs --calibrate automatiskt (tar lång tid)
    calibration_file = CALIBRATION_DIR / "calibration.json"
    if args.calibrate or not calibration_file.exists():
        calibration = calibrate(model, args.reuse)
    else:
        calibration = json.loads(calibration_file.read_text())

    if args.evaluate:
        print(evaluate(model).to_string(index=False))

    summaries = []
    for recording in args.recordings:
        summary = process(recording, model, calibration, args.reuse)
        print_summary(summary)
        summaries.append(summary)

    if summaries:
        columns = ["recording", "tree", "apples_counted", "count_low", "count_high", "fully_visible_apples",
                   "median_diameter_mm", "mean_weight_g", "estimate_kg", "kg_low", "kg_high"]
        # ? utf-8-sig så att Excel visar å, ä, ö rätt
        pd.DataFrame(summaries)[columns].round(1).to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
