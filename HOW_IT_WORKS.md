# How the apple yield system works

This explains, step by step, how we go from a video of an apple tree to "about 100 apples, about 11.6 kg". It covers what each step does and why it is done that way. The code is `Main/pipeline_3d.py` (computation) and `Main/visualize.py` (videos and images).

---

## The big picture

A camera is moved slowly past a tree. It records two things at the same time:

- **A colour video**: ordinary pictures, like a phone camera (1280 x 720 pixels).
- **A depth video**: for every pixel, *how far away* that point is, in millimetres. This is what makes it an "RGB-D" camera (RGB = colour, D = depth). We use an Orbbec Gemini 2.

From that, the system answers three questions:

1. **How many apples are on the tree?** (counting)
2. **How big is a typical apple?** (sizing)
3. **So how many kilograms is that?** (yield)

```
 colour + depth video
        │
        ▼
 ┌──────────────────────┐
 │ 1. Prepare the data  │  line up depth with colour, fill small gaps
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 2. Find apples (AI)  │  YOLO draws a box around every apple in every frame
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 3. Clean up boxes    │  remove duplicate boxes, remove far-away "apples"
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 4. Camera motion     │  how far did the camera move between frames?
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 5. Count each apple  │  "strip sampling": every apple counted once
 │    once              │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 6. Measure clear     │  only apples seen clearly, using box size + distance
 │    apples in mm      │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 7. Yield             │  count × average apple weight = kg
 └──────────────────────┘
```

The hard part is not finding apples in one picture. It is **counting each apple exactly once** across hundreds of frames, where the same apple shows up again and again, sometimes hidden behind a leaf, sometimes not.

---

## Step 1: Preparing the data ("preprocessing")

**What is *not* done:** we do not brighten, sharpen, or colour-correct the images. We tried that (brightening, CLAHE contrast boosting) and it did not help the detector, so the colour frames go into the AI exactly as recorded.

**What *is* done** is preparing the depth data, because it does not come ready to use:

- **Lining up depth with colour.** The depth sensor and the colour sensor are two separate cameras a few centimetres apart, with different lenses and resolutions (depth is 1280 x 800, colour 1280 x 720). So pixel (500, 300) in the depth image is *not* the same point as pixel (500, 300) in the colour image. The camera's `meta.json` file tells us exactly where each sensor sits and how its lens bends light (the "intrinsics" and "extrinsics"). Using that, every depth pixel is turned into a 3D point in space and then projected into the colour camera's view. After this, each colour pixel has a distance attached.
  *Why:* the AI finds apples in the colour image, and we then need to know how far away *that* apple is.

- **Filling small gaps.** That projection leaves a fine grid of empty pixels (like a sieve). Each empty pixel gets the nearest surface's distance from its close neighbours.
  *Why:* otherwise a small apple might land in a hole and get no distance at all.

---

## Step 2: Finding apples (the AI)

### What the AI is

We use **YOLO11** (medium size), a well-known object detector. You give it a picture, it returns boxes: "there is an apple here, and I am 87% sure". It is a neural network, a large mathematical function with millions of adjustable numbers, which has learned what objects look like by being shown many labelled examples.

### Where its knowledge comes from

1. **Pre-training (done by others).** YOLO comes already trained on COCO, a public collection of over 100,000 everyday photos labelled with 80 kinds of objects. "Apple" is one of them (class number 47). So out of the box it already roughly knows what an apple is, but mostly from kitchen and market photos, not from apples hanging in a tree among leaves.

2. **Fine-tuning (done by us, `train.py`).** We took 7 frames from tree 1, let the model guess the apples, and then corrected every box by hand in X-AnyLabeling. Then we trained the model further on those frames. Details that matter:
   - We **kept the "apple" class** from COCO instead of starting a new one, so we *adjust* what it already knows rather than teaching from zero.
   - We **froze the first 10 layers**. The early layers detect basic things like edges and textures that are useful for any picture; only the later, more "apple-specific" layers are adjusted. With only 7 images, this prevents the model from just memorising them.
   - **Augmentation:** during training the images are randomly flipped, scaled, slightly rotated, made lighter/darker, and stitched together in mosaics. That turns 7 pictures into many varied ones, so the model learns "apple" and not "these exact 7 pictures".

   *Result:* on frames the model never saw during training, the fine-tuned model scores **F1 0.855** against hand-marked apples (the pretrained one scored 0.80). F1 is a single number combining "how many real apples did it find" and "how many of its boxes are real apples"; 1.0 would be perfect.

### How it is run

- Every frame is fed to the model at **1920 pixels wide** (the frames are 1280 wide, so they are enlarged). Apples are often small in the picture; enlarging gives the model more pixels per apple and it finds more of the small ones.
- The model reports boxes down to 20% confidence. Two thresholds are then used:
  - **≥ 45% confidence: the box can be counted** as an apple.
  - **20-45%: only used to follow** an apple from frame to frame (e.g. when a leaf half covers it for a moment), never counted on its own.

---

## Step 3: Cleaning up the detections

### Duplicate boxes on one apple

The detector sometimes puts 2-5 boxes on the same apple (one around the whole apple, one around its brighter half, and so on). The standard cleanup inside YOLO misses these when the boxes have different sizes. So we add a rule: **if more than half of a box lies inside a more confident box, it is a duplicate and is dropped.** Two real neighbouring apples almost never overlap that much.

### Distance to each apple

For each box, we take the distances of the pixels in the **middle of the box** and use the median (the middle value). The centre is used because the edges of a box often include leaves or sky behind the apple. If the apple itself has no depth reading, the distance of its surroundings is used as a fallback.

### The 3-metre limit

Anything farther than **3.0 m** is treated as background and not counted. This removes things the AI mistakes for apples in the garden behind the tree: red buckets, dishes, rakes, apples on the *next* tree. Detections with no depth at all are kept (we don't throw away an apple just because the depth sensor missed it).
*Effect:* on the hand-marked frames, this made the detections more accurate (precision +1.5 to 2.7 points) and lost at most one real apple per frame.

---

## Step 4: Working out how the camera moved

To count each apple once, we need to know where every apple is *on the tree*, not just where it is in the picture. For that, we need to know how far the camera moved between frames.

**The trick: let the apples vote.** Between two frames, compare every apple in frame A with every apple in frame B, and note the displacement ("this one would have moved 23 px left"). Most pairs are random nonsense, but the pairs that really are the same apple all agree on roughly the same displacement. The displacement most pairs agree on is the camera's movement. If there is no clear winner (too few apples, a blurry frame), we assume the camera kept moving as it did a moment ago.

Adding these steps up gives the camera's position through the whole pass. Checked against a completely different method (optical flow, which tracks all the pixels), the two agree within 2%.

### Depth makes this better: near things move faster

When you look out of a train window, nearby fence posts fly past while distant hills barely move. The same happens here: when the camera travels sideways, an apple 1 m away crosses the picture much faster than one 2.5 m away.

We measured this on the recordings: **apple movement = 1 + β × (tree distance / apple distance − 1)**. In plain words: each apple moves a bit faster or slower than the average, depending on whether it is closer or farther than the tree's average distance. β (beta) says how much of the camera's movement is sideways travel (β = 1) versus turning on the spot (β = 0, where everything moves equally). Measured β was mostly 0.6-0.85.

So each apple gets its own speed, based on its own distance. This is the main way depth helps the *counting*.

---

## Step 5: Counting each apple once ("strip sampling")

This is the core of the system.

**The problem:** an apple stays in view for many frames as the camera passes. If we counted every apple in every frame, we would count it dozens of times. We tried "tracking" (giving each apple an ID and following it), but apples disappear behind leaves and reappear, and the trackers kept treating them as new apples: 2-6 times too many.

**The solution: don't track; divide the tree into strips.**

Imagine laying the tree out as one long panorama. Now:

1. Pick every K-th frame, e.g. every 5th.
2. Each picked frame "owns" the vertical strip of the tree around its own centre, up to halfway to the next picked frame.
3. In each picked frame, count **only the apples in its own strip**.

```
 tree panorama:   |  strip A  |  strip B  |  strip C  |  strip D  |
 picked frames:      frame 0     frame 5     frame 10    frame 15
```

Since the strips don't overlap and together cover the whole tree, every apple falls into exactly one strip and is counted exactly once, *without ever needing to recognise it as the same apple in another frame*. Each apple is also counted in the frame where it is closest to the middle of the picture, which is where it is seen most clearly.

With depth (step 4), each apple's strip position is computed using its own speed, so a near apple is placed correctly even though it moves faster through the picture.

**Doing it 12 times.** The result depends a little on which frames you happen to pick. So we do it with K = 3, 5 and 8, and with different starting frames, giving 12 variants. The reported count is their **average**; the lowest and highest variant (plus a 10% margin for detector errors) give the range.

**How well it works:**
- Inside the views of the hand-marked frames, it matches the hand count: 70 vs 68, 48 vs 45, 74-78 vs 76.
- An "audit video" numbers every counted apple; watching it showed about 1 double count per pass.

**A known extra:** the system also counts apples that are mostly hidden, where only a sliver shows. People usually don't count those. A rule "only count apples where at least 55% of the apple's width is visible" (using depth to know how big the apple *should* look at that distance) brings test_images from 100 down to 81, closer to what a person would say. It is not switched on yet; for yield, the hidden apples are still real apples.

---

## Step 6: Measuring apple size

### How a size in pixels becomes a size in mm

An apple that is 50 pixels wide could be a small apple close up or a big apple far away. With the distance from the depth camera we can tell which:

**real size ≈ size in pixels × distance ÷ focal length**

(the focal length, about 689 pixels for this camera, comes from `meta.json`). A small correction is added because depth measures the apple's front surface, not its centre.

### Only measure apples that are clearly visible

A half-hidden apple gives a box that is too small, so only apples that pass all of these are measured ("fully visible"):

- seen in at least 7 frames around the counted frame,
- always at least 60% confidence,
- round (box no more than 1.2 times wider than tall, or the other way),
- box size steady between frames (varies at most 10%; a leaf moving over it would change the size),
- distance steady between frames (at most 6% variation),
- nothing in front of it in the depth image (at most 15% of the apple covered),
- not cut off at the edge of the picture.

On the test recording of tree 1, 27 of the 100 apples pass.

### Calibration with the real apple

The raw measurements came out about 13% too small (boxes are drawn slightly tight, and the depth sensor sees the apple's front, among other things). To correct this, one apple was picked, measured and weighed by hand: **7.0 x 6.5 x 6.0 cm, 109 g**, described as average size for tree 1.

We then measured 92 fully visible apples in six other tree-1 recordings (not the test recording) and found one correction factor, **1.153**, that makes their typical (median) size match 65 mm. This factor is used for all recordings.

*Check:* on the test recording, which was not used for this, the median comes out 67 mm and 116 g, close to the reference.

### From size to weight

Treat the apple as a sphere: volume from the diameter, then weight = volume × density. The density (0.758 g/cm³) is chosen so that a 65 mm sphere weighs exactly 109 g, matching the reference apple. (A real apple is denser than that, but it is not a perfect sphere; this one number covers both.)

---

## Step 7: Yield

**Yield = number of apples × average weight of the fully visible apples in that recording.**

The range combines the lowest/highest count and the uncertainty in the average weight (resampling the measured apples many times to see how much the average jumps around). It is a "plausible range", not a strict statistical guarantee.

| Recording | Tree | Apples (range) | Measured apples | Avg. apple | Yield (range) |
|---|---|---|---|---|---|
| test_images | 1 | 100 (83-122) | 27 | 116 g | 11.6 kg (8.7-15.8) |
| Tree2 | 2 | 50 (40-61) | 6 | 141 g | 7.1 kg (4.4-10.6) |
| 1dceb275 (2nd pass, tree 2) | 2 | 43 (33-52) | 8 | 127 g | 5.5 kg (3.5-8.0) |

---

## Why it works (in short)

- **The AI only needs to do one thing well: find apples in a single picture.** That is a standard task, and fine-tuning on a few local frames was enough to get good at it.
- **Counting avoids the hardest problem (recognising the same apple again).** Strip sampling splits the tree into non-overlapping pieces, so each apple is counted once by construction.
- **Depth fixes the problems 2D can't:** it throws away far-away false apples, gives near and far apples their correct speed, and turns pixel sizes into millimetres.
- **One real, weighed apple anchors the whole size scale**, and it checks out on a recording that was not used to set it.

## What is tried and dropped (and why)

| Approach | Why it was dropped |
|---|---|
| Red-colour thresholding | Breaks with lighting, shadows, pale apples (tree 2), and red things in the garden |
| Circle detection (Hough) | Leaves and gaps between leaves look like circles too |
| Brightening / contrast boosting before the AI | Made no difference to detection |
| Object trackers (BoT-SORT) and grouping by identity | Lost apples behind leaves and recounted them: 2-3x too many |
| Remembering every apple's position over the whole pass, or tracking each apple by optical flow | Split apples into several "apples": 3-6x too many |
| Second AI that checks each box crop | Did not improve accuracy |
| Remembering apples ±2 frames around each counted frame | Worked, but counted 16% more than people do (it adds half-hidden apples) |

## Honest limitations

- The **whole-pass counts have not been checked against a full hand count** of a pass. Single frames have (about 7% count error).
- The whole **size scale rests on one weighed apple**, assumed to be from tree 1 and average.
- **Tree 2's weights come from only 6-8 measured apples**, hence the wide ranges.
- The AI was **fine-tuned on tree 1 only**. It still counts the odd leaf and misses some small apples in shade.
- **Mostly hidden apples are counted**, which people usually don't do (see step 5).
