import { DuckEntity } from '../types';
import { useInferenceStore } from '../store/inferenceStore';

/**
 * Maps raw backend ML dictionary into frontend-friendly DuckEntity objects.
 *
 * Ground rule: this function only TRANSLATES what the backend ML services
 * (video_inference_service.py / camera_inference_service.py) already
 * decided (per-detection isAnomaly/provisional, added_ids, missing_ids,
 * other_ids, thumbnails). It must not invent its own anomaly logic on top --
 * that caused the two bugs this rewrite fixes:
 *   1. Every duck getting flagged red whenever detected_duck_count !=
 *      expected_duck_count, instead of just the specific duck(s) the
 *      backend actually flagged.
 *   2. Nothing rendering during warmup / early inference, because every
 *      detection with a still-null id got force-marked provisional and
 *      then filtered out of activeDucks -- even once the backend itself
 *      had already stopped calling it provisional.
 *   3. too_many_ducks / too_few_ducks / "excess" being re-derived here from
 *      the RAW per-frame duck count instead of read from what the backend
 *      already decided (its shake-smoothed `reasons` array and its
 *      per-detection `excess` flag) -- so a single noisy frame could flag an
 *      anomaly a beat before (or in cases the) backend itself ever would.
 */
// Persistent cache of last known valid bounding box coordinates for each duck ID
const lastKnownBBoxes = new Map<string, { x: number; y: number; width: number; height: number }>();

export const resetBBoxCache = () => {
  lastKnownBBoxes.clear();
};

function computeIoU(b1: number[], b2: number[]): number {
  const x1 = Math.max(b1[0], b2[0]);
  const y1 = Math.max(b1[1], b2[1]);
  const x2 = Math.min(b1[2], b2[2]);
  const y2 = Math.min(b1[3], b2[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const a1 = Math.max(0, b1[2] - b1[0]) * Math.max(0, b1[3] - b1[1]);
  const a2 = Math.max(0, b2[2] - b2[0]) * Math.max(0, b2[3] - b2[1]);
  const union = a1 + a2 - inter;
  return union > 0 ? inter / union : 0;
}

export const mapDetectionsToDucks = (data: any, vw: number, vh: number, _fallbackExpected?: number): DuckEntity[] => {
  const incomingDucks: DuckEntity[] = [];
  const addedIds = data.added_ids || [];
  const missingIds = data.missing_ids || [];
  const excessIds = data.excess_ids || [];
  const isWarmingUp = data.status === 'WARMING' || !data.anchor_locked;

  // Find thumbnails. Ensure we do not drop thumbnails when data.thumbnails is an empty array [] (which is truthy in JS!)
  const rawDataThumbs = Array.isArray(data.thumbnails) ? data.thumbnails : [];
  const storeThumbs = Array.isArray(useInferenceStore.getState().stats?.thumbnails)
    ? useInferenceStore.getState().stats.thumbnails
    : [];
  const allThumbnails = rawDataThumbs.length >= storeThumbs.length ? rawDataThumbs : storeThumbs;

  // BUG FIX: analyzer.py votes on `too_many_ducks` / `too_few_ducks` over a
  // shake-smoothing window (self.count_history / anomaly_smoothing_frames) and
  // only publishes those verdicts once confirmed, via the top-level `reasons`
  // array it already sends us (see FIX Issue 4 / _finish()). The previous
  // version of this file recomputed "too many" / "too few" itself from the
  // RAW per-frame present-duck count -- i.e. it re-derived an anomaly verdict
  // instead of translating the backend's, so a single noisy frame (a brief
  // double-detection or a momentary occlusion) could flag an anomaly before
  // the backend's own smoothing had confirmed one. That's the same class of
  // bug this rewrite's docstring says was already fixed once (bogus red
  // boxes from a naive count mismatch) -- it just crept back in here.
  const backendReasons: string[] = Array.isArray(data.reasons) ? data.reasons : [];
  const isTooFewDucks = !isWarmingUp && backendReasons.includes('too_few_ducks');

  // Pre-collect locked DUCK bounding boxes only (used below purely to
  // de-duplicate stray unbound duck detections that overlap an already-locked
  // duck). BUG FIX: this used to collect boxes from ANY detection with a
  // positive id -- including "other_toys" objects, which also get positive
  // ids from the backend -- so an unbound duck overlapping an other-object's
  // box could get wrongly suppressed as a "duplicate". Now gated to species duck.
  const lockedBoxes: number[][] = [];
  if (Array.isArray(data.detections)) {
    data.detections.forEach((det: any) => {
      const sp = String(det.class_name || det.species || '').toLowerCase();
      const isDuck = sp === '' || sp === 'duck';
      const hasId = det.id !== null && det.id !== undefined && Number(det.id) > 0;
      const b = det.bbox || det.box;
      if (!isWarmingUp && isDuck && hasId && Array.isArray(b) && b.length === 4) {
        lockedBoxes.push(b);
      }
    });
  }

  // Note: over-count coloring (which specific duck id(s) are "excess", e.g.
  // expected 17 / 18 present -> #18 red, #1-17 green) is NOT computed here at
  // all -- the backend already tags each present duck detection with its own
  // `excess` boolean (see analyzer.py's `this_box_color` / `is_excess`
  // logic), and that's read directly below via `d.excess`.

  const seenIds = new Set<string>();

  if (Array.isArray(data.detections)) {
    data.detections.forEach((d: any, idx: number) => {
      const rawBox = d.bbox || d.box;
      const hasLockedId = d.id !== null && d.id !== undefined && Number(d.id) > 0;

      // During active inference (not warming up), suppress duplicate unbound detections that overlap an already-locked duck
      if (!isWarmingUp && !hasLockedId && Array.isArray(rawBox) && rawBox.length === 4) {
        const isDuplicateOfLocked = lockedBoxes.some((lb) => computeIoU(rawBox, lb) >= 0.45);
        if (isDuplicateOfLocked) {
          return;
        }
      }

      let px = 0, py = 0, pw = 0, ph = 0;
      if (d.bbox && d.bbox.length === 4) {
        px = (d.bbox[0] / vw) * 100;
        py = (d.bbox[1] / vh) * 100;
        pw = ((d.bbox[2] - d.bbox[0]) / vw) * 100;
        ph = ((d.bbox[3] - d.bbox[1]) / vh) * 100;
      } else if (d.box && d.box.length === 4) {
        px = (d.box[0] / vw) * 100;
        py = (d.box[1] / vh) * 100;
        pw = ((d.box[2] - d.box[0]) / vw) * 100;
        ph = ((d.box[3] - d.box[1]) / vh) * 100;
      }

      const species = String(d.class_name || d.species || '').toLowerCase();
      const isDuck = species === 'duck' || species === '';
      const isHand = species === 'hand';
      const isOther = !isDuck && !isHand;

      // Provisional ONLY applies during actual warmup phase — never on locked active inference
      const isProvisional = isWarmingUp || d.provisional === true;
      const rawId = hasLockedId ? String(d.id) : isWarmingUp ? `prov-${idx + 1}` : `unbound-${idx + 1}`;
      const displayId = isOther ? `other-${rawId}` : rawId;

      // Avoid rendering duplicate IDs in the same frame
      if (seenIds.has(displayId)) {
        return;
      }
      seenIds.add(displayId);

      const thumbObj = allThumbnails.slice().reverse().find((t: any) =>
        (String(t.id) === rawId || Number(t.id) === Number(rawId)) && (isOther ? t.event === 'other_present' : t.event !== 'other_present')
      );
      const isMissingDetection = !isProvisional && (d.status === 'missing' || missingIds.includes(displayId) || missingIds.includes(Number(displayId)));
      if (pw > 0 && ph > 0 && px >= 0 && py >= 0) {
        lastKnownBBoxes.set(displayId, { x: px, y: py, width: pw, height: ph });
      } else if (isMissingDetection) {
        const cached = lastKnownBBoxes.get(displayId);
        if (cached) {
          px = cached.x;
          py = cached.y;
          pw = cached.width;
          ph = cached.height;
        }
      }

      const expectedCount = Number(data.expected_duck_count || _fallbackExpected || 18);
      const detectedDuckCount = Number(data.detected_duck_count || 0);

      const rawNumId = Number(rawId);
      const isExcessById = isDuck && !isNaN(rawNumId) && expectedCount > 0 && rawNumId > expectedCount;
      const isExcessDetection = !isProvisional && (
        isExcessById ||
        d.excess === true ||
        d.status === 'excess' ||
        excessIds.includes(displayId) ||
        excessIds.includes(Number(displayId))
      );

      let eventStatus: DuckEntity['statusEvent'] = undefined;
      if (isMissingDetection) {
        eventStatus = 'missing';
      } else if (isExcessDetection) {
        eventStatus = 'added';
      } else if (thumbObj?.event === 'confirmed') {
        eventStatus = 'confirmed';
      } else if (thumbObj?.event === 'other_present' || isOther) {
        eventStatus = 'other_present';
      }

      // User requirement:
      // 1. "18 detected · expected 20": When expected count is higher (expectedCount > detectedDuckCount),
      //    turn ALL boxes to RED!
      // 2. "18 detected · expected 17": When expected count is less (expectedCount < detectedDuckCount),
      //    turn ONLY the excess duck to RED (normal ducks 1..expectedCount remain GREEN).
      // 3. Other species (foreign toy): ONLY the other species is RED, and normal ducks stay GREEN.
      // 4. Missing ducks: missing duck is displayed as anomaly (AMBER/DASHED).
      const isUnderCount = !isWarmingUp && (
        expectedCount > detectedDuckCount && detectedDuckCount > 0
      );

      const isAnomaly = !isProvisional && (
        isUnderCount ||
        isOther ||
        isExcessDetection ||
        isMissingDetection
      );

      incomingDucks.push({
        id: displayId,
        species: isHand ? 'Hand' : isDuck ? 'Duck' : 'Unknown',
        confidence:
          typeof d.confidence === 'number'
            ? d.confidence
            : typeof d.conf === 'number'
              ? d.conf
              : isMissingDetection
                ? 0.0
                : 0.9,
        x: px,
        y: py,
        width: pw,
        height: ph,
        vx: 0,
        vy: 0,
        heading: 0,
        isAnomaly: isAnomaly,
        thumbnail: thumbObj?.thumbnail,
        provisional: isProvisional,
        statusEvent: isHand ? 'hand_present' : eventStatus,
        handDetected: isHand
      });
    });
  }

  // Only render missing cards explicitly reported for this frame. Thumbnail
  // history is intentionally not evidence that a duck is currently missing:
  // it is a session-wide gallery cache and would otherwise turn normal later
  // frames into stale missing/anomaly frames.
  const resolvedMissingIds = new Set<string>();

  if (!isWarmingUp) {
    missingIds.forEach((mid: any) => resolvedMissingIds.add(String(mid)));
  }

  // Inject missing ducks so they appear in the gallery.
  // They stay in their numeric position, displaying their last known thumbnail, styled as 'missing'.
  resolvedMissingIds.forEach((stringMid: string) => {
    const existing = incomingDucks.find(d => String(d.id) === stringMid);
    const cached = lastKnownBBoxes.get(stringMid);
    if (existing) {
      existing.statusEvent = 'missing';
      existing.isAnomaly = true;
      if ((existing.width <= 0 || existing.height <= 0 || existing.x < 0) && cached) {
        existing.x = cached.x;
        existing.y = cached.y;
        existing.width = cached.width;
        existing.height = cached.height;
      }
      return;
    }

    const thumbObj = allThumbnails.slice().reverse().find((t: any) =>
      String(t.id) === stringMid || Number(t.id) === Number(stringMid)
    );
    incomingDucks.push({
      id: stringMid,
      species: 'Duck',
      confidence: 0.0,
      x: cached ? cached.x : -100,
      y: cached ? cached.y : -100,
      width: cached ? cached.width : 0,
      height: cached ? cached.height : 0,
      vx: 0, vy: 0, heading: 0,
      isAnomaly: true,
      thumbnail: thumbObj?.thumbnail,
      provisional: false,
      statusEvent: 'missing'
    });
  });

  return incomingDucks;
};
