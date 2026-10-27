# Teaching Aerial Imagery to See Height: Tracking Philadelphia's Gentle Densification with LiDAR-Supervised Foundation Models

**Xuanyu Zhou · AI Urban Sustainability · Final Project Proposal**

## 1. Background and Research Questions

Adding homes inside the existing city is one of the most sustainable ways to grow. Infill and vertical additions use existing streets, transit and utilities, avoid greenfield development, and cut per-capita travel and energy use. Philadelphia is a good place to study it. Its rowhouse neighborhoods are densifying one lot at a time: vacant lots are filled with new rowhouses, and existing homes gain an extra story (locally called "pop-ups"). Pop-ups are visible and contested enough that City Council has moved to restrict them in some neighborhoods [VERIFY: cite specific ordinance or drop this clause]. Yet the city has no consistent, building-level record of *where* the built fabric is actually growing. Permits record intent, not outcomes. Some permitted projects are never built, and some changes happen outside the permit system.

Airborne LiDAR measures building height directly, and Philadelphia has five LiDAR collections (2008, 2010, 2015, 2018, 2022). But LiDAR is flown only every few years and the latest is from March 2022. Aerial orthoimagery, by contrast, is flown almost every year (through 2025) at about 15 cm resolution, but contains no height. Recent work shows that a vision foundation model can learn to infer height from a single image when supervised by LiDAR: Meta and WRI's Canopy Height Maps v2 fine-tune a DINOv3 backbone on LiDAR-labeled imagery and report R² = 0.86 for tree height. We apply the same idea to buildings, using Philadelphia's multi-year LiDAR archive as free supervision.

- **RQ1 (AI):** Can a DINOv3 model fine-tuned on LiDAR-labeled aerial imagery estimate building height accurately enough to detect a one-story change (~3 m) in a year and a place it has never seen?
- **RQ2 (Urban):** Where, and how fast, has Philadelphia densified through new infill and vertical additions between 2018 and 2025? How is this growth distributed relative to transit access and neighborhood socioeconomic change?
- **RQ3 (Governance):** How well do detected changes line up with building permits? What share of permits result in visible construction, and how much change happens without a matching permit?

## 2. Data

| Data | Years | Role |
|---|---|---|
| City LiDAR point clouds (PASDA / OpenDataPhilly) | 2008, 2010, 2015, 2018, 2022 | Height labels; LiDAR-based change detection |
| City aerial orthoimagery (PASDA, ~15 cm) | annually 2008–2025 (gaps: 2013, 2021) | Model input |
| L&I building permits (OpenDataPhilly / Carto) | 2014–2025 (93–98% geocoded to parcels per year) | Validation and RQ3 |
| Parcel polygons (2016, 2025) and building footprints (2017) | — | Analysis units |
| ACS 5-year, SEPTA stops | 2013–2023 | RQ2 context |

## 3. AI Methods

**Stage 1: LiDAR change detection (completed in pilot).** For each parcel, I build normalized surface models (0.5 m) from the 2018 and 2022 LiDAR, remove vegetation using multi-return structure, align the two epochs vertically, and classify each parcel as new construction, vertical addition, rear extension or demolition. These LiDAR-derived changes are the reference labels for Stage 3.

**Stage 2: Height-from-imagery model (core AI component).** I pair each LiDAR year with same-year aerial imagery and fine-tune a satellite-pretrained DINOv3 backbone with a dense regression head to predict a per-pixel height map. Residual offsets between imagery and LiDAR (mainly building lean) are corrected by tile-level cross-correlation, and the loss tolerates 1–2 px misalignment. Baselines: a frozen DINOv3 with a linear head, and a U-Net trained from ImageNet weights.

**Data splits.** The test set is held out in *both time and space*. The model trains on 2010/2015/2018 imagery (and 2008 if LiDAR quality permits) in training zones, with separate spatial validation zones. It is tested once on **2022** imagery in held-out zones, including the pilot area (ZIP 19146), against 2022 LiDAR. The prediction set is 2023–2025 imagery, which has no LiDAR.

**Stage 3: Change detection from predicted heights.** Predicted heights are aggregated to building/parcel medians and differenced between years. If per-image errors are too large, a bi-temporal (Siamese) variant that directly predicts height change will be added.

**Evaluation.**
1. Height accuracy on 2022: building-level MAE and R².
2. Change detection, 2018→2022: recall and precision against the LiDAR-detected changes.
3. Indirect validation for 2023–2025: recall on post-2022 completed new-construction and addition permits, plus manual review of a random sample of detections.

## 4. Pilot Results (Feasibility)

A pilot in ZIP 19146 (Point Breeze / Graduate Hospital; 18,182 parcel units) confirms that Stage 1 works.
- **Recall:** LiDAR detects **96.6%** [94.7, 98.3] of new-construction permits and **87.1%** [80.6, 93.5] of vertical-addition permits issued and completed between the two LiDAR flights.
- **Precision:** In a blind manual review of 50 random detections, precision was **85.1%** [74.5, 95.7]. All new-construction and vertical-addition detections were real; errors came only from the rear-extension rule (cars and trees), which a contiguity filter largely removes.
- **Change counts, 2018–2022:** **456** new buildings and **398** pop-ups. Pop-ups cluster in Graduate Hospital; new construction clusters in Point Breeze.
- **Permits:** Every detected pop-up had a related permit on its own or an adjacent parcel, and manual checks confirmed only 4 changes without any permit. This suggests that in this area, large structural change is well captured by permits. The project therefore focuses on *where densification happens*, not on enforcement.

*(Figure 1: 2018→2022 change map of ZIP 19146 on a 100 m grid, with one pop-up shown in aerial imagery and a LiDAR profile.)*

## 5. Expected Outcomes

1. **Model:** a LiDAR-supervised building-height model for Philadelphia aerial imagery, with an honest, time- and space-held-out evaluation of whether it can detect one-story change.
2. **Maps:** building-height maps for 2018–2025 and a densification map (new infill and pop-ups) for the study area. This extends the record three years beyond the last LiDAR flight.
3. **Urban analysis:** densification rates by neighborhood and their relation to transit access and socioeconomic change (RQ2), plus a permit-to-construction comparison (RQ3).
4. **Interactive map:** aggregated results only. No individual address is labeled with permit status.

**Risks:** Single-image height errors may approach the one-story signal. To mitigate this, I aggregate per building and keep the bi-temporal fallback. Imagery also varies across years (camera, season, sun angle), which the multi-year training set is designed to cover.
