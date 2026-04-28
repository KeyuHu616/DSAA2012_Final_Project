# Presentation Script (5 Minutes)

---

## **[Slide 1: Title Page]** (~15 seconds)
**[Keyu]**

"Good morning, everyone. Today we're presenting our Multi-Frame Story Generation System. Our team consists of three members: myself Keyu Hu, Siqi Chen, and Zhenzhuo Li. Our system transforms raw text scripts into consistent multi-frame image sequences."

---

## **[Slide 2: Outline]** (~10 seconds)
**[Keyu]**

"Here's our agenda: Pipeline overview, LLM Processor, Case Studies, Evaluation, Limitations, Future Work, and Q&A."

---

## **[Slide 3: Pipeline Overview]** (~30 seconds)
**[Keyu]**

"Our system follows a four-layer pipeline. Input is raw text with scene tags. The LLM processor analyzes the script for character and scene consistency. Output is a storyboard of generated images."

---

## **[Slide 4: System Architecture]** (~20 seconds)
**[Keyu]**

"We built four core components: Script Director for LLM parsing, Asset Anchor for character portraits, Core Generator using SDXL, and Evaluation Hub with CLIP and LPIPS metrics. On 32 test cases, we achieved 100% success rate with average CLIP Score of 0.288 and consistency of 0.386."

---

## **[Slide 5: Phase 1 - Story Analysis]** (~40 seconds)
**[Siqi]**

"The LLM Processor has three phases. Phase One is Story Analysis. The LLM identifies characters and assigns unique IDs - for example, Lily becomes Lily underscore 001. It extracts visual anchors: clothing, hairstyle, and color scheme. Actions are decomposed per frame."

---

## **[Slide 6: Phase 2 - Prompt Generation]** (~35 seconds)
**[Siqi]**

"Phase Two generates SDXL prompts. Based on the analysis, it creates detailed prompts ensuring cross-frame consistency. For multi-character scenes, it enforces LEFT and RIGHT spatial positioning to avoid confusion."

---

## **[Slide 7: Phase 3 - Refinement]** (~30 seconds)
**[Siqi]**

"Phase Three refines the output. It verifies consistency - detecting clothing drift. It resolves pronouns like 'he' or 'she' to specific character names. It uses visual description prefix matching to prevent character duplication - solving the 'two Milos' problem."

---

## **[Slide 8-11: Case Studies - Success Examples]** (~40 seconds)
**[Zhenzhuo]**

"Here are our success cases. Olivia shows excellent consistency - auburn hair and outfit maintained across frames. Lily preserves her round glasses and kitchen setting. Ethan maintains beach environment and lighting. Even dynamic scenes like Mark riding a bike show smooth transitions."

---

## **[Slide 12: Quantitative Results]** (~30 seconds)
**[Keyu]**

"Our evaluation uses two metrics: CLIP Score measures text-image alignment, target above 0.25. LPIPS Consistency measures inter-frame identity, target above 0.30. The table shows our results on both success cases and challenge cases like multi-character and animal scenes."

---

## **[Slide 13: Metrics Explanation]** (~20 seconds)
**[Keyu]**

"CLIP Score uses OpenCLIP ViT-L-14 to measure similarity between generated images and text prompts. LPIPS uses learned perceptual features to measure identity consistency between consecutive frames."

---

## **[Slide 14-16: Limitations - Challenge Cases]** (~45 seconds)
**[Siqi]**

"Now let's look at our limitations. Case 03 features a cat and dog - animal characters show inconsistent features and multi-character tracking is difficult. Case 06 has Jack and Sara - we see clothing changes between scenes and face identity drift. Case 13 is a robot, but SDXL generates it as a human, overriding our prompts."

---

## **[Slide 17: Summary of Limitations]** (~20 seconds)
**[Siqi]**

"In summary: we excel at single human character stories, scene consistency, and lighting. But we face challenges with animals, multi-character scenes, and abstract entities like robots."

---

## **[Slide 18: Future Work]** (~30 seconds)
**[Zhenzhuo]**

"For future work, we plan to integrate IP-Adapter and ControlNet for stronger character anchoring, implement cross-frame attention mechanisms, and use face embedding-based consistency loss. For non-human characters, we will develop specialized prompts and fine-tune SDXL. Long-term, we aim to extend to video generation with temporal consistency and integrate text-to-speech for narrated story output."

---

## **[Slide 19: Questions & Discussion]** (~25 seconds)
**[Keyu]**

"Our key contributions are: LLM-driven script parsing with visual anchors, CLIP-based character portrait anchoring, compressed visual memory bank for consistency, and a quantitative evaluation pipeline. Thank you! Are there any questions?"

---

## **[Backup Slides 20-21]** (If needed, skip unless asked)

**[Zhenzhuo]** - "For technical details, the portrait pipeline extracts visual descriptions, generates portraits with SDXL, and stores CLIP features for cross-frame consistency."

**[Zhenzhuo]** - "The Memory Bank compresses visual features to 128 dimensions with importance-weighted retrieval and temporal decay."

---

## Timing Summary

| Slide | Speaker | Content | Estimated Time |
|-------|---------|---------|----------------|
| 1 | Keyu | Title Page | 15s |
| 2 | Keyu | Outline | 10s |
| 3 | Keyu | Pipeline Overview | 30s |
| 4 | Keyu | System Architecture | 20s |
| 5 | Siqi | Phase 1 - Story Analysis | 40s |
| 6 | Siqi | Phase 2 - Prompt Generation | 35s |
| 7 | Siqi | Phase 3 - Refinement | 30s |
| 8-11 | Zhenzhuo | Case Studies (Success) | 40s |
| 12 | Keyu | Quantitative Results | 30s |
| 13 | Keyu | Metrics Explanation | 20s |
| 14-16 | Siqi | Limitations (Challenges) | 45s |
| 17 | Siqi | Summary of Limitations | 20s |
| 18 | Zhenzhuo | Future Work | 30s |
| 19 | Keyu | Q&A | 25s |
| **Total** | | | **~5 min 30 sec** |

---

## Speaker Notes

### Keyu
- Keep opening brief - audience cares about results
- Emphasize 100% success rate and 32 test cases
- Results table now shows both success AND challenge cases - good contrast
- Evaluation metrics are straightforward, don't over-explain
- Wrap up Q&A gracefully

### Siqi
- LLM phases are the core innovation - explain clearly
- Limitations section now has concrete examples: animals, multi-character, robots
- Be honest about challenges but frame as opportunities for Future Work
- Connect limitations to the Future Work section

### Zhenzhuo
- Case studies: point to specific visual details on the slides
- Future Work: show concrete technical direction
- Keep technical but accessible
