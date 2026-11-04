# ตรวจและแก้ implementation เทียบโค้ดผู้เขียน — 4 พฤศจิกายน 2026

ค่าเริ่มต้น `configs/pet.json`, `books.json`, `beauty.json` ใช้ pipeline ใหม่
`implementation: author_release_v1` ใน `itdrec/reference/`
เก็บ config ก่อนแก้ไว้ใน `configs/legacy/` และยังโหลด checkpoint เก่าได้
ผลเก่า `*_paper_settings_seed0` เป็นผลของ legacy implementation ไม่ใช่ผล reproduce paper

**สถานะ: ตรวจความเทียบเท่ากับโค้ดเผยแพร่ได้ แต่ยังยืนยันผล Table II ไม่ได้**
โค้ดเผยแพร่เองมีข้อผิดพลาดและพฤติกรรมที่ขัดกับคำอธิบายใน paper
การเปลี่ยน architecture เพื่อหลีกเลี่ยงข้อผิดพลาดเชิงแนวคิดต้องระบุว่าเป็น model variant
จึงไม่แอบเปลี่ยนแล้วอ้างว่าตรงต้นฉบับหรือรับประกันคะแนนจะสูงขึ้น

## แหล่งที่ตรวจ

Repository: https://github.com/yifeiHuang623/ITDRec

Commit ที่ตรึง: `af804c01186a492050555007916de4f3e68ba5b8`

- [โมเดล](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/models/ITDRec.py)
- [DataLoader](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/utils/DataLoader.py)
- [Trainer](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/train_event_prediction.py)
- [Gaussian target](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/utils/utils.py)
- [Early stopping](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/utils/EarlyStopping.py)

ใช้ PDF ที่ผู้ใช้ให้สำหรับ Table I, Figure 8 และการตรวจข้อความขัดแย้ง
ไม่มี preprocessing script หรือ processed CSV ที่การทดลองใช้ใน commit นี้

## เปลี่ยนอะไร

| ส่วน | Legacy | Reference ใหม่ตามโค้ดเผยแพร่ |
|---|---|---|
| Item embedding | Normal(std=0.02), ไม่ scale | ค่าเริ่มต้น PyTorch Embedding, history คูณ √item_dim |
| Time encoding | คู่ sin/cos และ scale | cosine ของ Linear(1,time_dim) เรียนรู้ทั้ง frequency/phase |
| Absolute time | วันจาก event แรกของ history ที่เหลือ | วันจาก event แรกของ history ที่เหลือ เช่นเดียวกัน |
| Relative time | t − t_latest | t − เวลาสุดท้ายของ cluster ก่อนหน้า; cluster แรกเป็น 0 |
| Encoder | TransformerEncoder มาตรฐาน | LayerNorm(Q) → MHA(Q,history,history) → residual → Conv1d FFN ตาม release |
| Padding | right padding + key mask | left padding; zero features ระหว่าง layers แต่ไม่มี key-padding mask ตาม release |
| CPM | candidate เดียวเป็น memory | TransformerDecoder 2 layers/4 heads มี target self-attention และ memory หลายตำแหน่ง |
| Heatmap MLP | hidden dimension → hidden dimension → bins | hidden dimension → bins → bins |
| Heatmap target | Gaussian density ต่อวัน | Gaussian บน grid normalize + uniform smoothing 0.1 |
| Sigma | √ค่าจากกราฟ | standard deviation 30 วันตาม `convert_time()` |
| ตัวอย่างฝึก | ทุก prefix เป็นหนึ่งตัวอย่าง | หนึ่งลำดับต่อผู้ใช้ ใช้คู่ next-item สูงสุด 32 ตำแหน่งท้าย |
| Positive target time | gap จาก event ล่าสุด | between-cluster gap ของ target (ไม่ clip ใน training ตาม release) |
| Negative time | Uniform[365,730] | positive target time + 365 ตาม release |
| Negative item | exclude prefix และ target | exclude สินค้าทั้ง train sequence ของผู้ใช้นั้น |
| Loss reduction | เฉลี่ย candidates รวมกัน | positive mean + negative mean; time loss ตัด target gap=0 |
| Optimizer | Adam + gradient clipping | Adam ไม่มี gradient clipping ตาม release |
| Training order | shuffle | ลำดับผู้ใช้คงที่ตาม release |
| Checkpoint | best validation NDCG@10 | Hit/Recall@5/10/20 ต้องไม่ลดลงทั้ง 6 ค่า; ties ผ่าน |

ไฟล์อ่านตามลำดับ: `reference/data.py` → `reference/model.py` → `reference/loss.py`
→ `reference/train.py` → `reference/evaluate.py`
`itdrec/train.py` และ `itdrec/evaluate.py` เลือก pipeline จาก version ใน config/checkpoint

## สิ่งที่พบในโค้ดต้นฉบับ และขอบเขตการแก้

1. **Inference mask ผิดขนาด:** history ยาว 32 แต่ queries/candidates มี 100
   ทำให้ `predict()` ของ release เกิด RuntimeError ใน PyTorch ที่ใช้ตรวจ
   แก้ mask ให้มีขนาดตาม query length และอยู่ device เดียวกัน
2. **Candidate time เหมือนกัน:** release ทำซ้ำ last history hidden เป็น query เดียวกันทุก candidate
   ทุก query เห็น memory ชุดเดียวกัน ไม่มี candidate-specific position/input ที่ query
   เมื่อปิด dropout ระหว่าง eval จึงได้ heatmap เหมือนกันทุกตำแหน่ง (ภายใน floating-point tolerance)
   และมักได้ argmax bin เดียวกันทั้งหมด การคูณ logits ด้วย decay เดียวกันไม่เปลี่ยน ranking
   **ข้อจำกัดนี้ยังอยู่ใน reference เพื่อรักษาการคำนวณของ release**
   มี `constant_time_user_fraction`, `mean_candidate_time_span_days`,
   `reranked_top10_changed_fraction` ในทุกผล validation/test เพื่อเปิดเผยพฤติกรรม
3. **Training time decoder เห็น future positive memory:** cross-attention memory ของ release
   มี positive next-items หลายตำแหน่งและไม่มี causal memory mask จึงมี train–inference mismatch
   รักษาพฤติกรรมนี้ไว้เพื่อ parity ไม่อ้างว่า time branch เป็น autoregressive ที่ไม่มี future information ทุกจุด
   Inference ของเราไม่ส่ง labels/target times เข้าโมเดล
4. **Empty temporal labels:** release ได้ NaN เมื่อ batch ไม่มี positive gap ที่ไม่เป็นศูนย์
   แก้เป็น auxiliary loss 0 และยังฝึก item loss ส่วนผู้ใช้ที่ train มี event เดียวไม่มีคู่ฝึกจึงข้าม
5. **Item IDs:** release แปลง item IDs ด้วย offset ที่อาจเหลือ reserved ID ใน negative pool
   ใช้ canonical IDs 1..N และ 0 padding ในข้อมูลของเรา ไม่สร้าง fake negative item
6. **Tied timestamp merge:** release merge cluster features ด้วย `(user,timestamp)` อาจเพิ่มแถวซ้ำ
   ถ้ามีหลายสินค้าที่ timestamp เดียวกัน ที่นี่คำนวณ cluster บนรายการ event โดยตรง จึงไม่ทำ Cartesian merge
7. **Evaluation time metric:** release จับคู่เวลาทำนายที่เรียงตาม rank กับ target ที่ยังอยู่ลำดับเดิม
   และแทนคำทำนาย ≥365 วันด้วย baseline mean
   ที่นี่จับคู่ตาม candidate ที่แท้จริง รายงาน MAE/RMSE เป็นวันโดยไม่แทนด้วย mean
   จึงไม่อ้างว่า time MAE เทียบตรงกับ absolute-time accuracy ใน release ได้
8. **Evaluation randomness:** release สุ่ม candidates ใหม่ทุก epoch ผ่าน global RNG
   ที่นี่ตรึง candidates ต่อ split/seed/user เพื่อไม่ให้ early stopping เปลี่ยนเพราะสุ่มชุดข้อสอบใหม่
   Training negatives ใช้ RNG แยกตาม seed/epoch/user; distribution ตรง แต่ไม่รับประกัน draw sequence เดียวกัน
9. **Ranking averaging:** ค่าหลักใช้ค่าเฉลี่ยของ batch means ตาม release และเพิ่ม `user_mean_*`
   เพื่อเปิดเผยค่าเฉลี่ยต่อผู้ใช้ด้วย; positives/negative duplicates ใช้ semantics ของ release
   กรณี candidate count ไม่เท่ากันแบ่ง batch ย่อยโดยไม่ตัด positive ทิ้ง
10. **Test isolation:** ประเมิน test หลังโหลด checkpoint ที่เลือกแล้วเท่านั้น
    ไม่ประเมิน test ทุก 10 epochs ตาม release เพื่อลดการใช้ test ปรับงาน

การแนะนำทั้ง catalog ใช้ query เดียวแทน queries ที่ซ้ำกันใน eval เพื่อหลีกเลี่ยง attention ขนาด N²
โดยยังให้ query เห็น candidate memory ทั้งชุด ผลเทียบกับ full repeated queries อยู่ใน tests
ไม่แยก candidate memory เป็น chunks เพราะจะเปลี่ยนผลของ reference model

## Config และข้อมูล

ใช้ alpha/bins จาก Figure 8 ของ Heatmap: Pet 0.5/30, Books 0.1/52, Beauty 0.5/52
sigma=30 วัน, smoothing=0.1, beta=0.01, learning rate=0.0001 ตาม release
ไม่ได้อ้างว่าเป็น joint optimum หรือ exact config ที่สร้าง Table II
ยังใช้ MPS, batch 200, max length 32, สูงสุด 100 epochs, patience 20, seed 0

Table I ของ PDF หน้า 8 (เลขหน้าวารสาร 6370):

| Dataset | Users | Items | Interactions |
|---|---:|---:|---:|
| Pet | 23,904 | 126,988 | 344,667 |
| Books | 13,236 | 196,868 | 363,143 |
| Beauty | 21,837 | 184,362 | 472,091 |

ทดลองตีความ “from 2021/2020” เป็นตั้งแต่ปีนั้นจนสิ้นสุดข้อมูล (`--year-mode since`)
และเก็บผู้ใช้ที่มีน้อยกว่า 3 clusters ไว้ฝึกตาม DataLoader ของผู้เขียน
เก็บใน `data/processed/<dataset>_reference_since` แยกจากข้อมูล single-year เก่า
การตัดซ้ำยังใช้ `(user,item,timestamp)`; ยังไม่มี source preprocessing เพื่อยืนยันว่าใช้ key เดียวกัน
การมีจำนวนตรง Table I ก็ยังไม่ยืนยันว่าเป็น records/splits เดียวกัน

ผลเตรียมจริงครั้งนี้ **ยังไม่ตรง Table I**:

| Dataset | Users ใหม่ | Items ใหม่ | Interactions ใหม่ |
|---|---:|---:|---:|
| Pet | 31,168 | 99,949 | 517,901 |
| Books | 27,109 | 378,369 | 997,530 |
| Beauty | 29,083 | 195,152 | 1,037,259 |

จึงไม่สรุปว่า `since` คือ preprocessing ต้นฉบับ และไม่สุ่มตัดข้อมูลเพื่อให้ตัวเลขดูตรงตาราง
ต้องยืนยันช่วงเวลา, dedup key, product ID (`parent_asin` หรือ ASIN), และลำดับการกรองกับ preprocessing จริง

```bash
.venv/bin/python -m itdrec.data.prepare --dataset pet --year-mode since \
  --retain-training-only --output data/processed/pet_reference_since
.venv/bin/python -m itdrec.data.prepare --dataset books --year-mode since \
  --retain-training-only --output data/processed/books_reference_since
.venv/bin/python -m itdrec.data.prepare --dataset beauty --year-mode since \
  --retain-training-only --output data/processed/beauty_reference_since
.venv/bin/python -m itdrec.reference.audit_data
```

รายงาน `outputs/reference_audit/datasets.json` เปรียบเทียบจำนวนจริงกับ Table I ทีละค่า
trainer บันทึกผล comparison และ hash ของข้อมูล/implementation ลง `config.json`

## หลักฐาน numerical parity

```bash
git clone https://github.com/yifeiHuang623/ITDRec /tmp/itdrec-author
git -C /tmp/itdrec-author checkout af804c01186a492050555007916de4f3e68ba5b8
.venv/bin/python -m itdrec.reference.check_parity --source-dir /tmp/itdrec-author
.venv/bin/python -m pytest -q
```

ตัวตรวจยืนยัน commit และตรวจว่า source files ไม่ถูกแก้ก่อน import
ใช้ weights/input เดียวกัน ปิด dropout แล้วเทียบ encoder, positive/negative logits,
time outputs, Gaussian targets, joint loss และ gradients ของ 69 parameter tensors
ผ่านด้วย max absolute error **5.960464477539063e-08** ใน fixture ที่ตรวจ
เทียบ inference 100 candidates กับ release ที่ patch เฉพาะ mask size ผ่านด้วย
รายงานอยู่ `outputs/reference_audit/parity.json`
นี่เป็นหลักฐานความเทียบเท่าทางตัวเลขใน fixture ไม่ใช่หลักฐานว่าได้ benchmark score เดียวกับ paper

## การรันทดสอบและการเริ่ม experiment ใหม่

```bash
# ตรวจ pipeline ก่อน: ใช้โมเดลขนาดจริง แต่จำกัดงานและแยก output
.venv/bin/python -m itdrec.train --config configs/pet.json \
  --epochs 1 --max-train-batches 1 --max-eval-users 3 \
  --output-dir outputs/reference_check_pet

# หากต้องการทดลอง author-release baseline เต็ม (ยังมีข้อจำกัด inference ที่ระบุข้างต้น)
caffeinate -i .venv/bin/python -m itdrec.train --config configs/pet.json
# เปลี่ยน pet เป็น books หรือ beauty สำหรับอีกสอง dataset
```

ผลเต็มไป `outputs/<dataset>_author_release_seed0` และไม่เขียนทับรันเดิม
ยังไม่ควรทุ่มรัน 5 seeds เพื่ออ้าง reproduce Table II ก่อนแก้ข้อสงสัยต้นฉบับ:
ต้องมี preprocessing ที่ใช้จริง, config ของแต่ละ dataset และคำชี้แจง candidate-conditioned inference
จากผู้เขียน การทำ candidate-specific query/memory ใหม่เองเป็น research variant ไม่ใช่ reproduction ที่ยืนยันแล้ว
