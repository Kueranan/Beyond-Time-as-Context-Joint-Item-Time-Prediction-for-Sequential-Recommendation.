# ITDRec — ทำนายสินค้าและเวลาของ interaction ถัดไป

## สิ่งที่แก้ใน branch นี้

- เพิ่ม pipeline `author_release_v1` โดยปรับ time encoding, encoder, CPM และ Heatmap
  ให้ตรวจเทียบกับโค้ดผู้เขียนได้ และแยก implementation เดิมไว้สำหรับ checkpoint เก่า
- ปรับ Heatmap target เป็น Gaussian แบบ normalized พร้อม smoothing และปรับ joint loss
  รวมถึงการสร้าง sequence และ negative time ให้เป็นไปตามโค้ดที่ผู้เขียนเผยแพร่
- แก้ขนาด attention mask ตอนประเมิน 100 candidates, การคำนวณ time loss เมื่อไม่มี
  time label ที่ใช้ได้ และการจัดการ item ID/เหตุการณ์ที่มี timestamp ซ้ำ
- ปรับการเตรียมข้อมูลให้เก็บผู้ใช้ที่มีน้อยกว่า 3 clusters สำหรับฝึก และตัดข้อมูลซ้ำ
  หลังอ่านข้อมูลเป็นชุด พร้อมเพิ่มรายงานเปรียบเทียบจำนวนกับ Table I
- แยก config Pet, Books และ Beauty และเลือก checkpoint ตามเกณฑ์ Hit/Recall
  ทั้ง 6 ค่าของโค้ดผู้เขียน โดยยังประเมิน checkpoint เวอร์ชันเดิมได้
- จับคู่เวลาทำนายกับสินค้าเป้าหมายในการประเมิน และเพิ่มตัวตรวจว่าเวลาที่ทำนาย
  เหมือนกันทุก candidate หรือไม่ รวมถึงสถิติการเปลี่ยนอันดับหลัง reranking
- เพิ่มโค้ดสร้างกราฟ item loss, time loss และ validation NDCG@10 จากผลการฝึก
- เพิ่มการทดสอบ pipeline และเครื่องมือเทียบผลคำนวณกับ repository ของผู้เขียน

รายละเอียดการตรวจและข้อจำกัดที่ยังเหลืออยู่ใน [REFERENCE_AUDIT.md](docs/REFERENCE_AUDIT.md)

โค้ด Python / PyTorch สำหรับศึกษา paper **Beyond Time as Context: Joint Item-Time
Prediction for Sequential Recommendation** โดย Huang et al., IEEE TKDE 2026
(DOI: `10.1109/TKDE.2026.3717641`)

โมเดลหลักคือ **ITDRec**: อ่านประวัติสินค้าและเวลาที่เกิดขึ้น แล้วให้ทั้งคะแนนแนะนำสินค้า
และระยะเวลาที่คาดว่าจะเกิด interaction กับสินค้านั้น แยกโมดูลเพื่ออ่าน อธิบาย
และเปลี่ยนส่วนประกอบได้ง่าย พร้อมคำอธิบายภาษาไทย

ค่าเริ่มต้นปัจจุบันใช้ **author_release_v1** ใน `itdrec/reference/` ซึ่งตรวจผลคำนวณ
กับ repository ผู้เขียนแบบตรึง commit แล้ว ส่วน implementation เดิมเก็บไว้เพื่ออ่านผลรันเก่า
อ่าน [REFERENCE_AUDIT.md](docs/REFERENCE_AUDIT.md) ก่อนรันจริง:
**ยังไม่ใช่ reproduction ที่ยืนยันผล paper** เพราะข้อมูลยังไม่ตรง Table I และ inference
ในโค้ดเผยแพร่มีปัญหาให้เวลาเหมือนกันทุก candidate ทำให้ reranking ไม่ช่วย

ใน workspace นี้ดาวน์โหลดข้อมูลจริงครบทั้ง 3 หมวดแล้ว (ประมาณ 1.54 GB)
ข้อมูลเดิมอยู่ใน `data/processed/pet`, `books`, `beauty`
ข้อมูลสำหรับตรวจ reference แยกเป็น `pet_reference_since`, `books_reference_since`, `beauty_reference_since`
มี checkpoint จากการทดลองจริง พร้อมรายงานใน [VERIFICATION.md](docs/VERIFICATION.md)

## เริ่มอ่านตรงไหน

1. [รายงานการแก้และข้อจำกัดต้นฉบับ](docs/REFERENCE_AUDIT.md)
2. [โมเดล reference](itdrec/reference/model.py) — encoder, CPM และ Heatmap ตาม release
3. [รายละเอียด dataset](docs/DATASETS.md) — แหล่งข้อมูล ช่วงปี และการแบ่งชุดข้อมูล
4. [เทียบสมการกับโค้ด](docs/PAPER_MAPPING.md) — จุดที่เหมือนและจุดที่ต้องตีความ
5. [ผลการตรวจและทดลองจริง](docs/VERIFICATION.md) — tests, สถิติข้อมูล และข้อจำกัดของผลรันสั้น

## โครงสร้างไฟล์

```text
configs/
  pet.json                  config และเส้นทางข้อมูล/ผลลัพธ์ของ Pet (MPS)
  books.json                config และเส้นทางข้อมูล/ผลลัพธ์ของ Books (MPS)
  beauty.json               config และเส้นทางข้อมูล/ผลลัพธ์ของ Beauty (MPS)
  smoke.json                ค่าขนาดเล็กสำหรับตรวจ pipeline บน CPU
  legacy/                   config ก่อนแก้ สำหรับ implementation เดิม
  reference/                สำเนา config author_release_v1 แยกตาม dataset
itdrec/
  reference/                pipeline ปัจจุบันที่ตรวจเทียบโค้ดผู้เขียน
    model.py                cosine time encoder, encoder layers, TransformerDecoder CPM, Heatmap
    data.py                 left padding, between-cluster time, หนึ่ง sequence ต่อผู้ใช้
    loss.py                 normalized Gaussian + smoothing และ source loss reductions
    train.py                trainer และ checkpoint selection ตาม release
    evaluate.py             ranking/time metrics และ diagnostics ของ reranking
    check_parity.py          เทียบผลคำนวณ/gradients กับ checkout ของผู้เขียน
    audit_data.py            เทียบจำนวนข้อมูลกับ Table I
  config.py                 ชนิดและเงื่อนไขของ hyperparameters
  data/
    download.py             ดาวน์โหลดข้อมูลจริงพร้อม SHA-256 manifest
    prepare.py              กรองปี จัดลำดับ แบ่ง temporal clusters
    dataset.py              สร้าง history / target / negative samples
  models/                   โมเดล legacy สำหรับ checkpoint เก่า
    time_encoding.py        เรียนรู้ representation ของเวลาแบบต่อเนื่อง
    encoder.py              Time-aware causal Transformer
    cpm.py                  Chronological Preference Mapping
    time_heads.py           Regression / Binary / Heatmap
    itdrec.py               โมเดลหลักที่รวมส่วนประกอบทั้งหมด
  losses.py                 Joint item-time objective
  metrics.py                Hit / Recall / NDCG แบบหลาย positive
  train.py                  ฝึก เลือก checkpoint จาก validation และทดสอบ
  evaluate.py               ประเมิน checkpoint เดิมซ้ำ
  recommend.py              แสดงสินค้าพร้อมเวลาที่ทำนาย
  explain.py                ไล่ดู tensor และการคำนวณจากข้อมูลจริงหนึ่งตัวอย่าง
tests/                      ทดสอบ split, leakage, loss, gradients และ metrics
docs/                       เอกสารอธิบายภาษาไทยและผลตรวจการทำงาน
data/raw/                   ไฟล์จริงที่ดาวน์โหลด (ไม่ใส่ Git)
data/processed/<dataset>/   sequences.json, items.json, stats.json (ไม่ใส่ Git)
outputs/<run>/              checkpoint, config, history, metrics (ไม่ใส่ Git)
```

## ติดตั้ง

ใช้ Python 3.10 ขึ้นไป แนะนำ virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

รันคำสั่งทั้งหมดจากโฟลเดอร์หลักของโปรเจกต์ ไม่ต้องติดตั้ง package แบบ editable
เครื่องนี้ได้เตรียม `.venv` ไว้แล้ว ใช้ `.venv/bin/python` แทน `python` ได้ทันที
เวอร์ชันที่ใช้ตรวจงานบันทึกใน `docs/VERIFICATION.md` และ `outputs/<run>/config.json`

## ดาวน์โหลดและเตรียมข้อมูล

paper ใช้ Amazon Reviews **2023** หมวด **Pet_Supplies**, **Books** และ
**Beauty_and_Personal_Care** ไม่ใช่ชุด Amazon 2014/2018 และไม่ใช่หมวด All_Beauty
ใช้ [pure IDs 0-core จาก McAuley Lab](https://amazon-reviews-2023.github.io/data_processing/0core.html)
ซึ่งมี `user_id,parent_asin,rating,timestamp` ครบสำหรับงานนี้

```bash
# ดาวน์โหลดทั้ง 3 หมวด หรือเปลี่ยน all เป็น pet / books / beauty
python -m itdrec.data.download --dataset all

# เลือกประวัติครบทั้งปีของผู้ใช้สูงสุด 200 คน สำหรับทดลองบน CPU
python -m itdrec.data.prepare --dataset pet --max-users 200 --output data/processed/pet_demo

# เตรียมข้อมูลสำหรับตรวจ reference (ยังไม่ยืนยันว่า records ตรง Table I)
python -m itdrec.data.prepare --dataset pet --year-mode since --retain-training-only --output data/processed/pet_reference_since
python -m itdrec.data.prepare --dataset books --year-mode since --retain-training-only --output data/processed/books_reference_since
python -m itdrec.data.prepare --dataset beauty --year-mode since --retain-training-only --output data/processed/beauty_reference_since
python -m itdrec.reference.audit_data
```

ตัวดาวน์โหลดใช้ไฟล์ `.part` ระหว่างรับข้อมูล แล้วเปลี่ยนชื่อเมื่อครบ
ไฟล์ที่ดาวน์โหลดแล้วจะถูกตรวจ hash และไม่โหลดซ้ำ หากการเชื่อมต่อขาดให้รันคำสั่งเดิม
โปรแกรมขอโหลดต่อด้วย HTTP Range; ถ้าเซิร์ฟเวอร์ไม่รองรับจะเริ่มไฟล์นั้นใหม่
ไม่ควรเปิดดาวน์โหลดหมวดเดียวกันพร้อมกันหลาย process

ค่าเริ่มต้นเลือก **เฉพาะปี 2021** และอย่างน้อย 10 interactions สำหรับ Pet;
**เฉพาะปี 2020** และอย่างน้อย 15 interactions สำหรับ Books/Beauty
ข้อความ “from 2021/2020” ใน paper ตีความเรื่องขอบเขตปีได้มากกว่าหนึ่งแบบ
ถ้าต้องการตั้งแต่ปีดังกล่าวจนสิ้นสุด dataset ให้ใช้ `--year-mode since`
ทุกตัวเลือกถูกบันทึกลง `stats.json`

## ฝึกโมเดล

```bash
# ทดสอบ pipeline อย่างรวดเร็ว: ขนาดโมเดลเล็ก, 2 epochs, จำกัด batches/evaluation users
python -m itdrec.train --config configs/smoke.json \
  --data-dir data/processed/pet_demo --output-dir outputs/pet_demo_heatmap

# ทดลอง author-release baseline: สูงสุด 100 epochs, patience 20
# อ่านข้อจำกัดเรื่องข้อมูลและ inference ใน REFERENCE_AUDIT.md ก่อน
python -m itdrec.train --config configs/pet.json
python -m itdrec.train --config configs/books.json
python -m itdrec.train --config configs/beauty.json

# เปลี่ยนเฉพาะหัวทำนายเวลา
python -m itdrec.train --config configs/smoke.json --predictor regression \
  --data-dir data/processed/pet_demo --output-dir outputs/pet_demo_regression
python -m itdrec.train --config configs/smoke.json --predictor binary \
  --data-dir data/processed/pet_demo --output-dir outputs/pet_demo_binary
```

config ของทั้งสาม dataset ใช้ `device: mps` สำหรับ GPU ของ Apple Silicon เป็นค่าเริ่มต้น
แต่ละไฟล์ระบุ `data_dir` และ `output_dir` ของตัวเอง จึงเลือก dataset ได้ด้วย `--config` อย่างเดียว
ใช้ `--device cpu` เพื่อรันบน CPU หรือ `--device cuda` เมื่อมี NVIDIA GPU
หากระบุ `--data-dir` หรือ `--output-dir` ในคำสั่ง จะใช้ค่าจากคำสั่งแทนค่าใน config
`smoke.json` ยังคงเป็น config ทดลองขนาดเล็กบน CPU

ทั้งสาม config ใช้ `implementation: author_release_v1` และ Heatmap เท่านั้น
alpha/bins จาก Figure 8, sigma เป็น standard deviation 30 วันและ smoothing 0.1 ตามโค้ดผู้เขียน
ค่ารวมยังไม่ใช่ joint optimum ที่ยืนยัน รายละเอียดอยู่ใน [REFERENCE_AUDIT.md](docs/REFERENCE_AUDIT.md)
ผลใหม่แยกไว้ที่ `outputs/<dataset>_author_release_seed0`
config เดิมใน [PAPER_SETTINGS.md](docs/PAPER_SETTINGS.md) ย้ายไว้ `configs/legacy/`
loss ต่างนิยาม/สเกลระหว่างเวอร์ชัน จึงไม่ควรเทียบขนาด loss ข้ามเวอร์ชันโดยตรง

reference เลือก `best.pt` เมื่อ validation Hit/Recall@5/10/20 ไม่ลดลงทั้ง 6 ค่า (ยอมรับ ties)
ตาม release แล้วจึงประเมิน test ส่วน legacy ยังเลือก best NDCG@10
มี `history.json`, `metrics.json`, และ `config.json` สำหรับอ่านผลย้อนหลัง
โปรแกรมจะไม่เขียนทับ `best.pt` ที่มีอยู่ ให้เลือกชื่อ output ใหม่เมื่อเริ่มทดลองใหม่

## Plot กราฟผลการเทรน

```bash
# อ่าน output_dir จาก configs/pet.json, books.json และ beauty.json
.venv/bin/python -m itdrec.plot

# เลือกรันเองได้หนึ่งหรือหลายโฟลเดอร์
.venv/bin/python -m itdrec.plot \
  --run-dir outputs/pet_paper_settings_seed0 outputs/books_paper_settings_seed0 \
  --output-dir outputs/plots_selected
```

โมดูล `itdrec/plot.py` อ่าน `history.json` และสร้างกราฟ 3 ส่วนแยกไฟล์:
`item_loss`, `time_loss` และ `validation_ndcg` โดยแต่ละกราฟมีเส้นแยกตามรัน
บันทึกทั้ง PNG และ PDF ใน `outputs/plots/` เป็นค่าเริ่มต้น
ดาวบนกราฟ validation ระบุ checkpoint ที่เลือกตาม pipeline ของรันนั้น
reference ใช้ Hit/Recall ทั้ง 6 ค่า ส่วน legacy ใช้ NDCG@10 สูงสุดครั้งแรก
หากประวัติยังไม่ครบ ดาวคือ checkpoint ที่เลือกเท่าที่มีข้อมูลในขณะนั้น
time loss เป็นค่าก่อนคูณ alpha; สเกลขึ้นกับ head/หน่วย target/config
จึงใช้ดูแนวโน้มภายในรัน ไม่ใช้ขนาด loss ข้าม dataset/config ตัดสินว่าโมเดลใดดีกว่า
คำสั่งนี้อ่านผลที่บันทึกไว้ ไม่เริ่มเทรนใหม่

## ประเมินผลและลองแนะนำสินค้า

```bash
python -m itdrec.evaluate --checkpoint outputs/pet_demo_heatmap/best.pt \
  --data-dir data/processed/pet_demo --split test

python -m itdrec.recommend --checkpoint outputs/pet_demo_heatmap/best.pt \
  --data-dir data/processed/pet_demo --user-index 0 --top-k 10

# checkpoint ที่ฝึกจาก train examples ของ Pet ชุดเต็ม 2 epochs ไว้แล้วใน workspace นี้
python -m itdrec.recommend --checkpoint outputs/pet_full_heatmap_2epochs/best.pt \
  --data-dir data/processed/pet --user-index 0 --top-k 10

python -m pytest -q
```

ถ้าต้องนำเสนอหรืออธิบายการทำงานทีละขั้น ให้ลอง
`python -m itdrec.explain --data-dir data/processed/pet_demo`
โปรแกรมแสดง input, shapes, คะแนน, เวลาทำนาย และ loss ของหนึ่งตัวอย่าง
โดยใช้โมเดลที่ยังไม่ฝึก เพื่อแสดงเส้นทางการคำนวณอย่างชัดเจน

`recommend` แสดงรหัสสินค้า `parent_asin`, คะแนน, จำนวนวันที่คาดว่าจะเกิด interaction
และ timestamp โดยใช้ประวัติก่อน test cluster เป็นค่าเริ่มต้น (`--history valid`)
ใช้ `--history all` เพื่อแนะนำหลังเหตุการณ์สุดท้ายที่มีอยู่ใน dataset
คะแนนจัดอันดับไม่ใช่ความน่าจะเป็นที่ calibrate แล้ว และเวลาคือการประมาณจากเวลารีวิว

รายงานมี Hit/Recall/NDCG@5,10,20 ทั้งก่อนและหลัง rerank (`raw_` คือก่อน rerank)
MAE/RMSE เวลาเป็นหน่วยวัน ประเมินเฉพาะสินค้าที่เป็น positive จริง
การประเมินจัดอันดับใช้ candidates ที่สุ่มไว้ตาม seed; การแนะนำจริงจัดอันดับทั้ง catalog
และตัดสินค้าที่อยู่ในประวัติออก จึงเป็นคนละ candidate protocol

## แหล่งอ้างอิง

- paper PDF ที่ผู้ใช้ให้ เป็นเอกสารหลักสำหรับโมเดลและ experimental protocol
- [Repository ของผู้เขียน](https://github.com/yifeiHuang623/ITDRec)
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
- [Pure-ID files](https://amazon-reviews-2023.github.io/data_processing/0core.html)

โค้ดในโปรเจกต์นี้เขียนใหม่ ไม่ได้คัดลอก repository ผู้เขียนมาเป็นส่วนหนึ่งของ package
ข้อมูลต้นฉบับอยู่ภายใต้เงื่อนไขของผู้เผยแพร่ dataset; LICENSE ของ repository นี้ใช้กับโค้ด
