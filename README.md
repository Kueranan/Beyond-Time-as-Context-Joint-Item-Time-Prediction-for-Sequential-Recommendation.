# ITDRec — ทำนายสินค้าและเวลาของ interaction ถัดไป

โค้ด Python / PyTorch สำหรับศึกษา paper **Beyond Time as Context: Joint Item-Time
Prediction for Sequential Recommendation** โดย Huang et al., IEEE TKDE 2026
(DOI: `10.1109/TKDE.2026.3717641`)

โมเดลหลักคือ **ITDRec**: อ่านประวัติสินค้าและเวลาที่เกิดขึ้น แล้วให้ทั้งคะแนนแนะนำสินค้า
และระยะเวลาที่คาดว่าจะเกิด interaction กับสินค้านั้น แยกโมดูลเพื่ออ่าน อธิบาย
และเปลี่ยนส่วนประกอบได้ง่าย พร้อมคำอธิบายภาษาไทย

นี่คือ implementation ที่เขียนใหม่จาก paper เพื่อให้เรียนรู้และทดลองได้
มีการระบุสมมติฐานและความต่างจากโค้ดผู้เขียนใน [PAPER_MAPPING.md](docs/PAPER_MAPPING.md)
ผลจากการรันสั้นใช้ตรวจว่า pipeline ทำงาน ไม่ใช่ผล reproduce ตารางใน paper

ใน workspace นี้ดาวน์โหลดข้อมูลจริงครบทั้ง 3 หมวดแล้ว (ประมาณ 1.54 GB)
และเตรียม split พร้อมใช้ไว้ใน `data/processed/pet`, `books`, `beauty`
มี checkpoint จากการทดลองจริง พร้อมรายงานใน [VERIFICATION.md](docs/VERIFICATION.md)

## เริ่มอ่านตรงไหน

1. [ภาพรวมและคำอธิบายทีละขั้น](docs/WALKTHROUGH_TH.md)
2. [โมเดลหลัก](itdrec/models/itdrec.py) — จุดเชื่อมทุกส่วน
3. [รายละเอียด dataset](docs/DATASETS.md) — แหล่งข้อมูล ช่วงปี และการแบ่งชุดข้อมูล
4. [เทียบสมการกับโค้ด](docs/PAPER_MAPPING.md) — จุดที่เหมือนและจุดที่ต้องตีความ
5. [ผลการตรวจและทดลองจริง](docs/VERIFICATION.md) — tests, สถิติข้อมูล และข้อจำกัดของผลรันสั้น

## โครงสร้างไฟล์

```text
configs/
  pet.json                  ค่าทดลองขนาดปกติ ใช้กับทั้ง 3 หมวดได้
  smoke.json                ค่าขนาดเล็กสำหรับตรวจ pipeline บน CPU
itdrec/
  config.py                 ชนิดและเงื่อนไขของ hyperparameters
  data/
    download.py             ดาวน์โหลดข้อมูลจริงพร้อม SHA-256 manifest
    prepare.py              กรองปี จัดลำดับ แบ่ง temporal clusters
    dataset.py              สร้าง history / target / negative samples
  models/
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

# ใช้ผู้ใช้ที่ผ่านเงื่อนไขทั้งหมดเพื่อทดลองขนาดปกติ
python -m itdrec.data.prepare --dataset pet
python -m itdrec.data.prepare --dataset books
python -m itdrec.data.prepare --dataset beauty
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

# ทดลองจริง: สูงสุด 100 epochs, early stopping patience 20
python -m itdrec.train --config configs/pet.json \
  --data-dir data/processed/pet --output-dir outputs/pet_heatmap

# เปลี่ยนเฉพาะหัวทำนายเวลา
python -m itdrec.train --config configs/smoke.json --predictor regression \
  --data-dir data/processed/pet_demo --output-dir outputs/pet_demo_regression
python -m itdrec.train --config configs/smoke.json --predictor binary \
  --data-dir data/processed/pet_demo --output-dir outputs/pet_demo_binary
```

ใช้ `--device cuda` เมื่อมี NVIDIA GPU หรือ `--device mps` สำหรับ Apple Silicon
ที่ PyTorch รองรับ; การตรวจงานครั้งนี้ใช้ CPU หากต้องการทดลอง Books/Beauty ให้เปลี่ยน
`--data-dir` และ `--output-dir` ให้ตรงหมวด โมเดลใช้โค้ดชุดเดียวกัน

ไฟล์ `best.pt` มาจาก validation NDCG@10 ที่ดีที่สุด แล้วจึงประเมิน test
มี `history.json`, `metrics.json`, และ `config.json` สำหรับอ่านผลย้อนหลัง
โปรแกรมจะไม่เขียนทับ `best.pt` ที่มีอยู่ ให้เลือกชื่อ output ใหม่เมื่อเริ่มทดลองใหม่

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
