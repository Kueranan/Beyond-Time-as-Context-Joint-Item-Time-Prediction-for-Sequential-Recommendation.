# ผลตรวจการทำงาน

ตรวจเมื่อวันที่ 30 ตุลาคม 2026 บน macOS / Apple Silicon โดยใช้ CPU
Python 3.13.3, PyTorch 2.15.0, NumPy 2.5.4 และ pytest 9.2.0

## สิ่งที่ตรวจแล้ว

- `python -m pytest -q`: **16 tests ผ่าน**
- `python -m compileall -q itdrec tests`: ผ่าน
- `git diff --check`: ผ่าน
- ตรวจ links ไปไฟล์ภายใน README และเอกสาร: ไม่พบลิงก์ขาด
- ตรวจ split boundaries ของทั้งสามหมวดทุกคน: ผ่านทั้งหมด
- โหลด checkpoint ของ demo Heatmap แล้วประเมินด้วย split/seed/user limit เดิม:
  ตัวเลขตรงกับ `metrics.json` ทุกค่า
- ทดลอง `itdrec.recommend` กับ checkpoint จริง: คืนสินค้าและเวลาทำนายได้
- ทดลอง `itdrec.explain` กับตัวอย่างจริง: shapes และ loss แสดงได้ครบ

tests ครอบคลุม temporal clustering, การกัน target ออกจาก history,
negative sampling บน catalog ขนาดเล็ก, ความละเอียด timestamp,
gradient และการลด loss ของทั้ง 3 heads, padding, candidate order/chunk invariance,
multi-positive metrics, checkpoint round-trip, mapping mismatch,
HTTP resume/restart และการตรวจไฟล์ดาวน์โหลดไม่ครบ

## Dataset ที่ดาวน์โหลดและเตรียมครบแล้ว

ทั้งสามไฟล์อยู่ใน `data/raw/` รวม **1,543,721,191 bytes (ประมาณ 1.54 GB)**
ตรวจขนาดและ SHA-256 เทียบ manifest แล้วครบทุกไฟล์
ขั้น prepare อ่าน gzip จนจบทุกไฟล์และสร้าง split ของผู้ใช้ทั้งหมดที่ผ่านเงื่อนไข

| หมวด | ขนาดไฟล์ gzip (bytes) | ปี | ผู้ใช้หลังกรอง | สินค้า | interactions |
|---|---:|---:|---:|---:|---:|
| Pet_Supplies | 385,011,761 | 2021 | 5,709 | 35,881 | 89,740 |
| Books | 602,182,123 | 2020 | 5,460 | 84,501 | 160,719 |
| Beauty_and_Personal_Care | 556,527,307 | 2020 | 3,513 | 37,855 | 108,767 |

ไฟล์พร้อมฝึกอยู่ที่ `data/processed/pet/`, `data/processed/books/`, `data/processed/beauty/`
แต่ละ directory มี `sequences.json`, `items.json`, `stats.json`

## ข้อมูล Pet ที่ประมวลผลจริง

ใช้ปี 2021 แบบ `single`, อย่างน้อย 10 interactions และ gap > 1 วันเริ่ม cluster ใหม่
อ่าน raw ทั้งหมด 16,597,408 แถว แล้วจึงกรองตามเงื่อนไข

| ชุด | ผู้ใช้หลังกรอง | สินค้า | interactions | train examples |
|---|---:|---:|---:|---:|
| `pet_demo` (เลือกสูงสุด 200 คนก่อน cluster filter) | 174 | 2,409 | 2,715 | 1,875 |
| `pet` (ผู้ใช้ที่ผ่านเกณฑ์ทั้งหมด) | 5,709 | 35,881 | 89,740 | 61,506 |

Demo มี positive test items ที่ไม่เคยเป็น positive ใน train ประมาณ 87.6%
สำหรับผู้ใช้ 50 คนที่นำมาประเมิน จึงไม่ใช่ตัวแทนประสิทธิภาพของชุดใหญ่

## ทดลองทั้ง 3 time heads บน Pet demo

ใช้ `configs/smoke.json`: 2 epochs, สูงสุด 20 batches/epoch,
batch 32, validation/test อย่างละ 50 ผู้ใช้, candidates 100, seed 0
เลือก checkpoint ด้วย validation NDCG@10

| Head | Best epoch | Test NDCG@10 | Test time MAE (วัน) |
|---|---:|---:|---:|
| Heatmap | 1 | 0.064253 | 29.826 |
| Regression | 1 | 0.079164 | 386.945 |
| Binary | 1 | 0.045586 | 141.251 |

checkpoint และรายงานอยู่ใน `outputs/pet_demo_<head>/`
ตัวเลขนี้ยืนยันว่าแต่ละ head ฝึก บันทึก ประเมิน และทำนายได้
ยังสรุปไม่ได้ว่า head ใดดีกว่า เพราะจำนวนขั้นฝึกน้อยและยังไม่ได้ tune สเกล loss
โดยเฉพาะ Regression/Binary ยังมีความคลาดเคลื่อนเวลาสูง

## ทดลอง config ขนาดปกติบน Pet ชุดเต็ม

```bash
python -m itdrec.train --config configs/pet.json \
  --data-dir data/processed/pet --output-dir outputs/pet_full_heatmap_2epochs \
  --epochs 2 --max-eval-users 200
```

ใช้ train examples ทั้ง 61,506 ตัวอย่างต่อ epoch แต่จำกัด validation/test ที่อย่างละ 200 คน
best checkpoint อยู่ epoch 2:

- Validation NDCG@10: **0.245741**
- Test Hit@10: **0.390000**
- Test Recall@10: **0.283426**
- Test NDCG@10 หลัง rerank: **0.198323**
- Test NDCG@10 ก่อน rerank: **0.200205**
- Test time MAE: **210.686 วัน**

การรันนี้ rerank ยังไม่ช่วยให้ test NDCG@10 สูงขึ้น และการทำนายเวลายังคลาดเคลื่อนมาก
จึงไม่ใช่หลักฐานยืนยัน improvement ตาม paper เป็นการตรวจระบบขนาดใหญ่ขึ้นในเวลาจำกัด
ไม่ใช่การฝึก 100 epochs หรือการเฉลี่ยผล 5 seeds

## อ่านผลและรันซ้ำ

ทดสอบ Heatmap บน Books และ Beauty จริงด้วย `configs/smoke.json` แล้วด้วย
(2 epochs, 20 batches/epoch, ประเมิน 50 ผู้ใช้ต่อ split):

| Run directory | Best epoch | Test NDCG@10 | Test time MAE (วัน) |
|---|---:|---:|---:|
| `outputs/books_heatmap_smoke` | 2 | 0.030286 | 80.896 |
| `outputs/beauty_heatmap_smoke` | 1 | 0.052138 | 34.258 |

เป็นผลตรวจ pipeline ด้วยข้อมูลจริงของแต่ละหมวด ไม่ใช่ benchmark ของโมเดลที่ฝึกจนจบ

ไฟล์ `outputs/<run>/config.json` เก็บค่าทดลอง เวอร์ชันไลบรารี สถิติข้อมูล
และ hash ของ sequences / item mapping
`history.json` เก็บ training loss และ validation metrics แต่ละ epoch
`metrics.json` เก็บผลของ checkpoint ที่เลือกพร้อมสถานะ `limited_run`

คำสั่ง README ใช้สร้างข้อมูลและทดลองซ้ำได้ โดยให้เลือก output directory ใหม่
เมื่อมี `best.pt` อยู่แล้ว การรันบน hardware/PyTorch ต่างเวอร์ชันอาจให้ตัวเลขต่างกันเล็กน้อย
ต้องใช้ preprocessing เดียวกันและปรับ hyperparameters ด้วย validation ก่อนเทียบกับงานต้นฉบับ
