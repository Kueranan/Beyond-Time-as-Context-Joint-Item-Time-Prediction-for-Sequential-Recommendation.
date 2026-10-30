# ข้อมูลที่ใช้

แหล่งข้อมูลคือ [Amazon Reviews 2023 ของ McAuley Lab](https://amazon-reviews-2023.github.io/)
ตาม Section V-A ของ paper ที่ผู้ใช้ให้

| ชื่อ CLI | หมวดที่ตรง paper | ปีเริ่มต้น | interactions ขั้นต่ำ |
|---|---|---:|---:|
| `pet` | Pet_Supplies | 2021 | 10 |
| `books` | Books | 2020 | 15 |
| `beauty` | Beauty_and_Personal_Care | 2020 | 15 |

ใช้ไฟล์ `.csv.gz` จากหน้า [Pure IDs 0-core](https://amazon-reviews-2023.github.io/data_processing/0core.html)
ส่วน **Before Data Splitting** ซึ่งทางผู้เผยแพร่ทำ de-duplication แล้ว
ไม่ใช้ split สำเร็จรูปของเว็บ เพราะ paper ใช้ cluster split ต่างออกไป

## ความหมายของข้อมูล

```text
user_id,parent_asin,rating,timestamp
<ผู้ใช้>,<สินค้าแม่>,<คะแนนรีวิว>,<Unix timestamp หน่วย milliseconds>
```

ใช้ `parent_asin` เป็นสินค้า ไม่ใช้ `asin` ของตัวเลือกย่อย
ใช้ทุก interaction เป็น implicit positive โดยไม่กรอง rating
`timestamp` เป็นเวลาที่เกี่ยวข้องกับรีวิว ไม่ใช่ transaction log ที่ยืนยันเวลาซื้อจริง
จึงอธิบาย output ว่าเวลาของ interaction และไม่อ้างว่าเป็นเวลาซื้อที่ทราบแน่นอน

## ขั้นตอนของ prepare.py

1. อ่าน gzip ทีละแถวลงฐาน SQLite ชั่วคราว และกรองช่วงปีด้วย UTC
2. ตัดแถวซ้ำ `(user,item,timestamp)` เพิ่มเติมจากการ deduplicate ของต้นทาง
3. เลือกผู้ใช้ที่มีจำนวน interaction ถึงขั้นต่ำหลังกรองปี
4. ถ้าใช้ `--max-users` ให้เลือกผู้ใช้ทั้งคนด้วย hash ของ seed กับ user ID
   การเลือกนี้รักษาประวัติครบของผู้ใช้ที่เลือก ไม่ตัดไฟล์กลางทาง
5. เรียงแต่ละผู้ใช้ตาม timestamp แล้วใช้รหัสสินค้าเป็นตัวตัดสินกรณีเวลาเท่ากัน
6. เริ่ม cluster ใหม่เมื่อ gap ของเหตุการณ์ติดกัน **มากกว่า** 1 วัน
7. ใช้ cluster สุดท้ายเป็น test, รองสุดท้ายเป็น validation และก่อนหน้านั้นเป็น train
8. ตัดผู้ใช้ที่มีน้อยกว่า 3 clusters หรือมี train น้อยกว่า 2 เหตุการณ์
9. แปลงรหัสสินค้าเป็นเลขเริ่มจาก 1; 0 สงวนไว้เป็น padding

ตัวอย่าง:

```text
เวลา:      Jan 1, Jan 2 | Jan 6, Jan 6 | Jan 12, Jan 12
cluster:       A       |      B       |       C
split:       train    | validation   |      test
```

ฝึกจากแต่ละ prefix ภายใน train เท่านั้น เช่น `[item1] -> item2`
validation ใช้ประวัติ train ทำนายทุกสินค้าใน cluster B พร้อมกัน
test ใช้ประวัติ train+validation ทำนายทุกสินค้าใน cluster C พร้อมกัน
ไม่ป้อนเหตุการณ์แรกของ test กลับไปเพื่อช่วยทำนายเหตุการณ์ถัดไปใน cluster เดียวกัน

## ขอบเขตปีที่ต้องทราบ

paper เขียน “records from 2021” และ “from 2020” โดยไม่ได้แสดงเงื่อนไขกรองวันที่
ที่ชัดเจนในข้อความที่ตรวจอ่าน จึงมีโหมดเลือกปีอย่างเปิดเผย:

- `--year-mode single` (ค่าเริ่มต้น): ตั้งแต่ 1 ม.ค. ถึงก่อน 1 ม.ค. ปีถัดไป
- `--year-mode since`: ตั้งแต่ 1 ม.ค. ของปีนั้นจนสิ้นสุดข้อมูลที่มี

ควรตรวจสถิติกับไฟล์ preprocessing ต้นฉบับของผู้เขียนก่อนอ้างว่า reproduce dataset ได้ตรงกัน
สถิติจาก pipeline นี้อ่านได้จาก `stats.json` ไม่ได้นำจำนวนใน Table I มาใส่แทนผลประมวลผลจริง

## Negative sampling และ catalog

ช่วงฝึกสุ่ม negatives โดยไม่เลือกสินค้าใน prefix และ positive ปัจจุบัน
ไม่อ่าน labels ใน validation/test เพื่อกำหนด negatives ของ train
อาจเกิด false negative สำหรับสินค้าที่ผู้ใช้ซื้อจริงในอนาคตได้ ซึ่งเป็นข้อจำกัดของ sampled implicit feedback
หากสินค้าเหลือไม่พอให้สุ่มซ้ำใน train ได้; หากไม่เหลือเลยจะ padding และ mask loss

ช่วง validation/test ทำตาม candidate protocol ใน paper:
เก็บ positives ทั้ง cluster แล้วเติม negatives ที่ไม่อยู่ในประวัติทั้งหมดของผู้ใช้
ให้ครบ 100 รายการ โดยสุ่มแบบไม่ซ้ำและคงที่เมื่อใช้ seed เดิม
ถ้า catalog เล็กเกินไปจะใช้เท่าที่มี และรายงาน `short_candidate_lists`
ถ้า positives เกิน 100 จะเก็บ positives ทั้งหมด และรายงานจำนวนกรณีดังกล่าว

catalog ใช้ IDs จากทุก split ของผู้ใช้ที่เก็บไว้ (transductive)
แต่ไม่ใช้ target interactions ฝึกโมเดล; สินค้าที่ไม่มี positive ใน train
ยังไม่มีหลักฐานการซื้อให้เรียนรู้ อาจปรากฏเป็น negative ระหว่างฝึกได้
รายงาน `cold_positive_fraction` เพื่อให้เห็นผลกระทบนี้
split ต่อผู้ใช้ตาม paper ไม่ใช่ global temporal split และไม่ได้รับประกัน
ว่า training events ของผู้ใช้อื่นเกิดก่อน test event ของผู้ใช้ทุกคน

## ไฟล์ที่ได้

- `data/raw/<Category>.csv.gz`: ข้อมูลจริงครบหมวดในรูปบีบอัด
- `data/raw/<Category>.csv.gz.json`: source URL, ขนาด และ SHA-256 ที่คำนวณตอนโหลด
- `data/processed/<name>/sequences.json`: ประวัติและ index ของ split ต่อผู้ใช้
- `data/processed/<name>/items.json`: mapping เลขสินค้าไป `parent_asin`
- `data/processed/<name>/stats.json`: จำนวนจริง ตัวเลือก preprocessing และ hash ต้นทาง

hash ช่วยตรวจว่าไฟล์ที่ใช้ยังตรงกับตอนดาวน์โหลด ไม่ใช่ checksum ที่ผู้เผยแพร่รับรอง
checkpoint เก็บ hash ของ sequences และ item mapping เพิ่มเติม เพื่อป้องกันนำไปใช้กับ ID mapping ผิดชุด
ข้อมูลและ checkpoint ถูก ignore จาก Git; เก็บเฉพาะโค้ดและเอกสารที่ใช้สร้างซ้ำ
