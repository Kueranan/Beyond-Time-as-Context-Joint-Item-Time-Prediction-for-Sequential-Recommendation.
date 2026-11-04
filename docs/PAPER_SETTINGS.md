# ค่าฝึกที่อ้างอิง paper — Heatmap

**เอกสารนี้เป็นประวัติ config legacy วันที่ 3 พฤศจิกายน 2026** ปัจจุบันไฟล์เหล่านี้ย้ายไป
`configs/legacy/` แล้ว; `configs/pet.json`, `books.json`, `beauty.json` ใช้ author_release_v1
ดู [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md) สำหรับโค้ด/ค่าปัจจุบันและข้อจำกัดที่ตรวจพบ
คำสั่งด้านล่างที่อ้าง root configs เป็นคำสั่งในอดีต ไม่ใช่ config legacy ปัจจุบัน

ตรวจ PDF ที่ผู้ใช้ให้วันที่ 2026-11-03: Figure 8 และ 11 อยู่หน้า PDF 12
(เลขหน้าวารสาร 6374), รายละเอียดการทดลองอยู่ Section V-A หน้า PDF 8–9
ใช้ Heatmap ตามการทดลองหลักใน Section V-B

นี่คือการนำค่าที่ดีที่สุดในกราฟ sensitivity แต่ละกราฟมารวมกัน
**paper ไม่ได้ยืนยันว่าชุดค่าที่รวมนี้เป็น joint optimum หรือ config ที่ใช้สร้าง Table II**
ผลที่รันด้วย implementation นี้ยังต้องตรวจจาก validation ของเรา

## ค่าแยกตาม dataset

| ค่า | Pet | Books | Beauty | ที่มา |
|---|---:|---:|---:|---|
| `alpha` | 0.5 | 0.1 | 0.5 | Figure 8(a), จุดที่ทำเครื่องหมาย max ของ H@10/R@10 |
| `time_bins` | 30 | 52 | 52 | Figure 8(c), จุด max ของ Heatmap |
| σ² (วัน²) | 120 | 30 | 7 | จุดสูงสุดของทั้ง H@10/R@10 ใน Figure 11 |
| `sigma_days` = √σ² | 10.954451150103322 | 5.477225575051661 | 2.6457513110645907 | แปลง variance เป็น standard deviation ที่โค้ดรับ |

**ข้อขัดแย้งของ sigma:** ข้อความ Section V-G กล่าวว่า Beauty เหมาะกับค่ามาก
แต่ Figure 11 วาดผลสูงสุดที่ 7; config นี้ยึดกราฟ ไม่ใช่ข้อความบรรยาย
caption และช่วงค่าที่ทดลองระบุ σ² จึงตีความเป็น variance ตาม Eq. (19)
ในทางกลับกัน โค้ดเผยแพร่ของผู้เขียนใช้ `sigma = 30` เป็น standard deviation
และไม่ได้มีไฟล์ผล sweep เพื่อยืนยันความหมายของแกน Figure 11
การแปลง √σ² จึงเป็นการตีความตามสมการ/คำบรรยายรูป ไม่ใช่ค่าที่ตรวจยืนยันจาก run ต้นฉบับ

## ค่าร่วม

| ค่า | ค่าที่ตั้ง | หลักฐาน/สถานะ |
|---|---|---|
| predictor | heatmap | การทดลองหลัก Section V-B |
| item_dim | 172 | Section V-A |
| max_length | 32 | Section V-A |
| batch_size | 200 | Section V-A |
| max_days | 365 | Section V-A |
| epochs / patience | 100 / 20 | Section V-A; 100 คือจำนวนสูงสุด |
| candidate_count | 100 | Section V-A |
| time_dim | 100 | default ใน repository ผู้เขียน |
| heads / layers | 2 / 2 | default ของ encoder ใน repository ผู้เขียน |
| cpm_heads / cpm_layers | 4 / 2 | TransformerDecoder ใน `models/ITDRec.py` ของผู้เขียน |
| dropout | 0.1 | default ใน repository ผู้เขียน |
| learning_rate | 0.0001 | default ใน repository ผู้เขียน |
| optimizer / weight_decay | Adam / 0 | default ใน repository ผู้เขียน; trainer นี้ใช้ Adam |
| beta | 0.01 | default `time_weight`; **ไม่พบหลักฐานว่าเป็นค่าที่ดีที่สุดจาก paper** |
| negatives | 1 | หนึ่ง negative ต่อ positive ใน DataLoader ของผู้เขียน |
| negative target days | Uniform[365,730] | paper Section IV-C / Table V; โค้ดผู้เขียนใช้ positive gap + 365 |
| device | mps | สำหรับ Mac เครื่องนี้; paper ใช้ RTX 3090 |
| seed | 0 | รันเดี่ยว; paper เฉลี่ย 5 runs โดยใช้ seeds 0–4 |
| max_train_batches / max_eval_users | 0 / 0 | ไม่จำกัดจำนวน batches/users |
| threads | 2 | ค่าการทำงานของเครื่องนี้ ไม่ใช่ hyperparameter ที่ paper ยืนยัน |

Source ผู้เขียนที่ตรวจ: commit `af804c01186a492050555007916de4f3e68ba5b8`:

- [utils/load_configs.py](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/utils/load_configs.py)
- [models/ITDRec.py](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/models/ITDRec.py)
- [utils/DataLoader.py](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/utils/DataLoader.py)
- [utils/utils.py](https://github.com/yifeiHuang623/ITDRec/blob/af804c01186a492050555007916de4f3e68ba5b8/utils/utils.py)

## หน่วย time loss และขอบเขตการอ้างผล

config ใหม่ใช้ `heatmap_density_units: "days"`: Gaussian target เป็น density ต่อวัน
ตาม Eq. (19): `exp(-0.5*((grid_days-target_days)/sigma_days)^2)/(sqrt(2*pi)*sigma_days)`
ไม่ normalize ให้ผลรวมทุก bin เท่ากับ 1 และไม่เพิ่ม label smoothing
โค้ดผู้เขียน normalize Gaussian บน grid และเพิ่ม uniform smoothing 0.1 จึงต่างจาก Eq. (19)
ยังไม่มีหลักฐานยืนยันว่ารูป sensitivity สร้างจาก Gaussian แบบใด

config/checkpoint เก่าที่ไม่มี field นี้ยังใช้ `"horizon"` ตามเดิม โดย density
สูงกว่าแบบต่อวัน `max_days` เท่าเมื่อใช้ sigma เดียวกัน
ดังนั้น **อย่าเทียบค่าตัวเลข time loss เก่ากับใหม่โดยตรง** การเปลี่ยนหน่วย target
ไม่ใช่หลักฐานว่าโมเดลเรียนรู้ดีขึ้น ให้ดู validation ranking และ time MAE ประกอบ

trainer นี้ยังใช้ MSE เฉลี่ยตาม bins และเฉลี่ย item/time loss ตาม candidates
ส่วนโค้ดผู้เขียนบวก positive mean และ negative mean และตัด zero-gap บางตำแหน่งออกจาก time loss
การมี alpha เท่ากันจึงไม่ได้รับประกัน gradient หรือผลเหมือนการทดลองต้นฉบับ
ข้อแตกต่างด้าน CPM, time encoding, training prefixes และ preprocessing ยังอยู่ตาม
[PAPER_MAPPING.md](PAPER_MAPPING.md) นี่จึงเป็น **config อ้างอิง paper สำหรับ implementation นี้**
ยังไม่ใช่การ reproduce ต้นฉบับครบทุกขั้นตอน และไม่รับประกันว่าดีกว่ารันเก่า

## คำสั่งเริ่มรันใหม่

รันจาก root ของโปรเจกต์ เลือกทีละ dataset เพื่อไม่ให้แย่ง GPU:

```bash
caffeinate -i .venv/bin/python -m itdrec.train --config configs/pet.json
caffeinate -i .venv/bin/python -m itdrec.train --config configs/books.json
caffeinate -i .venv/bin/python -m itdrec.train --config configs/beauty.json
```

ผลไปที่ `outputs/pet_paper_settings_seed0`, `outputs/books_paper_settings_seed0`,
`outputs/beauty_paper_settings_seed0` แยกจากรันเก่าและ smoke tests
`caffeinate -i` ป้องกัน idle sleep ระหว่างรัน ไม่ได้รับประกันการทำงานเมื่อพับฝา Mac
checkpoint เลือกจาก validation NDCG@10; patience 20 จึงอาจหยุดก่อน epoch 100
ค่าที่ใช้จริงบันทึกใน `outputs/<run>/config.json` และ `best.pt`

หากต้องการครบ 5 seeds ตาม paper ให้ใช้ชื่อ output ใหม่แยกแต่ละ seed เช่น Books:

```bash
for seed in 0 1 2 3 4; do
  caffeinate -i .venv/bin/python -m itdrec.train --config configs/books.json \
    --seed "$seed" --output-dir "outputs/books_paper_5seeds/seed_${seed}" || break
done
```

การเปลี่ยน `--seed` ไม่เปลี่ยนชื่อ output อัตโนมัติ จึงต้องระบุ `--output-dir` ด้วย
เมื่อครบแล้วจึงรายงานค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐานของ metrics จากทั้ง 5 runs
อย่าเปลี่ยน config เหล่านี้เป็น Binary/Regression แล้วอ้างว่าใช้ค่าที่ดีที่สุดของ head นั้น
