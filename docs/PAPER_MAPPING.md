# สมการใน paper อยู่ตรงไหนในโค้ด

อ้างอิง PDF ที่ผู้ใช้ให้และ [โค้ดผู้เขียน](https://github.com/yifeiHuang623/ITDRec)
ตรวจ source `models/ITDRec.py` เพื่อเทียบรายละเอียด แต่ implementation นี้เขียนใหม่
ให้ใช้ interface เดียวกันทั้ง training และ inference และแยกส่วนอธิบายได้ง่าย

| ส่วนใน paper | โค้ด | ความหมาย |
|---|---|---|
| IV-A, Eq. 1 | `models/time_encoding.py` | learnable frequencies ของ sin/cos |
| IV-A, Eq. 2–5 | `models/encoder.py` | concatenation ของ item, absolute time, relative time แล้วเข้า causal Transformer |
| IV-B, Eq. 6 | `ITDRec.decode()` | dot product ของ history ส่วน item กับ candidate embedding |
| IV-B, Eq. 7–10 | `models/cpm.py` | history เป็น query; candidate เป็น key/value |
| IV-C, Eq. 11–12 | `RegressionHead` | ประมาณช่วงเวลาแบบจำนวนจริงและใช้ MSE |
| IV-C, Eq. 13–17 | `BinaryHead` | cumulative binary classifiers และ first threshold crossing |
| IV-C, Eq. 18–21 | `HeatmapHead` | Gaussian prototype, MSE และ argmax bin |
| IV-D, Eq. 22–24 | `losses.py` | BCE ของ time-decayed item logits บวก time loss |
| IV-D, Eq. 25–26 | `ITDRec.decode()` | rerank ด้วยเวลาที่ทำนาย แล้วจัดอันดับจากมากไปน้อย |
| V-A | `data/prepare.py`, `data/dataset.py` | temporal clusters และ 100-candidate evaluation |

## รายละเอียดที่ต้องตีความและความต่างที่ตั้งใจ

1. **Continuous time encoding.** สมการ (1) เขียน sin/cos จำนวน `dT` คู่
   แต่ข้อความระบุ output กว้าง `dT`; ที่นี่กำหนดความถี่ `dT/2` ตัวเพื่อให้ output กว้าง `dT` จริง
   โค้ดผู้เขียนใช้ cosine พร้อม learnable bias ขณะที่ที่นี่ใช้คู่ sin/cos ตามข้อความสมการ
   ทั้ง absolute และ relative encoding ใช้ frequency parameters ร่วมกัน

2. **Relative time.** ใช้เวลาแต่ละเหตุการณ์ลบเวลาล่าสุดของ prefix ที่เห็น (`t - t_recent`)
   ส่วน absolute ใช้ `t - t_first` ของประวัติหลัง truncate ตามนิยามใน IV-A
   ใน paper มีอีกประโยคกล่าวถึง gap ของเหตุการณ์ติดกัน; implementation นี้ยึดนิยาม `t - t_recent`
   ไม่ใช้เวลา future target คำนวณ input การฝึกสร้างแต่ละ prefix แยกกัน
   จึงไม่มี future timestamp หลุดเข้าไปใน representation ของตัวอย่างฝึก

3. **Encoder.** ใช้ PyTorch `TransformerEncoderLayer` แบบ causal พร้อม residual/LayerNorm มาตรฐาน
   รายละเอียด residual ใน Eq. (5) และการ normalize ใน repository ผู้เขียนไม่เหมือนกันทุกจุด
   ใช้ right padding พร้อม key-padding mask เพื่อไม่ให้เกิด attention แถวที่ถูก mask ทั้งหมด
   เลือก hidden state ณ เหตุการณ์จริงสุดท้ายด้วย `lengths - 1`

4. **CPM.** ยึดสมการที่เขียนแบบ candidate-wise: history เป็น query,
   embedding ของ candidate เดียวเป็น key/value แยกกันในแต่ละคู่ผู้ใช้–สินค้า
   ตามคณิตศาสตร์ softmax ที่มี key เดียวจะมีค่า 1 จึงเป็น candidate-conditioned residual
   ไม่ใช่ attention ที่เลือกหลายเหตุการณ์ในประวัติ และ projection ของ query/key
   จะไม่มี gradient ที่มีความหมายจาก attention weights ในกรณี singleton นี้
   ต้องระบุข้อจำกัดนี้ ไม่อธิบายเกินสิ่งที่โค้ดคำนวณจริง
   โมดูลไม่มีการผสม query positions นอกเหนือจาก encoder จึงคำนวณเฉพาะ query สุดท้ายได้
   โดยให้ผลเหมือนคำนวณทุก history query แล้วเลือกตัวสุดท้าย
   repository ผู้เขียนใช้ TransformerDecoder และ memory หลาย candidate/ตำแหน่ง
   ซึ่งมี behavior ต่างกัน; ที่นี่เลือกเวอร์ชันที่ไม่เปลี่ยนคำทำนายเมื่อสลับ candidate order
   หรือแบ่ง catalog เป็น chunks มี test ตรวจ invariant นี้

5. **Regression.** ใช้ Softplus ที่ output เพื่อบังคับเวลาต้องไม่ติดลบ
   MSE คำนวณกับ `days / max_days` แทนวันตรง ๆ เพื่อควบคุมสเกล gradient
   นั่นทำให้สเกล loss ต่างจากการใช้วันอยู่ `max_days²` เท่า; ค่า alpha ต้อง tune ให้เข้ากัน

6. **Binary.** labels เป็น cumulative `I(bin_boundary >= target_time)` ตาม Eq. (15)
   ไม่ใช่ one-hot classification ใช้ BCEWithLogits และเลือก boundary แรกที่ sigmoid >= 0.5
   ถ้าไม่ผ่านทุก bin คืน `2 * max_days` classifiers อิสระไม่ได้รับประกัน monotonic probabilities

7. **Heatmap.** ข้อความกำหนด negatives ใน `[max_days,2*max_days]`
   แต่ g(i) ที่อ้างข้ามจาก Binary ครอบคลุมแค่ horizon เดียว
   ที่นี่ใช้ grid `[0,2*max_days)` ตามช่วงที่โค้ดผู้เขียนใช้ decode heatmap
   และใช้ **ตำแหน่ง argmax** ไม่ใช่ค่าความสูงตามถ้อยคำที่อาจอ่านคลาดเคลื่อนใน Eq. (21)
   Gaussian เป็น density ในพิกัด `time / max_days`, sigma ก็แปลงหน่วยตามกัน
   output เป็น real-valued heatmap trained with MSE ไม่ใช่ softmax probability mass
   การเปลี่ยนหน่วย density ทำให้สเกล time loss และค่า alpha ต่างจากการใช้ density ในหน่วยวัน

8. **Loss และ reranking.** ใช้ `item_logit * exp(-beta * days)` ตาม Eq. (22), (25)
   train ใช้ label time; inference ใช้ predicted time เท่านั้น
   ถ้า item logit ติดลบ การ decay ทำให้คะแนนเข้าใกล้ศูนย์และอาจอันดับสูงขึ้น
   จึงไม่รับประกันว่าสินค้าที่ทำนายว่าจะเกิดเร็วขึ้นจะอันดับดีกว่าเสมอ
   เก็บรูปสมการเดิมไว้และรายงาน raw ranking ควบคู่เพื่อให้ตรวจผลจริงได้
   ถ้าจะเปลี่ยนเป็น decay ของ probability หรือ subtractive penalty ต้องนับเป็น model variant

9. **Training examples.** สร้างทุก prefix ใน train เป็นตัวอย่าง next-item แยกกัน
   โค้ดผู้เขียนคำนวณ loss หลายตำแหน่งใน sequence พร้อมกัน
   เป้าหมาย next-item เหมือนกัน แต่ batching, dropout และ optimizer trajectory ไม่เหมือนกัน
   ไม่ได้เพิ่ม baseline 10 ตัวที่ paper ใช้ เพราะขอบเขตงานนี้คือโมเดลหลัก ITDRec และ 3 time heads

10. **Experimental settings.** `configs/pet.json` ใช้ item embedding 172, sequence 32,
    batch 200, horizon 365 วัน, สูงสุด 100 epochs และ patience 20 ตามค่าที่ paper ระบุ
    time dimension, layers, dropout, bins, sigma, alpha, beta, learning rate และจำนวน negatives
    เป็นค่าทดลองที่ตั้งอย่างเปิดเผย ไม่อ้างว่าทุกค่าได้มาจาก paper
    ใช้ validation NDCG@10 เลือก checkpoint; paper ไม่ได้ระบุ metric เลือก checkpoint ชัดเจนในข้อความที่อ่าน
    หากจะเทียบผล paper ต้อง tune บน validation และรัน seeds 0–4 แล้วรายงานค่าเฉลี่ย

11. **Time evaluation.** ใช้ positive candidates ทั้ง cluster และเวลาจากท้าย history
    เป็นการวัดแบบทราบสินค้าจริง (item-conditioned) ไม่ใช่ joint top-1 event accuracy
    รายงาน MAE/RMSE เทียบเวลาจริงที่ไม่ clip พร้อม MAE ของ target ที่ clip horizon แยกอีกค่า
    คะแนน rank แบบ sampled 100 candidates ห้ามเทียบตรงกับ all-item ranking

12. **Data preparation.** ใช้ pure IDs ที่ต้นทาง deduplicate มาแล้ว
    ช่วงปีและการตัด tie timestamps ต้องระบุให้ชัด ดู `DATASETS.md`
    ไม่อ้างว่าจำนวนผู้ใช้/สินค้าเท่ากับ Table I โดยยังไม่มี preprocessing ต้นฉบับยืนยัน

การมีข้อแตกต่างเหล่านี้ทำให้โปรเจกต์เหมาะกับการอ่านโค้ด อธิบายแนวคิด
และเริ่มทดลองต่อ การอ้างผล reproduce แบบตรงต้นฉบับต้องตรวจ preprocessing,
CPM interpretation และ hyperparameters เพิ่มเติมด้วย
