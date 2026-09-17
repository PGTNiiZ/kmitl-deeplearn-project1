# Thai Character Recognition

โปรเจกต์จำแนกภาพตัวอักษรและตัวเลขไทย 72 คลาสด้วย PyTorch, `timm` และ Optuna

สมาชิกใหม่ให้เริ่มที่ **[คู่มือใช้โปรเจกต์และทำงานร่วมกัน](docs/00-team-guide.md)**
คู่มือนี้อธิบายตั้งแต่ติดตั้ง เตรียมข้อมูล รัน notebook ทดลองโมเดล ส่งผล และทำงาน
ร่วมกันผ่าน Git แบบทีละขั้น

## เริ่มแบบสั้นที่สุด

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
cp -n .env.example .env
python -m src.download_data
python -m jupyterlab
```

จากนั้นรับ dataset และโฟลเดอร์ `data/splits` จากผู้ดูแลข้อมูลตาม
[คู่มือทีม](docs/00-team-guide.md) แล้วเปิด `notebooks/02-model-family-search-lab.ipynb`
เลือก kernel `.venv` รันตั้งแต่ต้น และตรวจว่าขึ้น `✅ REAL DATA พร้อมใช้งาน`
ก่อนเริ่มเทรน

ถ้าต้องการทบทวนเนื้อหาที่เรียนก่อน ให้เริ่มจาก
[00-cnn-concepts-from-class.ipynb](notebooks/00-cnn-concepts-from-class.ipynb)

## กติกาหลักของทีม

- ทุกคนใช้ dataset และ `data/splits` ชุดเดียวกัน
- เปลี่ยนทีละปัจจัยต่อหนึ่งการทดลอง
- เลือก checkpoint จาก validation macro F1 และรายงาน accuracy ประกอบ
- ใช้คะแนนจาก artifact ที่บันทึกไว้ ไม่คัดลอกจากหน้าจอ
- ไม่ commit `.env`, dataset, weights หรือ `results/`

## ลำดับไฟล์

เอกสารและ notebook เรียงชื่อแบบ `00-file-name` แยกเลขในแต่ละโฟลเดอร์
ดู [สารบัญไฟล์และลำดับการอ่าน](docs/06-file-tree.md) เพื่อเลือกไฟล์ที่ต้องใช้

## โครงสร้างที่ใช้บ่อย

```text
configs/       config ของแต่ละการทดลอง
data/          dataset cache และ split กลาง
docs/          คู่มือ แผนงาน และสรุปผล
notebooks/     notebook กลางและ template ของสมาชิก
results/       ผลทดลองในเครื่อง
src/           โค้ด audit, split, train, search และ inference
tests/         ชุดทดสอบ
weights/       checkpoint ที่คัดเลือกแล้ว
```

แผนการทดลองโดยละเอียดอยู่ที่ [02-team-experiment-plan.md](docs/02-team-experiment-plan.md)
และบันทึกผลที่ทีมรับรองอยู่ที่ [04-experiments.md](docs/04-experiments.md)
