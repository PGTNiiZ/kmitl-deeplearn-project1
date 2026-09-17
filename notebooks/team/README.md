# แบ่ง 41 โมเดลให้ทีม 6–7 คน

ใช้ [03-team-model-template.ipynb](../03-team-model-template.ipynb) เป็น template กลาง หนึ่งครั้งของ Run All เท่ากับหนึ่งงานและเทรนหนึ่งโมเดล

งาน 1 คือ Custom CNN baseline งาน 2–41 คือ 40 model families ที่คัดจาก Notebook 02 รายการจริงอ่านจาก [team_family_assignments.json](../../configs/experiments/team_family_assignments.json)

## ตั้งค่าก่อนรับงาน

ทุกคนแก้ Cell 2:

```python
OWNER = "ชื่อผู้รัน"
TEAM_SIZE = 7
MEMBER_ID = 1
ROUND = 1
JOB_ID = None
```

ทุกคนต้องใช้ repository, `data/splits/`, `MODE`, seed, image size, batch size และสูตรฝึกเดียวกัน ส่วน path และ device เปลี่ยนตามเครื่องได้

## วิธีที่ 1: แจกตามรอบ

ปล่อย `JOB_ID = None` ระบบคำนวณงานด้วยสูตร:

```text
job_id = (ROUND - 1) * TEAM_SIZE + MEMBER_ID
```

ตัวอย่างทีม 7 คน:

| รอบ | สมาชิก 1 | สมาชิก 2 | สมาชิก 3 | สมาชิก 4 | สมาชิก 5 | สมาชิก 6 | สมาชิก 7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 2 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
| 3 | 15 | 16 | 17 | 18 | 19 | 20 | 21 |
| 4 | 22 | 23 | 24 | 25 | 26 | 27 | 28 |
| 5 | 29 | 30 | 31 | 32 | 33 | 34 | 35 |
| 6 | 36 | 37 | 38 | 39 | 40 | 41 | ไม่มีงาน |

ทีม 6 คนใช้หลักเดียวกัน รอบ 1–6 ได้งาน 1–36 และรอบ 7 สมาชิก 1–5 ได้งาน 37–41 ส่วนสมาชิก 6 ไม่มีงาน

เมื่อส่งผลรอบปัจจุบันแล้ว เพิ่ม `ROUND` หนึ่งค่า จากนั้น Restart Kernel และ Run All ใหม่

## วิธีที่ 2: หยิบงานว่างเอง

Cell 5 แสดงตารางงาน 1–41 หากต้องการให้คนที่ว่างรับงานถัดไปเอง ให้จองเลขในตารางแชร์ของทีมก่อน แล้วตั้ง:

```python
JOB_ID = 23
```

เมื่อ `JOB_ID` เป็นตัวเลข ระบบจะไม่ใช้สูตรรอบ การจองใน Google Sheet หรือเอกสารกลางยังจำเป็น เพราะ notebook ของแต่ละเครื่องไม่สามารถรู้พร้อมกันว่าเครื่องอื่นกำลังรันงานใด

แนะนำตารางแชร์ที่มีคอลัมน์ `job_id`, `family`, `owner`, `status`, `result_path` โดย status ใช้ `available`, `running`, `complete`, `failed`

## ส่งและรวมผล

ผลแยกตาม `results/team_family_search/team-N/member-XX/job-YY/` จึงไม่เขียนทับงานรอบอื่น ส่งทั้งโฟลเดอร์ที่ Cell สุดท้ายพิมพ์ รวม checkpoint, protocol, status, screening, notes และ handoff

ผู้รวมผลต้องตรวจ `job_id` 1–41 ไม่ซ้ำและไม่ขาด งานสำเร็จต้องมี `complete=true` และ checkpoint จริง งาน failed ต้องเปิดให้รับใหม่ ห้ามแทนคะแนนที่หายด้วยศูนย์

ก่อนรวม leaderboard ให้ตรวจ split hash, mode, config, versions และ trainer hash ว่าตรงกัน ผลทั้งหมดเป็น validation screening ยังต้องประเมิน unseen test ก่อนสรุปผลสุดท้าย
