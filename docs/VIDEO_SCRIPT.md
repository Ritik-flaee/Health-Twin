# HealthTwin video script

Aim for about 22 minutes so the cut still clears a 20-minute requirement. The submission form also accepts 15–20 minutes. Record the slide deck and the running dashboard. Speak as if a clinician and an engineer are both in the room.

Before you start:

1. The title slide already says team Ritik Awachat, Vijay Patil School of Management, solo. Say that aloud.
2. From the project folder, run `streamlit run app/dashboard.py`.
3. The app opens on **Who should get a message?** Reyansh Joshi is first. Press **Step 2: open Reyansh Joshi**. You land on Wednesday 9 September 2026, 08:00.
4. Leave **Demo only: show what really happened** off until the script asks for the later reading.

## 0:00–2:00 — Who you are and what you built

Open `docs/HealthTwin_Presentation.pptx` on the title slide.

Say your name, Ritik Awachat, and Vijay Patil School of Management. Say you are a solo full-time student. Say the project name: HealthTwin. One sentence on the job it does: it joins a clinic chart to a wearable stream and warns when blood sugar is likely to hit 180 before the next two hours are over.

Say, clearly, that every patient in the demo is synthetic.

## 2:00–4:30 — Why this problem

Slides 01 and 02.

Type 2 diabetes in India is common, and the usual clinic number is an average over months. The decision you chose is smaller and more useful: the next two hours. Mention that 180 mg/dL is a conventional post-meal line, not a diagnosis.

Describe the person the twin is for: an adult already known to have Type 2 diabetes, with a CGM and a watch, and a doctor who cannot watch every trace. The screen is for that doctor. It does not prescribe.

## 4:30 — Architecture

Open the architecture slide in the presentation. It is also `docs/HealthTwin_Architecture.pdf`.

Point left to right: clinic chart, wearable stream, model, dashboard. Say features stop at the current minute. Say the two warnings: glucose at or above 180, and a drop under 70.

## 5:00–8:30 — The sandbox

Slide 03. If you skipped the diagram, show slide 04 now.

Explain why you did not use real records: the DPDP Act, and HIPAA where a dataset would have come with it. Walk the two streams:

- Static: demographics, HbA1c, fasting glucose, medicines, blood pressure, lipids, kidney function, family history, simulated genetic flags.
- Dynamic: 15-minute glucose, heart rate, HRV, steps, sleep stage.

Say the generator makes spikes depend on HbA1c, medicines, short sleep, and low steps, and that those causes are not given to the model as a column called "meal". The model has to read them off the chart and the sensors. Mention the seed, 2026, so the cohort can be rebuilt.

## 8:30–12:30 — Model, split, and results

Slides 05, 06, and 07. Slide 04 was the architecture diagram.

Explain the label: the maximum glucose over the next eight readings. Positive at 180 or above. Features stop at the current minute. The split is by patient, 70 / 15 / 15. The threshold is chosen on validation patients and locked before the test patients are scored.

Put the numbers on screen and say them slowly:

- Fusion ROC-AUC **0.918** on the 23 unseen people, glucose-sensor only **0.828**.
- When glucose is still under 140: fusion **0.873**, sensor only **0.670**.
- Precision **0.66**, recall **0.74** at the fusion threshold of **0.58**.
- Peak error about **17 mg/dL**.

Say why the second row is the one you care about. If the reading is already 210, you do not need a twin to notice. The gain shows up while the reading is still ordinary, because the clock, recent steps, and HbA1c still say a spike is coming.

Say what the importance table showed: time of day and the current glucose move the score most, then the 30-minute slope, recent steps, and HbA1c. Do not claim sleep is the dominant input. It is in the model. It is not the top feature.

## 12:30–18:30 — The dashboard

Show slide 08, the list and the walk side by side. Then switch to the browser. Do this in order, and pause on each view.

1. Start on the list for Wednesday 9 September 2026, 08:00. Read the four counts: **Message them 10**, **Low sugar likely 0**, **Already too high 3**, **No message 10**. Say this is the product: a doctor sees who to message before opening a chart. Reyansh Joshi is first because his sugar is still ordinary and a rise is likely. Press **Step 2: open Reyansh Joshi**.

2. Read the sentence, then the four tiles. He is 47, in Mumbai, Type 2 diabetes, HbA1c **7.8%**, blood pressure **145** systolic. Glucose about **117 mg/dL**. Chance it shoots up **96%**. How high it may get **217**. Chance it drops too low **0%**. Open the health record. Blood pressure is context. This is not a heart-attack forecast.

3. Point at the green line, **So far**, and the orange estimate. The peak on the chart is labelled **217**. Turn on **Demo only: show what really happened**. The grey line climbs to about **268 mg/dL**. The estimate is lower and still clears 180. Turn the grey line off. That is the clinic view.

4. Read the reasons: usual meal time, about 2,283 steps, HbA1c 7.8%, borderline sleep, no diabetes medicine on record. Say the buttons are a question for the doctor, not an order.

5. Press **Walk 30 min**. The chance falls from **96%** to **87%**, and the peak from **217** to **201**. The sentence says it was 96%. The green line does not move. Press **Reset**. Then press **Meal 90 min later** and show it move to about **80%**. Press **Reset** again. Nothing here is a prescription.

6. Press **Back**. In **Examples and the technical review**, press **Open a quiet example**. Aarav Joshi, Saturday 12 September, 01:00, glucose about **78**, and the sentence says **Sugar is 78 now. No warning right now.** **Meal 90 min later** is hidden because it is night. Say the app is not alarming on every row.

6b. To show the low-sugar warning, go back to the list and move the clinic time to **Sun 6 Sep, 22:45**. **Step 2** opens Kiara Menon, P-080, age 52, Pune. Another Kiara is also on the list, so use the ID. Sugar is about **75**, and the chance it drops under 70 is about **99%**. The box is **What the chart shows**, and it says sugar fell from about 168 to 75. The low-point tile is a grey **No estimate**. Then use **Back** and return the clock to Wednesday 9 September, 08:00 if you still need the list.

7. On the open person, open **Examples and the technical review** and read the ROC-AUC line: **0.92** for the full twin and **0.83** for the glucose sensor only, on the 23 people the model had not seen. Point at **Top 8 inputs the model relied on (practice data)**. Time of day and current sugar lead. Sleep is in the model. It is not the top bar.

## 18:30–21:30 — Limits and what you would do next

Slide 09.

Say these limits without being asked:

- The score is on a generator your team wrote. It is not a clinical validation.
- Meals follow a day, so the clock is doing real work. A next version should take meal logs or detect meals.
- The genetic flags are simulated.
- The sketch is not a whole-night physiological model.
- Nothing here should be used to change a dose.

Then say what a serious next step would be: an open or synthetic set you did not design yourselves, a prospective check, and a clinician reviewing alerts before anyone acts.

## 21:30–22:30 — Close

Slide 10.

Restate the pattern: two streams, two warnings (a spike at 180 and a drop under 70), and one screen a doctor can stand in front of. Thank the jury. Stop the recording only after a full breath of silence so the end is not cut.

## If something on screen does not match this script

Do not retrain before you record. These numbers match the saved models. If a figure disagrees, you are not on Reyansh Joshi at Wednesday 9 September 2026, 08:00. The bookmark is `data/demo_moment.json`.
