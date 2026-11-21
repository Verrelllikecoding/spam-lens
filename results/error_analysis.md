# Error analysis: DistilBERT (models/distilbert)

All 32 test-set errors of the DistilBERT model in `models/distilbert` (Colab run, threshold 0.50): 10 false positives (ham flagged as spam) and 22 false negatives (spam missed). Every error was read and given one category. The taxonomy is the one from the first review (`results/archive/error_analysis.md`), plus "Plain promotion missed" for ordinary adverts that the model let through. Counts: `results/tables/error_analysis.csv`.

| Category | Count | FP | FN |
|---|---:|---:|---:|
| Conversational spam / romance / job scam | 11 | 0 | 11 |
| Short or ambiguous message | 7 | 6 | 1 |
| Transactional / OTP resembles promotion | 5 | 2 | 3 |
| Plain promotion missed | 3 | 0 | 3 |
| Probable published-label noise | 3 | 1 | 2 |
| Non-English or code-mixed | 2 | 1 | 1 |
| Obfuscated wording | 1 | 0 | 1 |
| **Total** | **32** | **10** | **22** |

## Anonymized examples

**Conversational spam (FN):** What is the color of your eyes and hair? Please write back, I'm eagerly waiting for your response.

**Conversational spam (FN):** Are you at home? you'll never guess what happened to me today.. This is my temporary number from now on you can save it in your contacts.

**Conversational spam (FN):** Dear customer, we kindly invite you to our head office tonight at [NUM]. and please bring an identification to verify the person.

**Short or ambiguous (FP):** Please submit your reports on time

**Short or ambiguous (FN):** [NUM]/[NUM] [NUM]tf[NUM]p

**Transactional / OTP resembles promotion (FP):** Order Confirmed Hi [NAME]! Your order ... is confirmed. We will deliver it between [NUM] - [NUM]. Track your order on the ...

**Transactional / OTP resembles promotion (FN):** delivery due tomorrow Friday [NUM] Jun [NUM]. if not home we'll leave it in a safe place See

**Plain promotion missed (FN):** DINE-IN open now at our [NAME] Branch!!! Visit now and enjoy [NUM]% off on all Dine-in & Takeaway orders.

**Probable label noise (FP, labelled ham):** Valentines offer! Recharge your [NAME] Mobile/Data card online and avail discounts ... Offer valid on online recharge above [MONEY]

**Non-English or code-mixed (FP):** Can apply for lol account transfer liao! Got free skins b4 [NUM] jan

**Obfuscated wording (FN):** | am [NAME]. | tried to call and left a voicemail concerning biz financing. Give me a ring back to discuss details.

## Main points

- Half of the missed spam (11 of 22) is conversational: dating, wrong-number, family-impersonation, or fake-appointment openers with no typical spam words.
- Most false alarms (6 of 10) are short or unusual normal messages, such as a reminder or a chain message.
- About 3 of 32 errors are probably wrong labels in the dataset, not model mistakes.
- 10 of the 12 examples quoted in the first review are also errors of this model, so the error types are stable between training runs.
