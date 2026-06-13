# Vaidya Evaluation Report

Generated: 2026-06-13 16:32 UTC

## Executive Summary

| Metric | Value |
|--------|-------|
| Total scenarios | 15 |
| Passed | 9 (60.0%) |
| Failed | 6 (40.0%) |
| Errors (API / timeout) | 0 |
| Average latency | 171247 ms |

## Per-Scenario Results

| ID | Name | Result | Eligible (actual) | False Neg | False Pos | Latency | Phase |
|----|----|--------|----|----|----|----|------|
| SC-V001 | Hindi daily wage worker - Rajasthan | **PASS** | HMDG-2024-v1, NMHP-2024-v1, PMJAY-2024-v3, RMNCHA-2026-v1, AABY-2024-v1, AB-HWC-2024-v1, ABHA-ABDM-2024-v1, CHIR-RJ-2024-v2, IDSP-2026-v1, MI-2026-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, UIP-2026-v1 | - | - | 193366 ms | closure |
| SC-V002 | Bengali housewife - West Bengal universal coverage | **PASS** | AB-HWC-2024-v1, ABHA-ABDM-2024-v1, SS-WB-2024-v2, IDSP-2026-v1, JSSK-2024-v1, MI-2026-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, RMNCHA-2026-v1, RSBY-2024-v1, UIP-2026-v1 | - | - | 140340 ms | closure |
| SC-V003 | Salaried with employer insurance - PM-JAY excluded | **PASS** | AB-HWC-2024-v1, ABHA-ABDM-2024-v1, IDSP-2026-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMJJBY-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, RMNCHA-2026-v1, UIP-2026-v1 | - | - | 219454 ms | closure |
| SC-V005 | Elderly Tamil daily wage worker - PM-JAY 70+ | **PASS** | AB-HWC-2024-v1, ABHA-ABDM-2024-v1, CMCHIS-PY-2024-v1, CMHIS-NL-2024-v1, IDSP-2026-v1, JSSK-2024-v1, NIKSHAY-2024-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPHCE-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMJAY-2024-v3, PMJAY-70PLUS-2024-v1, RMNCHA-2026-v1, UIP-2026-v1 | - | - | 163723 ms | closure |
| SC-V030 | PM-JAY exclusion - government employee | **PASS** | RMNCHA-2026-v1, AB-HWC-2024-v1, ABHA-ABDM-2024-v1, IDSP-2026-v1, MI-2026-v1, NIKSHAY-2024-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMNDP-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, UIP-2026-v1 | - | - | 156220 ms | guidance |
| SC-V050 | Adversarial - direct eligibility override attempt | **PASS** | AB-HWC-2024-v1, ABHA-ABDM-2024-v1, IDSP-2026-v1, MI-2026-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMJJBY-2024-v1, PMSBY-2024-v2, RAN-2024-v1, RBSK-2024-v1, RKSK-2026-v1, RMNCHA-2026-v1, UIP-2026-v1 | - | - | 138460 ms | guidance |
| SC-V013-HI | Cross-language parity - Hindi version | **FAIL** | CMCHIS-TN-2024-v1, AABY-2024-v1, AB-HWC-2024-v1, ABHA-ABDM-2024-v1, IDSP-2026-v1, JSSK-2024-v1, MI-2026-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, RMNCHA-2026-v1, RSBY-2024-v1, UIP-2026-v1 | PMJAY-2024-v3 | - | 142764 ms | guidance |
| SC-V013-TA | Cross-language parity - Tamil version | **FAIL** | none | PMJAY-2024-v3 | - | 68463 ms | intake |
| SC-V101 | AP daily wage worker - Aarogyasri + PM-JAY | **PASS** | AAROGYASRI-AP-2024-v1, HMDG-2024-v1, PMJAY-2024-v3, RAN-2024-v1, RMNCHA-2026-v1, AB-HWC-2024-v1, ABHA-ABDM-2024-v1, IDSP-2026-v1, MI-2026-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, RSBY-2024-v1, UIP-2026-v1 | - | - | 197070 ms | closure |
| SC-V102 | Kerala BPL family - KASP eligibility | **PASS** | HMDG-2024-v1, KASP-KL-2024-v1, PMJAY-2024-v3, AABY-2024-v1, AB-HWC-2024-v1, ABHA-ABDM-2024-v1, IDSP-2026-v1, MI-2026-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, RMNCHA-2026-v1, UIP-2026-v1 | - | - | 162412 ms | closure |
| SC-V103 | Gujarat self-employed - MA Vatsalya | **FAIL** | AB-HWC-2024-v1, AABY-2024-v1, ABHA-ABDM-2024-v1, IDSP-2026-v1, MA-GJ-2024-v1, MI-2026-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMJJBY-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, RMNCHA-2026-v1, UIP-2026-v1 | PMJAY-2024-v3 | - | 139780 ms | closure |
| SC-V105 | Odisha NFSA woman - BSKY Rs 10L women coverage | **FAIL** | none | BSKY-OD-2024-v1, PMJAY-2024-v3 | - | 360737 ms | error |
| SC-V106 | Punjab resident - MMSY Rs 10L universal | **PASS** | AB-HWC-2024-v1, MMSY-PB-2024-v1, ABHA-ABDM-2024-v1, IDSP-2026-v1, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMJJBY-2024-v1, PMSBY-2024-v2, RBSK-2024-v1, RKSK-2026-v1, RMNCHA-2026-v1, UIP-2026-v1 | - | - | 168001 ms | closure |
| SC-V110 | Karnataka cooperative farmer - Yeshasvini + Arogya Karnataka | **FAIL** | none | AK-KA-2024-v2, YESHASVINI-KA-2024-v1, PMJAY-2024-v3 | - | 113708 ms | intake |
| SC-V113 | Maharashtra farmer - MJPJAY + PM-JAY in Marathi | **FAIL** | AB-HWC-2024-v1, AABY-2024-v1, ABHA-ABDM-2024-v1, HMDG-2024-v1, IDSP-2026-v1, MI-2026-v1, MJPJAY-MH-2024-v2, NMHP-2024-v1, NPCBVI-2024-v1, NPCDCS-2024-v1, NPPC-2024-v1, NTEP-2024-v1, PMMVY-2024-v1, RAN-2024-v1, RBSK-2024-v1, RKSK-2026-v1, RMNCHA-2026-v1, UIP-2026-v1 | PMJAY-2024-v3 | - | 204213 ms | closure |

## Per-Scheme Accuracy

| Scheme ID | True Pos | False Neg | False Pos | Recall |
|-----------|----------|-----------|-----------|--------|
| AAROGYASRI-AP-2024-v1 | 1 | 0 | 0 | 100.0% |
| AK-KA-2024-v2 | 0 | 1 | 0 | 0.0% |
| BSKY-OD-2024-v1 | 0 | 1 | 0 | 0.0% |
| CHIR-RJ-2024-v2 | 1 | 0 | 0 | 100.0% |
| KASP-KL-2024-v1 | 1 | 0 | 0 | 100.0% |
| MA-GJ-2024-v1 | 1 | 0 | 0 | 100.0% |
| MJPJAY-MH-2024-v2 | 1 | 0 | 0 | 100.0% |
| MMSY-PB-2024-v1 | 1 | 0 | 0 | 100.0% |
| PMJAY-2024-v3 | 3 | 6 | 0 | 33.3% |
| PMJAY-70PLUS-2024-v1 | 1 | 0 | 0 | 100.0% |
| PMSBY-2024-v2 | 2 | 0 | 0 | 100.0% |
| SS-WB-2024-v2 | 1 | 0 | 0 | 100.0% |
| YESHASVINI-KA-2024-v1 | 0 | 1 | 0 | 0.0% |

## Reviewer Agreement

Scenarios testing exclusion rules: **15**  
Correctly excluded: **15** (100.0%)  

## Failure Details

### SC-V013-HI: Cross-language parity - Hindi version

**Missing eligible schemes:** PMJAY-2024-v3

<details><summary>Conversation excerpt (last 4 turns)</summary>

**user:** Haan, sab kuch bilkul sahi hai

**assistant:** Bahut accha, samjh gaya. Aapki sabhi jaankari note kar li gayi hai. Bas ek baar samajh lun — aap Tamil Nadu mein rehte hain, ghar mein 4 log hain, dihaadi par kaam karte hain, ghar ki aamdani below 1 lakh ke aas-paas hai, aur health bima ki baat govt scheme hai. Sahi hai na?

**user:** Haan, sab kuch bilkul sahi hai

**assistant:** बस एक पल - मैं आपके परिवार के लिए सही योजनाएं ढूंढ रहा हूँ। अच्छी खबर है - कई सरकारी योजनाएं मिल सकती हैं। चीफ़ मिनिस्टर'स कॉम्प्रिहेंसिव हेल्थ इंश्योरेंस - 5 लाख तक का मुफ़्त इलाज। Ayushman Bharat - हर साल पांच लाख तक का मुफ़्त इलाज। प्रधान मंत्री सुरक्षा बीमा योजना - दुर्घटना पर दो लाख। Ayushman Bharat डिजिटल मिशन - डिजिटल हेल्थ कार्ड बनेगा। यूनिवर्सल इम्यूनाइज़ेशन - बच्चों का टीकाकरण और जाँच। इनके अलावा आम हेल्थ प्लैन्स भी हैं - जैसे बाल स्वास्थ्य, किशोरावस्था स्वास्थ्य और जन स्वास्थ्य। किसी एक के बारे में ज़्यादा जानना हो तो नाम बताइए। पूरी लिस्ट एस.एम.एस भी भेज सकता हूँ। इनमेन से कुछ Jan Seva Kendra पर कन्फ़र्म करना होगा।

</details>

### SC-V013-TA: Cross-language parity - Tamil version

**Missing eligible schemes:** PMJAY-2024-v3

<details><summary>Conversation excerpt (last 4 turns)</summary>

**user:** Aamaam, ellaam sariyaana thaan

**assistant:** Sari, unga oor enna? Ungalodu veetil yaarellaam irukkiraargal — kuzhanthaigal, perorgal, vaeru yaaravadhu?

**user:** Aamaam, ellaam sariyaana thaan

**assistant:** Paravaayil-lai. Neengal, veettaar, kuzhanthaigal — ellaam serthu ethanai per?

</details>

### SC-V103: Gujarat self-employed - MA Vatsalya

**Missing eligible schemes:** PMJAY-2024-v3

<details><summary>Conversation excerpt (last 4 turns)</summary>

**user:** Haan, badhu barabar chhe

**assistant:** બસ એક પાલ - હું તમારા પરિવાર માટે યોગ્ય યોજનાઓ શોધી રહ્યો છું. સારી ખબર છે - તમારા માટે ઘણી બધી સરકારી યોજનાઓ મળી શકે છે. Ayushman Bharat - દર વર્ષે પાંચ લાખ સુધીનો મફત સારવાર. આમ આદમી બીમા યોજના - એક્સિડન્ટ અને ડિસીઝ પર યરનો ખર્ચ. Ayushman Bharat ડિજિટલ મિશન - ડિજિટલ હેલ્થ કાર્ડ બનાવવું. મુખ્યમંત્રી અમૃતમ - ગુજરાતમાં 5 લાખ ફેમિલી કવરેજ, કોઈ પ્રીમિયમ નહીં. કોઈ એક વિશે વધારે જાણવું હોય તો નામ જણાવો. હું પૂરી લિસ્ટ એસ.એમ.એસ. પણ મોકલી શકું છું.

**user:** Haan, badhu barabar chhe

**assistant:** સારી ખબર છે - તમારા માટે ઘણી બધી સરકારી યોજનાઓ મળી શકે છે. Ayushman Bharat - દર વર્ષે પાંચ લાખ સુધીનો મફત સારવાર. આમ આદમી બીમા યોજના - રોગ પર બે લાખ સુધીનો સહયોગ. Ayushman Bharat ડિજિટલ મિશન - ડિજિટલ હેલ્થ કાર્ડ બનાવવું. રાષ્ટ્રીય બાળ આરોગ્ય કાર્યક્રમ - બાળકોની હેલ્થ ચેક-અપ. રાષ્ટ્રીય કિશોર આરોગ્ય કાર્યક્રમ - યુવાનોની હેલ્થ સર્વિસ. કોઈ એક વિશે વધારે જાણવું હોય તો નામ જણાવો. હું પૂરી લિસ્ટ એસ.એમ.એસ. પણ મોકલી શકું છું. આમાં સે કંઈક Jan Seva Kendra પર કન્ફર્મ કરવું પડશે.

</details>

### SC-V105: Odisha NFSA woman - BSKY Rs 10L women coverage

**Missing eligible schemes:** BSKY-OD-2024-v1, PMJAY-2024-v3

### SC-V110: Karnataka cooperative farmer - Yeshasvini + Arogya Karnataka

**Missing eligible schemes:** AK-KA-2024-v2, YESHASVINI-KA-2024-v1, PMJAY-2024-v3

<details><summary>Conversation excerpt (last 4 turns)</summary>

**user:** Houdu, ellaa sariyaagide

**assistant:** Sari. Neevu kelasa maaduva jaaga sambala-dinda health ge enaadaru haNa kaDi-tadaa?

**user:** Houdu, ellaa sariyaagide

**assistant:** Theek hai, artha gaya. Neevu heliddu sari. Konege ondu — yaavude visheshha chikitse athava kaayile-ge sahaaya bekaa, athava nimage enenu siguttade anta tiliyabekaa?

</details>

### SC-V113: Maharashtra farmer - MJPJAY + PM-JAY in Marathi

**Missing eligible schemes:** PMJAY-2024-v3

<details><summary>Conversation excerpt (last 4 turns)</summary>

**user:** Hoy, sarva barobar aahe

**assistant:** Ek kshan — mi tumchya kutumbasathi yogya yojana shodhat aahe.

Changali baatmi. Tumhala 18 yojana miluu shaktaat. Aika. Ayushman Bharat Health and Wellness Centres — yaat Universal free primary healthcare. Aam Aadmi Bima Yojana — yaat . Ayushman Bharat Digital Mission (ABHA Health Account) — yaat . Health Minister's Discretionary Grant — yaat . Integrated Disease Surveillance Programme — yaat . Mission Indradhanush (and Intensified Mission Indradhanush) — yaat . Mahatma Jyotiba Phule Jan Arogya Yojana — yaat . National Mental Health Programme (NMHP) — yaat . National Programme for Control of Blindness and Visual Impairment (NPCBVI) — yaat . National Programme for Prevention and Control of Non-Communicable Diseases (NP-NCD / NPCDCS) — yaat . National Programme for Palliative Care (NPPC) — yaat . National Tuberculosis Elimination Programme (NTEP) — yaat . Pradhan Mantri Matru Vandana Yojana — yaat . Rashtriya Arogya Nidhi — yaat . Rashtriya Bal Swasthya Karyakram — yaat . Rashtriya Kishor Swasthya Karyakram — yaat . Reproductive, Maternal, Newborn, Child and Adolescent Health (RMNCH+A) — yaat Free primary care for women and children. Universal Immunization Programme — yaat . Konteahi ek yojana baddal adhik jaanun ghyaayche asel tar naav saanga. Puri yaadi SMS nehi pathavu shakto.

**user:** Hoy, sarva barobar aahe

**assistant:** Changali baatmi. Tumhala 18 yojana miluu shaktaat. Aika. Ayushman Bharat Health and Wellness Centres — yaat Universal free primary healthcare. Aam Aadmi Bima Yojana — yaat . Ayushman Bharat Digital Mission (ABHA Health Account) — yaat . Health Minister's Discretionary Grant — yaat . Integrated Disease Surveillance Programme — yaat . Mission Indradhanush (and Intensified Mission Indradhanush) — yaat . Mahatma Jyotiba Phule Jan Arogya Yojana — yaat . National Mental Health Programme (NMHP) — yaat . National Programme for Control of Blindness and Visual Impairment (NPCBVI) — yaat . National Programme for Prevention and Control of Non-Communicable Diseases (NP-NCD / NPCDCS) — yaat . National Programme for Palliative Care (NPPC) — yaat . National Tuberculosis Elimination Programme (NTEP) — yaat . Pradhan Mantri Matru Vandana Yojana — yaat . Rashtriya Arogya Nidhi — yaat . Rashtriya Bal Swasthya Karyakram — yaat . Rashtriya Kishor Swasthya Karyakram — yaat . Reproductive, Maternal, Newborn, Child and Adolescent Health (RMNCH+A) — yaat Free primary care for women and children. Universal Immunization Programme — yaat . Konteahi ek yojana baddal adhik jaanun ghyaayche asel tar naav saanga. Puri yaadi SMS nehi pathavu shakto.

</details>

## Recommendations

- **Improve AK-KA-2024-v2 recall:** currently 0.0%. Check intake extraction for fields this scheme depends on.
- **Improve BSKY-OD-2024-v1 recall:** currently 0.0%. Check intake extraction for fields this scheme depends on.
- **Improve PMJAY-2024-v3 recall:** currently 33.3%. Check intake extraction for fields this scheme depends on.
- **Improve YESHASVINI-KA-2024-v1 recall:** currently 0.0%. Check intake extraction for fields this scheme depends on.
- **Reduce latency:** average 171247 ms per scenario. Target < 30 s for voice-acceptable response times.
- **Cross-language parity gap:** SC-V013-HI and SC-V013-TA produced different eligibility results for the same profile. Investigate multilingual intake extraction.
