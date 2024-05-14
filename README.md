# Uwierzytelnianie na podstawie biometrii behawioralnej

Projekt realizowany w ramach kursu Biometria na Politechnice Wrocławskiej. Więcej informacji [tutaj](https://www.syga.ai.pwr.edu.pl/courses/bio/P2.pdf).


# Środowisko

Projekt postawiony został na środowisku Python 3.8 virtual environment, inne wersje Pythona mogą działać, ale nie były sprawdzane

# Dane

[LINK TRAIN](https://drive.google.com/file/d/1Lgv1oxXlzoDudMPcmbfjX0-C8nIB7Bm7/view?usp=sharing)
[LINK TEST](https://drive.google.com/file/d/11mO9XGT7M_66HTFQu1PNLSDoKk-e0xrw/view?usp=sharing)

Struktura:
Dane treningowe, użytkownicy wdrożeni

```
vox1
├── vox1
│   ├── Name_Surname
│   │   ├── profile
│   │   └── test_positive
│   ├── Name_Surname
│   │   ├── profile
│   │   └── test_positive
|   ...
```

Dane testowe, użytkownicy spoza bazy
```
vox1_test
├── vox1_test
│   ├── Name_Surname
│   │   └── files
│   ├── Name_Surname
│   │   └── files
|   ...
```

# Sposób uruchomienia

>> streamlit run src/app.py
