# American Sign Language (ASL) Recognition System

[PL] Projekt badawczy oraz implementacja systemu rozpoznawania liter amerykańskiego języka migowego (ASL) z wykorzystaniem Głębokich Sieci Neuronowych (CNN vs DNN). Projekt zoptymalizowany pod kątem akceleracji sprzętowej na architekturze **Apple Silicon M4 (TensorFlow-Metal)**.

[EN] A research project and implementation of an American Sign Language (ASL) letter recognition system using Deep Neural Networks (CNN vs DNN). Optimized for hardware acceleration on the **Apple Silicon M4 (TensorFlow-Metal)** architecture.

---

# 🛠️ Environment & Requirements / Środowisko i Wymagania

## Core Versions / Główne wersje

- **Python:** `3.10.x` / `3.11.x`
- **TensorFlow:** `2.13.0+` (Keras 3)
- **TensorFlow-Metal:** Apple GPU Acceleration Plugin

---

## Installation / Instalacja

```bash
# System dependency for drawing network plots
brew install graphviz

# Python libraries
pip install tensorflow tensorflow-metal pydot matplotlib numpy pillow scikit-learn
```

---

# DATASET

---

## 🇵🇱 Polski

Projekt wykorzystuje publicznie dostępny zbiór danych **ASL Alphabet** udostępniony na platformie Kaggle:

[ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

### Opis zbioru danych

Dataset zawiera obrazy przedstawiające gesty dłoni odpowiadające literom amerykańskiego języka migowego (ASL). Zbiór składa się z **29 klas**:

- 26 liter alfabetu (`A-Z`)
- `SPACE`
- `DELETE`
- `NOTHING`

### Główne właściwości

- **Liczba obrazów treningowych:** ~87 000
- **Rozdzielczość obrazów:** `200x200`
- **Typ danych:** kolorowe fotografie gestów dłoni (RGB)
- **Liczba klas:** 29
- **Format zbioru:** struktura katalogowa kompatybilna z klasyfikacją obrazów

---

## 🇬🇧 English

The project uses the publicly available **ASL Alphabet** dataset hosted on Kaggle:

[ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

### Dataset Overview

The dataset contains images representing hand gestures corresponding to letters of the American Sign Language (ASL) alphabet. It consists of **29 classes**:

- 26 alphabet letters (`A-Z`)
- `SPACE`
- `DELETE`
- `NOTHING`

### Key Properties

- **Training images:** ~87,000
- **Image resolution:** `200x200`
- **Image type:** RGB photographs of hand gestures
- **Number of classes:** 29
- **Dataset format:** folder-based image classification structure


# 📂 Scripts Overview / Przegląd Skryptów

---

## `prep_images.py`

### 🇵🇱 Polski

**Co robi:**  
Standaryzuje surowe zdjęcia do formatu wejściowego sieci neuronowej.

**W skrócie:**  
Zmienia rozmiar wszystkich obrazów do `200x200` pikseli oraz konwertuje je do skali szarości (*grayscale*).

### 🇬🇧 English

**What it does:**  
Standardizes raw images for neural network input.

**In short:**  
Resizes all images to `200x200` pixels and converts them to grayscale.

---

## `prep_dataset.py`

### 🇵🇱 Polski

**Co robi:**  
Buduje potok danych dla procesu treningowego.

**W skrócie:**  
Tworzy obiekty `tf.data.Dataset`.

### 🇬🇧 English

**What it does:**  
Builds and optimizes the data pipeline for training.

**In short:**  
Creates `tf.data.Dataset` objects.

---

## `cnn_model.py`

### 🇵🇱 Polski

**Co robi:**  
Definiuje, trenuje i zapisuje architekturę sieci splotowej CNN.

**W skrócie:**  
Buduje zaawansowany model CNN z warstwami konwolucyjnymi oraz automatyczną normalizacją wejścia poprzez `Rescaling(1.0 / 255.0)`.

**Wynik:** ~96% accuracy.

### 🇬🇧 English

**What it does:**  
Defines, trains, and saves the Convolutional Neural Network architecture.

**In short:**  
Builds an advanced CNN model with convolutional layers and automatic input normalization using `Rescaling(1.0 / 255.0)`.

**Result:** ~96% accuracy.

---

## `dnn_model.py`

### 🇵🇱 Polski

**Co robi:**  
Definiuje, trenuje i zapisuje klasyczną sieć gęstą DNN.

**W skrócie:**  
Spłaszcza obraz do wektora `40,000` cech i przetwarza go przez głębokie warstwy typu `Dense`.

**Rola:** Model bazowy (*baseline comparison*).

**Wynik:** ~16% accuracy.

### 🇬🇧 English

**What it does:**  
Defines, trains, and saves the classic Deep Neural Network architecture.

**In short:**  
Flattens the image into a `40,000`-feature vector and passes it through deep `Dense` layers.

**Role:** Baseline comparison model.

**Result:** ~16% accuracy.

---

## `draw_network.py`

### 🇵🇱 Polski

**Co robi:**  
Generuje wizualny diagram architektury modelu.

**W skrócie:**  
Wczytuje plik `.keras` i przy użyciu Graphviz zapisuje strukturę warstw oraz wymiary tensorów do pliku `.png`.

### 🇬🇧 English

**What it does:**  
Generates a visual flowchart of the model architecture.

**In short:**  
Loads a `.keras` file and uses Graphviz to save the layer structure and tensor shapes into a `.png` file.

---

## `test_on_validation.py`

### 🇵🇱 Polski

**Co robi:**  
Przeprowadza szczegółową ewaluację statystyczną modeli.

**W skrócie:**  
Generuje pełny *Classification Report* zawierający wskaźniki:

- Precision
- Recall
- F1-Score

oraz średnie zbiorcze dla całego zbioru walidacyjnego.

### 🇬🇧 English

**What it does:**  
Performs a detailed statistical evaluation of the models.

**In short:**  
Generates a full *Classification Report* including:

- Precision
- Recall
- F1-Score

and aggregate averages for the entire validation dataset.

---

## `test_model.py`

### 🇵🇱 Polski

**Co robi:**  
Służy do szybkiej weryfikacji modelu na pojedynczych, zewnętrznych próbkach.

**W skrócie:**  
Pozwala wskazać surowy plik obrazu (0–255 pixel scale), np. z telefonu lub kamerki, aby sprawdzić rzeczywistą odpowiedź modelu.

### 🇬🇧 English

**What it does:**  
Used for quick model verification on single external samples.

**In short:**  
Allows passing a raw image file (0–255 pixel scale), e.g., from a phone or webcam, to test the network's real-world inference performance.