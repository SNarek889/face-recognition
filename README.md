# Face Recognition System / Դեմքերի Ճանաչման Համակարգ

## Նկարագրություն / Description

**Հայերեն:**
Այս նախագիծը դեմքերի ճանաչման համակարգ է, որը թույլ է տալիս՝
- Մշակել HEIF և JPEG ֆորմատի նկարներ
- Ավտոմատ հայտնաբերել դեմքերը նկարներում
- Կտրել և պահպանել միայն դեմքերի մասերը
- Ուսուցանել CNN մոդել ձեր սեփական նկարների վրա

**English:**
This project is a face recognition system that allows you to:
- Process HEIF and JPEG format images
- Automatically detect faces in images
- Crop and save only face regions
- Train a CNN model on your own images

---

## Տեղադրում / Installation

### 1. Պահանջներ / Requirements
```bash
python >= 3.8
```

### 2. Տեղադրել անհրաժեշտ գրադարանները / Install dependencies
```bash
pip install -r requirements.txt
```

---

## Օգտագործում / Usage

### Քայլ 1: Նկարների Պատրաստում / Step 1: Prepare Images

Տեղադրեք ձեր նկարները `data/raw/` թղթապանակում, յուրաքանչյուր անձի համար առանձին թղթապանակ՝

Place your images in `data/raw/` folder, separate folder for each person:

```
data/raw/
├── person1/
│   ├── photo1.heic
│   ├── photo2.jpg
│   └── photo3.heic
├── person2/
│   ├── photo1.jpg
│   └── photo2.heic
└── person3/
    └── photo1.jpg
```

### Քայլ 2: Նկարների Մշակում / Step 2: Preprocess Images

Այս քայլում կատարվում է՝
- HEIF → JPEG փոխակերպում
- Դեմքերի հայտնաբերում
- Դեմքերի կտրում և պահպանում

Run preprocessing to detect and crop faces:

```bash
python src/data_preprocessing.py
```

Մշակված նկարները կպահվեն `data/processed/` թղթապանակում։

Processed images will be saved in `data/processed/` folder.

### Քայլ 3: Մոդելի Ուսուցում / Step 3: Train Model

```bash
python src/train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

Պարամետրեր / Parameters:
- `--epochs`: Ուսուցման էպոխաների քանակը / Number of training epochs
- `--batch_size`: Batch-ի չափը / Batch size
- `--learning_rate`: Ուսուցման արագությունը / Learning rate

### Քայլ 4: Մոդելի Օգտագործում / Step 4: Use Model

```python
from src.model import FaceRecognitionCNN
from src.utils import predict_face, load_model
import torch

# Բեռնել մոդելը / Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, label_names = load_model('models/checkpoints/best_model.pth', num_classes=3, device=device)

# Կանխատեսել / Predict
image_path = 'path/to/new/face_image.jpg'
predicted_person, confidence = predict_face(model, image_path, label_names, device)
print(f"Predicted: {predicted_person} (Confidence: {confidence:.2%})")
```

---

## Նախագծի Կառուցվածք / Project Structure

```
face-recognition/
├── requirements.txt          # Անհրաժեշտ գրադարաններ / Dependencies
├── README.md                 # Փաստաթղթեր / Documentation
├── config.py                 # Կարգավորումներ / Configuration
├── data/
│   ├── raw/                  # Սկզբնական նկարներ / Original images
│   └── processed/            # Մշակված դեմքեր / Processed faces
├── models/
│   └── checkpoints/          # Պահպանված մոդելներ / Saved models
└── src/
    ├── __init__.py
    ├── data_preprocessing.py # Նկարների մշակում / Image preprocessing
    ├── model.py              # CNN ճարտարապետություն / CNN architecture
    ├── dataset.py            # Dataset loader
    ├── train.py              # Ուսուցում / Training
    └── utils.py              # Օգնական ֆունկցիաներ / Helper functions
```

---

## Տեխնոլոգիաներ / Technologies

- **Python 3.8+**
- **PyTorch**: Deep Learning framework
- **OpenCV**: Դեմքերի հայտնաբերում / Face detection
- **pillow-heif**: HEIF ֆորմատի աջակցություն / HEIF format support

---

## Լիցենզիա / License

MIT License

---

## Հեղինակ / Author

SNarek889