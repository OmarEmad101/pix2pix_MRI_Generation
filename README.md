## 📂 Project Structure  

```plaintext
📂 pix2pix_MRI_Generation/
│── 📂 models/                 # Contains all model variants  
│── 📂 results/                # Stores trained models & outputs (to be created automatically)  
│── 📂 logs/                   # TensorBoard logs (to be created automatically)  
│── main.py                 # Runs all models  
│── train_utils.py          # Helper functions for training  
│── requirements.txt        # Dependencies (TensorFlow, NumPy, etc.)  
│── README.md               # Project documentation
```


## 🛠️ Setup Instructions  

### 1️⃣ **Download & Extract Dataset**  
- **Google Drive Link**: [📥 Download Dataset](https://drive.google.com/file/d/1h7r9xhHN30vYsieR4i3m6rMJ5Fg1CStj/view?usp=drive_link)  
- After extraction, the dataset folder should be structured as:
```plaintext
📂 model_data/
│── 📂 model_images/
│── 📂 model_masks/
```
📌 **Make sure you have the correct dataset structure before running the code.**  

---

### 2️⃣ **Install Dependencies**  
Make sure you have **Python 3.8 - 3.11** installed. Then, install required libraries:  
```bash
pip install -r requirements.txt
```

### 3️⃣ **Set Dataset Path in `main.py`**  

Before running, open **`main.py`** and update this line to match your dataset path:  

```python
DATA_DIR = "path/to/your/extracted/model_data"
```

### 4️⃣ **Run the Training Script 🚀**  

Once everything is set up, run the following command:  

```bash
python main.py
```



