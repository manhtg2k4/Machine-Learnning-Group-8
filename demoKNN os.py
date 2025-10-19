import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ----------------------------
# Hàm load dữ liệu ảnh (grayscale + chuẩn hóa)
# ----------------------------
def load_data(base_dir, img_size=(64, 64)):
    data = []
    labels = []
    file_paths = []
    classes = os.listdir(base_dir)

    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for file in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file)
            if os.path.isfile(file_path):
                img = cv2.imread(file_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # chuyển sang grayscale
                img = cv2.resize(img, img_size)               # resize ảnh
                feature = img.flatten().astype("float32") / 255.0  # chuẩn hóa về [0,1]
                data.append(feature)
                labels.append(cls)
                file_paths.append(file_path)

    return np.array(data), np.array(labels), file_paths

# ----------------------------
# Load dữ liệu
# ----------------------------
X_train, y_train, _ = load_data("dataset_split/train")
X_test, y_test, test_paths = load_data("dataset_split/test")

print("Số mẫu train:", len(X_train))
print("Số mẫu test :", len(X_test))

# ----------------------------
# Thử nhiều giá trị k để tìm k tốt nhất
# ----------------------------
best_acc = 0
best_k = 0
best_pred = None

for k in [3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    #Dự đoán 
    y_pred = knn.predict(X_test)
    #Độ chính xác 
    acc = accuracy_score(y_test, y_pred)
    print(f"K={k}, Độ chính xác: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_k = k
        best_pred = y_pred

print(f"\n✅ K tốt nhất: {best_k}, Accuracy cao nhất: {best_acc*100:.2f}%")

# ----------------------------
# Hiển thị một số ảnh test với dự đoán tốt nhất
# ----------------------------
plt.figure(figsize=(12, 6))

for i in range(12):  # hiển thị 12 ảnh đầu tiên
    img = cv2.imread(test_paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Thật: {y_test[i]}\nDự đoán: {best_pred[i]}",
              color="green" if y_test[i] == best_pred[i] else "red")

plt.tight_layout()
plt.show()
