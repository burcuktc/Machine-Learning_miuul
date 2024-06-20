#Lojistik Regresyon
#Amaç, bir sınıflandırma problemi için bağımlı ve bağımsız değişkenler arasındaki ilişkiyi doğrusal modellemektir.
#Nasıl? Gerçek değerler ile tahmin edilen değerler arasındaki farklara ilişkin log loss değerini minimum yapabilecek ağırlıkları bularak.
#z=b+w1x1+w2x2+..+wpxp
#yi=1/(1+e^(-z)) (doğrusal formdan gelen sayıyı 0 ile 1 arasına dönüştürür)

#Sınıflandırma Problemlerinde Başarı Değerlendirme
#Confusion Matrix:
#Doğru yapılan işler true, pozitif ve negatif yapmam gerekenden farklı yaptığım kısımlar, 1 pozitif kısmı temsil eder.
#Accuracy:Doğru sınıflandırma oranıdır.  (TP+TN)/(TP+TN+FP+FN)
#Precision:Pozitif sınıf (1) tahminlerinin başarı oranıdır. (Tahmin ettiklerimde ne kadar başarılıyım)  (TP/TP+FP)
#Recall:Doğru sınıfın  (1) doğru tahmin edilme oranıdır. TP/(TP+FN) (Gerçekteki pozitif sınıfın ne kadarının doğru tahmin edildiğini ifade eder)
#Precision tahminlerin başarısına odaklanmıştır. Recall gerçekleri yakalama başarısına odaklanmıştır.
#Elimizdeki sınıflandırma problemi dengeli sınıf dağılımına sahipse accuracy kullanılabilir, dengeli değilse recall ve precision değerlerine bakmak gerekir.
#F1 Score: Precision ve recall değerlerinin harmonik ortalamasıdır. 2*(Precision*Recall)/(Precision+recall)
#classification threshold: tahmin edilen sınıflar belirli bir olasılık değerine sahiptir önceden belirlenmiş eşik değere göre 1 veya 0 olarak belirlenir.
#Olası eşik değer değişimlerine karşılık başarı değişimleri nedir sorusunu cevaplamak için ROC eğrisi kullanılır.

#ROC curve:
#x ekseni true pozitif rate (precision), y ekseni false pozitif rate
#AUC:Area Under Curve: ROC eğrisinin tek bir sayısal değerle ifade edilişidir. ROC eğrisi altında kalan alandır. Tüm olası sınıflandırma eşikleri için toplu bir performans ölçüsüdür.

#Sınıflandırma problemlerinde veri seti dengeli mi dengesiz mi öncelikle buna bakılır. Dengesizse recall,precision değerine F1 scoreuna bakılır.  AUC'ye bakılır.

#LOG LOSS:
#Bir başarı metriğidir. Aynı zamanda düşmesini istediğimiz, optimize etmek istediğimiz amaç fonksiyonudur.
#Entropi çeşitliliktir

######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

#Gerçekleştirilecek adımlar
# 1. Exploratory Data Analysis: veri setini tanımaya yönelik keşifçi veri analizi ve veri önişleme adımları
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit #eşik değer hesaplamak için fonksiyon

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False #eşik değer hesapladıktan sonra değişkende aykırı değer var mı yok mu hesaplamak için fonksiyon

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit #aykırı değer varsa aykırı değerleri silme ama hesaplanan eşik değerlerle değiştir fonksiyonu

pd.set_option('display.max_columns', None) #tüm sütunları göster
pd.set_option('display.float_format', lambda x: '%.3f' % x) #virgülden sonra 3 bas. göster
pd.set_option('display.width', 500)

######################################################
# Exploratory Data Analysis (keşifçi veri analizi EDA)
######################################################

df = pd.read_csv("C:/Users/asus/Desktop/machine learning/machine_learning/datasets/diabetes.csv")
df.head()
"""Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin    BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0 33.600                     0.627   50        1
1            1       85             66             29        0 26.600                     0.351   31        0
2            8      183             64              0        0 23.300                     0.672   32        1
3            1       89             66             23       94 28.100                     0.167   21        0
4            0      137             40             35      168 43.100                     2.288   33        1
"""
df.shape "Out[11]: (768, 9)"

df["Outcome"].value_counts()
"""Out[12]: 
0    500
1    268""" #0 sınıfından 500, 1 sınıfından 268 tane

sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df) #oranlara bakmak için
"""Out[13]: 
0   65.104
1   34.896""" #veri setinin %34ünde 1 sınıfı var.

# Feature'ların Analizi (bağımsız değişkenlerin analizi)
##########################
df.describe().T
df.head()

#sayısal değişkenleri görselleştirmek için histogram veya kutu grafiği kullanıırız.
#Histogram ilgili sayısal değişkenin değerlerini belirli aralıklarda ne kadar gözlenme frekansı var diye gösterir.
#Kutu grafik ise ilgili sayısal değişkenin değerleri küçükten büyüğe sıraladıktan sonra değişkenin dağılımı ile ilgili bilgi verir.
df["BloodPressure"].hist(bins=20) #yaş
plt.xlabel("BloodPressure")
plt.show() #örneğin 60-70 aralığında kan basıncına sahip .. kadar kişi var bilgisini verir
#bunu tüm değişkenlerde yapmak için :

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]


##########################
# Target ve feature #birlikte analiz
##########################
df.groupby("Outcome").agg({"Pregnancies": "mean"})
df.groupby("Outcome").agg({"Pregnancies": "mean"})
"""Out[10]: 
         Pregnancies
Outcome             
0              3.298
1              4.866"""
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)
# Data Preprocessing (Veri Ön İşleme)
######################################################
df.shape
df.head()

df.isnull().sum()

df.describe().T

for col in cols:
    print(col, check_outlier(df, col)) #aykırı değerleri yukarda belirlediğimiz fonk ile bakıyoruz. sadece insulin değişkeninde aykırı değer varmış.

replace_with_thresholds(df, "Insulin") #insulin değişkeninde var olan aykırı değerleri eşik değerle değiştiriyoruz

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]]) #değişken standartlaştırma, ölçeklendirme

df.head(
# Model & Prediction
######################################################

y = df["Outcome"]#bağımlı değişken

X = df.drop(["Outcome"], axis=1) #bağımsız değişkenler

log_model = LogisticRegression().fit(X, y) #model kuruldu

log_model.intercept_ #sabit
log_model.coef_ #ağırlıklar

y_pred = log_model.predict(X)

y_pred[0:10] #tahmin edilen değerler

y[0:10] #gerçek değerler


# Model Evaluation (model başarı değerlendirme)
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))


# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.83939


######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)


######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])



# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327

######################################################
# Prediction for A New Observation
######################################################

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)