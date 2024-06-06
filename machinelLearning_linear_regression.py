#Makine Öğrenmesi:Bilgisayarların insanlara benzer şekilde öğrenmesini sağlamak amacıyla çeşitli algoritma ve tekniklerin geliştirilmesi için çalışılan bilimsel çalışma alanıdır.

#Bağımlı değişkeni yani hedeflediğimiz değişkeni sayısal olan problemlere regresyon problemi denir.
#Model kurmak bağımlı ve bağımsız değişkenler arasındaki ilişkiyi çıkarmak.
#Değişken türleri
#Sayısal değişkenler:Yaş, metrekare,fiyat gibi değişkenlerdir.Kesikli ondalıklı olabilir
#Kategorik:kadın erkek,hayatta kaldı-kalmadı,hasta-hasta değil,futbol takımları gibi bir sınıfı, kategoriyi ifade eden değişkenlerdir.
#nominal:sınıflar arasında fark yoktur, kadın-erkek, futbol takımları gibi
#ordinal:Sınıflar arasında fark vardır, eğitim durumları gibi
#supervised denetimli öğrenme türlerinde; bağımlı değişken(target,dependent,output,response) ilgilendiğimiz hedef değişkene bağımlı değişken denir.
#örneğin hastalık tahmini durumunda hasta olup olmama bağımlı değişkendir, ev fiyat tahmininde evlerin fiyatları bağımlı değişkendir
#bağımsız değişken(independent,feature,input,column,explonatary):ilgilendiğimiz problemdeki bağımlı değişkeni oluşturduğunu varsaydığımız diğer ifadeyle targetın oluşmasında etki ettiğini varsaydığımız değişkendir.

#Öğrenme türleri
#1)Denetimli (Supervised) 2)denetimsiz(Unsupervised) 3)pekiştirmeli(Reinforcement)
#pekiştirmeli:deneme yanılma yoluyla yanlış yaptıklarından pekiştirerek öğrenme. (Örneğin boş odada bir robotun kapıyı bulmaya çalışması,çocuğun sobaya dokunarak yakıyor olduğunu öğrenmesi, otonom araçarda örneğin virajlarda belirli bir hız,fren,gaz,manevraları yapmayı öğrenmeli)
#Denetimli öğrenme: veri setinde labellar yer aıyorsa denetimli öğrenmedir. Bağımlı target değişken var.
#Denetimsiz öğrenme: verisetinde labellar yoksa denetimsiz öğrenmedir. Target yok.

#Problem türleri:
#Regresyon problemlerinde bağımlı değişken sayısaldır.
#Sınıflandırma problemlerinde bağımlı değişken kategoriktir.

#Model başarı değerlendirme yöntemleri
#Tahminlerin ne kadar başarılı olduğunu değerlendirir.
#Regresyon problemlerinde başarı değerlendirme yöntemleri:mean square error (mse) (ne kadar küçükse başarı o kadar iyidir),root mean sqare error (rmse),mutlak ort. hata (MAE)
#Sınıflandırma modellerinde başarı değerlendirme yöntemleri:accuracy=doğru sınıflandırma sayısı/toplam sınıflandırılan gözlem sayısı

#Model doğrulama yöntemleri:
#Holdout yöntemi(Sınama seti yöntemi) veri seti test ve eğitim olarak bölünür.
#K katlı çapraz doğrulama (K fold cross validation): Veri setini örneğin 5 parçaya bölüp, 5 parçanın her iterasyonda 4ü ile eğitim 1i ile test şeklinde olabilir. En son hataların ort. alınarak cross val. hata elde edilir
#veya eğitim ve test olarak veri seti ikiye ayrıldıktan sonra sadece eğitim kısmına yukardaki metod uygulanabilir(4 parça ile model kur 1 parça ile test et, başka bir 4 parça ile model kur 1 parça ile test et şeklinde) ve en son hiç görmediği veri setindeki performansı değerlendirilir.

#Yanlılık varyans değiş tokuşu
#overfitting (aşırı öğrenme): modelin veriyi öğrenmesidir, ezberlemesidir, yüksek varyans. veriyi değil verinin yapısını, örüntüyü, ilişkileri öğrenmesini bekleriz.
#underfitting: modelin veriyi öğrenememe problemidir.Yüksek yanlılık.
#Aşırı öğrenmeye düştüğünüzü nasıl anlarsınız? **
#Eğitim seti ve test setinin model karmaşıklığı ve tahmin hatası çerçevesinde birlikte değerlendirilmesiyle tespit edilir. Eğitim seti ve test setindeki hata değişimleri incelenir bu iki hatanın birbirinden ayrılmaya başladığı nokta itibariyle aşırı öğrenme başlamıştır. Aşırı öğrenmeyi tespit ettiğimiz noktada model karmaşıklığını iterasyon süresini vs durdurursak önüne geçebilriiz.veri setinin boyutu artırılabilir, feature selection yapılabilir.

"""Doğrusal (Linear Regresyon)"""
#Amaç, bağımlı ve bağımsız değişken/değişkenler arasındaki ilişkiyi doğrusal olarak modellemektir.
#yi=b+wxi
#yi=b+w1x1+w2x2+w3x3+...wpxp
#Gerçek değerler ile tahmin edilen değerler arasındaki farkların karelerinin toplamını/ortalamasını minimum yapabilecek sabit(b) ve ağırlık(w) değerlerini bularak

#Regresyon Modellerinde Başarı Değerlendirme Metrikleri (MSE,RMSE,MAE)
#MSE YÖNTEMİ (MEAN SQUARE ERROR)
#RMSE (ROOT MEAN SQUARE ERROR)
#MAE (MEAN ABSOLUTE ERROR)

#Parametrelerin Tahmin Edilmesi (Ağırlıkların Bulunması)
#En küçük hatayı verecek olan sabiti ve ağırlığı bulmaya çalışıyoruz.
#1)Analitik Yöntem:Normal denklemler yöntemi(En küçük kareler yöntemi, OLS)
#Bu yöntem matris formunda, türeve dayalı bir yöntemdir. Bu yöntem kullanıldığında ortaya çıkacak sonuç üzerinde analitik, neden sonuç yorumları yapılabilir.
#2)Optimizasyon çözümü:Gradient Descent:
#Parametrelerin değerlerini iteratif bir şekilde değiştirerek çalışır.

#Doğrusal Regresyon için Gradient Descent
#Gradient Descent metodu, bir fonksiyonun minimum yapabileceği parametre değerlerini bulabilmektir.
#İlgili fonksiyonun, ilgili parametreye göre kısmi türevini alır
#bu türev sonucuna göre güncelleme işlemi yapar, elde edilen türev (gradient) değeri ilgili fonksiyonun max. artış yönünü verir. Max. artış yönünü veren gradient'in tersine doğru belirli bir şiddet ile (alfa) giderek parametrennin eski değerinde değişiklik yaparak her iteradyonda hatanın azalmasını sağlar.
#Yani; gradyanın negatifi olarak tanımlanan 'en dik iniş' yönünde iteratif olarak parametre değerlerini güncelleyerek ilgili fonksiyonun minimum değerini verebilecek parametreleri bulur.
#Makine öğrenmesi için gradient descent: Cost fonksiyonunu minimize edebilecek parametreleri bulmak için kullanılır.

#Sales Prediction with Linear Regression (Satış Tahmin)
#Satış tahmin modeli gerçekleştireceğiz. Bu model, çeşitli kanallarda yapılan reklam harcamaları neticesinde ne kadar satış elde edildiğini ifade edecektir.
#Önce simple linear regression iki değişkenli regresyon modeli kuracağız. Daha sonra veri setinde bulunan 5 değişken ile model kuracağız.
##########################
#Sales Prediction With Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x) #virgülden sonra iki basamak göster.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Simple Linear Regression with OLS Using Scikit-Learn

df = pd.read_csv("C:/Users/asus/Desktop/machine learning/machine_learning/datasets/advertising.csv")

     """df
Out[5]: 
        TV  radio  newspaper  sales
0    230.1   37.8       69.2   22.1
1     44.5   39.3       45.1   10.4
2     17.2   45.9       69.3    9.3
3    151.5   41.3       58.5   18.5
4    180.8   10.8       58.4   12.9
..     ...    ...        ...    ...
195   38.2    3.7       13.8    7.6
196   94.2    4.9        8.1    9.7
197  177.0    9.3        6.4   12.8
198  283.6   42.0       66.2   25.5
199  232.1    8.6        8.7   13.4
[200 rows x 4 columns]
"""

df.shape #kaç gözlem var?
#Out[6]: (200, 4)

X=df[["TV"]] #tüm dataframe içerisinden sadece tv değişkeninin sales'a etkisini inceleyeceğiz. Daha sonra bu model denklemini grafik yardımıyla değerlendireceğiz.
y=df[["sales"]] #bağımlı değişken

#######
#Model
####
reg_model=LinearRegression().fit(X,y)
# y_hat = b + w*TV

# sabit (b - bias-intercept)
reg_model.intercept_[0]
#Out[9]: 7.032593549127693

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]
#Out[10]: 0.047536640433019764

##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150
#Out[11]: 14.163089614080658

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r") #regresyon modeli görselleştirmek için seaborn'dan regplot komutunu kullandık. X, bağımsız değişken; Y, bağımlı değişkendir.

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}") #virgülden sonra 2 basamak al yuvarla
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310) #x ekseni -10 ile 310 arasında
plt.ylim(bottom=0) #y ekseni 0dan başla
plt.show()

##########################
# Tahmin Başarısı
##########################

# MSE

#mean_squared_error(y, y_pred) ; MSE metodu gerçek değerleri ve tahmin edilen değerleri ver. Bunların farkları ve kareleri lınır toplayıp ortalaması alınarak ort. hata değeri bulunur. Fakat elimizde tahmin edilen değerler yok
y_pred = reg_model.predict(X) #tahmin edilen değerler için regresyon modelini kullanıyoruz. predict metodu ile bağımsız değişkenleri sorup bağımlı değişkenleri tahmin edip y_pred olarak kaydettik.
mean_squared_error(y, y_pred)
# 10.51 #bu değerin iyi olup olmadığını bilmiyoruz. Olmasını istediğimiz şey bu değerin mümkün olduğunca en küçük değere gitmesidir. Öğrenmek için bağımlı değişkenin ortalamasına bakıyoruz.
y.mean()
#14.02 bağımlı değişkenin ortalaması
y.std()
# 5.22 bu durumda 19 -9 arasında değerler değişiyor gibi görünüyor. bu durumda ort. hata büyük gibi görünüyor. 1-1.5 civarı olsa daha iyi olurdu.

# RMSE

np.sqrt(mean_squared_error(y, y_pred)) #RMSE için numpydan MSEnin karekökünü alıyoruz.
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE
#doğrusal regresyon modelinde modelin başarısına ilişkin bir metriktir. Bu değer verisetindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir. yani bu verisetinde tv değişkeninin satış değişkenindeki değişikliği açıklama yüzdesidir.
reg_model.score(X, y)
#0.61 Yani bu modelde bağımsız değişkenler bağımlı değişkenin %61ini açıklayabilmektedir.
#değişken sayısı arttıkça r kare artmaya meğillidir.


######################################################
# Multiple Linear Regression
######################################################
#Birden fazla bağımsız değişken örneği

df = pd.read_csv("C:/Users/asus/Desktop/machine learning/machine_learning/datasets/advertising.csv")

"""df
Out[12]: 
        TV  radio  newspaper  sales
0   230.10  37.80      69.20  22.10
1    44.50  39.30      45.10  10.40
2    17.20  45.90      69.30   9.30
3   151.50  41.30      58.50  18.50
4   180.80  10.80      58.40  12.90
..     ...    ...        ...    ...
195  38.20   3.70      13.80   7.60
196  94.20   4.90       8.10   9.70
197 177.00   9.30       6.40  12.80
198 283.60  42.00      66.20  25.50
199 232.10   8.60       8.70  13.40
[200 rows x 4 columns]"""

X= df.drop('sales', axis=1) #bağımsız değişkenler
"""X
Out[14]: 
        TV  radio  newspaper
0   230.10  37.80      69.20
1    44.50  39.30      45.10
2    17.20  45.90      69.30
3   151.50  41.30      58.50
4   180.80  10.80      58.40
..     ...    ...        ...
195  38.20   3.70      13.80
196  94.20   4.90       8.10
197 177.00   9.30       6.40
198 283.60  42.00      66.20
199 232.10   8.60       8.70
[200 rows x 3 columns]"""

y=df[["sales"]] #bağımlı değişken

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=20, random_state=1)

#train ile model kurup test seti ile test edeceğiz.
reg_model=LinearRegression()
reg_model.fit(X_train,y_train)

veya #reg_model=LinearRegression().fit(X_train,y_train)

# sabit (b - bias)
reg_model.intercept_ #array([2.95666177])

# coefficients (w - weights)
reg_model.coef_ #array([[ 0.04647758,  0.18534601, -0.00231602]])
##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90 sabit
# 0.0468431 , 0.17854434, 0.00258619 ağırlıklar

# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002 (model denklemi)

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619 #5.88

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN RKARE
reg_model.score(X_train, y_train) #0.89

# Test RMSE
y_pred = reg_model.predict(X_test) #train üzerinden kurduğumuz modele test setini soruyoruz.
np.sqrt(mean_squared_error(y_test, y_pred)) #bağımlı değişkenin gerçek değeri y_test, tahmin edilen değeri y_pred
# 1.41

# Test RKARE
reg_model.score(X_test, y_test)


# 10 Katlı CV RMSE
# #10 katlı çapraz doğrulama
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
"""cross_val_score :   array([-3.56038438, -3.29767522, -2.08943356, -2.82474283, -1.3027754 ,
       -1.74163618, -8.17338214, -2.11409746, -3.04273109, -2.45281793])"""
# mean 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71



######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################
#Cost function MSE'yi minimuma getirmeye çalışıyoruz. W değerlerini değiştirerek hatanın min. olduğu noktayı bulmaya çalışyoruz.
# Cost function MSE


