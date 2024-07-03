--Veritabanı,Veritabanı Yönetim Sistemi, İlişkisel Veritabanı, SQL Dili, Transact SQL 
--SQL (Structred Query Language-Yapısal Sorgulama Dili)
--Veritabanı:Verileri listeler halinde tablo ve satırlarda tutan yapılardır. Veritabanının temel bileşenleri; tablolar,satırlar,sütunlar,indexler
--Her veritabanı yönetim sistemi bir veritabanıdır. Excel bir veritabanı iken, MS SQL,MYSQL,POSTGRESQL bir veritabanı yönetim sistemidir.
--veritabanından veriyi okurken bir takım kaynaklara ihtiyaç duyarız: CPU,RAM gibi ham haldeki veriyi sorgulamak ve analiz etmek için kullaınılır. Ancak bir veritabanı yönetim sisteminde bir veritabanı sunucu ile konuşuruz.
--veriyi getirmesini onun anlayacağı bir dil ile söyleriz ve bu sistem tamamen kendi kaynaklarını kullanarak veriyi bize gönderir. Biz veritabanı sunucunun bu işi nasıl yaptığı ile ilgilenmeyiz.
--Oracle, MSSQL,MYSQL, postgreSQL birer veritabanı yönetim sistemidir. Bu sistemler SQL dilini de içeren kendilerine özel diller geliştirmişlerdir. Örneğin oracle için PL SQL, msSQL için T-SQL dilleri mevcuttur.
--Veritabanı  sunucu:bir donanım değil yazılımdır. Veritabanı yönetim sistemlerinin yaptığı her şey veritabanı sunucu üzerinde gerçekleşir.
-----Veritabanı sunucu ile haberleşme----
--İstemci bilgisayar bir veritabanı sunucudan bir veriyi sorgulamak istiyor. Öncelikle veritabanı sunucuya bağlanması gerekir. Yine bunun için hem sunucu hem istemcinin aynı ağda olması gerekir. 
--İstemci bilgisayar veritaban sunucuya bir bağlantı isteği gönderir (kullanıcı adı ve şifre vererek) bağlantı isteği sunucu tarafından kontrol edilir sistemde bu kullanıcı adı ve şifreye ait kullanıcı yok ise sistem bağlantıyı otomatik reddeder. Var ise bağlantıyı kabul eder ve üzerinde işlem gerçekleştirmesine izin verir.
 
 --İlişkisel Veritabanı (RDMS)
 --Tekrar eden verileri tekilleştirmek amacıyla yapılandırılan veritabanı sistemleridir.
