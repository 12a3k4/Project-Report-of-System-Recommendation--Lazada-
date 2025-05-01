# Project-Report-of-System-Recommendation-(Lazada)

# Laporan Proyek Machine Learning - Ali Akbar Said

Final Submission dari course Machine Learning Terapan (Dicoding).

Dibuat dengan tujuan untuk melengkapi Learning Objective dari "Laskar AI 2025".

## Project Overview

Saat ini, industri e-commerce seperti Lazada berkembang sangat pesat di Indonesia, dengan beragam pilihan produk dari berbagai kategori. Namun, banyaknya produk ini membuat konsumen kesulitan memilih produk yang sesuai dengan kebutuhan mereka. Oleh karena itu, sistem rekomendasi menjadi solusi penting untuk membantu pengguna menemukan produk yang relevan dengan preferensi mereka.

Sistem rekomendasi yang akan dibangun dalam proyek ini berfokus pada pendekatan berbasis konten (Content-Based Filtering), menggunakan algoritma TF-IDF (Term Frequency-Inverse Document Frequency) dan Cosine Similarity. Tujuannya adalah untuk mengidentifikasi produk yang serupa berdasarkan fitur seperti nama produk dan deskripsi produk, sehingga pengalaman pengguna saat berbelanja menjadi lebih personal dan efisien.

## Business Understanding

### Problem Statements
Beberapa pertanyaan utama yang ingin dijawab melalui proyek ini adalah:
- Bagaimana sistem rekomendasi berbasis kemiripan produk dapat membantu pengguna menemukan produk lain yang relevan?
- Bagaimana sistem ini dapat memberikan rekomendasi berdasarkan nama, kategori, rating, dan harga produk?

### Goals
Tujuan utama dari proyek analisis prediktif ini adalah untuk menjawab pertanyaan-pertanyaan di atas, beberapa tujuan spesifik yang ingin dicapai adalah sebagai berikut:
- Menyediakan rekomendasi produk yang relevan dengan teknik Content-Based Filtering menggunakan kesamaan kategori produk
- Memanfaatkan informasi kategori dan rating produk untuk menampilkan produk serupa yang lebih relevan bagi pengguna
- Memanfaatkan informasi kategori dan total rekomendasi tiap produk untuk menampilkan produk serupa yang lebih relevan bagi pengguna

### Solution statements
Solusi yang diusulkan adalah membangun sistem rekomendasi berbasis Content-Based Filtering dengan menggunakan TF-IDF untuk mengekstraksi fitur dari teks nama produk dan deskripsi produk, lalu menghitung kesamaan antar produk menggunakan Cosine Similarity.

## Data Understanding
Dataset berisi 3961 baris dan 6 kolom, dataset terdiri dari Kolom-kolomnya berisi brand_name, product_name, product_id, beauty_point_earned, price_range, price_by_combinations, url, active_date, default_category, categories, rating_types_str, average_rating, total_reviews, average_rating_by_types, total_recommended_count, total_repurchase_maybe_count, total_repurchase_no_count, total_repurchase_yes_count, total_in_wishlist. Variabel yang akan digunakan pada kasus kali ini sebagai parameter rekomendasi adalah variabel default_category. Kondisi data masih belum bersih dengan ditandai masih adanya missing values.

Referensi:
Nugroho. "Product Data E-Commerce Lazada". Tautan: [https://www.kaggle.com/datasets/intodarkmoon/product-data-e-commerce-lazada]. Diakses pada 27 April 2025.

Variabel-variabel pada dataset adalah sebagai berikut:
- Name: nama produk
- Rating Score: penilaian produk oleh customer
- Total Review: Berapa banyak customer yang melakukan review
- Seller Name: Nama penjual.
- Location: lokasi toko offline / gudang.
- Price (IDR): Harga barang dalam mata uang Rupiah.

Jadi, dalam dataaset terdiri dari 3796 nama produk yang berbeda, 304 nama toko yang berbeda, 510 nilai rating yang berbeda, dan 69 lokasi pedagang yang berbeda

## Data Preparation
### Teknik Data Preparation yang Digunakan
Pada proyek ini, beberapa teknik data preparation diterapkan untuk memastikan data siap digunakan dalam tahap pemodelan. Berikut adalah langkah-langkah yang dilakukan:
#### 1. Mengatasi Missing Values
Pada tahap ini, terdapat banyak missing values di dua kolom, yaitu 'Rating Score' dan 'Total Review'. Oleh sebab itu, baris pada dataset yang memiliki missing value pada kolom tersebut akan di drop, karena memiliki missing values dan nilai yang tidak terlalu berpengaruh terhadap tujuan rekomendasi.

#### 2. Proses TF-IDF 
Pada tahap ini, TF-IDF (Term Frequency-Inverse Document Frequency) digunakan sebagai metode ekstraksi fitur untuk mengubah data kategori produk menjadi representasi vektor yang dapat digunakan sebagai input dalam model rekomendasi. TF-IDF menghitung bobot dari setiap kata di dalam kategori produk berdasarkan frekuensinya atau kemunculannnya dalam seluruh isi dokumen serta jarang atau seringnya kata tersebut muncul dalam seluruh isi dokumen, proses ini membantu sistem dalam menentukan kata-kata yang paling penting untuk mewakili kategori. Dalam hal ini, tf.get_feature_names_out(), akan mendapat kata-kata unik di seluruh kategori, lalu mentransformasikannya dengan menjadi matriks TF-IDF dan terakhir menyusunnya dalam bentuk dataframe dengan mengambil sekitar 20 sampel acak untuk melihat evaluasi awal dengan indeks baris product_name. Seluruh proses TF-IDF akan digunakan sebagai parameter untuk menghitung cosine similarity pada tahap awal pemodelan. Dalam hal ini, TF_IDF yang menghitung kategori produk dapat digunakan untuk dua algoritma berbeda.

## Modeling
### Model Development: Sistem Rekomendasi dengan Content-Based Filtering dengan Pendekatan Cosine Similarity berdasarkan Kategori Produk
Tahapan ini menjelaskan model sistem rekomendasi yang dibangun untuk menyelesaikan masalah rekomendasi produk di dataset. Model ini akan menghasilkan top-N recommendations berdasarkan konten produk, dengan pendekatan yang memanfaatkan rating produk dari pengguna. Model ini dipilih untuk mengukur tingkat kemiripan antarproduk berdasarkan kategori, dan parameter rating tertinggi, digunakan cosine similarity pada representasi TF-IDF yang menggambarkan kata-kata penting dari kategori produk.
#### 1. Cosine Similarity
Cosine similarity adalah metode untuk mengukur kesamaan antar dua vektor. Dalam sistem rekomendasi ini, vektor dari setiap produk mewakili bobot kata dalam kategori produk tersebut. Cosine similarity menghitung kemiripan antar produk dengan menilai sudut kosinus antara dua vektor. Nilai cosine similarity berkisar antara -1 hingga 1, di mana nilai 1 menunjukkan kesamaan maksimal (arah vektor yang sama), dan nilai 0 menunjukkan tidak ada kesamaan. Matriks TF-IDF dari tahap sebelumnya digunakan untuk menghitung cosine similarity, yang menghasilkan matriks kemiripan antar produk. Matriks ini dibentuk dalam data frame dengan product_name sebagai indeks dan kolom, sehingga dapat dilihat nilai dari kemiripan setiap produk dengan produk lainnya.
#### 2. Pembuatan Model dengan Content Based Filtering
Algoritma ini menggunakan Conten Based Filtering, yang menggabungkan TF-IDF dari kategori produk dan kategori  untuk menghasilkan representasi vektor yang lebih kaya. Berikut ini adalah langkah-langkahnya:
- ekstraksi fitur, diambil dari data kategori (dari kolom Name) dan data TF-IDF dari kategori produk digabungkan menjadi satu matriks fitur,
- perhitungan kemiripan, dengan cosine similarity diterapkan pada matriks fitur gabungan, menghitung kemiripan antar produk berdasarkan kategori,
- fungsi rekomendasi, fungsi products_recommendations menerima parameter nama_produk, yang digunakan untuk mencari indeks kemiripan pada produk lainnya. Fungsi ini juga menggunakan parameter similarity_data yang berisi data cosine similarity dan items, data produk yang ingin ditampilkan.
- urutan kemiripan, produk yang paling mirip diurutkan menggunakan argpartition pada hasil cosine similarity, menghasilkan top-N produk. Menambahkan urutan berdasarkan rating tertinggi
- output, return akan mengembalikan recommendations, yaitu dataframe produk-produk yang mirip, menampilkan 'Name', 'Location', 'Seller Name'.

### Model Development: Sistem Rekomendasi dengan Content-Based Filtering dengan Pendekatan Cosine Similarity berdasarkan Rata-Rata Rating 
Tahapan ini menjelaskan model sistem rekomendasi yang dibangun untuk menyelesaikan masalah rekomendasi produk di dataset. Model ini akan menghasilkan top-N recommendations berdasarkan konten produk, dengan pendekatan yang memanfaatkan kategori produk yang berkolerasi dengan rating tertinggi. Model ini dipilih untuk mengukur tingkat kemiripan antarproduk berdasarkan kategori dan rating yang paling tinggi, dan digunakan cosine similarity pada representasi TF-IDF yang menggambarkan kata-kata penting dari kategori produk. Pada kasus kali ini, dataset tidak memiliki data pengguna, oleh karena itu, digunakan algoritma content-based filtering, dengan tujuan memberikan rekomendasi produk dengan rating tertinggi dari nama produk yang telah digunkaan sebelumnya. Hal ini, juga berlaku pada rekomendasi produk berdasarkan Total rekomendasi, dengan harapan output akan memunculkan 
#### 1. Cosine Similarity
Cosine similarity adalah metode untuk mengukur kesamaan antar dua vektor, yang dalam sistem rekomendasi ini menggunakan vektor dari setiap produk untuk merepresentasikan bobot kata dalam kategori produk dan nilai rata-rata rating produk. Pada model ini, bobot kata dari TF-IDF kategori produk dikombinasikan dengan nilai rating rata-rata produk untuk menghasilkan fitur gabungan menggunakan combined_features. Dengan cosine_similarity, kemiripan dihitung berdasarkan kombinasi ini, sehingga produk yang tidak hanya mirip dalam kategori, tetapi juga dalam penilaian pengguna (rating) dapat lebih diutamakan. Hasilnya adalah matriks cosine similarity, dalam bentuk data frame dengan product_name sebagai indeks dan kolom, yang memperlihatkan kemiripan antar produk.
#### 2. Pembuatan Model dengan Content Based Filtering
Algoritma ini tetap menggunakan Conten Based Filtering tetapi parameternya menggunakan parameter Collaborative-Based Filtering, yang menggabungkan TF-IDF dari kategori produk dan rata-rata rating dari pengguna untuk menghasilkan representasi vektor yang lebih kaya. Berikut ini adalah langkah-langkahnya:
- ekstraksi fitur, diambil dari data rating (dari kolom average_rating) dan data TF-IDF dari kategori produk digabungkan menjadi satu matriks fitur,
- perhitungan kemiripan, dengan cosine similarity diterapkan pada matriks fitur gabungan, menghitung kemiripan antar produk berdasarkan kategori dan rating,
- fungsi rekomendasi, fungsi products_recommendations_by_rating menerima parameter nama_produk, yang digunakan untuk mencari indeks kemiripan pada produk lainnya. Fungsi ini juga menggunakan parameter similarity_data yang berisi data cosine similarity dan items, data produk yang ingin ditampilkan.
- urutan kemiripan, produk yang paling mirip diurutkan menggunakan argpartition pada hasil cosine similarity, menghasilkan top-N produk. Menambahkan urutan berdasarkan rating tertinggi, setelah daftar top-N produk terpilih, data diurutkan lagi berdasarkan average_rating secara menurun, sehingga produk dengan rating lebih tinggi ditampilkan lebih awal. 
- output, return akan mengembalikan recommendations, yaitu dataframe produk-produk yang mirip, menampilkan 'Name', 'Location', 'Seller Name', 'Rating Score', diurutkan berdasarkan kemiripan dan rating tertinggi di antara produk tersebut.

### Model Development: Sistem Rekomendasi dengan Content-Based Filtering dengan Pendekatan Cosine Similarity berdasarkan Price (IDR)
Tahapan ini menjelaskan model sistem rekomendasi yang dibangun untuk menyelesaikan masalah rekomendasi produk di dataset. Model ini akan menghasilkan top-N recommendations berdasarkan konten produk, dengan pendekatan yang memanfaatkan kategori produk yang berkolerasi dengan total rekomendasi dari pengguna lain. Model ini dipilih untuk mengukur tingkat kemiripan antarproduk berdasarkan kategori dan total rekomendasi yang paling banyak, dan digunakan cosine similarity pada representasi TF-IDF yang menggambarkan kata-kata penting dari kategori produk. Pada kasus kali ini, dataset tidak memiliki data pengguna, oleh karena itu, digunakan algoritma content-based filtering, dengan tujuan memberikan rekomendasi produk dengan total rekomendasi dari nama produk yang telah digunkaan sebelumnya. 
#### 1. Cosine Similarity
Cosine similarity adalah metode untuk mengukur kesamaan antar dua vektor, yang dalam sistem rekomendasi ini menggunakan vektor dari setiap produk untuk merepresentasikan bobot kata dalam kategori produk dan nilai rata-rata rating produk. Pada model ini, bobot kata dari TF-IDF kategori produk dikombinasikan dengan nilai total rekomendasi produk untuk menghasilkan fitur gabungan menggunakan combined_features. Dengan cosine_similarity, kemiripan dihitung berdasarkan kombinasi ini, sehingga produk yang tidak hanya mirip dalam kategori, tetapi juga dalam total rekomendasi dari pengguna lain dapat lebih diutamakan. Hasilnya adalah matriks cosine similarity, dalam bentuk data frame dengan product_name sebagai indeks dan kolom, yang memperlihatkan kemiripan antar produk.
#### 2. Pembuatan Model dengan Content Based Filtering
Algoritma ini tetap menggunakan Conten Based Filtering tetapi parameternya menggunakan parameter Collaborative-Based Filtering, yang menggabungkan TF-IDF dari kategori produk dan rata-rata rating dari pengguna untuk menghasilkan representasi vektor yang lebih kaya. Berikut ini adalah langkah-langkahnya:
- ekstraksi fitur, diambil dari data total rekomendasi (dari kolom total_recommended_count) dan data TF-IDF dari kategori produk digabungkan menjadi satu matriks fitur,
- perhitungan kemiripan, dengan cosine similarity diterapkan pada matriks fitur gabungan, menghitung kemiripan antar produk berdasarkan kategori dan total rekomendasi,
- fungsi rekomendasi, fungsi products_recommendations_by_count_recom menerima parameter nama_produk, yang digunakan untuk mencari indeks kemiripan pada produk lainnya. Fungsi ini juga menggunakan parameter similarity_data yang berisi data cosine similarity dan items, data produk yang ingin ditampilkan.
- urutan kemiripan, produk yang paling mirip diurutkan menggunakan argpartition pada hasil cosine similarity, menghasilkan top-N produk. Menambahkan urutan berdasarkan total rekomendasi produk terbanyak, setelah daftar top-N produk terpilih, data diurutkan lagi berdasarkan total_recommended_count secara menurun, sehingga produk dengan total rekomendasi lebih tinggi ditampilkan lebih awal. 
- output, return akan mengembalikan recommendations, yaitu dataframe produk-produk yang mirip, menampilkan 'Name', 'Location', 'Seller Name', 'Price (IDR)', diurutkan berdasarkan kemiripan dan rating tertinggi di antara produk tersebut.

#### 3. Hasil Top-N Recommendation 
##### Sistem Rekomendasi dengan Content-Based Filtering dengan Pendekatan Cosine Similarity berdasarkan Kategori Produk (produk yang dicari: OPPO A58 6/128GB Garansi Resmi Indonesia)
| Index | Nama Produk                                     | Lokasi             | Nama Penjual          |
|-------|-------------------------------------------------|--------------------|------------------------|
| 0     | OPPO A58 6/128GB Garansi Resmi                  | Kota Malang        | Alibabastore.id        |
| 1     | OPPO A58 NFC (8GB+128GB) - GARANSI RESMI        | Kota Jakarta Barat | Duniagadgetku          |
| 2     | OPPO A58 NFC - RAM 8+8GB / ROM 128GB - GARANSI  | Kab. Sidoarjo      | NISFU GADGET           |
| 3     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI        | Kota Medan         | J3 SHOP                |
| 4     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI        | Kota Medan         | KSTORE87               |
| 5     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI        | Kota Medan         | J3 SHOP                |
| 6     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI        | Kota Medan         | KSTORE87               |
| 7     | HP OPPO A58 RAM 6GB ROM 128GB                   | Kab. Bekasi        | Pelangipelangi Shop    |
| 8     | Oppo A18 4/128GB Garansi Resmi Oppo Indonesia   | Kota Bandung       | Zabeela Store          |
| 9     | Oppo A38 6/128GB 4/128GB Garansi Resmi Oppo In  | Kota Bandung       | Zabeela Store          |

##### Sistem Rekomendasi dengan Content-Based Filtering dengan Pendekatan Cosine Similarity berdasarkan Rata-Rata Rating (produk yang dicari: OPPO A58 6/128GB Garansi Resmi Indonesia)
| Index | Nama Produk                                           | Lokasi             | Nama Penjual            | Rating Score |
|-------|-------------------------------------------------------|--------------------|--------------------------|--------------|
| 0     | OPPO A58 6/128GB Garansi Resmi                         | Kota Malang        | Alibabastore.id          | 5.0          |
| 1     | OPPO A58 NFC (8GB+128GB) - GARANSI RESMI               | Kota Jakarta Barat | Duniagadgetku            | 5.0          |
| 2     | OPPO A58 NFC - RAM 8+8GB / ROM 128GB - GARANSI         | Kab. Sidoarjo      | NISFU GADGET             | 5.0          |
| 4     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI               | Kota Medan         | KSTORE87                 | 5.0          |
| 10    | OPPO A18 4/128GB Garansi Resmi Indonesia               | Kota Surabaya      | Apollo Gadget Store      | 5.0          |
| 6     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI               | Kota Medan         | KSTORE87                 | 5.0          |
| 7     | HP OPPO A58 RAM 6GB ROM 128GB                          | Kab. Bekasi        | Pelangipelangi Shop      | 5.0          |
| 8     | Oppo A18 4/128GB Garansi Resmi Oppo Indonesia          | Kota Bandung       | Zabeela Store            | 5.0          |
| 11    | OPPO A38 RAM 4/128GB & 6/128GB (EXTENDED RAM)          | Kab. Tangerang     | Rajalaku Store.Id        | 5.0          |
| 9     | Oppo A38 6/128GB 4/128GB Garansi Resmi Oppo Indonesia  | Kota Bandung       | Zabeela Store            | 5.0          |

##### Sistem Rekomendasi dengan Content-Based Filtering dengan Pendekatan Cosine Similarity berdasarkan Harga / Price (produk yang dicari: OPPO A58 6/128GB Garansi Resmi Indonesia)
| Index | Nama Produk                                           | Lokasi             | Nama Penjual            | Harga (IDR) |
|-------|-------------------------------------------------------|--------------------|--------------------------|-------------|
| 3     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI               | Kota Medan         | J3 SHOP                  | 2,469,000   |
| 5     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI               | Kota Medan         | J3 SHOP                  | 2,469,000   |
| 1     | OPPO A58 NFC (8GB+128GB) - GARANSI RESMI               | Kota Jakarta Barat | Duniagadgetku            | 2,449,000   |
| 0     | OPPO A58 6/128GB Garansi Resmi                         | Kota Malang        | Alibabastore.id          | 2,399,000   |
| 7     | HP OPPO A58 RAM 6GB ROM 128GB                          | Kab. Bekasi        | Pelangipelangi Shop      | 2,399,000   |
| 4     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI               | Kota Medan         | KSTORE87                 | 2,125,000   |
| 6     | OPPO A58 6/128 & 8/128 NEW GARANSI RESMI               | Kota Medan         | KSTORE87                 | 2,125,000   |
| 9     | Oppo A38 6/128GB 4/128GB Garansi Resmi Oppo Indonesia  | Kota Bandung       | Zabeela Store            | 1,799,000   |
| 11    | OPPO A38 RAM 4/128GB & 6/128GB (EXTENDED RAM)          | Kab. Tangerang     | Rajalaku Store.Id        | 1,675,000   |
| 2     | OPPO A58 NFC - RAM 8+8GB / ROM 128GB - GARANSI         | Kab. Sidoarjo      | NISFU GADGET             | 1,599,000   |

#### Kelebihan dan Kekurangan dari Content-Based Filtering
Kelebihan:
- Personalisasi, Content-based filtering dapat memberikan rekomendasi yang lebih personal kepada konsumen dengan mempertimbangkan preferensi dan minat mereka terhadap kategori atau fitur produk tertentu.
- Tidak terikat oleh konsumen lain, sistem ini tidak bergantung pada interaksi konsumen atau pengguna lain, sehingga dapat berfungsi dengan baik, ketika data memiliki interaksi pengguna yang terbatas (cold-start problem).
- Kualitas rekomendasi, dengan memanfaatkan konten dan deskripsi produk, sistem dapat merekomendasikan produk yang relevan berdasarkan kesamaan fitur-fitur yang ada, meningkatkan relevansi dan kualitas rekomendasi.

Kekurangan:
- Terbatas pada konten saja, rekomendasi hanya didasarkan pada konten produk, sehingga kurang bervariasi. Konsumen mungkin tidak mendapatkan produk yang berada di luar kategori yang dipilih, meskipun produk tersebut mungkin sesuai dengan preferensi mereka.
- Keterbatasan dalam penemuan, model ini dapat mengarahkan kepada konsumen dengan hanya melihat produk yang mirip dengan yang telah mereka pilih sebelumnya, sehingga mengurangi kemungkinan menemukan produk baru yang berbeda.
- Pengabaian interaksi antar pengguna, Content-based filtering tidak mempertimbangkan preferensi atau perilaku pengguna yang lebih mendalam, hal ini bisa berakibat pada rekomendasi yang bisa saja tidak sepenuhnya sesuai dengan keinginan pengguna.

Dengan memanfaatkan ketiga pendekatan ini, sistem rekomendasi di produk-produk pada Lazada dapat memberikan saran yang lebih baik dan relevan kepada pengguna, dan meningkatkan pengalaman berbelanja secara keseluruhan.

## Evaluation
Model sistem rekomendasi produk di Lazada menggunakan pendekatan Content-Based Filtering berbasis TF-IDF dan cosine similarity menunjukkan hasil yang cukup baik. Untuk mengevaluasi performa sistem, metrik evaluasi Precision@k digunakan, yang mengukur proporsi rekomendasi yang relevan di antara semua rekomendasi yang diberikan. Dalam konteks ini, Precision@k mengukur seberapa banyak dari k rekomendasi teratas yang relevan untuk pengguna. Dengan formula:

Precision@k = Jumlah produk relevan di top-k/𝑘

Dengan cara kerja, yaitu menghitung jumlah produk yang relevan dari k rekomendasi teratas, dibagi dengan jumlah seluruh rekomendasi yang diberikan. Pada pengujian rekomendasi untuk kategori produk "OPPO A58 6/128GB Garansi Resmi Indonesia" dari 10 item teratas yang direkomendasikan, semua produk memiliki kategori yang sama, menghasilkan precision sebesar 10/10 atau 100%.

Model rekomendasi berbasis Content-Based Filtering yang dikembangkan telah menunjukkan dampak positif dalam memenuhi Business Understanding, terutama dalam menjawab Problem Statements dan mencapai Goals yang diharapkan.
#### Menjawab Problem Statements:
Sistem ini telah berhasil memberikan rekomendasi produk yang memiliki kesamaan kategori dengan produk yang telah dipilih oleh pengguna. Dengan menggunakan TF-IDF untuk mengekstraksi kata kunci dalam kategori produk dan cosine similarity untuk mengukur kesamaan antar produk, model dapat mengarahkan pengguna pada produk serupa, membantu pengguna menemukan item lain yang relevan. Dengan memasukkan rating sebagai faktor tambahan, model ini juga mampu mempertimbangkan kualitas dan popularitas produk saat memberikan rekomendasi. Pengguna yang melihat rekomendasi berdasarkan kategori juga dapat mempertimbangkan produk dengan rating tinggi, yang meningkatkan kemungkinan mereka menemukan produk yang lebih berkualitas. Walaupun fokus utama pemodelan adalah pada Content-Based Filtering berdasarkan kategori, kombinasi informasi kategori dan popularitas dari total rekomendasi memberikan pengguna wawasan mengenai produk yang sering direkomendasikan, yang juga membantu mereka dalam pengambilan keputusan.
#### Mencapai Goals:
Proyek ini berhasil mencapai tujuan untuk menyediakan rekomendasi yang relevan melalui pendekatan berbasis kesamaan kategori produk, menggunakan TF-IDF dan cosine similarity. Ini membuat model dapat menampilkan produk yang serupa secara konten, memudahkan pengguna dalam mencari produk alternatif. Dengan menambahkan rating sebagai parameter, model tidak hanya mengandalkan kesamaan kategori, tetapi juga mempertimbangkan kualitas produk melalui nilai rating, yang berhasil meningkatkan relevansi rekomendasi. Selain itu, penggunaan total rekomendasi dari pengguna lain memberikan dimensi tambahan bagi pengguna dalam melihat popularitas produk, yang mendukung keputusan pembelian mereka.
#### Solusi Statement:
Solusi staement berdampak besar dalam sistem rekomendasi ini, yang memudahkan pengguna menemukan produk yang sesuai dengan minat mereka dengan lebih cepat dan efisien, yang berpotensi meningkatkan tingkat konversi dan pengalaman belanja, dengan benar-benar memanfaatkan representasi tekstual produk melalui teknik TF-IDF (Term Frequency-Inverse Document Frequency) dalam mengukur pentingnya kata-kata dalam kategori produk dan kesamaan antar produk akan dihitung menggunakan cosine similarity.

Secara keseluruhan, model rekomendasi berbasis Content-Based Filtering ini telah memberikan dampak positif terhadap pemenuhan Business Understanding, menjawab Problem Statements, mencapai Goals, dan memberikan solusi yang tepat bagi pengguna dalam menemukan produk yang mereka butuhkan.
