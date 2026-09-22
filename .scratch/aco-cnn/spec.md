# ACO-CNN hyperparameter tuning

Status: ready-for-agent

## Problem Statement

Pengguna ingin membangun program Ant Colony Optimization (ACO) untuk mencari hyperparameter terbaik bagi Convolutional Neural Network (CNN) yang digunakan dalam klasifikasi gambar. Jurnal Purnomo et al. (2024), “Metaheuristics Approach for Hyperparameter Tuning of Convolutional Neural Network”, menjadi sumber inspirasi utama, tetapi tidak menjelaskan cukup banyak detail untuk direplikasi secara eksak.

Artikel menyebutkan MNIST, 20 ant, probabilitas pemilihan yang hanya bergantung pada pheromone, evaporation rate `rho = 0.25`, reinforcement konstan `+0.5`, dan delapan hyperparameter dengan delapan pilihan masing-masing. Artikel tidak menjelaskan secara lengkap nilai awal pheromone, pembagian data, jumlah epoch, stopping criterion, waktu update pheromone, atau ant mana yang melakukan reinforcement.

Tanpa pemisahan yang eksplisit, implementasi dapat mencampurkan fakta jurnal dengan asumsi pengembangan, menggunakan test set selama tuning, atau menghasilkan kode ACO yang sulit diuji tanpa melatih CNN berkali-kali.

## Solution

Bangun pipeline ACO-CNN yang dapat membedakan tiga mode eksperimen:

1. `paper_literal`, yaitu interpretasi ant-by-ant terhadap pseudocode jurnal: ant mengambil candidate, candidate dievaluasi, pheromone mengalami evaporation, lalu pilihan ant diperkuat sebelum ant berikutnya dibuat.
2. `paper_conventional`, yaitu interpretasi iteration-based: seluruh ant dibuat dan dievaluasi menggunakan pheromone pada awal iterasi, kemudian pheromone mengalami evaporation sekali dan pilihan seluruh ant yang valid diperkuat.
3. `improved`, yaitu mode pengembangan dengan search space yang lebih kecil, validation protocol eksplisit, evaporation sekali per iterasi, dan reinforcement berbobot fitness hanya untuk iteration-best.

Gunakan MNIST sebagai benchmark awal, fixed CNN architecture yang sama pada semua mode, validation accuracy sebagai fitness, dan test set yang tidak pernah digunakan selama pencarian. Pisahkan algoritma ACO dari evaluator candidate melalui satu seam utama: optimizer menerima evaluator candidate yang dapat berupa evaluator sintetis untuk smoke test atau evaluator CNN/MNIST untuk eksperimen nyata.

Catat setiap trial dan evolusi pheromone/probability dalam log yang dapat dianalisis. Setelah tuning selesai, latih model baru menggunakan seluruh 60.000 data training-development selama `best_epoch` dari candidate terbaik, lalu evaluasi satu kali pada 10.000 data test.

## User Stories

1. As a peneliti, I want to mengetahui dengan jelas bagian mana yang merupakan fakta jurnal, bagian mana yang merupakan interpretasi, dan bagian mana yang merupakan pengembangan proyek, so that laporan eksperimen tidak mengklaim asumsi sebagai isi jurnal.
2. As a peneliti, I want to menjalankan mode `paper_literal`, so that saya dapat mengamati konsekuensi membaca pseudocode jurnal secara literal.
3. As a peneliti, I want to menjalankan mode `paper_conventional`, so that saya dapat menggunakan interpretasi iteration-based yang lebih stabil untuk perbandingan utama berbasis jurnal.
4. As a peneliti, I want to menjalankan mode `improved`, so that saya dapat menguji strategi reinforcement yang menggunakan kualitas candidate secara eksplisit.
5. As a pengguna, I want to memilih mode melalui command line, so that setiap eksperimen dapat direproduksi tanpa mengubah kode sumber.
6. As a peneliti, I want semua mode menggunakan fixed CNN architecture yang sama, so that perbedaan mode tidak disebabkan oleh perubahan arsitektur convolutional.
7. As a peneliti, I want input MNIST diproses sebagai gambar grayscale berukuran 28×28×1, so that representasi data sesuai dengan dataset yang digunakan.
8. As a peneliti, I want data resmi MNIST dibagi menjadi 50.000 training, 10.000 validation, dan 10.000 test, so that fitness dapat dihitung tanpa menggunakan test set.
9. As a peneliti, I want split training-validation dibuat secara stratified dan deterministik menggunakan dataset seed 2024, so that semua run menggunakan validation set yang sama.
10. As a peneliti, I want test set tidak disentuh selama ACO, so that test accuracy tetap menjadi evaluasi akhir yang independen.
11. As a peneliti, I want paper-faithful modes mempertahankan delapan dimensi search space jurnal, so that implementasi tetap dapat dibandingkan dengan deskripsi artikel.
12. As a peneliti, I want kesalahan penulisan `linier` pada artikel direpresentasikan sebagai identifier implementasi `linear` sambil mempertahankan label asli dalam metadata, so that provenance search space tidak hilang.
13. As a peneliti, I want improved mode menggunakan 5.184 candidate configurations, so that eksperimen awal tidak menghadapi search space paper sebesar 16.777.216 konfigurasi.
14. As a peneliti, I want improved mode mencari dense-layer widths, dropout, batch size, activation, optimizer, dan learning rate, so that hyperparameter training dan dense architecture dapat dioptimalkan bersama.
15. As a peneliti, I want improved mode menggunakan sparse categorical crossentropy sebagai loss tetap, so that baseline improved tidak tercampur dengan ketidakcocokan berbagai loss.
16. As a peneliti, I want paper-faithful modes mencoba loss yang tercantum dalam artikel tanpa mengubah output atau label secara otomatis, so that konfigurasi yang gagal dapat diamati sebagai failed trial.
17. As a peneliti, I want loss yang dapat dijalankan secara teknis tetapi tidak lazim untuk sparse multiclass diberi semantic warning, so that technical success tidak disamakan dengan methodological validity.
18. As a peneliti, I want setiap hyperparameter memiliki pheromone vector sendiri, so that representasi discrete ACO sesuai dengan model yang dijelaskan artikel.
19. As a peneliti, I want semua pheromone diinisialisasi dengan nilai 1.0, so that starting preference uniform dan asumsi yang tidak dijelaskan jurnal terdokumentasi.
20. As a peneliti, I want probability candidate dihitung dengan normalisasi pheromone per hyperparameter, so that setiap option probability valid dan jumlahnya satu.
21. As a peneliti, I want pheromone memiliki configurable minimum value, so that evaporation tidak membuat option kehilangan probabilitas sepenuhnya.
22. As a peneliti, I want `paper_literal` melakukan evaporation setelah setiap ant, so that interpretasi literal dapat diuji dan dampaknya terhadap dinamika pheromone terlihat.
23. As a peneliti, I want ant berikutnya pada `paper_literal` melihat pheromone yang telah diubah ant sebelumnya, so that lifecycle ant-by-ant benar-benar terwakili.
24. As a peneliti, I want `paper_conventional` melakukan evaporation tepat sekali per iterasi, so that seluruh ant dalam iterasi disampling dari pheromone yang sama.
25. As a peneliti, I want `paper_conventional` memperkuat option semua ant yang memiliki fitness valid dengan reinforcement konstan 0.5, so that mode tersebut tetap dekat dengan formula jurnal.
26. As a peneliti, I want `improved` memperkuat hanya iteration-best, so that pheromone lebih fokus pada candidate terbaik pada iterasi tersebut.
27. As a peneliti, I want reinforcement improved dihitung sebagai Q dikali validation accuracy dengan Q sebesar 1.0, so that pengaruh fitness dapat dijelaskan secara langsung.
28. As a peneliti, I want failed trial tidak mengubah pheromone melalui reinforcement, so that candidate tanpa fitness valid tidak dianggap solusi yang baik.
29. As a peneliti, I want evaporation tetap terjadi ketika seluruh ant dalam iterasi gagal, so that lifecycle mode tetap konsisten dan eksperimen dapat melanjutkan iterasi berikutnya.
30. As a peneliti, I want run yang seluruh trial-nya gagal diberi status failed, so that sistem tidak menghasilkan final model tanpa candidate valid.
31. As a peneliti, I want candidate fitness didefinisikan sebagai validation accuracy maksimum, so that objective ACO eksplisit dan tidak menggunakan test accuracy.
32. As a peneliti, I want CNN candidate menggunakan early stopping berdasarkan validation accuracy dengan patience 2 dan restore best weights, so that training kandidat tidak membuang budget ketika performa berhenti membaik.
33. As a peneliti, I want best epoch dipilih berdasarkan validation accuracy, lalu validation loss sebagai tie-break, so that final retraining memiliki aturan yang deterministik.
34. As a peneliti, I want validation accuracy, validation loss, training accuracy, dan best epoch disimpan untuk setiap successful trial, so that kualitas candidate dapat diaudit.
35. As a peneliti, I want framework default learning rate digunakan pada paper-faithful modes, so that learning rate tidak diam-diam menambah dimensi search space jurnal.
36. As a peneliti, I want effective learning rate tetap dicatat pada paper-faithful modes, so that konfigurasi training aktual tetap diketahui.
37. As a peneliti, I want improved mode memilih learning rate dari kandidat yang telah ditentukan, so that pengembangan dapat menguji pengaruh learning rate secara eksplisit.
38. As a peneliti, I want trial seed ditentukan secara stabil dari run seed, mode, dan canonical configuration, so that candidate identik dalam kondisi sama memiliki seed yang sama.
39. As a peneliti, I want trial seed tidak bergantung pada iteration atau ant ID, so that cache candidate tidak berubah hanya karena candidate muncul di lokasi berbeda.
40. As a peneliti, I want deterministic TensorFlow diminta untuk smoke, pilot, dan main experiment, so that hasil lebih mudah direproduksi.
41. As a peneliti, I want warning determinism yang tidak dapat diterapkan dicatat dalam metadata, so that keterbatasan reproducibility tidak tersembunyi.
42. As a pengguna, I want cache dapat diaktifkan atau dimatikan, so that saya dapat memilih antara audit perilaku mentah dan efisiensi eksperimen.
43. As a peneliti, I want cache key membedakan mode, run seed, configuration, dataset split, epoch policy, dan training policy, so that hasil stochastic dari kondisi berbeda tidak tertukar.
44. As a peneliti, I want cache hit dianggap sebagai evaluasi valid untuk ranking dan pheromone update, so that optimasi cache tidak mengubah arti kejadian ant.
45. As a peneliti, I want tie-breaking menggunakan validation accuracy, validation loss, training time, lalu ant ID, so that iteration-best dan global-best deterministik.
46. As a peneliti, I want satu baris log mewakili satu ant/candidate trial, so that semua percobaan dapat dianalisis kembali.
47. As a peneliti, I want pheromone history disimpan dalam format long dengan phase dan update step, so that perubahan pheromone per ant pada `paper_literal` dapat divisualisasikan.
48. As a peneliti, I want failed trial menyimpan failure reason dan fitness kosong, so that error tidak disamarkan sebagai accuracy nol.
49. As a peneliti, I want cache hit dibedakan dari training success melalui status dan cache flag, so that runtime dan perilaku optimasi dapat diaudit.
50. As a peneliti, I want smoke test dapat menjalankan ACO dengan evaluator sintetis tanpa TensorFlow, so that aturan pheromone dapat diverifikasi cepat.
51. As a peneliti, I want evaluator CNN/MNIST dihubungkan melalui seam yang sama dengan evaluator sintetis, so that test algoritma tidak perlu melatih CNN.
52. As a peneliti, I want smoke budget berisi 2 ant, 2 iterasi, dan maksimal 2 epoch, so that pipeline dasar dapat diverifikasi sebelum eksperimen besar.
53. As a peneliti, I want pilot budget berisi 10 ant, 10 iterasi, dan maksimal 10 epoch, so that konvergensi awal dan runtime dapat diperiksa.
54. As a peneliti, I want main experiment paper-conventional dan improved menggunakan 20 ant, 20 iterasi, maksimal 10 epoch, dan seed 42, 43, 44, so that perbandingan utama memiliki budget yang sepadan.
55. As a peneliti, I want paper-literal diperlakukan sebagai diagnostic experiment, so that biaya komputasinya tidak menghambat primary comparison.
56. As a peneliti, I want final model baru dilatih pada 60.000 training-development images selama best epoch, so that model yang diuji tidak sekadar checkpoint candidate selama search.
57. As a peneliti, I want final model dievaluasi satu kali pada 10.000 test images, so that test accuracy tidak bocor ke proses tuning.
58. As a peneliti, I want hasil setiap seed dilaporkan secara individual, so that variasi stochastic tidak disembunyikan oleh satu angka agregat.
59. As a peneliti, I want aggregate configuration dipilih menggunakan mean validation accuracy pada seed yang sama, so that konfigurasi dari seed berbeda mendapat perbandingan yang adil.
60. As a peneliti, I want test accuracy tidak pernah digunakan untuk memilih aggregate configuration, so that status test set sebagai evaluasi independen terjaga.
61. As a peneliti, I want laporan menyatakan bahwa paper-conventional versus improved adalah pipeline comparison, so that perbedaan search space tidak ditafsirkan sebagai ablation update rule murni.
62. As a peneliti, I want metadata versi Python, TensorFlow/Keras, dan NumPy disimpan, so that eksperimen dapat direproduksi pada lingkungan yang sesuai.

## Implementation Decisions

- Sistem menggunakan domain vocabulary `ant`, `candidate`, `search space`, `pheromone`, `selection probability`, `fitness`, `iteration`, `trial`, dan `final model` sebagaimana didefinisikan dalam domain context.
- Satu fixed CNN architecture digunakan di semua mode: input grayscale 28×28×1, dua convolutional block, dropout feature-map, pooling, flatten, batch normalization, dua dense layer yang ukurannya dapat dicari, dropout dense, dan output softmax 10 kelas.
- ACO optimizer dipisahkan dari candidate evaluator melalui satu interface dependency-injection. Optimizer bertanggung jawab atas sampling, pheromone lifecycle, ranking, cache coordination, dan logging; evaluator bertanggung jawab atas evaluasi candidate dan metadata training.
- Evaluator sintetis menjadi adapter untuk smoke test. Evaluator CNN/MNIST menjadi adapter untuk eksperimen nyata. Keduanya mengembalikan hasil trial dengan status, fitness, metrik, epoch, waktu, dan failure metadata.
- Search space paper-faithful mempertahankan delapan hyperparameter jurnal dan delapan option per hyperparameter. Label artikel `linier` dipetakan ke identifier `linear`, tetapi provenance label tetap disimpan.
- Search space improved menggunakan empat pilihan dense units pertama dan kedua, tiga pilihan masing-masing dropout, tiga batch size, dua activation, dua optimizer, dan tiga learning rate. Loss improved tetap sparse categorical crossentropy.
- Pheromone disimpan sebagai vector per hyperparameter. Option yang tidak digunakan pada row padding tidak ikut probability atau update.
- Pheromone initial value adalah 1.0 dan merupakan assumption implementation karena tidak ditentukan artikel. Minimum pheromone default adalah 1e-12 dan dapat dikonfigurasi.
- Selection probability adalah pheromone yang dinormalisasi pada setiap hyperparameter. Probability harus tetap valid setelah initialization dan setiap update.
- `paper_literal` melakukan sampling, evaluasi, evaporation, dan reinforcement untuk setiap ant secara berurutan. Ant berikutnya menggunakan pheromone setelah update ant sebelumnya.
- `paper_conventional` melakukan sampling seluruh ant sebelum update, mengevaluasi seluruh ant, melakukan satu evaporation per iterasi, lalu memperkuat option setiap ant valid dengan nilai konstan 0.5.
- `improved` melakukan sampling dan evaluasi seluruh ant, memilih iteration-best, melakukan satu evaporation, lalu menambahkan reinforcement Q dikali validation accuracy hanya pada option iteration-best. Q bernilai 1.0 dan rho bernilai 0.25.
- Candidate failed tidak memiliki fitness dan tidak menerima reinforcement. Evaporation mode tetap berjalan sesuai lifecycle apabila seluruh ant pada iterasi gagal. Run tanpa successful atau cached valid candidate berakhir failed dan tidak melakukan final retraining.
- Cache hit diperlakukan sebagai hasil evaluasi valid untuk ranking dan pheromone update, tetapi tidak melakukan training ulang. Cache dibatasi oleh kondisi stochastic dan training policy yang relevan.
- Early stopping memonitor validation accuracy dengan mode maksimum, patience 2, min delta nol, dan restore best weights. Best epoch menggunakan validation loss sebagai tie-break jika validation accuracy sama.
- Paper-faithful mode menggunakan learning rate default optimizer framework dan hanya mencatat effective learning rate. Improved mode mencari learning rate secara eksplisit.
- Dataset split dibuat satu kali secara stratified dengan seed 2024: 50.000 training, 10.000 validation, dan 10.000 test resmi yang untouched selama tuning.
- Final retraining menggunakan model baru, seluruh 60.000 training-development images, jumlah epoch dari best candidate, tanpa early stopping, kemudian satu evaluasi pada test set.
- Primary statistical comparison hanya mencakup paper-conventional dan improved, masing-masing tiga seed dan budget 20 ant × 20 iterasi. Paper-literal digunakan untuk smoke, pilot, dan diagnostic run kecuali sumber daya memungkinkan eksperimen tambahan.
- Logging menyediakan satu record per trial dan record long-format per snapshot pheromone/probability. Metadata minimal mencakup mode, family, seed, iteration, ant ID, candidate ID, hyperparameters, fitness, metrik training, best epoch, waktu, status, failure reason, semantic warning, cache hit, dan best flags.
- CLI menyediakan pemilihan mode, jumlah ant, jumlah iterasi, maksimum epoch, seed, cache, synthetic evaluator, dan output location agar budget dapat dibedakan antara smoke, pilot, dan main experiment.
- Implementasi harus mempertahankan perbedaan antara fakta artikel, interpretasi pseudocode, dan pengembangan proyek dalam dokumentasi serta metadata eksperimen.

## Testing Decisions

- Test berfokus pada external behavior di seam optimizer-evaluator, bukan pada detail private implementation atau susunan internal array yang tidak terlihat pengguna.
- Test utama menggunakan evaluator sintetis agar cepat, deterministik, dan tidak membutuhkan TensorFlow atau dataset MNIST. Evaluator CNN diuji terpisah sebagai adapter integration boundary.
- Probability behavior: setiap hyperparameter yang memiliki option harus menghasilkan probability yang jumlahnya satu, bernilai positif, dan tetap valid setelah evaporation/reinforcement.
- Candidate behavior: setiap ant harus menghasilkan satu nilai untuk setiap hyperparameter dan candidate ID harus stabil untuk konfigurasi canonical yang sama.
- Mode behavior: `paper_literal` harus memperlihatkan update snapshot per ant; `paper_conventional` harus memperlihatkan satu evaporation dan satu snapshot update per iterasi; `improved` hanya memperkuat iteration-best.
- Failure behavior: failed trial tidak dapat menjadi best dan tidak menghasilkan reinforcement; seluruh iterasi gagal tetap dapat melakukan evaporation; run tanpa hasil valid berakhir failed.
- Cache behavior: cache hit harus mengembalikan fitness valid tanpa memanggil evaluator ulang, tetap dapat dipakai dalam ranking, dan tetap mengikuti aturan reinforcement mode aktif.
- Ranking behavior: tie-breaking harus mengikuti validation accuracy, validation loss, training time, lalu ant ID.
- Logging behavior: trial log dan pheromone history harus memuat status, update step, phase, dan nilai yang cukup untuk merekonstruksi perubahan pheromone.
- Seed behavior: konfigurasi yang sama dengan mode dan run seed sama harus menghasilkan trial seed sama; perubahan mode atau run seed harus menghasilkan trial seed berbeda; iteration dan ant ID tidak boleh memengaruhi trial seed.
- Configuration behavior: search space paper-faithful memiliki delapan dimensi dengan delapan pilihan; improved memiliki 5.184 kombinasi; mode dan parameter budget yang tidak valid ditolak secara eksplisit.
- Dataset/evaluator integration test memverifikasi bentuk input, split sizes, preprocessing normalization, fixed CNN output 10 kelas, dan final evaluation yang tidak dipanggil selama candidate search.
- Final training integration test memverifikasi model baru menggunakan training-development data, best epoch, dan test evaluation satu kali.
- Existing prior art adalah smoke test mekanisme ACO yang sudah ada di workspace. Test tersebut boleh diperluas atau dirapikan, tetapi tidak boleh mengubah test menjadi verifikasi implementation detail yang tidak berdampak pada perilaku eksternal.

## Out of Scope

- Reproduksi kode asli penulis jurnal secara eksak, karena detail implementasi artikel tidak tersedia.
- Klaim bahwa fixed CNN architecture, initial pheromone 1.0, dataset split 50k/10k, epoch policy, dan stopping policy berasal dari jurnal.
- Full neural architecture search untuk jumlah convolutional block, filter count, kernel size, pooling strategy, atau topology CNN.
- Dataset gambar selain MNIST pada fase awal.
- Penggunaan test accuracy sebagai fitness, signal pheromone, tie-break, atau pemilih konfigurasi.
- Adapter otomatis yang mengubah output layer atau label encoding untuk setiap loss paper-faithful.
- Controlled ablation murni yang hanya membedakan pheromone update tetapi menggunakan search space identik antara paper-conventional dan improved.
- Eksperimen utama paper-literal tiga seed sebagai kewajiban; mode ini bersifat diagnostic kecuali sumber daya tersedia.
- Optimasi distributed training, multi-GPU orchestration, deployment model, serving API, dan user interface.
- Menetapkan hasil accuracy, runtime, atau convergence sebelum CNN/MNIST benar-benar dijalankan pada environment yang memiliki TensorFlow.
- Menghapus atau menganggap final implementasi yang sudah telanjur ada sebagai bukti bahwa seluruh spec telah tervalidasi.

## Further Notes

- Perbandingan paper-conventional dan improved adalah pipeline comparison. Search space paper-conventional memiliki 16.777.216 kemungkinan, sedangkan improved memiliki 5.184 dan tambahan learning rate; perbedaan hasil tidak boleh ditafsirkan sebagai dampak update pheromone saja.
- Eksperimen dijalankan bertahap: smoke, pilot, main, lalu extended opsional. Hasil smoke tidak boleh dipakai sebagai hasil ilmiah.
- Jika TensorFlow belum tersedia, smoke test optimizer harus tetap dapat berjalan dengan evaluator sintetis. Eksperimen CNN nyata harus memberikan pesan dependency yang jelas, bukan gagal pada import yang tidak informatif.
- Artefak eksperimen harus menyimpan environment metadata dan konfigurasi lengkap agar hasil dapat diaudit.
- Spec ini menjadi sumber kerja untuk pemecahan tiket berikutnya. Implementasi yang sudah ada dianggap draft yang harus diverifikasi terhadap spec, bukan alasan untuk melewati ticketing atau testing.
