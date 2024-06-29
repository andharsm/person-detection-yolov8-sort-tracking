import streamlit as st

def display_overview():
  # data akusisi
  st.write("""
  ## Data Acquisition
  Dataset yang digunakan yaitu video footage kerumunan pejalan kaki di tempat umum, video tersebut didapatkan dari platform YouTube [Shopping, People, Commerce, Mall, Many, Crowd, Walking Free Stock video footage YouTube.](https://www.youtube.com/watch?v=WvhYuDvH17I)

  Dataset berdurasi 13.6s.
  """)
  
  st.image('person_detection_yolov8\deploy\images\youtube_dataset.jpg', caption='Person Detection in Public Area Footage')

  # ekstraksi image
  st.write("""
  ## Ekstraksi Image
  Dataset video diekstrak menjadi 5 frame/s, dari 13.6s didapatkan 69 frame. Proses ekstraksi frame dilakukan menggunakan Roboflow.
  """)

  st.image('person_detection_yolov8\deploy\images\ekstraksi_frame.jpg', caption='Ekstraksi Image')

  # image labeling
  st.write("""
  ## Image Labeling
  Proses pelabelan dilakukan secara manual satu per satu terhadap objek person (orang) pada setiap frame, proses ini dilakukan diroboflow.

  Dataset yang sudah diberi label dapat digunakan secara publik dilink berikut: [Person Detection Roboflow](https://universe.roboflow.com/object-detection-b0jxc/people-detection-jhker)
  """)

  st.image('person_detection_yolov8\deploy\images\dataset_roboflow.jpg', caption='Dataset in Roboflow')

  # modeling
  st.write("""
  ## Modeling
  Proses modeling dilakukan di Google Colab, model yang digunakan dalam objek deteksi ini adalah model YOLOv8.
  """)

  # evaluasi
  st.write('## Evaluasi')

  st.image('person_detection_yolov8\\deploy\\images\\result_train.png', caption='Hasil Pelatihan')
  st.write("""
  Secara keseluruhan, terlihat bahwa selama pelatihan, nilai Box Loss, Class Loss, dan DFL Loss terus menurun, menunjukkan peningkatan performa model. Precision, Recall, dan mAP juga mengalami peningkatan yang signifikan, terutama setelah Epoch 10, yang menandakan model semakin akurat dalam mendeteksi objek
  """)

  st.image('person_detection_yolov8\deploy\images\pr_curve.png', caption='Kurva PR')
  st.write("""
  Kurva Precision-Recall ini menunjukkan bahwa model YOLO yang dilatih sangat efektif dalam mendeteksi orang, dengan nilai precision dan recall yang sangat tinggi, yaitu 0.981, menghasilkan kinerja keseluruhan yang sangat baik.
  """)

  st.image('person_detection_yolov8\deploy\images\confusion_matrix.png', caption='Confusion Matrix')
  st.write("""
  Confusion matrix ini menunjukkan bahwa model YOLO yang dilatih sangat baik dalam mendeteksi orang, dengan jumlah prediksi benar (true positives) yang jauh lebih tinggi dibandingkan dengan jumlah kesalahan (false positives dan false negatives). Model ini mampu mendeteksi 379 dari 428 objek "people" dengan benar, hanya melakukan kesalahan kecil dalam klasifikasi
  """)

  # Demo Projek
  st.write('## Demo Projek')
  if st.button('Demo Projek'):
    st.session_state.menu_selection = "Demo Project"
    st.rerun()

