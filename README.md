\# Proyecto: \*Celebrity Look‑alije\* con FaceNet (facenet‑pytorch)



\*\*Objetivo:\*\* Dada una foto (tu rostro), devolver el \*\*Top‑k\*\* de celebridades del dataset que más se parecen, usando \*\*embeddings\*\* faciales (512‑D) de un modelo preentrenado.



\*\*Pipeline:\*\*

1\) Detección + alineación: `MTCNN` (facenet-pytorch)

2\) Embeddings: `InceptionResnetV1(pretrained="vggface2")`

3\) Clasificación ligera (fine‑tuning): una capa `Linear(512 → N)` \*\*o\*\* recuperación por similitud (cosine) con centroides.



> Nota: Los modelos preentrenados de facenet‑pytorch fueron entrenados con caras de \\\*\\\*160×160\\\*\\\*; rinden mejor si primero recortas la cara con MTCNN.

