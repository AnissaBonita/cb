import os
import gc
import re
import cv2
import numpy as np
import gradio as gr
import torch
import traceback
import math
import time
import ast
import argparse
import zipfile
from collections import defaultdict
from facexlib.utils.misc import download_from_url
from basicsr.utils.realesrganer import RealESRGANer
from utils.dataops import auto_split_upscale

input_images_limit = 350
# Define URLs and their corresponding local storage paths
face_models = {
      "GFPGANv1.4.pth": ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                           "https://github.com/TencentARC/GFPGAN/",
"""GFPGAN: Hacia la restauración de rostros ciegos en el mundo real y el escalado de imágenes con un prior facial generativo.
GFPGAN tiene como objetivo desarrollar un algoritmo práctico para la restauración de rostros en el mundo real.
Aprovecha información previa rica y diversa encapsulada en una GAN facial preentrenada (por ejemplo, StyleGAN2) para la restauración de rostros a ciegas."""],

   "RestoreFormer++.ckpt": ["https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt",
                           "https://github.com/wzhouxiff/RestoreFormerPlusPlus",
"""RestoreFormer++: Hacia la restauración facial ciega en el mundo real a partir de pares clave-valor no degradados.
RestoreFormer++ es una extensión de RestoreFormer. Propone restaurar una imagen facial degradada con fidelidad y
realismo mediante el uso de los potentes mecanismos de atención totalmente espaciales para modelar la abundante información contextual en el rostro y
su interacción con información previa de alta calidad orientada a la reconstrucción."""],

   "CodeFormer.pth": ["https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                           "https://github.com/sczhou/CodeFormer",
"""CodeFormer: Hacia una restauración robusta de rostros ciegos con Codebook Lookup Transformer (NeurIPS 2022).
CodeFormer es un modelo basado en Transformer diseñado para abordar el complejo problema de la restauración de rostros a ciegas, donde las entradas suelen estar gravemente degradadas.
Al plantear la restauración facial como una tarea de predicción de código, este enfoque garantiza tanto una mejor correspondencia entre las entradas degradadas y las salidas como la generación de rostros visualmente ricos y de alta calidad.
"""],

   "GPEN-BFR-512.pth": ["https://huggingface.co/akhaliq/GPEN-BFR-512/resolve/main/GPEN-BFR-512.pth",
                           "https://github.com/yangxy/GPEN",
"""GPEN: Red neuronal GAN ​​integrada para la restauración de rostros de personas ciegas en entornos reales.
GPEN aborda la restauración facial ciega (BFR, por sus siglas en inglés) mediante la integración de una GAN en una DNN en forma de U, combinando la capacidad de la GAN para generar imágenes de alta calidad con la extracción de características de la DNN.
Este diseño reconstruye la estructura global, los detalles finos y los fondos a partir de datos de entrada degradados.
Sencillo pero eficaz, GPEN supera a los métodos más avanzados, ofreciendo resultados realistas incluso para imágenes muy degradadas."""],

   "GPEN-BFR-1024.pt": ["https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model.pt",
                           "https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/files",
"""Igual que GPEN pero para una resolución de 1024."""],



   "GPEN-BFR-2048.pt": ["https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model-2048.pt",
                           "https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/files",
"""Igual que GPEN pero con resolución para 2048."""],



   # legacy model
   "GFPGANv1.3.pth": ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                         "https://github.com/TencentARC/GFPGAN/","Igual que GFPGAN pero modelo heredado"],
   "GFPGANv1.2.pth": ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
                         "https://github.com/TencentARC/GFPGAN/","Igual que GFPGAN pero modelo heredado"],
   "RestoreFormer.ckpt": ["https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer.ckpt",
                         "https://github.com/wzhouxiff/RestoreFormerPlusPlus","Igual que RestoreFormer++ pero con modelo heredado"],
}
upscale_models = {
   # SRVGGNet(Compact)
   "realesr-general-x4v3.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                               "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.3.0",
"""Eliminación de compresión, escalador general, JPEG, realista, investigación, restauración xinntao: añade realesr-general-x4v3 y realesr-general-wdn-x4v3. Son modelos muy pequeños para escenas generales y pueden ser más robustos. Pero, al ser modelos pequeños, su rendimiento puede ser limitado."""],

   "realesr-animevideov3.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                               "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0",
"""Anime, Dibujos animados, Eliminación de compresión, Escalador general, JPEG, Realista, Investigación, Restauración
xinntao: actualiza el modelo RealESRGAN AnimeVideo-v3, que puede lograr mejores resultados con una velocidad de inferencia más rápida."""],
  
   "4xLSDIRCompact.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact/4xLSDIRCompact.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact",
"""Realista
Phhofm: Amplía fotos pequeñas de buena calidad a 4 veces su tamaño. Este es mi primer modelo de escalado de fotos autodidacta."""],
   
   "4xLSDIRCompactC.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompactC/4xLSDIRCompactC.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompactC",
"""Eliminación de compresión, JPEG, Realista, Restauración
Phhofm: Escalador de fotos 4x que maneja la compresión JPG. Estoy intentando extender mi modelo anterior para que pueda manejar la compresión (JPG 100-30) modificando manualmente el conjunto de datos de entrenamiento, ya que 4xLSDIRCompact no puede manejar la compresión. Usa esto en lugar de 4xLSDIRCompact si tu foto tiene compresión (como una imagen de la web)."""],
       
   "4xLSDIRCompactR.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompactC/4xLSDIRCompactR.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompactC",
"""Eliminación de compresión, realista, restauración Phhofm: Escalador de fotos 4x que maneja compresión JPG, ruido y ligeras degradaciones. Al extender mi último modelo 4xLSDIRCompact a Real-ESRGAN, es decir, entrenado con datos sintéticos para manejar más tipos de degradaciones, debería poder manejar compresión, ruido y ligeras desenfoques."""],

   "4xLSDIRCompactN.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactC3.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3",
"""Realista Phhofm: Mejora la calidad de las fotos de entrada hasta 4 veces su tamaño. El 4xLSDIRCompact original está un poco más entrenado, no puede manejar la degradación. Estoy lanzando la Serie 3 de mis modelos 4xLSDIRCompact. En general, mi sugerencia es que, si tienes imágenes de entrada de buena calidad, uses 4xLSDIRCompactN3; de lo contrario, prueba 4xLSDIRCompactC3, que podrá manejar la compresión JPG y un poco de desenfoque, o bien 4xLSDIRCompactCR3, que es una interpolación entre C3 y R3 para poder manejar un poco de ruido adicional."""],

   "4xLSDIRCompactC3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactC3.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3",
"""Extracción de compresión, JPEG, Realista, Restauración Phhofm: Amplía el tamaño de las fotos comprimidas hasta 4 veces su tamaño original. Admite compresión JPG (30-100).Estoy lanzando la Serie 3 de mis modelos 4xLSDIRCompact. En general, mi sugerencia es que, si tienes imágenes de entrada de buena calidad, uses 4xLSDIRCompactN3; de lo contrario, prueba 4xLSDIRCompactC3, que podrá manejar la compresión JPG y un poco de desenfoque, o bien 4xLSDIRCompactCR3, que es una interpolación entre C3 y R3 para poder manejar un poco de ruido adicional."""],

   "4xLSDIRCompactR3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactR3.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3",
"""Realista, Restauración Phhofm: Mejora la resolución de fotos (degradadas) a 4 veces su tamaño. Entrenado con datos sintéticos, diseñado para manejar un mayor grado de degradación. Estoy lanzando la Serie 3 de mis modelos 4xLSDIRCompact. En general, mi sugerencia es que, si tienes imágenes de entrada de buena calidad, uses 4xLSDIRCompactN3; de lo contrario, prueba 4xLSDIRCompactC3, que podrá manejar la compresión JPG y un poco de desenfoque, o bien 4xLSDIRCompactCR3, que es una interpolación entre C3 y R3 para poder manejar un poco de ruido adicional."""],

   "4xLSDIRCompactCR3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactCR3.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3",
"""Phhofm: Estoy lanzando la Serie 3 de mis modelos 4xLSDIRCompact. En general, mi sugerencia es que, si tienes imágenes de entrada de buena calidad, uses 4xLSDIRCompactN3; de lo contrario, prueba 4xLSDIRCompactC3, que podrá manejar la compresión JPG y un poco de desenfoque, o bien 4xLSDIRCompactCR3, que es una interpolación entre C3 y R3 para poder manejar un poco de ruido adicional."""],

   "2xParimgCompact.pth": ["https://github.com/Phhofm/models/releases/download/2xParimgCompact/2xParimgCompact.pth",
                               "https://github.com/Phhofm/models/releases/tag/2xParimgCompact",
"""Realista Phhofm: Un modelo compacto de escalado de fotos 2x basado en ImagePairs de Microsoft. Este fue uno de los primeros modelos que comencé a entrenar y ahora lo he terminado para su lanzamiento. Como se puede ver en los ejemplos, este modelo afectará a los colores."""],

   "1xExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xExposureCorrection_compact.pth",
                                        "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact",
"""Restauración Phhofm: Este modelo está pensado como un experimento para ver si Compact puede usarse para entrenar fotos y corregir la exposición usando las pérdidas de píxeles, perceptuales, de color y LDL. No hay pérdida de brillo. Aun así, parece funcionar."""],
  
   "1xUnderExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xUnderExposureCorrection_compact.pth",
                                             "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact",
"""Restauración Phhofm: Este modelo está pensado como un experimento para ver si Compact puede usarse para entrenar con imágenes subexpuestas y corregir la exposición utilizando las pérdidas de píxeles, perceptuales, de color y LDL. No hay pérdida de brillo. Aun así, parece funcionar."""],
  
   "1xOverExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xOverExposureCorrection_compact.pth",
                                            "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact",
"""Restauración Phhofm: Este modelo está pensado como un experimento para ver si Compact puede usarse para entrenar con imágenes sobreexpuestas y corregir la exposición utilizando las pérdidas de píxeles, perceptuales, de color y LDL. No hay pérdida de brillo. Aun así, parece funcionar."""],

   "2x-sudo-UltraCompact.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-sudo-UltraCompact.pth",
                               "https://openmodeldb.info/models/2x-sudo-UltraCompact",
"""Anime, Dibujos animados, Restauración sudo: Restauración de animación en tiempo real y realización de tareas como eliminación de desenfoque y artefactos de compresión. Mi primer intento de crear un modelo de escalado 2x en tiempo real, aplicando al mismo tiempo el aprendizaje del profesor al alumno. (Profesor: RealESRGANv2-animevideo-xsx2.pth)"""],

   "2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth": ["https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
                                                 "https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-SuperUltraCompact",
"""Anime, Eliminación de compresión, Restauración
la base de datos: Modelos Real-ESRGAN Compactos/Ultracompactos/SuperUltracompactos en tiempo real (2x) diseñados para escalar anime de 1080p a 4K. El objetivo de estos modelos es abordar los problemas de escalado, desenfoque, sobreenfoque y artefactos de compresión durante el escalado para ofrecer un resultado que parezca como si el anime se hubiera masterizado originalmente en resolución 4K."""],

   "2x_AnimeJaNai_HD_V3_UltraCompact.pth": ["https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
                                            "https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-UltraCompact",
"""Anime, Eliminación de compresión, Restauración la base de datos: Modelos Real-ESRGAN Compactos/Ultracompactos/SuperUltracompactos en tiempo real (2x) diseñados para escalar anime de 1080p a 4K. El objetivo de estos modelos es abordar los problemas de escalado, desenfoque, sobreenfoque y artefactos de compresión durante el escalado para ofrecer un resultado que parezca como si el anime se hubiera masterizado originalmente en resolución 4K."""],

   "2x_AnimeJaNai_HD_V3_Compact.pth": ["https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
                                                 "https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-Compact",
"""Anime, Eliminación de compresión, Restauración
la base de datos: Modelos Real-ESRGAN Compactos/Ultracompactos/SuperUltracompactos en tiempo real (2x) diseñados para escalar anime de 1080p a 4K. El objetivo de estos modelos es abordar los problemas de escalado, desenfoque, sobreenfoque y artefactos de compresión durante el escalado para ofrecer un resultado que parezca como si el anime se hubiera masterizado originalmente en resolución 4K."""],

   #RRDBNet
   "RealESRGAN_x4plus_anime_6B.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                                     "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4",
"""Anime, Dibujos animados, Eliminación de compresión, Escalador general, JPEG, Realista, Investigación, Restauración
xinntao: Añadimos RealESRGAN_x4plus_anime_6B.pth, que está optimizado para imágenes de anime con un tamaño de modelo mucho menor. Más detalles y comparaciones con waifu2x se encuentran en anime_model.md"""],

   "RealESRGAN_x2plus.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                                     "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1",
""""Eliminación de compresión, escalador general, JPEG, realista, investigación, restauración"
xinntao: Añadir modelo RealESRGAN_x2plus.pth"""],

   "RealESRNet_x4plus.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
                                     "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.1",
"""Eliminación de compresión, escalador general, JPEG, realista, investigación, restauración"
xinntao: Esta versión es principalmente para almacenar modelos preentrenados y archivos ejecutables."""],

   "RealESRGAN_x4plus.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                                     "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0",
"""Eliminación de compresión, escalador general, JPEG, realista, investigación, restauración"
xinntao: Esta versión es principalmente para almacenar modelos preentrenados y archivos ejecutables."""],

   # ESRGAN(oldRRDB)
   "4x-AnimeSharp.pth": ["https://huggingface.co/utnah/esrgan/resolve/main/4x-AnimeSharp.pth?download=true",
                        "https://openmodeldb.info/models/4x-AnimeSharp",
"""Anime, Dibujos animados, Texto Kim2091: Interpolación entre 4x-UltraSharp y 4x-TextSharp-v0.5. Funciona de maravilla con anime. También mejora el tamaño del texto, pero es mucho mejor con contenido de anime."""],

   "4x_IllustrationJaNai_V1_ESRGAN_135k.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP",
                                              "https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2",
"""Anime, Dibujos animados, Eliminación de compresión, Eliminación de medios tonos, Escalador general, JPEG, Manga, Restauración
la base de datos: Modelo para imágenes en color, incluyendo portadas de manga e ilustraciones a color, arte digital, arte de novelas visuales, libros de arte y más.
La versión DAT2 es la de mayor calidad, pero también la más lenta. Consulte la versión ESRGAN para un rendimiento más rápido."""],

   "2x-sudo-RealESRGAN.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-sudo-RealESRGAN.pth",
                              "https://openmodeldb.info/models/2x-sudo-RealESRGAN",
"""Anime, Dibujos animados sudo: Intenté crear el mejor modelo 2x posible para dibujos. Creo que lo logré.
Y sí, son casi 3,8 millones de iteraciones (probablemente un récord que nadie superará aquí), y me llevó casi medio año entrenarlo.
Puede ocurrir que en un borde haya un patrón irregular. Para ello, puedes usar relleno o recorte. Mi objetivo era lograr una buena calidad perceptiva sin ampliar las imágenes al 400%. Dado que RealESRGAN tiene un factor de ampliación de 4x, reduje la escala de estas imágenes con una función bicúbica. Preentrenado: Pretrained_Model_G: RealESRGAN_x4plus_anime_6B.pth / RealESRGAN_x4plus_anime_6B.pth (sudo_RealESRGAN2x_3.332.758_G.pth)"""],
  
   "2x-sudo-RealESRGAN-Dropout.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-sudo-RealESRGAN-Dropout.pth",
                              "https://openmodeldb.info/models/2x-sudo-RealESRGAN-Dropout",
"""Anime, Dibujos animados
sudo: Intenté crear el mejor modelo 2x posible para dibujos. Creo que lo logré. Y sí, son casi 3,8 millones de iteraciones (probablemente un récord que nadie superará aquí), y me llevó casi medio año entrenarlo. Puede ocurrir que en un borde haya un patrón irregular. Para ello, puedes usar relleno o recorte. Mi objetivo era lograr una buena calidad perceptiva sin ampliar las imágenes al 400%. Dado que RealESRGAN tiene un factor de ampliación de 4x, reduje la escala de estas imágenes con una función bicúbica. Preentrenado: Pretrained_Model_G: RealESRGAN_x4plus_anime_6B.pth / RealESRGAN_x4plus_anime_6B.pth (sudo_RealESRGAN2x_3.332.758_G.pth)"""],

   "4xNomos2_otf_esrgan.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos2_otf_esrgan/4xNomos2_otf_esrgan.pth",
                              "https://github.com/Phhofm/models/releases/tag/4xNomos2_otf_esrgan",
"""Eliminación de compresión, JPEG, Realista, Restauración
Phhofm: Restauración, modelo ESRGAN 4x para fotografía, entrenado utilizando el pipeline de degradación Real-ESRGAN otf."""],

   "4xNomosWebPhoto_esrgan.pth": ["https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_esrgan/4xNomosWebPhoto_esrgan.pth",
                              "https://github.com/Phhofm/models/releases/tag/4xNomosWebPhoto_esrgan",
"""Realista, Restauración
Phhofm: Restauración, modelo ESRGAN 4x para fotografía, entrenado con ruido realista, desenfoque de lente, recompresión de jpg y webp.
Versión ESRGAN de 4xNomosWebPhoto_RealPLKSR, entrenada con el mismo conjunto de datos y de la misma manera."""],

   "4x_foolhardy_Remacri.pth": ["https://civitai.com/api/download/models/164821?type=Model&format=PickleTensor",
                              "https://openmodeldb.info/models/4x-Remacri",
"""Original
FoolhardyVEVO: Una creación de BSRGAN con más detalles y menos suavizado, hecha mediante la interpolación de modelos IRL como Siax,
Superscale, Superscale Artisoft, Pixel Perfect, etc. De esta forma, cosas como la piel y otros detalles no se vuelven borrosos ni difusos."""],

   "4x_foolhardy_Remacri_ExtraSmoother.pth": ["https://civitai.com/api/download/models/164822?type=Model&format=PickleTensor",
                              "https://openmodeldb.info/models/4x-Remacri",
"""ExtraSuavizante
FoolhardyVEVO: Una creación de BSRGAN con más detalles y menos suavizado, hecha mediante la interpolación de modelos IRL como Siax,
Superscale, Superscale Artisoft, Pixel Perfect, etc. De esta forma, cosas como la piel y otros detalles no se vuelven borrosos ni difusos."""],

   # DATNet
   "4xNomos8kDAT.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos8kDAT/4xNomos8kDAT.pth",
                                            "https://openmodeldb.info/models/4x-Nomos8kDAT",
"""Anime, Eliminación de compresión, Escalador general, JPEG, Realista, Restauración
Phhofm: Un escalador de fotos 4x con compresión JPG OTF, desenfoque y redimensionamiento, entrenado con el conjunto de datos Nomos8k_sfw de musl para SR realista, esta vez basado en la arquitectura DAT, como un ajuste fino del modelo DAT 4x oficial."""],

   "4x-DWTP-DS-dat2-v3.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-DWTP-DS-dat2-v3.pth",
                                            "https://openmodeldb.info/models/4x-DWTP-DS-dat2-v3",
"""Dehalftone, Restauración
umzi.x.dead: Modelo de despantallado DAT, diseñado para reducir las discrepancias en los mosaicos debido a la excesiva pérdida de la primera versión, al tiempo que elimina la eliminación de la textura del papel"""],

   "4xBHI_dat2_real.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_real/4xBHI_dat2_real.pth",
                                            "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real",
"""Eliminación de compresión, JPEG, realista
Phhofm: Modelo de escalado 4x dat2 para imágenes web y realistas. Maneja ruido realista, cierto desenfoque realista y (re)compresión webp y jpg. Entrenado con mi conjunto de datos BHI (390.035 mosaicos de entrenamiento) con un subconjunto LR degradado."""],

   "4xBHI_dat2_otf.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_otf/4xBHI_dat2_otf.pth",
                                            "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_otf",
"""Eliminación de compresión, JPEG
Phhofm: Modelo de escalado de datos 4x2, entrenado con la canalización real-esrgan otf en mi conjunto de datos bhi. Maneja el ruido y la compresión."""],

   "4xBHI_dat2_multiblur.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_multiblurjpg/4xBHI_dat2_multiblur.pth",
                                            "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_multiblurjpg",
"""Phhofm: el punto de control 4xBHI_dat2_multiblur (entrenado con 250000 iteraciones), que no puede manejar la compresión pero podría ofrecer una salida ligeramente mejor en una entrada no degradada."""],

   "4xBHI_dat2_multiblurjpg.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_multiblurjpg/4xBHI_dat2_multiblurjpg.pth",
                                            "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_multiblurjpg",
"""Eliminación de compresión, JPEG
Phhofm: Modelo de escalado de datos 4x2, entrenado con algoritmos de escalado down_up, linear, cubic_mitchell, lanczos, gauss y box, algunos desenfoques promedio, gaussiano y anisotrópico y compresión jpg. Entrenado en mi conjunto de datos BHI sisr."""],

   "4x_IllustrationJaNai_V1_DAT2_190k.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP",
                                            "https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2",
"""Anime, Dibujos animados, Eliminación de compresión, Eliminación de medios tonos, Escalador general, JPEG, Manga, Restauración
la base de datos: Modelo para imágenes en color, incluyendo portadas de manga e ilustraciones a color, arte digital, arte de novelas visuales, libros de arte y más. La versión DAT2 es la de mayor calidad, pero también la más lenta. Consulte la versión ESRGAN para un rendimiento más rápido."""],

   "4x-PBRify_UpscalerDAT2_V1.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_UpscalerDAT2_V1/4x-PBRify_UpscalerDAT2_V1.pth",
                                     "https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_UpscalerDAT2_V1",
"""Eliminación de compresión, DDS, texturas de juegos, restauración
Kim2091: Otro modelo más de la serie PBRify_Remix. Se trata de un nuevo escalador que sustituye al modelo anterior 4x-PBRify_UpscalerSIR-M_V2.
Este modelo supera con creces la calidad del anterior, con una generación de detalles mucho más natural y una mejor reconstrucción de líneas y bordes."""],

   "4xBHI_dat2_otf_nn.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_otf_nn/4xBHI_dat2_otf_nn.pth",
                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_otf_nn",
"""Eliminación de compresión, JPEG
Phhofm: Modelo de escalado de datos 4x2, entrenado con el pipeline real-esrgan otf pero sin ruido, en mi conjunto de datos bhi. Maneja redimensionamientos y compresión jpg."""],

   # HAT
   "4xNomos8kSCHAT-L.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-L.pth",
                             "https://openmodeldb.info/models/4x-Nomos8kSCHAT-L",
"""Anime, Eliminación de compresión, Escalador general, JPEG, Realista, Restauración
Phhofm: Escalador de fotos 4x con compresión JPG OTF y desenfoque, entrenado con el conjunto de datos Nomos8k_sfw de musl para SR realista. Dado que es un modelo grande, el escalado podría tardar un tiempo."""],

   "4xNomos8kSCHAT-S.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-S.pth",
                             "https://openmodeldb.info/models/4x-Nomos8kSCHAT-S",
"""Anime, Eliminación de compresión, Escalador general, JPEG, Realista, Restauración
Phhofm: escalador de fotos 4x con compresión JPG OTF y desenfoque, entrenado con el conjunto de datos Nomos8k_sfw de musl para una versión/modelo HAT-S realista."""],

   "4xNomos8kHAT-L_otf.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos8kHAT-L_otf/4xNomos8kHAT-L_otf.pth",
                             "https://openmodeldb.info/models/4x-Nomos8kHAT-L-otf",
"""Rostros, Escalador general, Realista, Restauración
Phhofm: escalador de fotos 4x entrenado con otf, maneja cierta compresión jpg, algo de desenfoque y algo de ruido."""],

   "4xBHI_small_hat-l.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_small_hat-l/4xBHI_small_hat-l.pth",
                             "https://github.com/Phhofm/models/releases/tag/4xBHI_small_hat-l",
"""Phhofm: Modelo de escalado 4x hat-l para entrada de buena calidad. Este modelo no maneja ninguna degradación.
Este modelo es bastante suave; intenté equilibrar la nitidez y la fidelidad/ausencia de artefactos.
Para obtener una salida un poco más nítida, pero que puede generar algunos artefactos, puede probar la versión 4xBHI_small_hat-l_sharp.
También se incluye en este lanzamiento, que aún podría parecer suave si estás acostumbrado a resultados más nítidos."""],

   # RealPLKSR_dysample
   "4xHFA2k_ludvae_realplksr_dysample.pth": ["https://github.com/Phhofm/models/releases/download/4xHFA2k_ludvae_realplksr_dysample/4xHFA2k_ludvae_realplksr_dysample.pth",
                                            "https://openmodeldb.info/models/4x-HFA2k-ludvae-realplksr-dysample",
"""Anime, Eliminación de compresión, Restauración
Phhofm: Un modelo de escalado Dysample RealPLKSR 4x para la resolución de imágenes individuales de anime."""],

   "4xArtFaces_realplksr_dysample.pth": ["https://github.com/Phhofm/models/releases/download/4xArtFaces_realplksr_dysample/4xArtFaces_realplksr_dysample.pth",
                                            "https://openmodeldb.info/models/4x-ArtFaces-realplksr-dysample",
"""ArtFaces
Phhofm: Un modelo de escalado 4x de Dysample RealPLKSR para arte / rostros pintados."""],

   "4x-PBRify_RPLKSRd_V3.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_RPLKSRd_V3/4x-PBRify_RPLKSRd_V3.pth",
                                            "https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_RPLKSRd_V3",
"""Eliminación de compresión, DDS, eliminación de bandas, eliminación de tramado, eliminación de halo, texturas de juegos, restauración
Kim2091: Esta actualización incorpora un nuevo modelo de escalado, 4x-PBRify_RPLKSRd_V3. Este modelo es aproximadamente 8 veces más rápido que el modelo DAT2 actual, a la vez que ofrece mayor calidad.
Produce detalles mucho más naturales, define las líneas y los bordes con mayor suavidad y elimina mejor los artefactos de compresión.
Como resultado de esas mejoras, el PBR también ha mejorado mucho. Tiende a ser más nítido y con menos artefactos definidos."""],

   "4xNomos2_realplksr_dysample.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos2_realplksr_dysample/4xNomos2_realplksr_dysample.pth",
                                            "https://openmodeldb.info/models/4x-Nomos2-realplksr-dysample",
"""Eliminación de compresión, JPEG, Realista, Restauración
Phhofm: Un modelo de escalado ascendente Dysample RealPLKSR 4x que fue entrenado con / maneja la compresión jpg hasta 70 en el conjunto de datos Nomosv2, preserva la profundidad de campo.
Este modelo afecta/satura los colores, lo cual se puede contrarrestar un poco utilizando la corrección de color wavelet, como se usa en estos ejemplos."""],

   # RealPLKSR
   "2x-AnimeSharpV2_RPLKSR_Sharp.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Sharp.pth",
                                       "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set",
"""Anime, Eliminación de compresión, Restauración
Kim2091: Este es mi primer modelo de anime en años. Espero que puedan encontrarle una buena utilidad.
RealPLKSR (mayor calidad, más lento) Sharp: Para fuentes muy degradadas. Los modelos Sharp tienen problemas con la profundidad de campo, pero son los mejores para eliminar artefactos.
"""],

   "2x-AnimeSharpV2_RPLKSR_Soft.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Soft.pth",
                                        "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set",
"""Anime, Eliminación de compresión, Restauración
Kim2091: Este es mi primer modelo de anime en años. Espero que puedan encontrarle una buena utilidad.
RealPLKSR (mayor calidad, más lento) Suave: Para fuentes más limpias. Los modelos suaves conservan la profundidad de campo, pero es posible que no eliminen otros artefactos tan bien."""],

   "4xPurePhoto-RealPLSKR.pth": ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-RealPLSKR.pth",
                                       "https://openmodeldb.info/models/4x-PurePhoto-RealPLSKR",
"""Generado por IA, Eliminación de compresión, JPEG, Realista, Restauración asterixcool: Experto en trabajar con gatos, cabello, fiestas y en crear imágenes nítidas. También tiene experiencia en redimensionar fotos y ampliar imágenes grandes y nítidas. También puede mejorar eficazmente imágenes de tamaño pequeño (hasta 300 píxeles en un lado, dependiendo del sujeto).
Con experiencia en experimentar con técnicas como el escalado ascendente con este modelo dos veces y luego reduciéndolo en un 50% para realzar los detalles, especialmente en rasgos como el cabello o los animales."""],

   "2x_Text2HD_v.1-RealPLKSR.pth": ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2x_Text2HD_v.1-RealPLKSR.pth",
                                       "https://openmodeldb.info/models/2x-Text2HD-v-1",
"""Eliminación de compresión, reducción de ruido, escalado general, JPEG, restauración, texto asterixcool: El modelo de alta gama está diseñado específicamente para mejorar las imágenes de texto de menor calidad, Mejorando su claridad y legibilidad al aumentar su tamaño al doble. Destaca por procesar textos de tamaño moderado, transformándolos eficazmente en escaneos legibles y de alta calidad. Sin embargo, el modelo puede encontrar dificultades al tratar con texto muy pequeño, ya que su rendimiento está optimizado para texto de un tamaño mínimo determinado. Para obtener los mejores resultados, Las imágenes de entrada deben contener texto que no sea excesivamente pequeño."""],

   "2xVHS2HD-RealPLKSR.pth": ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2xVHS2HD-RealPLKSR.pth",
                                       "https://openmodeldb.info/models/2x-VHS2HD",
"""Eliminación de compresión, eliminación de halo, realista, restauración, fotograma de vídeo
asterixcool: Un modelo avanzado de grabación VHS diseñado para mejorar la calidad del vídeo reduciendo artefactos como halos, imágenes fantasma y patrones de ruido. Optimizado principalmente para resolución PAL (NTSC también podría funcionar bien)."""],

   "4xNomosWebPhoto_RealPLKSR.pth": ["https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth",
                                       "https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR",
"""Realista, Restauración
Phhofm: Modelo 4x RealPLKSR para fotografía, entrenado con ruido realista, desenfoque de lente, recompresión de jpg y webp."""],

   # DRCT
   "4xNomos2_hq_drct-l.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_drct-l/4xNomos2_hq_drct-l.pth",
                                       "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_drct-l",
"""Escalador general, realista
Phhofm: Un modelo de escalado ascendente drct-l 4x, similar a los modelos 4xNomos2_hq_atd, 4xNomos2_hq_dat2 y 4xNomos2_hq_mosr, entrenado para su uso con datos de entrada no degradados para proporcionar una salida de buena calidad."""],

   # ETC
   "4xNomos2_hq_atd.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_atd/4xNomos2_hq_atd.pth",
                                        "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_atd",
"""Escalador general, realista
Phhofm: Un modelo de escalado ascendente atd 4x, similar a los modelos 4xNomos2_hq_dat2 o 4xNomos2_hq_mosr, entrenado para su uso con datos de entrada no degradados para proporcionar una salida de buena calidad."""],

   #MoSR
   "4xNomos2_hq_mosr.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_mosr/4xNomos2_hq_mosr.pth",
                                        "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_mosr",
"""Escalador general, realista
Phhofm: Un modelo de escalado MoSR 4x, diseñado para entradas no degradadas, ya que este modelo fue entrenado con entradas no degradadas para proporcionar una salida de buena calidad."""],
  
   "2x-AnimeSharpV2_MoSR_Sharp.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_MoSR_Sharp.pth",
                                        "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set",
"""Anime, Eliminación de compresión, Restauración
Kim2091: Este es mi primer modelo de anime en años. Espero que puedan encontrarle una buena utilidad. MoSR (menor calidad, más rápido), Sharp: Para fuentes muy degradadas. Los modelos Sharp tienen problemas con la profundidad de campo, pero son los mejores para eliminar artefactos."""],
  
   "2x-AnimeSharpV2_MoSR_Soft.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_MoSR_Soft.pth",
                                        "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set",
"""Anime, Eliminación de compresión, Restauración
Kim2091: Este es mi primer modelo de anime en años. Espero que puedan encontrarle una buena utilidad. MoSR (menor calidad, más rápido), Soft: para fuentes más limpias. Los modelos Soft conservan la profundidad de campo, pero es posible que no eliminen otros artefactos."""],

   #SRFormer
   "4xNomos8kSCSRFormer.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCSRFormer/4xNomos8kSCSRFormer.pth",
                                            "https://github.com/Phhofm/models/releases/tag/4xNomos8kSCSRFormer",
"""Anime, Eliminación de compresión, Escalador general, JPEG, Realista, Restauración Phhofm: escalador de fotos 4x con compresión JPG OTF y desenfoque, entrenado con el conjunto de datos Nomos8k_sfw de musl para SR realista."""],

   "4xFrankendataFullDegradation_SRFormer460K_g.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1PZrj-8ofxhORv_OgTVSoRt3dYi-BtiDj",
                                                   "https://openmodeldb.info/models/4x-Frankendata-FullDegradation-SRFormer",
"""Eliminación de compresión, reducción de ruido, realismo, restauración Crustáceo D: Escalador realista 4x que también puede funcionar para uso general. Se entrenó con degradación aleatoria OTF con un rango de degradación que abarca desde muy bajo hasta muy alto, incluyendo desenfoque, ruido y compresión.Entrenado con el mismo conjunto de datos Frankendata que utilicé para el modelo de preentrenamiento."""],

   "4xFrankendataPretrainer_SRFormer400K_g.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1SaKvpYYIm2Vj2m9GifUMlNCbmkE6JZmr",
                                                   "https://openmodeldb.info/models/4x-FrankendataPretainer-SRFormer",
"""Realista, Restauración Crustáceo D: Escalador realista 4x que también puede funcionar para uso general. Se entrenó con degradación aleatoria OTF con un rango de degradación que abarca desde muy bajo hasta muy alto, incluyendo desenfoque, ruido y compresión. Entrenado con el mismo conjunto de datos Frankendata que utilicé para el modelo de preentrenamiento."""],

   "1xFrankenfixer_SRFormerLight_g.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1UJ0iyFn4IGNhPIgNgrQrBxYsdDloFc9I",
                                                 "https://openmodeldb.info/models/1x-Frankenfixer-SRFormerLight",
"""Realista, Restauración
Crustáceo D: Un modelo 1x diseñado para reducir artefactos y restaurar detalles en imágenes escaladas con 4xFrankendata_FullDegradation_SRFormer. Posiblemente también funcione con otros modelos de escalado."""],
}

example_list = ["images/a01.jpg", "images/a02.jpg", "images/a03.jpg", "images/a04.jpg", "images/bus.jpg", "images/zidane.jpg", 
                "images/b01.jpg", "images/b02.jpg", "images/b03.jpg", "images/b04.jpg", "images/b05.jpg", "images/b06.jpg", 
                "images/b07.jpg", "images/b08.jpg", "images/b09.jpg", "images/b10.jpg", "images/b11.jpg", "images/c01.jpg",  
                "images/c02.jpg", "images/c03.jpg", "images/c04.jpg", "images/c05.jpg", "images/c06.jpg", "images/c07.jpg", 
                "images/c08.jpg", "images/c09.jpg", "images/c10.jpg"]

def get_model_type(model_name):
    # Define model type mappings based on key parts of the model names
    model_type = "other"
    if any(value in model_name.lower() for value in ("4x-animesharp.pth", "sudo-realesrgan", "remacri")):
        model_type = "ESRGAN"
    elif "srformer" in model_name.lower():
        model_type = "SRFormer"
    elif ("realplksr" in model_name.lower() and "dysample" in model_name.lower()) or "rplksrd" in model_name.lower():
        model_type = "RealPLKSR_dysample"
    elif any(value in model_name.lower() for value in ("realplksr", "rplksr", "realplskr")):
        model_type = "RealPLKSR"
    elif any(value in model_name.lower() for value in ("realesrgan", "realesrnet")):
        model_type = "RRDB"
    elif any(value in model_name.lower() for value in ("realesr", "compact")):
        model_type = "SRVGG"
    elif "esrgan" in model_name.lower():
        model_type = "ESRGAN"
    elif "dat" in model_name.lower():
        model_type = "DAT"
    elif "hat" in model_name.lower():
        model_type = "HAT"
    elif "drct" in model_name.lower():
        model_type = "DRCT"
    elif "atd" in model_name.lower():
        model_type = "ATD"
    elif "mosr" in model_name.lower():
        model_type = "MoSR"
    return f"{model_type}, {model_name}"

typed_upscale_models = {get_model_type(key): value[0] for key, value in upscale_models.items()}


class Upscale:
    def __init__(self,):
        self.scale         = 4
        self.modelInUse    = ""
        self.realesrganer  = None
        self.face_enhancer = None

    def initBGUpscaleModel(self, upscale_model):
        upscale_type, upscale_model = upscale_model.split(", ", 1)
        download_from_url(upscale_models[upscale_model][0], upscale_model, os.path.join("weights", "upscale"))
        self.modelInUse = f"_{os.path.splitext(upscale_model)[0]}"
        netscale = 1 if any(sub in upscale_model.lower() for sub in ("x1", "1x")) else (2 if any(sub in upscale_model.lower() for sub in ("x2", "2x")) else 4)
        model = None
        half = True if torch.cuda.is_available() else False
        if upscale_type:
            # The values of the following hyperparameters are based on the research findings of the Spandrel project.
            # https://github.com/chaiNNer-org/spandrel/tree/main/libs/spandrel/spandrel/architectures
            from basicsr.archs.rrdbnet_arch import RRDBNet
            loadnet = torch.load(os.path.join("weights", "upscale", upscale_model), map_location=torch.device('cpu'), weights_only=True)
            if 'params_ema' in loadnet or 'params' in loadnet:
                loadnet = loadnet['params_ema'] if 'params_ema' in loadnet else loadnet['params']

            if upscale_type == "SRVGG":
                from basicsr.archs.srvgg_arch import SRVGGNetCompact
                body_max_num = self.find_max_numbers(loadnet, "body")
                num_feat     = loadnet["body.0.weight"].shape[0]
                num_in_ch    = loadnet["body.0.weight"].shape[1]
                num_conv     = body_max_num // 2 - 1
                model        = SRVGGNetCompact(num_in_ch=num_in_ch, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=netscale, act_type='prelu')
            elif upscale_type == "RRDB" or upscale_type == "ESRGAN":
                if upscale_type == "RRDB":
                    num_block = self.find_max_numbers(loadnet, "body") + 1
                    num_feat  = loadnet["conv_first.weight"].shape[0]
                else:
                    num_block = self.find_max_numbers(loadnet, "model.1.sub")
                    num_feat  = loadnet["model.0.weight"].shape[0]
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, num_grow_ch=32, scale=netscale, is_real_esrgan=upscale_type == "RRDB")
            elif upscale_type == "DAT":
                from basicsr.archs.dat_arch import DAT
                half = False

                in_chans   = loadnet["conv_first.weight"].shape[1]
                embed_dim  = loadnet["conv_first.weight"].shape[0]
                num_layers = self.find_max_numbers(loadnet, "layers") + 1
                depth      = [6] * num_layers
                num_heads  = [6] * num_layers
                for i in range(num_layers):
                    depth[i] = self.find_max_numbers(loadnet, f"layers.{i}.blocks") + 1
                    num_heads[i] = loadnet[f"layers.{i}.blocks.1.attn.temperature"].shape[0] if depth[i] >= 2 else \
                                   loadnet[f"layers.{i}.blocks.0.attn.attns.0.pos.pos3.2.weight"].shape[0] * 2

                upsampler        = "pixelshuffle" if "conv_last.weight" in loadnet else "pixelshuffledirect"
                resi_connection  = "1conv" if "conv_after_body.weight" in loadnet else "3conv"
                qkv_bias         = "layers.0.blocks.0.attn.qkv.bias" in loadnet
                expansion_factor = float(loadnet["layers.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim)

                img_size = 64
                if "layers.0.blocks.2.attn.attn_mask_0" in loadnet:
                    attn_mask_0_x, attn_mask_0_y, _attn_mask_0_z = loadnet["layers.0.blocks.2.attn.attn_mask_0"].shape
                    img_size = int(math.sqrt(attn_mask_0_x * attn_mask_0_y))

                split_size = [2, 4]
                if "layers.0.blocks.0.attn.attns.0.rpe_biases" in loadnet:
                    split_sizes = loadnet["layers.0.blocks.0.attn.attns.0.rpe_biases"][-1] + 1
                    split_size = [int(x) for x in split_sizes]

                model = DAT(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, split_size=split_size, depth=depth, num_heads=num_heads, expansion_factor=expansion_factor, 
                            qkv_bias=qkv_bias, resi_connection=resi_connection, upsampler=upsampler, upscale=netscale)
            elif upscale_type == "HAT":
                half = False
                from basicsr.archs.hat_arch import HAT
                in_chans = loadnet["conv_first.weight"].shape[1]
                embed_dim = loadnet["conv_first.weight"].shape[0]
                window_size = int(math.sqrt(loadnet["relative_position_index_SA"].shape[0]))
                num_layers = self.find_max_numbers(loadnet, "layers") + 1
                depths      = [6] * num_layers
                num_heads   = [6] * num_layers
                for i in range(num_layers):
                    depths[i] = self.find_max_numbers(loadnet, f"layers.{i}.residual_group.blocks") + 1
                    num_heads[i] = loadnet[f"layers.{i}.residual_group.overlap_attn.relative_position_bias_table"].shape[1]
                resi_connection = "1conv" if "conv_after_body.weight" in loadnet else "identity"

                compress_ratio = self.find_divisor_for_quotient(embed_dim, loadnet["layers.0.residual_group.blocks.0.conv_block.cab.0.weight"].shape[0],)
                squeeze_factor = self.find_divisor_for_quotient(embed_dim, loadnet["layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight"].shape[0],)

                qkv_bias = "layers.0.residual_group.blocks.0.attn.qkv.bias" in loadnet
                patch_norm = "patch_embed.norm.weight" in loadnet
                ape = "absolute_pos_embed" in loadnet

                mlp_hidden_dim = int(loadnet["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0])
                mlp_ratio = mlp_hidden_dim / embed_dim
                upsampler = "pixelshuffle"

                model = HAT(img_size=64, patch_size=1, in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, compress_ratio=compress_ratio,
                            squeeze_factor=squeeze_factor, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape, patch_norm=patch_norm,
                            upsampler=upsampler, resi_connection=resi_connection, upscale=netscale,)
            elif "RealPLKSR" in upscale_type:
                from basicsr.archs.realplksr_arch import realplksr
                half = False if "RealPLSKR" in upscale_model else half
                use_ea       = "feats.1.attn.f.0.weight" in loadnet
                dim          = loadnet["feats.0.weight"].shape[0]
                num_feats    = self.find_max_numbers(loadnet, "feats") + 1
                n_blocks     = num_feats - 3
                kernel_size  = loadnet["feats.1.lk.conv.weight"].shape[2]
                split_ratio  = loadnet["feats.1.lk.conv.weight"].shape[0] / dim
                use_dysample = "to_img.init_pos" in loadnet

                model = realplksr(upscaling_factor=netscale, dim=dim, n_blocks=n_blocks, kernel_size=kernel_size, split_ratio=split_ratio, use_ea=use_ea, dysample=use_dysample)
            elif upscale_type == "DRCT":
                half = False
                from basicsr.archs.DRCT_arch import DRCT

                in_chans    = loadnet["conv_first.weight"].shape[1]
                embed_dim   = loadnet["conv_first.weight"].shape[0]
                num_layers  = self.find_max_numbers(loadnet, "layers") + 1
                depths      = (6,) * num_layers
                num_heads   = []
                for i in range(num_layers):
                    num_heads.append(loadnet[f"layers.{i}.swin1.attn.relative_position_bias_table"].shape[1])

                mlp_ratio       = loadnet["layers.0.swin1.mlp.fc1.weight"].shape[0] / embed_dim
                window_square   = loadnet["layers.0.swin1.attn.relative_position_bias_table"].shape[0]
                window_size     = (math.isqrt(window_square) + 1) // 2
                upsampler       = "pixelshuffle" if "conv_last.weight" in loadnet else ""
                resi_connection = "1conv" if "conv_after_body.weight" in loadnet else ""
                qkv_bias        = "layers.0.swin1.attn.qkv.bias" in loadnet
                gc_adjust1      = loadnet["layers.0.adjust1.weight"].shape[0]
                patch_norm      = "patch_embed.norm.weight" in loadnet
                ape             = "absolute_pos_embed" in loadnet

                model = DRCT(in_chans=in_chans,  img_size= 64, window_size=window_size, compress_ratio=3,squeeze_factor=30,
                    conv_scale= 0.01, overlap_ratio= 0.5, img_range= 1., depths=depths, embed_dim=embed_dim, num_heads=num_heads, 
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape, patch_norm=patch_norm, use_checkpoint=False,
                    upscale=netscale, upsampler=upsampler, resi_connection=resi_connection, gc =gc_adjust1,)
            elif upscale_type == "ATD":
                half = False
                from basicsr.archs.atd_arch import ATD
                in_chans    = loadnet["conv_first.weight"].shape[1]
                embed_dim   = loadnet["conv_first.weight"].shape[0]
                window_size = math.isqrt(loadnet["relative_position_index_SA"].shape[0])
                num_layers  = self.find_max_numbers(loadnet, "layers") + 1
                depths      = [6] * num_layers
                num_heads   = [6] * num_layers
                for i in range(num_layers):
                    depths[i] = self.find_max_numbers(loadnet, f"layers.{i}.residual_group.layers") + 1
                    num_heads[i] = loadnet[f"layers.{i}.residual_group.layers.0.attn_win.relative_position_bias_table"].shape[1]
                num_tokens          = loadnet["layers.0.residual_group.layers.0.attn_atd.scale"].shape[0]
                reducted_dim        = loadnet["layers.0.residual_group.layers.0.attn_atd.wq.weight"].shape[0]
                convffn_kernel_size = loadnet["layers.0.residual_group.layers.0.convffn.dwconv.depthwise_conv.0.weight"].shape[2]
                mlp_ratio           = (loadnet["layers.0.residual_group.layers.0.convffn.fc1.weight"].shape[0] / embed_dim)
                qkv_bias            = "layers.0.residual_group.layers.0.wqkv.bias" in loadnet
                ape                 = "absolute_pos_embed" in loadnet
                patch_norm          = "patch_embed.norm.weight" in loadnet
                resi_connection     = "1conv" if "layers.0.conv.weight" in loadnet else "3conv"

                if "conv_up1.weight" in loadnet:
                    upsampler = "nearest+conv"
                elif "conv_before_upsample.0.weight" in loadnet:
                    upsampler = "pixelshuffle"
                elif "conv_last.weight" in loadnet:
                    upsampler = ""
                else:
                    upsampler = "pixelshuffledirect"

                is_light = upsampler == "pixelshuffledirect" and embed_dim == 48
                category_size = 128 if is_light else 256

                model = ATD(in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, category_size=category_size,
                            num_tokens=num_tokens, reducted_dim=reducted_dim, convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape,
                            patch_norm=patch_norm, use_checkpoint=False, upscale=netscale, upsampler=upsampler, resi_connection='1conv',)
            elif upscale_type == "MoSR":
                from basicsr.archs.mosr_arch import mosr
                n_block         = self.find_max_numbers(loadnet, "gblocks") - 5
                in_ch           = loadnet["gblocks.0.weight"].shape[1]
                out_ch          = loadnet["upsampler.end_conv.weight"].shape[0] if "upsampler.init_pos" in loadnet else in_ch
                dim             = loadnet["gblocks.0.weight"].shape[0]
                expansion_ratio = (loadnet["gblocks.1.fc1.weight"].shape[0] / loadnet["gblocks.1.fc1.weight"].shape[1]) / 2
                conv_ratio      = loadnet["gblocks.1.conv.weight"].shape[0] / dim
                kernel_size     = loadnet["gblocks.1.conv.weight"].shape[2]
                upsampler       = "dys" if "upsampler.init_pos" in loadnet else ("gps" if "upsampler.in_to_k.weight" in loadnet else "ps")

                model = mosr(in_ch = in_ch, out_ch = out_ch, upscale = netscale, n_block = n_block, dim = dim,
                            upsampler = upsampler, kernel_size = kernel_size, expansion_ratio = expansion_ratio, conv_ratio = conv_ratio,)
            elif upscale_type == "SRFormer":
                half = False
                from basicsr.archs.srformer_arch import SRFormer
                in_chans   = loadnet["conv_first.weight"].shape[1]
                embed_dim  = loadnet["conv_first.weight"].shape[0]
                ape        = "absolute_pos_embed" in loadnet
                patch_norm = "patch_embed.norm.weight" in loadnet
                qkv_bias   = "layers.0.residual_group.blocks.0.attn.q.bias" in loadnet
                mlp_ratio  = float(loadnet["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0] / embed_dim)

                num_layers = self.find_max_numbers(loadnet, "layers") + 1
                depths     = [6] * num_layers
                num_heads  = [6] * num_layers
                for i in range(num_layers):
                    depths[i] = self.find_max_numbers(loadnet, f"layers.{i}.residual_group.blocks") + 1
                    num_heads[i] = loadnet[f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table"].shape[1]

                if "conv_hr.weight" in loadnet:
                    upsampler = "nearest+conv"
                elif "conv_before_upsample.0.weight" in loadnet:
                    upsampler = "pixelshuffle"
                elif "upsample.0.weight" in loadnet:
                    upsampler = "pixelshuffledirect"
                resi_connection = "1conv" if "conv_after_body.weight" in loadnet else "3conv"

                window_size = int(math.sqrt(loadnet["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].shape[0])) + 1

                if "layers.0.residual_group.blocks.1.attn_mask" in loadnet:
                    attn_mask_0 = loadnet["layers.0.residual_group.blocks.1.attn_mask"].shape[0]
                    patches_resolution = int(math.sqrt(attn_mask_0) * window_size)
                else:
                    patches_resolution = window_size
                    if ape:
                        pos_embed_value = loadnet.get("absolute_pos_embed", [None, None])[1]
                        if pos_embed_value:
                            patches_resolution = int(math.sqrt(pos_embed_value))

                img_size = patches_resolution
                if img_size % window_size != 0:
                    for nice_number in [512, 256, 128, 96, 64, 48, 32, 24, 16]:
                        if nice_number % window_size != 0:
                            nice_number += window_size - (nice_number % window_size)
                        if nice_number == patches_resolution:
                            img_size = nice_number
                            break

                model = SRFormer(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, 
                             qkv_bias=qkv_bias, qk_scale=None, ape=ape, patch_norm=patch_norm, upscale=netscale, upsampler=upsampler, resi_connection=resi_connection,)

        if model:
            self.realesrganer = RealESRGANer(scale=netscale, model_path=os.path.join("weights", "upscale", upscale_model), model=model, tile=0, tile_pad=10, pre_pad=0, half=half)
        elif upscale_model:
            import PIL
            from image_gen_aux import UpscaleWithModel
            class UpscaleWithModel_Gfpgan(UpscaleWithModel):
                def cv2pil(self, image):
                    ''' OpenCV type -> PIL type
                    https://qiita.com/derodero24/items/f22c22b22451609908ee
                    '''
                    new_image = image.copy()
                    if new_image.ndim == 2:  # Grayscale
                        pass
                    elif new_image.shape[2] == 3:  # Color
                        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
                    elif new_image.shape[2] == 4:  # Transparency
                        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
                    new_image = PIL.Image.fromarray(new_image)
                    return new_image

                def pil2cv(self, image):
                    ''' PIL type -> OpenCV type
                    https://qiita.com/derodero24/items/f22c22b22451609908ee
                    '''
                    new_image = np.array(image, dtype=np.uint8)
                    if new_image.ndim == 2:  # Grayscale
                        pass
                    elif new_image.shape[2] == 3:  # Color
                        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    elif new_image.shape[2] == 4:  # Transparency
                        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
                    return new_image

                def enhance(self, img, outscale=None):
                    # img: numpy array
                    h_input, w_input = img.shape[0:2]
                    pil_img = self.cv2pil(img)
                    pil_img = self.__call__(pil_img)
                    cv_image = self.pil2cv(pil_img)
                    if outscale is not None and outscale != float(netscale):
                        interpolation = cv2.INTER_AREA if outscale < float(netscale) else cv2.INTER_LANCZOS4
                        cv_image = cv2.resize(
                            cv_image, (
                                int(w_input * outscale),
                                int(h_input * outscale),
                            ), interpolation=interpolation)
                    return cv_image, None

            device = "cuda" if torch.cuda.is_available() else "cpu"
            upscaler = UpscaleWithModel.from_pretrained(os.path.join("weights", "upscale", upscale_model)).to(device)
            upscaler.__class__ = UpscaleWithModel_Gfpgan
            self.realesrganer = upscaler


    def initFaceEnhancerModel(self, face_restoration, face_detection):
        model_rootpath = os.path.join("weights", "face")
        model_path = os.path.join(model_rootpath, face_restoration)
        download_from_url(face_models[face_restoration][0], face_restoration, model_rootpath)
        
        self.modelInUse = f"_{os.path.splitext(face_restoration)[0]}" + self.modelInUse
        from gfpgan.utils import GFPGANer
        resolution = 512
        channel_multiplier = None
        
        if face_restoration and face_restoration.startswith("GFPGANv1."):
            arch = "clean"
            channel_multiplier = 2
        elif face_restoration and face_restoration.startswith("RestoreFormer"):
            arch = "RestoreFormer++" if face_restoration.startswith("RestoreFormer++") else "RestoreFormer"
        elif face_restoration == 'CodeFormer.pth':
            arch = "CodeFormer"
        elif face_restoration.startswith("GPEN-BFR-"):
            arch = "GPEN"
            channel_multiplier = 2
            if "1024" in face_restoration:
                arch = "GPEN-1024"
                resolution = 1024
            elif "2048" in face_restoration:
                arch = "GPEN-2048"
                resolution = 2048
        
        self.face_enhancer = GFPGANer(model_path=model_path, upscale=self.scale, arch=arch, channel_multiplier=channel_multiplier, model_rootpath=model_rootpath, det_model=face_detection, resolution=resolution)


    def inference(self, gallery, face_restoration, upscale_model, scale: float, face_detection, face_detection_threshold: any, face_detection_only_center: bool, outputWithModelName: bool, save_as_png: bool, progress=gr.Progress()):
        try:
            if not gallery or (not face_restoration and not upscale_model):
                raise ValueError("Invalid parameter setting")
            
            gallery_len = len(gallery)
            print(face_restoration, upscale_model, scale, f"gallery length: {gallery_len}")

            timer = Timer()  # Create a timer
            self.scale = scale
            
            progressTotal = gallery_len + 1
            progressRatio = 0.5 if upscale_model and face_restoration else 1
            print(f"progressRatio: {progressRatio}")
            current_progress = 0
            progress(0, desc="Initializing models...")
            if upscale_model:
                self.initBGUpscaleModel(upscale_model)
                current_progress += progressRatio/progressTotal;
                progress(current_progress, desc="BG upscale model initialized.")
                timer.checkpoint(f"Initialize BG upscale model")

            if face_restoration:
                self.initFaceEnhancerModel(face_restoration, face_detection)
                current_progress += progressRatio/progressTotal;
                progress(current_progress, desc="Face enhancer model initialized.")
                timer.checkpoint(f"Initialize face enhancer model")
                
            timer.report()

            if not outputWithModelName:
                self.modelInUse = ""
                
            files = []
            # Create zip files for each output type
            unique_id = str(int(time.time())) # Use timestamp for uniqueness
            zip_cropf_path    = f"output/{unique_id}_cropped_faces{self.modelInUse}.zip"
            zipf_cropf         = zipfile.ZipFile(zip_cropf_path, 'w', zipfile.ZIP_DEFLATED)
            zip_restoref_path = f"output/{unique_id}_restored_faces{self.modelInUse}.zip"
            zipf_restoref     = zipfile.ZipFile(zip_restoref_path, 'w', zipfile.ZIP_DEFLATED)
            zip_cmp_path      = f"output/{unique_id}_cmp{self.modelInUse}.zip"
            zipf_cmp          = zipfile.ZipFile(zip_cmp_path, 'w', zipfile.ZIP_DEFLATED)
            zip_restore_path  = f"output/{unique_id}_restored_images{self.modelInUse}.zip"
            zipf_restore      = zipfile.ZipFile(zip_restore_path, 'w', zipfile.ZIP_DEFLATED)

            is_auto_split_upscale = True
            # Dictionary to track counters for each filename
            name_counters = defaultdict(int)
            for gallery_idx, value in enumerate(gallery):
                img_path = None
                try:
                    if value is None or not value:
                        print(f"Warning: Invalid gallery item at index {gallery_idx}. Skipping.")
                        continue
                    img_path = str(value[0]) # value is often a list/tuple like [filepath, caption]
                    img_name = os.path.basename(img_path)
                    basename, extension = os.path.splitext(img_name)
                    # Increment the counter for the current name if it appears multiple times
                    name_counters[img_name] += 1
                    if name_counters[img_name] > 1:
                        basename = f"{basename}_{name_counters[img_name] - 1:02d}"

                    img_cv2 = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_UNCHANGED) # numpy.ndarray

                    if img_cv2 is None:
                        print(f"Warning: Could not read or decode image '{img_path}'. Skipping this image.")
                        # Skip this iteration and process the next image
                        continue

                    img_mode = "RGBA" if len(img_cv2.shape) == 3 and img_cv2.shape[2] == 4 else None
                    if len(img_cv2.shape) == 2:  # for gray inputs
                        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR)
                    print(f"> Processing image {gallery_idx:02d}, Shape: {img_cv2.shape}")

                    bg_upsample_img = None
                    if upscale_model and self.realesrganer and hasattr(self.realesrganer, "enhance"):
                        bg_upsample_img, _ = auto_split_upscale(img_cv2, self.realesrganer.enhance, self.scale) if is_auto_split_upscale else self.realesrganer.enhance(img_cv2, outscale=self.scale)
                        current_progress += progressRatio/progressTotal;
                        progress(current_progress, desc=f"Image {gallery_idx:02d}: Background upscaling...")
                        timer.checkpoint(f"Image {gallery_idx:02d}: Background upscale section")

                    if face_restoration and self.face_enhancer:
                        cropped_faces, restored_aligned, bg_upsample_img = self.face_enhancer.enhance(img_cv2, has_aligned=False, only_center_face=face_detection_only_center, paste_back=True, bg_upsample_img=bg_upsample_img, eye_dist_threshold=face_detection_threshold)
                        # save faces
                        if cropped_faces and restored_aligned:
                            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_aligned)):
                                # save cropped face
                                save_crop_path = f"output/{basename}_{idx:02d}_cropped_faces{self.modelInUse}.png"
                                self.imwriteUTF8(save_crop_path, cropped_face)
                                zipf_cropf.write(save_crop_path, arcname=os.path.basename(save_crop_path))
                                # save restored face
                                save_restore_path = f"output/{basename}_{idx:02d}_restored_faces{self.modelInUse}.png"
                                self.imwriteUTF8(save_restore_path, restored_face)
                                zipf_restoref.write(save_restore_path, arcname=os.path.basename(save_restore_path))
                                # save comparison image
                                save_cmp_path = f"output/{basename}_{idx:02d}_cmp{self.modelInUse}.png"
                                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                                self.imwriteUTF8(save_cmp_path, cmp_img)
                                zipf_cmp.write(save_cmp_path, arcname=os.path.basename(save_cmp_path))

                                files.append(save_crop_path)
                                files.append(save_restore_path)
                                files.append(save_cmp_path)
                        current_progress += progressRatio/progressTotal;
                        progress(current_progress, desc=f"Image {gallery_idx:02d}: Face enhancement...")
                        timer.checkpoint(f"Image {gallery_idx:02d}: Face enhancer section")

                    restored_img = bg_upsample_img
                    timer.report() # Report time for this image

                    # Handle cases where image processing might still result in None
                    if restored_img is None:
                       print(f"Warning: Processing resulted in no image for '{img_path}'. Skipping output.")
                       continue

                    # Determine the file extension for the output image based on user preference and image properties.
                    if save_as_png:
                        # Force PNG output for the best quality, as requested by the user.
                        final_extension = ".png"
                    else:
                        # Use original logic: PNG for images with an alpha channel (RGBA), otherwise use the original extension or default to jpg.
                        final_extension = ".png" if img_mode == "RGBA" else (extension if extension else ".jpg")
                    save_path = f"output/{basename}{self.modelInUse}{final_extension}"
                    self.imwriteUTF8(save_path, restored_img)
                    zipf_restore.write(save_path, arcname=os.path.basename(save_path))

                    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                    files.append(save_path)
                except RuntimeError as error:
                    print(f"Runtime Error while processing image {gallery_idx} ({img_path or 'unknown path'}): {error}")
                    print(traceback.format_exc())
                except Exception as general_error:
                    print(f"An unexpected error occurred while processing image {gallery_idx} ({img_path or 'unknown path'}): {general_error}")
                    print(traceback.format_exc())
                    # Can still choose to continue to process the next image
                    continue

            progress(1, desc=f"Processing complete.")
            timer.report_all()  # Print all recorded times for the whole batch
            # Close zip files
            zipf_cropf.close()
            zipf_restoref.close()
            zipf_cmp.close()
            zipf_restore.close()
        except Exception as error:
            print(f"Global exception occurred: {error}")
            print(traceback.format_exc())
            return None, None
        finally:
            if hasattr(self, 'face_enhancer') and self.face_enhancer:
                self.face_enhancer._cleanup()
            # Free GPU memory and clean up resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
                
        return files, [zip_cropf_path, zip_restoref_path, zip_cmp_path, zip_restore_path]  if face_restoration else [zip_restore_path]


    def find_max_numbers(self, state_dict, findkeys):
        if isinstance(findkeys, str):
            findkeys = [findkeys]
        max_values = defaultdict(lambda: None)
        patterns = {findkey: re.compile(rf"^{re.escape(findkey)}\.(\d+)\.") for findkey in findkeys}
    
        for key in state_dict:
            for findkey, pattern in patterns.items():
                if match := pattern.match(key):
                    num = int(match.group(1))
                    max_values[findkey] = max(num, max_values[findkey] if max_values[findkey] is not None else num)

        return tuple(max_values[findkey] for findkey in findkeys) if len(findkeys) > 1 else max_values[findkeys[0]]

    def find_divisor_for_quotient(self, a: int, c: int):
        """
        Returns a number `b` such that `a // b == c`.
        If `b` is an integer, return it as an `int`, otherwise return a `float`.
        """
        if c == 0:
            raise ValueError("c cannot be zero to avoid division by zero.")

        b_float = a / c

        # Check if b is an integer
        if b_float.is_integer():
            return int(b_float)

        # Try using ceil and floor
        ceil_b = math.ceil(b_float)
        floor_b = math.floor(b_float)

        if a // ceil_b == c:
            return ceil_b if ceil_b == b_float else float(ceil_b)
        if a // floor_b == c:
            return floor_b if floor_b == b_float else float(floor_b)

        # account for rounding errors
        if c == a // b_float:
            return b_float
        if c == a // (b_float - 0.01):
            return b_float - 0.01
        if c == a // (b_float + 0.01):
            return b_float + 0.01

        raise ValueError(f"Could not find a number b such that a // b == c. a={a}, c={c}")

    def imwriteUTF8(self, save_path, image): # `cv2.imwrite` does not support writing files to UTF-8 file paths.
        img_name = os.path.basename(save_path)
        _, extension = os.path.splitext(img_name)
        is_success, im_buf_arr = cv2.imencode(extension, image)
        if (is_success): im_buf_arr.tofile(save_path)

class Timer:
    def __init__(self):
        self.start_time  = time.perf_counter()  # Record the start time
        self.checkpoints = [("Start", self.start_time)]  # Store checkpoints

    def checkpoint(self, label="Checkpoint"):
        """Record a checkpoint with a given label."""
        now = time.perf_counter()
        self.checkpoints.append((label, now))

    def report(self, is_clear_checkpoints = True):
        # Determine the max label width for alignment
        max_label_length = max(len(label) for label, _ in self.checkpoints)

        prev_time = self.checkpoints[0][1]
        for label, curr_time in self.checkpoints[1:]:
            elapsed = curr_time - prev_time
            print(f"{label.ljust(max_label_length)}: {elapsed:.3f} seconds")
            prev_time = curr_time
        
        if is_clear_checkpoints:
            self.checkpoints.clear()
            self.checkpoint()  # Store checkpoints

    def report_all(self):
        """Print all recorded checkpoints and total execution time with aligned formatting."""
        print("\n> Execution Time Report:")

        # Determine the max label width for alignment
        max_label_length = max(len(label) for label, _ in self.checkpoints) if len(self.checkpoints) > 0 else 0

        prev_time = self.start_time
        for label, curr_time in self.checkpoints[1:]:
            elapsed = curr_time - prev_time
            print(f"{label.ljust(max_label_length)}: {elapsed:.3f} seconds")
            prev_time = curr_time
        
        total_time = self.checkpoints[-1][1] - self.start_time
        print(f"{'Total Execution Time'.ljust(max_label_length)}: {total_time:.3f} seconds\n")

        self.checkpoints.clear()

    def restart(self):
        self.start_time  = time.perf_counter()  # Record the start time
        self.checkpoints = [("Start", self.start_time)]  # Store checkpoints


def get_selection_from_gallery(selected_state: gr.SelectData):
    """
    Extracts the selected image path and caption from the gallery selection state.

    Args:
        selected_state (gr.SelectData): The selection state from a Gradio gallery, 
                                        containing information about the selected image.

    Returns:
        tuple: A tuple containing:
               - str: The file path of the selected image.
               - str: The caption of the selected image.
               If `selected_state` is None or invalid, it returns `None`.
    """
    if not selected_state:
        return selected_state

    return (selected_state.value["image"]["path"], selected_state.value["caption"])

def limit_gallery(gallery):
    """
    Ensures the gallery does not exceed input_images_limit.
    
    Args:
        gallery (list): Current gallery images.

    Returns:
        list: Trimmed gallery with a maximum of input_images_limit images.
    """
    return gallery[:input_images_limit] if input_images_limit > 0 and gallery else gallery

def append_gallery(gallery: list, image: str):
    """
    Append a single image to the gallery while respecting input_images_limit.

    Parameters:
    - gallery (list): Existing list of images. If None, initializes an empty list.
    - image (str): The image to be added. If None or empty, no action is taken.

    Returns:
    - list: Updated gallery.
    """
    if gallery is None:
        gallery = []
    if not image:
        return gallery, None
    
    if input_images_limit == -1 or len(gallery) < input_images_limit:
        gallery.append(image)

    return gallery, None


def extend_gallery(gallery: list, images):
    """
    Extend the gallery with new images while respecting the input_images_limit.
    
    Parameters:
    - gallery (list): Existing list of images. If None, initializes an empty list.
    - images (list): New images to be added. If None, defaults to an empty list.
    
    Returns:
    - list: Updated gallery with the new images added.
    """
    if gallery is None:
        gallery = []
    if not images:
        return gallery
    
    # Add new images to the gallery
    gallery.extend(images)

    # Trim gallery to the specified limit, if applicable
    if input_images_limit > 0:
        gallery = gallery[:input_images_limit]

    return gallery

def remove_image_from_gallery(gallery: list, selected_image: str):
    """
    Removes a selected image from the gallery if it exists.

    Args:
        gallery (list): The current list of images in the gallery.
        selected_image (str): The image to be removed, represented as a string 
                              that needs to be parsed into a tuple.

    Returns:
        list: The updated gallery after removing the selected image.
    """
    if not gallery or not selected_image:
        return gallery

    selected_image = ast.literal_eval(selected_image) # Use ast.literal_eval to parse text into a tuple in remove_image_from_gallery.
    # Remove the selected image from the gallery
    if selected_image in gallery:
        gallery.remove(selected_image)
    return gallery

def main():
    if torch.cuda.is_available():
        # Sets the VRAM limit for the GPU. Fine-tune VRAM usage.
        # Higher values allow larger image tiles to be processed at once, which is faster. If you see the 'depth' in the console log increase frequently, try raising this value.
        # Lower it for complex models that need more memory overhead.
        # This setting is locked in once processing starts. To apply a new value, you must restart the entire application.
        torch.cuda.set_per_process_memory_fraction(0.900, device='cuda:0')
        # set torch options to avoid get black image for RTX16xx card
        # https://github.com/CompVis/stable-diffusion/issues/69#issuecomment-1260722801
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    # Ensure the target directory exists
    os.makedirs('output', exist_ok=True)

    title = "Caritas Moraditas 💙💜💜 Caritas Moraditas"
    description = "Modelos pa upscaling y mejora de rostros 💙💜💙💙.<br> Llevatelos todos... Copia el espacio."
    
    # Custom CSS to set the height of the gr.Dropdown menu
    css = """
    ul.options {
        max-height: 500px !important;  /* Set the maximum height of the dropdown menu */
        overflow-y: auto !important;   /* Enable vertical scrolling if the content exceeds the height */
    }
    div.progress-level div.progress-level-inner {
        text-align: left !important;
        width: 55.5% !important;
    }
    """

    upscale = Upscale()
    
    rows = []
    tmptype = None
    upscale_model_tables = []
    for key, _ in typed_upscale_models.items():
        upscale_type, upscale_model = key.split(", ", 1)
        if tmptype and tmptype != upscale_type:#RRDB ESRGAN
            speed = "Fast" if tmptype == "SRVGG" else ("Slow" if any(value == tmptype for value in ("DAT", "HAT", "DRCT", "ATD", "SRFormer")) else "Normal")
            upscale_model_header = f"| Upscale Model | Info, Type: {tmptype}, Model execution speed: {speed} | Download URL |\n|------------|------|--------------|"
            upscale_model_tables.append(upscale_model_header + "\n" + "\n".join(rows))
            rows.clear()
        tmptype = upscale_type
        value = upscale_models[upscale_model]
        row = f"| [{upscale_model}]({value[1]}) | " + value[2].replace("\n", "<br>") + " | [download]({value[0]}) |"
        rows.append(row)
    speed = "Fast" if tmptype == "SRVGG" else ("Slow" if any(value == tmptype for value in ("DAT", "HAT", "DRCT", "ATD", "SRFormer")) else "Normal")
    upscale_model_header = f"| Upscale Model Name | Info, Type: {tmptype}, Model execution speed: {speed} | Download URL |\n|------------|------|--------------|"
    upscale_model_tables.append(upscale_model_header + "\n" + "\n".join(rows))

    with gr.Blocks() as demo:
        gr.Markdown(value=f"<h1 style=\"text-align:center;\">CaritasMoraditas</h1><br>{description}")
        with gr.Row():
            with gr.Column(variant="panel"):
                submit = gr.Button(value="Submit", variant="primary", size="lg")
                # Create an Image component for uploading images
                input_image                = gr.Image(label="Pon una imagen", type="filepath", format="png", height=150)
                with gr.Row():
                    upload_button          = gr.UploadButton("Sube varias imagenes", file_types=["image"], file_count="multiple", size="sm")
                    remove_button          = gr.Button("Remove Selected Image", size="sm")
                input_gallery              = gr.Gallery(columns=5, rows=5, show_share_button=False, interactive=True, height="500px", label="Gallery that displaying a grid of images" + (f"(The online environment image limit is {input_images_limit})" if input_images_limit > 0 else ""))
                face_model                 = gr.Dropdown([None]+list(face_models.keys()), type="value", value='GFPGANv1.4.pth', label='Face Restoration version', info="Face Restoration and RealESR can be freely combined in different ways, or one can be set to \"None\" to use only the other model. Face Restoration is primarily used for face restoration in real-life images, while RealESR serves as a background restoration model.")
                upscale_model              = gr.Dropdown([None]+list(typed_upscale_models.keys()), type="value", value='SRVGG, realesr-general-x4v3.pth', label='UpScale version')
                upscale_scale              = gr.Number(label="Factor de escala", value=4)
                face_detection             = gr.Dropdown(["retinaface_resnet50", "YOLOv5l", "YOLOv5n"], type="value", value="retinaface_resnet50", label="Face Detection type")
                face_detection_threshold   = gr.Number(label="Face eye dist threshold", value=10, info="A threshold to filter out faces with too small an eye distance (e.g., side faces).")
                face_detection_only_center = gr.Checkbox(value=False, label="Face detection only center", info="If set to True, only the face closest to the center of the image will be kept.")
                with_model_name            = gr.Checkbox(label="Output image files name with model name", value=True)
                # Add a checkbox to always save the output as a PNG file for the best quality.
                save_as_png                = gr.Checkbox(label="Always save output as PNG", value=True, info="If enabled, all output images will be saved in PNG format to ensure the best quality. If disabled, the format will be determined automatically (PNG for images with transparency, otherwise JPG).")

                # Define the event listener to add the uploaded image to the gallery
                input_image.change(append_gallery, inputs=[input_gallery, input_image], outputs=[input_gallery, input_image])
                # When the upload button is clicked, add the new images to the gallery
                upload_button.upload(extend_gallery, inputs=[input_gallery, upload_button], outputs=input_gallery)
                # Event to update the selected image when an image is clicked in the gallery
                selected_image = gr.Textbox(label="Selected Image", visible=False)
                input_gallery.select(get_selection_from_gallery, inputs=None, outputs=selected_image)
                # Trigger update when gallery changes
                input_gallery.change(limit_gallery, input_gallery, input_gallery)
                # Event to remove a selected image from the gallery
                remove_button.click(remove_image_from_gallery, inputs=[input_gallery, selected_image], outputs=input_gallery)

                with gr.Row():
                    clear = gr.ClearButton(
                        components=[
                            input_gallery,
                            face_model,
                            upscale_model,
                            upscale_scale,
                            face_detection,
                            face_detection_threshold,
                            face_detection_only_center,
                            with_model_name,
                            save_as_png,
                        ], variant="secondary", size="lg",)
            with gr.Column(variant="panel"):
                gallerys = gr.Gallery(type="filepath", label="Output (The whole image)", format="png")
                outputs = gr.File(label="Download the output ZIP file")
        with gr.Row(variant="panel"):
            # Generate output array
            output_arr = []
            for file_name in example_list:
                output_arr.append([file_name,])
            gr.Examples(output_arr, inputs=[input_image,], examples_per_page=20)
        with gr.Row(variant="panel"):
            # Convert to Markdown table
            header = "| Face Model Name | Info | Download URL |\n|------------|------|--------------|"
            rows = [
                f"| [{key}]({value[1]}) | " + value[2].replace("\n", "<br>") + f" | [download]({value[0]}) |"
                for key, value in face_models.items()
            ]
            markdown_table = header + "\n" + "\n".join(rows)
            gr.Markdown(value=markdown_table)

        for table in upscale_model_tables:
            with gr.Row(variant="panel"):
                gr.Markdown(value=table)

        submit.click(
            upscale.inference, 
            inputs=[
                input_gallery,
                face_model,
                upscale_model,
                upscale_scale,
                face_detection,
                face_detection_threshold,
                face_detection_only_center,
                with_model_name,
                save_as_png,
            ],
            outputs=[gallerys, outputs],
        )
    
    demo.queue(default_concurrency_limit=1)
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_images_limit", type=int, default=350)
    args = parser.parse_args()
    input_images_limit = args.input_images_limit
    main()