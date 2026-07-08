#Aplicação de Inspeção de Placas de Circuito Impresso com IA

#Realização das importações necessárias

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Inspeção AQI - Semicondutores", layout="wide")

st.title("🔍 Sistema Inteligente de Inspeção de Semicondutores")
st.markdown("Detecção automática de anomalias em placas utilizando Visão Computacional.")

from ultralytics import YOLO

@st.cache_resource
def carregar_unet():
    return tf.keras.models.load_model(
        "unet_final.keras",
        compile=False
    )

@st.cache_resource
def carregar_yolo_small():
    return YOLO("yolo26s.pt")

@st.cache_resource
def carregar_yolo_medium():
    return YOLO("yolo26m.pt")

@st.cache_resource
def carregar_yolo_large():
    return YOLO("yolo26l.pt")

@st.cache_resource
def carregar_rtdetr():
    return YOLO("rtdetr.pt")

# Carregue os modelos
modelo_unet = carregar_unet()

modelo_yolo_s = carregar_yolo_small()
modelo_yolo_m = carregar_yolo_medium()
modelo_yolo_l = carregar_yolo_large()

modelo_transformer = carregar_rtdetr()

# BARRA LATERAL (CONTROLES)
st.sidebar.image("https://cdn-icons-png.flaticon.com/128/18263/18263101.png", width=60)
st.sidebar.title("AQI System")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Sobre o Sistema:**\n\n"
    "Módulo de Inspeção Automática de Qualidade (AQI). "
    "Utiliza redes neurais profundas para detectar anomalias e defeitos micrométricos "
    "em placas semicondutoras na linha de montagem."
)
st.sidebar.markdown("---")
st.sidebar.header("Painel de Controle")

modelo_escolhido = st.sidebar.selectbox(

"Modelo",

[
"U-Net",

"YOLO26 Small",

"YOLO26 Medium",

"YOLO26 Large",

"RT-DETR"
]

)

alpha = st.sidebar.slider("Transparência da Máscara", 0.0, 1.0, 0.5)

# ÁREA PRINCIPAL E INFERÊNCIA
arquivo_upload = st.file_uploader("Faça o upload da imagem da placa (.jpg, .png)", type=["jpg", "png", "jpeg"])

if arquivo_upload is not None:
    # Ler a imagem carregada
    image = Image.open(arquivo_upload)
    img_array = np.array(image)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original (Câmera da Esteira)")
        st.image(img_array, use_column_width=True)
        
    with col2:
        st.subheader("Análise da IA (Overlay de Defeitos)")
        
        # Início do processamento
        inicio_inferencia = time.time()
        r = None
        resultado = None
        mask_bin = None
        mask_bin_original = None
        
        with st.spinner(f"Processando com {modelo_escolhido}..."):
            # 1. Pré-processamento (Igual ao seu Unet.py)
            img_resized = cv2.resize(img_array, (256, 256))
            img_input = img_resized / 255.0
            img_input = np.expand_dims(img_input, axis=0) # Adiciona batch dimension
            
            # 2. Inferência (Exemplo com U-Net)
            if "U-Net" in modelo_escolhido:
                predicao = modelo_unet.predict(img_input)[0]
                mask_bin = (predicao > 0.5).astype(np.uint8) # Máscara binária
            elif modelo_escolhido == "YOLO26 Small":
                resultado = modelo_yolo_s.predict(img_array,conf=0.25,verbose=False)
                r = resultado[0]
                st.subheader("Defeitos encontrados")
                for box in r.boxes:
                    classe = int(box.cls)
                    confianca = float(box.conf)
                    nome = r.names[classe]

                st.write(f"• {nome} ({confianca:.1%})")
                imagem = r.plot()
                st.image(imagem)
            elif modelo_escolhido == "YOLO26 Medium":
                resultado = modelo_yolo_m.predict(img_array,conf=0.25,verbose=False)
                r = resultado[0]
                st.subheader("Defeitos encontrados")
                for box in r.boxes:
                    classe = int(box.cls)
                    confianca = float(box.conf)
                    nome = r.names[classe]
                imagem = r.plot()
                st.image(imagem)
            elif modelo_escolhido == "YOLO26 Large":
                resultado = modelo_yolo_l.predict(img_array,conf=0.25,verbose=False)
                r = resultado[0]
                st.subheader("Defeitos encontrados")
                for box in r.boxes:
                    classe = int(box.cls)
                    confianca = float(box.conf)
                    nome = r.names[classe]
                imagem = r.plot()
                st.image(imagem)
            elif modelo_escolhido == "RT-DETR":
                resultado = modelo_transformer.predict(img_array,conf=0.25,verbose=False)
                r = resultado[0]
                st.subheader("Defeitos encontrados")
                for box in r.boxes:
                    classe = int(box.cls)
                    confianca = float(box.conf)
                    nome = r.names[classe]
                imagem = r.plot()
                st.image(imagem)
            else:
                # Placeholder para YOLO/ViT
                mask_bin = np.zeros((256, 256, 1), dtype=np.uint8)
                
            if modelo_escolhido == "U-Net":
                mask_bin_original = cv2.resize(mask_bin,(img_array.shape[1], img_array.shape[0]),interpolation=cv2.INTER_NEAREST)

                overlay = img_array.copy()

                overlay[mask_bin_original == 1] = [128,0,128]

                resultado = cv2.addWeighted(img_array,1-alpha,overlay,alpha,0)

            else:

                resultado = r.plot()


            tempo_inferencia = time.time() - inicio_inferencia

            st.image(resultado, use_column_width=True)
            if modelo_escolhido == "U-Net":

                predicao = modelo_unet.predict(img_input)[0]
                mask_bin = (predicao > 0.5).astype(np.uint8)

                mask_bin_original = cv2.resize(mask_bin,(img_array.shape[1], img_array.shape[0]),interpolation=cv2.INTER_NEAREST)

                overlay = img_array.copy()

                overlay[mask_bin_original == 1] = [128,0,128]

                resultado = cv2.addWeighted(img_array,1-alpha,overlay,alpha,0)

            st.image(resultado,use_column_width=True)
            



    # MÉTRICAS DE RESULTADO
    st.markdown("### 📊 Relatório de Inspeção")
    pixels_defeituosos = 0
    m1, m2, m3 = st.columns(3)
    cor_status = "red" if pixels_defeituosos > 0 else "green"

    if "U-Net" in modelo_escolhido:
            pixels_defeituosos = np.sum(mask_bin_original)
            status = "REPROVADA (Defeito Encontrado)" if pixels_defeituosos > 0 else "APROVADA (Sem Defeitos)"
            m1.metric(label="Status da Placa", value=status)
            m2.metric(label="Tempo de Inferência", value=f"{tempo_inferencia:.3f} s")
            m3.metric(label="Área Afetada", value=f"{pixels_defeituosos} pixels")

    elif modelo_escolhido == "YOLO26 Small":
        quantidade = len(r.boxes)
        m1.metric("Defeitos encontrados",quantidade)
        classes = []
        st.subheader("Defeitos encontrados")
        for box in r.boxes:
            nome = r.names[int(box.cls)]
            classes.append(nome)
            st.write(classes)
        m3.metric(label="Área Afetada", value=f"{pixels_defeituosos} pixels")
    elif modelo_escolhido == "YOLO26 Medium":
        quantidade = len(r.boxes)
        m1.metric("Defeitos encontrados",quantidade)
        classes = []
        st.subheader("Defeitos encontrados")
        for box in r.boxes:
            nome = r.names[int(box.cls)]
            classes.append(nome)
            st.write(classes)
        m2.metric(label="Tempo de Inferência", value=f"{tempo_inferencia:.3f} s")
        m3.metric(label="Área Afetada", value=f"{pixels_defeituosos} pixels")

    elif modelo_escolhido == "YOLO26 Large":
        quantidade = len(r.boxes)
        m1.metric("Defeitos encontrados",quantidade)
        classes = []
        st.subheader("Defeitos encontrados")
        for box in r.boxes:
            nome = r.names[int(box.cls)]
            classes.append(nome)
            st.write(classes)
        m2.metric(label="Tempo de Inferência", value=f"{tempo_inferencia:.3f} s")
        m3.metric(label="Área Afetada", value=f"{pixels_defeituosos} pixels")

    elif modelo_escolhido == "RT-DETR":
        quantidade = len(r.boxes)
        m1.metric("Defeitos encontrados",quantidade)
        classes = []
        st.subheader("Defeitos encontrados")
        for box in r.boxes:
            nome = r.names[int(box.cls)]
            classes.append(nome)
            st.write(classes)
        m2.metric(label="Tempo de Inferência", value=f"{tempo_inferencia:.3f} s")
        m3.metric(label="Área Afetada", value=f"{pixels_defeituosos} pixels")