#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de Navegação de Drone Indoor com Detecção de QR Code e Pose Estimation.
Autor: Engenheiro de Visão Computacional (Simulado)
Plataforma: Raspberry Pi 4
"""

import cv2
import numpy as np
import sys

def main():
    # Inicializa a captura de vídeo
    # 0 geralmente é a câmera padrão (CSI ou USB no RPi)
    cap = cv2.VideoCapture(0)

    # Verifica se a câmera abriu corretamente
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        sys.exit()

    # Define o detector de QR Code
    detector = cv2.QRCodeDetector()

    # --- Configurações do Mundo Real ---
    # Tamanho do QR Code em metros (10.0 cm = 0.1 m)
    QR_CODE_SIZE = 0.1
    half_size = QR_CODE_SIZE / 2.0

    # Define os pontos 3D dos cantos do QR Code no sistema de coordenadas do objeto.
    # A ordem deve corresponder à ordem retornada pelo detector (geralmente TL, TR, BR, BL).
    # Vamos assumir que o centro do QR Code é (0, 0, 0).
    # Eixo X para a direita, Eixo Y para baixo (para alinhar com imagem), Eixo Z apontando para fora do QR code (ou para dentro).
    # O padrão OpenCV geralmente assume sistema destro.
    # Pontos: Topo-Esquerda, Topo-Direita, Baixo-Direita, Baixo-Esquerda
    object_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ], dtype=np.float32)

    # --- Configuração da Câmera (Matriz Intrinseca) ---
    # Valores aproximados. Para precisão, você DEVE calibrar sua câmera e substituir estes valores.
    # Exemplo para Raspberry Pi Camera V2 (aproximado)
    # Vamos inicializar dinamicamente baseada na resolução do primeiro frame capturado
    ret, frame = cap.read()
    if not ret:
        print("Erro: Falha ao capturar frame inicial.")
        sys.exit()

    h, w = frame.shape[:2]

    # Aproximação: Focal length é muitas vezes próximo à largura da imagem em pixels
    focal_length = w
    center_x = w / 2
    center_y = h / 2

    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # Coeficientes de distorção (assumindo 0 se não calibrado)
    dist_coeffs = np.zeros((4, 1))

    print("Iniciando loop de detecção. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Aviso: Frame não capturado.")
            break

        # Detecta e decodifica o QR Code
        # retval: booleano (detectou ou não)
        # decoded_info: string com dados
        # points: array com coordenadas dos cantos (bbox)
        # straight_qrcode: imagem retificada do QR Code
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)

        detected = False

        # O detector pode retornar True mas pontos vazios ou vice-versa em algumas versões,
        # mas detectAndDecodeMulti retorna um booleano de sucesso e uma lista de infos.
        if retval:
            detected = True

            # points geralmente vem no formato (N, 4, 2) onde N é numero de QRs. Pegamos o primeiro.
            # Se detectAndDecode (singular) fosse usado, seria (1, 4, 2) ou (4, 2).
            # Vamos garantir que temos os pontos no formato correto (4, 2) float32
            qr_points = points[0]
            qr_points = qr_points.astype(np.float32)

            # Desenha a caixa VERDE ao redor do QR Code
            # Converte para int para desenhar polígono
            pts_int = qr_points.astype(np.int32)
            pts_int = pts_int.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_int], True, (0, 255, 0), 3)

            # --- Pose Estimation ---
            # Resolve o problema Perspective-n-Point
            # Calcula a rotação (rvec) e translação (tvec) do objeto em relação à câmera
            success, rvec, tvec = cv2.solvePnP(object_points, qr_points, camera_matrix, dist_coeffs)

            if success:
                # tvec contém x, y, z em metros (porque definimos o objeto em metros)
                # Converter para cm para exibição
                x_cm = tvec[0][0] * 100
                y_cm = tvec[1][0] * 100
                z_cm = tvec[2][0] * 100

                # Exibe as coordenadas na tela
                label = f"X: {x_cm:.1f}cm Y: {y_cm:.1f}cm Z: {z_cm:.1f}cm"
                cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Opcional: Desenhar eixos 3D projetados no QR code para validação visual
                axis_length = 0.05 # 5cm
                axis_points = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]).reshape(-1,3)
                imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

                # Centro do QR code na imagem
                corner = tuple(pts_int[0].ravel())

                # O OpenCV desenha: Azul (Z?), Verde (Y?), Vermelho (X?) - Cores BGR
                # Geralmente: X=Azul, Y=Verde, Z=Vermelho no padrão projectPoints drawing tutorials, mas opencv é BGR.
                # Vamos desenhar simples linhas
                origin_pt = (int(corner[0]), int(corner[1])) # Topo-esquerda como origem visual aproximada ou projetar (0,0,0)

                # Projetar o centro (0,0,0)
                center_point_3d = np.float32([[0,0,0]])
                center_point_2d, _ = cv2.projectPoints(center_point_3d, rvec, tvec, camera_matrix, dist_coeffs)
                center_x_2d = int(center_point_2d[0][0][0])
                center_y_2d = int(center_point_2d[0][0][1])
                center_tuple = (center_x_2d, center_y_2d)

                # Desenhar um ponto no centro
                cv2.circle(frame, center_tuple, 5, (255, 0, 255), -1)


        if not detected:
            # Desenha retângulo VERMELHO nas bordas (alerta)
            thickness = 20
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), thickness)

            # Escreve "BUSCANDO..." no centro
            text = "BUSCANDO..."
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 3

            # Calcula tamanho do texto para centralizar
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = (w - text_w) // 2
            text_y = (h + text_h) // 2

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

        # Exibe o frame resultante
        cv2.imshow('Drone Navigation System', frame)

        # Sai se 'q' for pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
