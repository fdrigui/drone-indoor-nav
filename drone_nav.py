#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de Navegação de Drone Indoor com Detecção de QR Code e Pose Estimation.
Autor: Engenheiro de Visão Computacional (Simulado)
Plataforma: Raspberry Pi 4
Refatorado: Adicionado Loguru e Tratamento de Erros
"""

import cv2
import numpy as np
import sys
from loguru import logger

def draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, length=0.05):
    """
    Desenha os eixos X (Vermelho), Y (Verde) e Z (Azul) sobre o objeto detectado.
    """
    # Define os pontos dos eixos no espaço do objeto
    axis_points = np.float32([
        [length, 0, 0],  # Eixo X
        [0, length, 0],  # Eixo Y
        [0, 0, -length]  # Eixo Z (Apontando para a câmera se Z negativo for "frente")
        # Nota: A direção depende da convenção. Normalmente Z sai do plano.
        # Ajustando para visualização padrão OpenCV:
        # X: Vermelho, Y: Verde, Z: Azul
    ]).reshape(-1, 3)

    # Projeta os pontos 3D para o plano 2D da imagem
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Projeta a origem (0,0,0)
    origin_point_3d = np.float32([[0,0,0]])
    origin_point_2d, _ = cv2.projectPoints(origin_point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    origin = tuple(origin_point_2d[0][0].astype(int))

    # Extrai os pontos projetados
    p_x = tuple(imgpts[0][0].astype(int))
    p_y = tuple(imgpts[1][0].astype(int))
    p_z = tuple(imgpts[2][0].astype(int))

    # Desenha as linhas
    # X axis (Red - BGR: 0, 0, 255)
    cv2.line(frame, origin, p_x, (0, 0, 255), 3)
    # Y axis (Green - BGR: 0, 255, 0)
    cv2.line(frame, origin, p_y, (0, 255, 0), 3)
    # Z axis (Blue - BGR: 255, 0, 0)
    cv2.line(frame, origin, p_z, (255, 0, 0), 3)

def main():
    # Configuração do Logger
    logger.remove() # Remove o handler padrão para evitar duplicação se rodar múltiplas vezes
    logger.add(sys.stderr, level="INFO") # Default para console
    logger.add("logs/drone_nav.log", rotation="5 MB", level="DEBUG")

    logger.info("Iniciando Sistema de Navegação de Drone...")

    # Inicializa a captura de vídeo
    # 0 geralmente é a câmera padrão (CSI ou USB no RPi)
    cap = cv2.VideoCapture(0)

    # Verifica se a câmera abriu corretamente
    if not cap.isOpened():
        logger.error("Erro Crítico: Não foi possível acessar a câmera.")
        sys.exit(1)

    # Define o detector de QR Code
    detector = cv2.QRCodeDetector()

    # --- Configurações do Mundo Real ---
    # Tamanho do QR Code em metros (10.0 cm = 0.1 m)
    QR_CODE_SIZE = 0.1
    half_size = QR_CODE_SIZE / 2.0

    # Define os pontos 3D dos cantos do QR Code no sistema de coordenadas do objeto.
    object_points = np.array([
        [-half_size, -half_size, 0], # Topo-Esquerda
        [ half_size, -half_size, 0], # Topo-Direita
        [ half_size,  half_size, 0], # Baixo-Direita
        [-half_size,  half_size, 0]  # Baixo-Esquerda
    ], dtype=np.float32)

    # --- Configuração da Câmera (Matriz Intrinseca) ---
    ret, frame = cap.read()
    if not ret:
        logger.error("Erro: Falha ao capturar frame inicial.")
        sys.exit(1)

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

    logger.info("Loop de detecção iniciado. Pressione 'q' para sair.")

    # Variável de estado para evitar spam no log
    last_status_found = False
    frame_count = 0

    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            logger.warning("Aviso: Frame não capturado.")
            break

        try:
            # Detecta e decodifica o QR Code
            retval, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)

            detected = False

            if retval:
                detected = True

                # Se mudou de estado (de não encontrado para encontrado), loga
                if not last_status_found:
                    logger.info(f"QR Code detectado! Dados: {decoded_info}")
                    last_status_found = True

                # points geralmente vem no formato (N, 4, 2). Pegamos o primeiro.
                qr_points = points[0].astype(np.float32)

                # Desenha a caixa VERDE ao redor do QR Code
                pts_int = qr_points.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_int], True, (0, 255, 0), 3)

                # --- Pose Estimation ---
                # Resolve o problema Perspective-n-Point
                success, rvec, tvec = cv2.solvePnP(object_points, qr_points, camera_matrix, dist_coeffs)

                if success:
                    # tvec contém x, y, z em metros
                    x_cm = tvec[0][0] * 100
                    y_cm = tvec[1][0] * 100
                    z_cm = tvec[2][0] * 100

                    # Exibe as coordenadas na tela
                    label = f"X: {x_cm:.1f}cm Y: {y_cm:.1f}cm Z: {z_cm:.1f}cm"
                    cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Chama função para desenhar eixos 3D
                    draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs)

                    # Log de sucesso (Throttle para evitar spam: a cada 30 frames ~ 1 seg)
                    if frame_count % 30 == 0:
                        logger.success(f"Pose Calculada: X={x_cm:.1f} Y={y_cm:.1f} Z={z_cm:.1f}")

            if not detected:
                # Se mudou de estado (de encontrado para não encontrado), loga
                if last_status_found:
                    logger.warning("QR Code perdido. Buscando...")
                    last_status_found = False

                # Desenha retângulo VERMELHO nas bordas
                thickness = 20
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), thickness)

                # Escreve "BUSCANDO..." no centro
                text = "BUSCANDO..."
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 3

                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = (w - text_w) // 2
                text_y = (h + text_h) // 2

                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

        except Exception as e:
            logger.error(f"Exceção durante processamento: {e}")
            # Não damos break para tentar recuperar no próximo frame, mas desenhamos alerta talvez?
            cv2.putText(frame, "ERRO DE PROCESSAMENTO", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Exibe o frame resultante
        cv2.imshow('Drone Navigation System', frame)

        # Sai se 'q' for pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Usuário solicitou encerramento.")
            break

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Sistema encerrado com sucesso.")

if __name__ == "__main__":
    main()
