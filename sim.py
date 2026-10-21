import cv2
from embedding import embedding

result = embedding("..\challenge_sim\defense\harbour.bmp", "..\challenge_sim\defense\clk.npy")

cv2.imshow("Watermarked", result)
