import cv2
import numpy as np

def segment_image_kmeans(image_path, K=10):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 400))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixel_values = img_rgb.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img_rgb.shape)

    cv2.imshow('Asl rasm', img)
    cv2.imshow(f'{K} segmentli rasm', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('segmentlangan_rasm.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

segment_image_kmeans(image_path='img1.jpg', K=5)