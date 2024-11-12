import cv2
import numpy as np

'''
## SIFT特征提取和匹配具体步骤:

1. 生成高斯差分金字塔(DOG金字塔)，尺度空间构建
2. 空间极值点检测（关键点的初步查探）
3. 稳定关键点的精确定位
4. 稳定关键点方向信息分配
5. 关键点描述
6. 特征点匹配

'''

# 绘制匹配点的函数，展示两张图像的特征匹配结果
def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    # 获取两张图像的高度和宽度
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # 创建一张拼接图，用于显示两张图像和匹配点连线
    combined_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
    combined_image[:height1, :width1] = img1  # 左侧显示图像1
    combined_image[:height2, width1:width1 + width2] = img2  # 右侧显示图像2

    # 获取匹配点的坐标
    # matches：这是 BFMatcher 的 KNN 匹配结果，
    # 包含了 queryIdx 和 trainIdx，分别对应图像 1 和图像 2 中匹配点的索引。
    points1 = np.int32([keypoints1[match.queryIdx].pt for match in matches])
    points2 = np.int32([keypoints2[match.trainIdx].pt for match in matches]) + (width1, 0)

    # 画线连接匹配点
    for (x1, y1), (x2, y2) in zip(points1, points2):
        cv2.line(combined_image, (x1, y1), (x2, y2), (0, 0, 255))  # 画红色线

    # 显示匹配结果
    cv2.namedWindow("Matched Points", cv2.WINDOW_NORMAL)
    cv2.imshow("Matched Points", combined_image)

# 读取灰度图像
img1 = cv2.imread("iphone1.png")
img2 = cv2.imread("iphone2.png")

# 初始化 SIFT 特征检测器
sift = cv2.SIFT_create()

# 检测和计算特征点及其描述符
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 初始化 BFMatcher 进行暴力匹配
bf_matcher = cv2.BFMatcher(cv2.NORM_L2)

# 进行 KNN 匹配，k=2 表示每个特征点找两个最佳匹配
knn_matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

'''
每个匹配对象(例如 matches[i][0] 和 matches[i][1])都是 DMatch 类型的对象,DMatch 包含以下几个重要属性:

queryIdx:源图像特征点的索引.
trainIdx:目标图像特征点的索引.
distance:描述符之间的距离,用于衡量匹配的相似度,距离越小表示越相似.
'''

# 筛选出较好的匹配点
good_matches = []
for m, n in knn_matches:
    if m.distance < 0.5 * n.distance:  # 如果第一匹配的距离小于第二匹配的0.5倍
        good_matches.append(m)

# 绘制并显示前20个匹配点
draw_matches(img1, keypoints1, img2, keypoints2, good_matches[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
