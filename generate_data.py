import numpy as np
# 数据生成函数
def generate_centered_box_data(num_samples):
    images = []
    vectors = []
    labels = []
    for _ in range(num_samples):
        # 创建空白图像
        img = np.zeros((64, 64), dtype=np.uint8)
        
        # 随机生成矩形的宽度和高度
        rect_w = np.random.randint(5, 15)
        rect_h = np.random.randint(5, 15)
        
        # 计算矩形左上角坐标，使矩形居中
        rect_x = (64 - rect_w) // 2
        rect_y = (64 - rect_h) // 2
        
        # 计算矩形右下角坐标
        rect_x2 = rect_x + rect_w
        rect_y2 = rect_y + rect_h
        
        # 在图像上绘制矩形
        img[rect_y:rect_y2, rect_x:rect_x2] = 255
        
        # 随机生成直线参数
        x1, y1 = np.random.randint(0, 64, size=2)
        x2, y2 = np.random.randint(0, 64, size=2)
        
        # 使用 Bresenham 算法绘制直线
        rr, cc = draw_line(x1, y1, x2, y2)
        img[cc, rr] = 128  # 用不同的灰度值表示
        
        # 判断直线是否与矩形重叠
        overlap = np.any(img[rect_y:rect_y2, rect_x:rect_x2] == 128)
        label = int(overlap)
        
        # 保存数据
        images.append(img)
        vectors.append([rect_x, rect_y, rect_w, rect_h, x1, y1, x2, y2])
        labels.append(label)
    return np.array(images), np.array(vectors), np.array(labels)

# 绘制直线的函数（Bresenham 算法）
def draw_line(x0, y0, x1, y1):
    from skimage.draw import line
    rr, cc = line(y0, x0, y1, x1)
    rr = np.clip(rr, 0, 63)
    cc = np.clip(cc, 0, 63)
    return cc, rr 

# 数据生成函数
def generate_data(num_samples):
    images = []
    vectors = []
    labels = []
    for _ in range(num_samples):
        # 创建空白图像
        img = np.zeros((64, 64), dtype=np.uint8)
        
        # 随机生成矩形参数
        rect_x = np.random.randint(5, 48)
        rect_y = np.random.randint(5, 48)
        rect_w = np.random.randint(5, 15)
        rect_h = np.random.randint(5, 15)
        
        # 确保矩形在图像内
        rect_x2 = min(rect_x + rect_w, 63)
        rect_y2 = min(rect_y + rect_h, 63)
        
        # 在图像上绘制矩形
        img[rect_y:rect_y2, rect_x:rect_x2] = 255
        
        # 随机生成直线参数
        x1, y1 = np.random.randint(0, 63, size=2)
        x2, y2 = np.random.randint(0, 63, size=2)
        
        # 使用 Bresenham 算法绘制直线
        rr, cc = draw_line(x1, y1, x2, y2)
        img[cc, rr] = 128  # 用不同的灰度值表示
        
        # 判断是否重叠
        overlap = np.any(img[rect_y:rect_y2, rect_x:rect_x2] == 128)
        label = int(overlap)
        
        # 保存数据
        images.append(img)
        vectors.append([rect_x, rect_y, rect_w, rect_h, x1, y1, x2, y2])
        labels.append(label)
    return np.array(images), np.array(vectors), np.array(labels) 

# 需要修改input_size
def generate_curve_data(num_samples, shape_center=(32, 32), shape_scale=10):
    images = []
    vectors = []
    labels = []
    for _ in range(num_samples):
        # 创建空白图像
        img = np.zeros((64, 64), dtype=np.uint8)
        
        # 生成贝塞尔曲线控制点
        # 控制点位于图像中心周围，通过shape_scale参数控制大小
        ctrl1 = (shape_center[0] + np.random.randint(-shape_scale, shape_scale),
                 shape_center[1] + np.random.randint(-shape_scale, shape_scale))
        ctrl2 = (shape_center[0] + np.random.randint(-shape_scale, shape_scale),
                 shape_center[1] + np.random.randint(-shape_scale, shape_scale))
        ctrl3 = (shape_center[0] + np.random.randint(-shape_scale, shape_scale),
                 shape_center[1] + np.random.randint(-shape_scale, shape_scale))
        
        # 绘制贝塞尔曲线并获取曲线上的点
        rr, cc = bezier_curve(ctrl1[0], ctrl1[1], ctrl2[0], ctrl2[1], ctrl3[0], ctrl3[1], 2)
        rr = np.clip(rr, 0, 63)
        cc = np.clip(cc, 0, 63)
        img[rr, cc] = 255

        # 将曲线上的点组合成坐标对列表
        curve_points = list(zip(cc, rr))  # x, y coordinates

        # 如果曲线点少于32个，重复采样直到长度为32
        while len(curve_points) < 32:
            curve_points.extend(curve_points)
        curve_points = curve_points[:32]

        # 将坐标对展平为一维列表
        flat_curve_points = [coord for point in curve_points for coord in point]  # [x0, y0, x1, y1, ..., x31, y31]

        # 随机生成直线参数
        x1, y1 = np.random.randint(0, 63, size=2)
        x2, y2 = np.random.randint(0, 63, size=2)
        
        # 使用 Bresenham 算法绘制直线
        line_rr, line_cc = draw_line(x1, y1, x2, y2)
        img[line_cc, line_rr] = 128  # 用不同的灰度值表示
        
        # 判断是否重叠
        overlap = np.any(img[rr, cc] == 128)
        label = int(overlap)
        
        # 保存数据
        images.append(img)
        vectors.append(flat_curve_points)  # 使用采样的32个点
        labels.append(label)
    return np.array(images), np.array(vectors), np.array(labels)