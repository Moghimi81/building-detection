import cv2
import numpy as np
import math
import os
from typing import List, Tuple, Optional

def save_image(image: np.ndarray, path: str, description: str = ""):
    """ذخیره تصویر با ایجاد پوشه در صورت نیاز"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    print(f"[ذخیره] {description} -> {path}")

def estimate_centers_with_saving(gray_img: np.ndarray,
                                 output_dir: str,
                                 base_name: str,
                                 min_area: int = 200,
                                 morph_kernel_size: int = 7) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    مرحله اول با ذخیره خروجی‌های میانی:
    - 1_otsu_binary.jpg
    - 2_after_remove_small.jpg
    - 3_after_morphology.jpg
    - 4_final_mask_with_centers.jpg
    همچنین تصویر با مراکز تخمینی ذخیره می‌شود.
    """
    # آستانه اوتسو
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_image(binary, os.path.join(output_dir, f"{base_name}_1_otsu_binary.jpg"), "خروجی آستانه اوتسو")

    # یافتن کانتورها و حذف نواحی کوچک (مرحله اول)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask1 = np.zeros_like(binary)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(mask1, [cnt], -1, 255, -1)
    save_image(mask1, os.path.join(output_dir, f"{base_name}_2_after_remove_small.jpg"), "پس از حذف نواحی کوچک (مرحله اول)")

    # عملیات مورفولوژی
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask_eroded = cv2.erode(mask1, kernel, iterations=2)
    mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=2)
    save_image(mask_dilated, os.path.join(output_dir, f"{base_name}_3_after_morphology.jpg"), "پس از فرسایش و گسترش")

    # حذف مجدد نواحی کوچک
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(binary)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    save_image(final_mask, os.path.join(output_dir, f"{base_name}_4_final_mask.jpg"), "ماسک نهایی پس از حذف مجدد")

    # ذخیره تصویر با مراکز تخمینی
    img_with_centers = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for (cx, cy) in centers:
        cv2.circle(img_with_centers, (cx, cy), 5, (255, 0, 0), -1)
    save_image(img_with_centers, os.path.join(output_dir, f"{base_name}_5_estimated_centers.jpg"), "مراکز تخمینی ساختمان")

    return centers, final_mask

def filter_short_edges(edge_img: np.ndarray, min_length: int = 50) -> np.ndarray:
    lines = cv2.HoughLinesP(edge_img, rho=1, theta=np.pi/180, 
                            threshold=30, minLineLength=min_length, maxLineGap=10)
    filtered = np.zeros_like(edge_img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(filtered, (x1, y1), (x2, y2), 255, 1)
    return filtered

def group_parallel_lines(lines: List[Tuple[float, float, int]], 
                         theta_thresh_deg: float = 10.0, 
                         rho_thresh: float = 10.0) -> List[List[int]]:
    n = len(lines)
    used = [False] * n
    groups = []
    for i in range(n):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        rho_i, theta_i, _ = lines[i]
        for j in range(i+1, n):
            if used[j]:
                continue
            rho_j, theta_j, _ = lines[j]
            if abs(theta_i - theta_j) < theta_thresh_deg and abs(rho_i - rho_j) < rho_thresh:
                group.append(j)
                used[j] = True
        groups.append(group)
    return groups

def find_rectangles_in_window_with_saving(gray_patch: np.ndarray,
                                          d_min: int, d_max: int,
                                          output_dir: str,
                                          patch_id: int,
                                          center: Tuple[int, int],
                                          theta_thresh: float = 10.0,
                                          eps_thresh: float = 0.5,
                                          alpha_thresh: float = 15.0) -> bool:
    """
    بررسی مستطیل در یک پنجره و ذخیره تصاویر میانی (لبه، خطوط هاف، گروه‌بندی)
    در صورت تشخیص مستطیل، یک تصویر با خطوط رسم شده ذخیره می‌شود.
    """
    # لبه‌یابی
    edges = cv2.Canny(gray_patch, 205, 240)
    save_image(edges, os.path.join(output_dir, f"patch_{patch_id}_center_{center[0]}_{center[1]}_1_edges.jpg"), f"لبه‌های پنجره {patch_id}")

    # بهبود لبه
    edges_filtered = filter_short_edges(edges, min_length=50)
    save_image(edges_filtered, os.path.join(output_dir, f"patch_{patch_id}_center_{center[0]}_{center[1]}_2_edges_filtered.jpg"), f"لبه‌های بهبودیافته {patch_id}")
    if np.sum(edges_filtered) == 0:
        return False

    # تبدیل هاف استاندارد
    lines_hough = cv2.HoughLines(edges_filtered, rho=1, theta=np.pi/180, 
                                 threshold=int(0.5 * d_min))
    if lines_hough is None:
        return False

    # استخراج خطوط
    line_list = []
    for line in lines_hough:
        rho, theta_rad = line[0]
        theta_deg = math.degrees(theta_rad)
        # مقدار رأی تقریبی (می‌توان از تعداد نقاط روی خط محاسبه کرد ولی برای سادگی)
        # در اینجا از تعداد خطوط به دست آمده استفاده نمی‌کنیم، یک مقدار فرضی
        line_list.append((rho, theta_deg, 1))  # votes تقریبی

    # گروه‌بندی خطوط موازی
    groups = group_parallel_lines(line_list, theta_thresh, 10.0)

    # ذخیره تصویر خطوط تشخیص داده شده (رسم روی تصویر لبه)
    line_drawing = cv2.cvtColor(edges_filtered, cv2.COLOR_GRAY2BGR)
    for (rho, theta_deg, _) in line_list:
        theta_rad = math.radians(theta_deg)
        a = math.cos(theta_rad)
        b = math.sin(theta_rad)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + (-b)), int(y0 + (a)))
        pt2 = (int(x0 - (-b)), int(y0 - (a)))
        cv2.line(line_drawing, pt1, pt2, (0, 255, 0), 1)
    save_image(line_drawing, os.path.join(output_dir, f"patch_{patch_id}_center_{center[0]}_{center[1]}_3_hough_lines.jpg"), f"خطوط هاف در پنجره {patch_id}")

    # پیدا کردن زوج‌های گسترش یافته
    extended_peaks = []
    for group in groups:
        if len(group) < 2:
            continue
        group_lines = [line_list[i] for i in group]
        for i in range(len(group_lines)):
            rho1, theta1, votes1 = group_lines[i]
            for j in range(i+1, len(group_lines)):
                rho2, theta2, votes2 = group_lines[j]
                if abs(rho1 - rho2) <= d_min:
                    continue
                if abs(votes1 - votes2) > eps_thresh * (votes1 + votes2) / 2:
                    continue
                alpha = (theta1 + theta2) / 2.0
                epsilon = abs(rho1 - rho2) / 2.0
                extended_peaks.append((alpha, epsilon))

    # بررسی زوج‌های عمود بر هم
    for i in range(len(extended_peaks)):
        alpha1, _ = extended_peaks[i]
        for j in range(i+1, len(extended_peaks)):
            alpha2, _ = extended_peaks[j]
            diff = abs(abs(alpha1 - alpha2) - 90.0)
            if diff < alpha_thresh:
                # مستطیل پیدا شد - ذخیره تصویر نهایی پنجره با نشانه‌گذاری
                rect_img = cv2.cvtColor(gray_patch, cv2.COLOR_GRAY2BGR)
                cv2.putText(rect_img, "Rectangle Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                save_image(rect_img, os.path.join(output_dir, f"patch_{patch_id}_center_{center[0]}_{center[1]}_4_RECTANGLE.jpg"), f"مستطیل تشخیص داده شده در پنجره {patch_id}")
                return True
    return False

def detect_buildings_with_saving(image_path: str,
                                 output_dir: str = "outputs",
                                 d_min: int = 30,
                                 d_max: int = 150,
                                 min_area: int = 200,
                                 visualize: bool = True) -> List[Tuple[int, int]]:
    """
    تشخیص ساختمان با ذخیره تمام مراحل میانی.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"تصویر {image_path} یافت نشد.")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ذخیره تصویر اصلی
    save_image(img_bgr, os.path.join(output_dir, f"{base_name}_0_original.jpg"), "تصویر اصلی")

    # مرحله 1: تخمین مراکز با ذخیره خروجی‌ها
    centers, _ = estimate_centers_with_saving(gray, output_dir, base_name, min_area)

    # مرحله 2: تأیید هر مرکز و ذخیره خروجی‌های پنجره‌ای
    confirmed = []
    h, w = gray.shape
    for idx, (cx, cy) in enumerate(centers):
        x1 = max(0, cx - d_max // 2)
        y1 = max(0, cy - d_max // 2)
        x2 = min(w, cx + d_max // 2)
        y2 = min(h, cy + d_max // 2)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        # پوشه اختصاصی برای پچ‌ها
        patch_dir = os.path.join(output_dir, f"patches_{base_name}")
        if find_rectangles_in_window_with_saving(patch, d_min, d_max, patch_dir, idx, (cx, cy)):
            confirmed.append((cx, cy))

    # ذخیره نتیجه نهایی
    result_img = img_bgr.copy()
    for (cx, cy) in confirmed:
        cv2.circle(result_img, (cx, cy), 6, (0, 0, 255), -1)
    save_image(result_img, os.path.join(output_dir, f"{base_name}_6_final_detected_buildings.jpg"), "مراکز نهایی ساختمان‌های تشخیص داده شده")

    if visualize:
        # نمایش مختصر (اختیاری) اما خروجی اصلی ذخیره می‌شود
        cv2.imshow("Final Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return confirmed

# مثال استفاده
if __name__ == "__main__":
    # مسیر تصویر خود را وارد کنید
    image_file = "Annotation 2026-06-15 173954.png"  # تغییر دهید
    if os.path.exists(image_file):
        detected = detect_buildings_with_saving(image_file, output_dir="detection_outputs",
                                                d_min=15, d_max=300, min_area=200)
        print(f"تعداد ساختمان‌های تشخیص داده شده: {len(detected)}")
    else:
        print("لطفاً مسیر تصویر معتبر را مشخص کنید.")