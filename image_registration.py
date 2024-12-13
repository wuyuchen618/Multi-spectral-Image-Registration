import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

def select_files():
    root = tk.Tk()
    root.withdraw()
    
    print("\nPlease select the moving image (image to be aligned)...")
    file1 = filedialog.askopenfilename(
        title='Select Moving Image (image to be aligned)',
        filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
    )
    if file1:
        print(f"Moving image selected: {file1}")
    else:
        print("No moving image selected")
        return None, None
        
    print("\nPlease select the fixed image (reference image)...")
    file2 = filedialog.askopenfilename(
        title='Select Fixed Image (reference image)',
        filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
    )
    if file2:
        print(f"Fixed image selected: {file2}")
    else:
        print("No fixed image selected")
        return None, None
    
    return file1, file2

def read_image(file_path):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"OpenCV reading failed, trying PIL: {file_path}")
            pil_img = Image.open(file_path)
            img = np.array(pil_img)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        print(f"Image loaded: shape={img.shape}, dtype={img.dtype}, range=[{np.min(img)}, {np.max(img)}]")
        return img
        
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def draw_alignment_vectors(fixed_img, moving_img, kp1, kp2, good_matches, mask, output_path):
    """
    繪製匹配點之間的偏移向量圖
    """
    # 創建彩色圖像來繪製向量
    vector_img = cv2.cvtColor(fixed_img.copy(), cv2.COLOR_GRAY2BGR)
    
    # 只取RANSAC後的內點
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
    
    # 為每個匹配點繪製箭頭
    for match in inlier_matches:
        # 取得匹配點座標
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        
        # 繪製箭頭
        cv2.arrowedLine(vector_img, pt1, pt2, 
                       color=(0, 255, 0),  # 綠色箭頭
                       thickness=1,         # 線條粗細
                       line_type=cv2.LINE_AA,
                       tipLength=0.2)       # 箭頭尖端長度
    
    # 保存結果
    cv2.imwrite(output_path, vector_img)
    return vector_img

def align_images_sift(moving, fixed):
    """
    使用SIFT對齊圖像
    """
    # 初始化SIFT檢測器
    sift = cv2.SIFT_create()
    
    # 檢測關鍵點和描述子
    kp1, des1 = sift.detectAndCompute(moving, None)
    kp2, des2 = sift.detectAndCompute(fixed, None)
    
    print(f"SIFT keypoints detected - Moving image: {len(kp1)}, Fixed image: {len(kp2)}")
    
    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"Good matches found: {len(good_matches)}")
    
    if len(good_matches) >= 4:
        # 獲取匹配點的坐標
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 計算單應性矩陣
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = [good_matches[i] for i in range(len(mask)) if mask[i]]
        print(f"RANSAC inliers: {len(inliers)}/{len(good_matches)}")
        
        # 對齊圖像
        aligned = cv2.warpPerspective(moving, H, (fixed.shape[1], fixed.shape[0]))
        
        # 生成匹配可視化
        match_img = cv2.drawMatches(moving, kp1, fixed, kp2, inliers, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                  
        return aligned, match_img, H, mask, good_matches, kp1, kp2, True
        
    else:
        print("Not enough good matches found")
        return None, None, None, None, None, None, None, False

def add_text_to_image(image, text, position=(100, 100)):
    """
    在圖片上添加文字
    """
    img_with_text = image.copy()
    if len(img_with_text.shape) == 2:  # 如果是灰階圖，轉為BGR
        img_with_text = cv2.cvtColor(img_with_text, cv2.COLOR_GRAY2BGR)
    
    # 添加白色文字，黑色邊框使文字在任何背景都清晰可見
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 2
    
    # 先畫黑色邊框
    cv2.putText(img_with_text, text, position, font, font_scale, (0, 0, 0), thickness + 1)
    # 再畫白色文字
    cv2.putText(img_with_text, text, position, font, font_scale, (255, 255, 255), thickness)
    
    return img_with_text

def main():
    # Select files
    file1, file2 = select_files()
    
    if file1 is None or file2 is None:
        print("Error: Both files must be selected")
        return
    
    # Get image names without extension
    moving_name = os.path.splitext(os.path.basename(file1))[0]
    fixed_name = os.path.splitext(os.path.basename(file2))[0]
    
    # Read images
    print("\nReading images...")
    img1 = read_image(file1)
    img2 = read_image(file2)
    
    if img1 is None or img2 is None:
        print("Error: Cannot read one or both images")
        return
        
    print("Successfully read both images")
    
    # Process images
    print("\nProcessing images...")
    aligned_img, match_img, H, mask, good_matches, kp1, kp2, success = align_images_sift(img1, img2)
    
    if success:
        # Create output directory with meaningful name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('results', f'{moving_name}_to_{fixed_name}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Add labels to images
        labeled_img1 = add_text_to_image(img1, f"Moving Image ({moving_name})")
        labeled_img2 = add_text_to_image(img2, f"Fixed Image ({fixed_name})")
        labeled_aligned = add_text_to_image(aligned_img, f"Aligned {moving_name}")
        
        # Save matching result with meaningful name
        match_img_labeled = add_text_to_image(match_img, f"Matching: {moving_name} -> {fixed_name}")
        cv2.imwrite(os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_matching.png'), match_img_labeled)
        
        # Save aligned image
        cv2.imwrite(os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_aligned.png'), labeled_aligned)
        
        # Generate and save vector visualization
        vector_img = draw_alignment_vectors(img2, img1, kp1, kp2, good_matches, mask,
                                         os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_vectors.png'))
        labeled_vector = add_text_to_image(vector_img,f"Alignment Vectors: {moving_name} -> {fixed_name}")
        cv2.imwrite(os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_vectors.png'), labeled_vector)
        
        # Generate and save blended result
        blended = cv2.addWeighted(img2, 0.5, aligned_img, 0.5, 0)
        labeled_blended = add_text_to_image(blended, f"Blended: {moving_name} + {fixed_name}")
        cv2.imwrite(os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_blended.png'), labeled_blended)
        
        # Create GIF showing the alignment process with labels
        images_for_gif = [
            Image.fromarray(cv2.cvtColor(labeled_img1, cv2.COLOR_BGR2RGB)),      # Moving image
            Image.fromarray(cv2.cvtColor(labeled_img2, cv2.COLOR_BGR2RGB)),      # Fixed image
            Image.fromarray(cv2.cvtColor(labeled_aligned, cv2.COLOR_BGR2RGB)),   # Aligned image
            Image.fromarray(cv2.cvtColor(labeled_blended, cv2.COLOR_BGR2RGB))    # Blended result
        ]
        
        # Save GIF
        images_for_gif[0].save(
            os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_sequence.gif'),
            save_all=True,
            append_images=images_for_gif[1:],
            duration=1500,
            loop=5
        )
        
        print(f"\nResults saved in {output_dir}")
        print("Successfully completed image alignment!")
    else:
        print("\nImage alignment failed")

if __name__ == "__main__":
    main()