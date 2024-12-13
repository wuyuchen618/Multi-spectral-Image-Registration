import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

def select_mode():
    """選擇匹配模式"""
    root = tk.Tk()
    root.withdraw()
    
    result = messagebox.askquestion("選擇模式", "是否要執行一對多匹配?\n選擇'是'執行一對多匹配\n選擇'否'執行一對一匹配")
    return result == 'yes'

def select_fixed_image():
    """選擇固定參考圖像"""
    root = tk.Tk()
    root.withdraw()
    
    print("\n請選擇固定參考圖像...")
    file = filedialog.askopenfilename(
        title='選擇固定參考圖像',
        filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
    )
    if file:
        print(f"已選擇固定參考圖像: {file}")
        return file
    else:
        print("未選擇固定參考圖像")
        return None

def get_moving_images(is_multi_mode, fixed_image_path):
    """獲取要匹配的圖像"""
    if is_multi_mode:
        # 獲取固定圖像所在目錄的所有圖像
        directory = os.path.dirname(fixed_image_path)
        fixed_name = os.path.basename(fixed_image_path)
        all_files = [f for f in os.listdir(directory) 
                    if f.endswith(('.tif', '.tiff', '.TIF', '.TIFF')) 
                    and f != fixed_name]
        return [os.path.join(directory, f) for f in all_files]
    else:
        # 選擇單一圖像
        root = tk.Tk()
        root.withdraw()
        
        print("\n請選擇要對齊的圖像...")
        file = filedialog.askopenfilename(
            title='選擇要對齊的圖像',
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        return [file] if file else []

def read_image(file_path):
    """讀取圖像"""
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"OpenCV讀取失敗，嘗試使用PIL: {file_path}")
            pil_img = Image.open(file_path)
            img = np.array(pil_img)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        print(f"圖像已載入: shape={img.shape}, dtype={img.dtype}, range=[{np.min(img)}, {np.max(img)}]")
        return img
        
    except Exception as e:
        print(f"讀取圖像時發生錯誤: {e}")
        return None

def draw_alignment_vectors(fixed_img, moving_img, kp1, kp2, good_matches, mask, output_path):
    """繪製匹配點之間的偏移向量圖"""
    vector_img = cv2.cvtColor(fixed_img.copy(), cv2.COLOR_GRAY2BGR)
    
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
    
    for match in inlier_matches:
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        
        cv2.arrowedLine(vector_img, pt1, pt2, 
                       color=(0, 255, 0),
                       thickness=1,
                       line_type=cv2.LINE_AA,
                       tipLength=0.2)
    
    cv2.imwrite(output_path, vector_img)
    return vector_img

def align_images_sift(moving, fixed):
    """使用SIFT對齊圖像"""
    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(moving, None)
    kp2, des2 = sift.detectAndCompute(fixed, None)
    
    print(f"已檢測SIFT關鍵點 - 移動圖像: {len(kp1)}, 固定圖像: {len(kp2)}")
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"找到的良好匹配點: {len(good_matches)}")
    
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = [good_matches[i] for i in range(len(mask)) if mask[i]]
        print(f"RANSAC內點: {len(inliers)}/{len(good_matches)}")
        
        aligned = cv2.warpPerspective(moving, H, (fixed.shape[1], fixed.shape[0]))
        
        match_img = cv2.drawMatches(moving, kp1, fixed, kp2, inliers, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                  
        return aligned, match_img, H, mask, good_matches, kp1, kp2, True
        
    else:
        print("未找到足夠的良好匹配點")
        return None, None, None, None, None, None, None, False

def add_text_to_image(image, text, position=(100, 100)):
    """在圖片上添加文字"""
    img_with_text = image.copy()
    if len(img_with_text.shape) == 2:
        img_with_text = cv2.cvtColor(img_with_text, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 2
    
    cv2.putText(img_with_text, text, position, font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(img_with_text, text, position, font, font_scale, (255, 255, 255), thickness)
    
    return img_with_text

def process_single_pair(moving_path, fixed_path, fixed_img):
    """處理單對圖像匹配"""
    moving_name = os.path.splitext(os.path.basename(moving_path))[0]
    fixed_name = os.path.splitext(os.path.basename(fixed_path))[0]
    
    print(f"\n處理圖像對: {moving_name} -> {fixed_name}")
    
    moving_img = read_image(moving_path)
    if moving_img is None:
        return None, None
        
    aligned_img, match_img, H, mask, good_matches, kp1, kp2, success = align_images_sift(moving_img, fixed_img)
    
    if success:
        # 創建輸出目錄
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('results', f'{moving_name}_to_{fixed_name}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加標籤
        labeled_moving = add_text_to_image(moving_img, f"Moving Image ({moving_name})")
        labeled_fixed = add_text_to_image(fixed_img, f"Fixed Image ({fixed_name})")
        labeled_aligned = add_text_to_image(aligned_img, f"Aligned {moving_name}")
        
        # 保存匹配結果
        match_img_labeled = add_text_to_image(match_img, f"Matching: {moving_name} -> {fixed_name}")
        cv2.imwrite(os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_matching.png'), match_img_labeled)
        
        # 保存對齊結果
        cv2.imwrite(os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_aligned.png'), labeled_aligned)
        
        # 生成和保存向量可視化
        vector_img = draw_alignment_vectors(fixed_img, moving_img, kp1, kp2, good_matches, mask,
                                         os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_vectors.png'))
        labeled_vector = add_text_to_image(vector_img, f"Alignment Vectors: {moving_name} -> {fixed_name}")
        cv2.imwrite(os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_vectors.png'), labeled_vector)
        
        # 生成和保存混合結果
        blended = cv2.addWeighted(fixed_img, 0.5, aligned_img, 0.5, 0)
        labeled_blended = add_text_to_image(blended, f"Blended: {moving_name} + {fixed_name}")
        cv2.imwrite(os.path.join(output_dir, f'{moving_name}_to_{fixed_name}_blended.png'), labeled_blended)
        
        return {
            'moving': labeled_moving,
            'fixed': labeled_fixed,
            'aligned': labeled_aligned,
            'blended': labeled_blended,
            'output_dir': output_dir,
            'names': (moving_name, fixed_name)
        }, True
    
    return None, False

def create_overlay_sequence(fixed_img, aligned_results, fixed_name):
    """
    創建漸進式疊圖序列，每一幀都是前面所有圖片的疊加結果
    """
    # 創建輸出目錄
    output_dir = os.path.join('results', f'all_to_{fixed_name}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 將固定圖像轉換為RGB格式
    base_img = fixed_img.copy()
    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    
    # 初始化GIF序列，從固定圖像開始
    overlay_frames = [Image.fromarray(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))]
    
    # 當前疊加狀態
    current_overlay = base_img.copy()
    
    # 保存每一步疊加的結果
    for idx, result in enumerate(aligned_results):
        # 獲取當前要疊加的對齊圖像
        aligned_img = result['aligned']
        if len(aligned_img.shape) == 2:
            aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_GRAY2BGR)
            
        # 計算當前疊加權重
        # 使用 1/(n+1) 的權重分配，確保所有圖像權重和為1
        weight = 1.0 / (idx + 2)  # +2 是因為包含了固定圖像
        
        # 更新疊加結果
        current_overlay = cv2.addWeighted(
            current_overlay,
            1.0 - weight,  # 前面圖像的總權重
            aligned_img,
            weight,        # 當前圖像的權重
            0
        )
        
        # 添加說明文字
        overlay_with_text = current_overlay.copy()
        text = f"Overlay: {idx + 1}/{len(aligned_results)} images"
        cv2.putText(
            overlay_with_text,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # 將當前疊加結果添加到序列中
        overlay_frames.append(Image.fromarray(cv2.cvtColor(overlay_with_text, cv2.COLOR_BGR2RGB)))
        
        # 保存每一步的疊加結果為PNG
        step_filename = f'overlay_step_{idx + 1}.png'
        cv2.imwrite(os.path.join(output_dir, step_filename), overlay_with_text)
    
    # 保存最終的疊加結果（不帶文字）
    cv2.imwrite(
        os.path.join(output_dir, 'final_overlay.png'),
        current_overlay
    )
    
    # 生成GIF動畫，展示漸進疊加過程
    gif_path = os.path.join(output_dir, 'progressive_overlay.gif')
    overlay_frames[0].save(
        gif_path,
        save_all=True,
        append_images=overlay_frames[1:],
        duration=500,  # 每幀顯示1秒
        loop=5
    )
    
    return output_dir

def main():
    # 選擇模式
    is_multi_mode = select_mode()
    
    # 選擇固定參考圖像
    fixed_path = select_fixed_image()
    if fixed_path is None:
        print("錯誤: 必須選擇固定參考圖像")
        return
        
    # 讀取固定參考圖像
    fixed_img = read_image(fixed_path)
    if fixed_img is None:
        print("錯誤: 無法讀取固定參考圖像")
        return
    
    # 獲取要匹配的圖像
    moving_paths = get_moving_images(is_multi_mode, fixed_path)
    if not moving_paths:
        print("錯誤: 未選擇要匹配的圖像")
        return
    
    fixed_name = os.path.splitext(os.path.basename(fixed_path))[0]
    
    # 處理所有圖像對
    successful_results = []
    for moving_path in moving_paths:
        result, success = process_single_pair(moving_path, fixed_path, fixed_img)
        if success:
            # 為每組匹配創建基本序列GIF
            sequence = [
                Image.fromarray(cv2.cvtColor(result['moving'], cv2.COLOR_BGR2RGB)),
                Image.fromarray(cv2.cvtColor(result['fixed'], cv2.COLOR_BGR2RGB)),
                Image.fromarray(cv2.cvtColor(result['aligned'], cv2.COLOR_BGR2RGB)),
                Image.fromarray(cv2.cvtColor(result['blended'], cv2.COLOR_BGR2RGB))
            ]
            
            # 保存每組的序列GIF
            gif_path = os.path.join(
                result['output_dir'], 
                f'{result["names"][0]}_to_{result["names"][1]}_sequence.gif'
            )
            sequence[0].save(
                gif_path,
                save_all=True,
                append_images=sequence[1:],
                duration=1500,
                loop=5
            )
            print(f"\n已保存序列GIF: {gif_path}")
            
            successful_results.append(result)
    
    # 如果是一對多模式且有成功的結果，創建疊圖序列
    if is_multi_mode and len(successful_results) > 1:
        overlay_dir = create_overlay_sequence(fixed_img, successful_results, fixed_name)
        print(f"已生成疊圖序列，保存在: {overlay_dir}")
        
    if successful_results:
        print("圖像對齊完成!")
    else:
        print("\n所有圖像對齊均失敗")

if __name__ == "__main__":
    main()