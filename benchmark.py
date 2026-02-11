import os
import cv2
import numpy as np
import pandas as pd
import torch
import gc
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')

import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict
from lightglue import SuperPoint, ALIKED, DISK, LightGlue
from sklearn.metrics import average_precision_score

# ==========================================
# 1. СТРУКТУРЫ ДАННЫХ
# ==========================================

@dataclass
class KeypointData:
    x: float
    y: float
    confidence: float

@dataclass
class RawMatch:
    point_sat: KeypointData
    point_drone: KeypointData
    score: float = 1.0 

# ==========================================
# 2. ВИЗУАЛИЗАЦИЯ
# ==========================================

class ResultVisualizer:
    def __init__(self, base_output_dir="vis_results", kp_scale_factor=40):
        self.base_output_dir = base_output_dir
        self.kp_scale_factor = kp_scale_factor
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

    def _draw_on_image(self, img, all_kpts: List[KeypointData], inlier_kpts: List[KeypointData], draw_all=True):
        vis = img.copy()
        # 1. Спутник: Красные (все) + Зеленые (инлайеры)
        if draw_all:
            for kp in all_kpts:
                r = max(1, min(int(kp.confidence * self.kp_scale_factor), 12))
                cv2.circle(vis, (int(kp.x), int(kp.y)), r, (255, 0, 0), -1)
        
        # 2. Инлайеры (Зеленым) - рисуются всегда поверх
        for kp in inlier_kpts:
            r = max(1, min(int(kp.confidence * self.kp_scale_factor), 12))
            cv2.circle(vis, (int(kp.x), int(kp.y)), r, (0, 255, 0), -1)
        return vis

    def save_analysis_plot(self, img_s, img_d, all_sat_kpts, raw_matches, inliers, model_name, frame_name):
        save_path = os.path.join(self.base_output_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        inliers_sat = [m.point_sat for m in inliers]
        inliers_drone = [m.point_drone for m in inliers]

        vis_s = self._draw_on_image(img_s, all_sat_kpts, inliers_sat, draw_all=True)
        vis_d = self._draw_on_image(img_d, [], inliers_drone, draw_all=False) # Только зеленые

        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.8])

        ax0 = fig.add_subplot(gs[:, 0]); ax0.imshow(vis_s); ax0.set_title("Satellite Map (All + Inliers)"); ax0.axis('off')
        ax1 = fig.add_subplot(gs[:, 1]); ax1.imshow(vis_d); ax1.set_title("Drone Image (Inliers Only)"); ax1.axis('off')

        # Гистограмма Confidence
        ax2 = fig.add_subplot(gs[0, 2])
        raw_confs = [m.point_sat.confidence for m in raw_matches]
        inl_confs = [m.point_sat.confidence for m in inliers]
        if raw_confs:
            ax2.hist(raw_confs, bins=25, alpha=0.4, color='red', label='Raw Matches')
            if inl_confs: ax2.hist(inl_confs, bins=25, alpha=0.7, color='green', label='Inliers')
            ax2.set_title("Confidence Distribution")
            ax2.set_xlabel("Confidence Value"); ax2.set_ylabel("Frequency"); ax2.legend()
        ax2.grid(True, alpha=0.2)

        # Гистограмма Scores
        ax3 = fig.add_subplot(gs[1, 2])
        raw_scores = [m.score for m in raw_matches]
        inl_scores = [m.score for m in inliers]
        if raw_scores:
            ax3.hist(raw_scores, bins=25, alpha=0.4, color='blue', label='Raw Scores')
            if inl_scores: ax3.hist(inl_scores, bins=25, alpha=0.7, color='cyan', label='Inlier Scores')
            ax3.set_title("Matching Scores Distribution")
            ax3.set_xlabel("Matching Score"); ax3.set_ylabel("Frequency"); ax3.legend()
        ax3.grid(True, alpha=0.2)

        plt.suptitle(f"Model: {model_name} | {frame_name} | Inliers: {len(inliers)}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{frame_name}.png"), dpi=100)
        plt.close()

    def save_summary_statistics(self, all_model_results: Dict[str, List[Dict]]):
        if not all_model_results: return
        save_path = os.path.join(self.base_output_dir, "summary_report")
        os.makedirs(save_path, exist_ok=True)
        
        data_list = []
        for model_name, frames in all_model_results.items():
            for f in frames:
                data_list.append({'Model': model_name.upper(), 'mAP': f['mAP'], 'Inliers': f['inliers']})
        df = pd.DataFrame(data_list)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        sns.boxplot(x='Model', y='mAP', data=df, ax=axes[0], palette="Set2")
        axes[0].set_title("Stability: mAP Distribution")
        sns.boxplot(x='Model', y='Inliers', data=df, ax=axes[1], palette="Set3", hue='Model', legend=False)
        axes[1].set_title("Stability: Inlier Count")
        
        plt.savefig(os.path.join(save_path, "global_comparison.png"))
        print(f"\n📊 Финальный отчет сохранен в: {save_path}")
        plt.close()

# ==========================================
# 3. МОДУЛИ ОБРАБОТКИ
# ==========================================

class GeometryVerifier:
    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def filter_matches(self, matches: List[RawMatch]) -> Tuple[List[RawMatch], List[RawMatch]]:
        if len(matches) < 4: return [], matches
        pts_s = np.array([[m.point_sat.x, m.point_sat.y] for m in matches], dtype=np.float32)
        pts_d = np.array([[m.point_drone.x, m.point_drone.y] for m in matches], dtype=np.float32)
        H, mask = cv2.findHomography(pts_s, pts_d, cv2.USAC_MAGSAC, self.threshold)
        inliers, outliers = [], []
        if mask is not None:
            mask = mask.ravel()
            for i, m in enumerate(matches):
                if mask[i]: inliers.append(m)
                else: outliers.append(m)
        return inliers, outliers

class MetricsCalculator:
    def calculate_reliability_metrics(self, all_sat_kpts: List[KeypointData], inliers: List[RawMatch]):
        if not all_sat_kpts: return {"mAP": 0.0, "inliers": 0}
        inlier_coords = set([(m.point_sat.x, m.point_sat.y) for m in inliers])
        y_true, y_scores = [], []
        for kp in all_sat_kpts:
            y_scores.append(kp.confidence)
            y_true.append(1 if (kp.x, kp.y) in inlier_coords else 0)
        ap = average_precision_score(y_true, y_scores) if sum(y_true) > 0 else 0
        return {"mAP": ap, "inliers": len(inliers)}
    
class ImagePreprocessor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # 1. Перевод в оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. Адаптивное выравнивание контраста (CLAHE)
        # Это самое важное для выявления мелких деталей на бетоне/земле
        gray = self.clahe.apply(gray)

        # 3. Растяжение гистограммы (Нормализация)
        # Гарантирует, что самый темный пиксель = 0, самый светлый = 255
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # 4. Превращение обратно в 3-канальное "псевдо-RGB"
        # Нужно, чтобы модели (SuperPoint/XFeat) не ругались на количество каналов
        img_pseudo_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # 5. Легкое размытие для удаления цифрового шума (опционально)
        # Помогает, если на фото дрона есть "зерно" (ISO шум)
        img_pseudo_rgb = cv2.GaussianBlur(img_pseudo_rgb, (3, 3), 0)

        return img_pseudo_rgb

# ==========================================
# 4. УНИВЕРСАЛЬНЫЙ БЕНЧМАРК
# ==========================================

class WaypointBenchmark:
    def __init__(self, device='cpu', vis_scale=40):
        self.device = device
        self.visualizer = ResultVisualizer(kp_scale_factor=vis_scale)
        self.configs = {"xfeat": "xfeat", "superpoint": "superpoint", "aliked": "aliked", "disk": "disk"}

    def execute(self, provider, verifier, metrics_calc, target_width=1024):
        preprocessor = ImagePreprocessor()
        final_summary = {}
        all_frames_data = {}

        for model_name, match_cfg in self.configs.items():
            print(f"\n🚀 Запуск: {model_name.upper()}")
            
            if model_name == "xfeat":
                model = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1024).to(self.device).eval()
            else:
                from lightglue import SuperPoint, ALIKED, DISK
                ext_class = {"superpoint": SuperPoint, "aliked": ALIKED, "disk": DISK}[model_name]
                extractor = ext_class(max_num_keypoints=1024).to(self.device).eval()
                matcher = LightGlue(features=match_cfg).to(self.device).eval()
            
            model_frame_stats = []
            frame_idx = 0
            
            for img_s_raw, img_d_raw, meta in provider.generator():
                if frame_idx == 5: break

                frame_idx += 1

                img_name = meta.get('file', f'frame_{frame_idx}').split('.')[0]
                
                # Масштабирование
                h, w = img_s_raw.shape[:2]
                scale = target_width / w
                img_s = cv2.resize(img_s_raw, (target_width, int(h * scale)))
                img_d = cv2.resize(img_d_raw, (target_width, int(h * scale)))

                img_s = preprocessor(img_s)
                img_d = preprocessor(img_d)

                img_s_t = torch.from_numpy(img_s).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                img_d_t = torch.from_numpy(img_d).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            
                with torch.inference_mode():
                    start_t = time.perf_counter()
                    if model_name == "xfeat":
                        f0 = model.detectAndCompute(img_s_t, top_k=1024)[0]
                        f1 = model.detectAndCompute(img_d_t, top_k=1024)[0]

                        f0['image_size'] = (img_s_t.shape[-1], img_s_t.shape[-2])
                        f1['image_size'] = (img_d_t.shape[-1], img_d_t.shape[-2])
                        
                        # Проверка на пустые значения для XFeat
                        if f0['keypoints'].shape[0] == 0 or f1['keypoints'].shape[0] == 0:
                            print(f"\r  [{frame_idx}] ⚠️ {img_name} | Точки не найдены! Пропуск...", end='')
                            model_frame_stats.append({"mAP": 0.0, "inliers": 0})
                            continue

                        k0 = f0['keypoints'].cpu().numpy()
                        s0 = f0['scores'].cpu().numpy().flatten() 

                        mkpts0, mkpts1, m_scores = model.match_lighterglue(f0, f1)
                        m_scores = m_scores.flatten() 

                        # mkpts0 = mkpts0.cpu().numpy()
                        # mkpts1 = mkpts1.cpu().numpy()
                        # m_scores = m_scores.cpu().numpy()

                        raw_matches = [RawMatch(KeypointData(m0[0], m0[1], s0[i]), 
                                       KeypointData(m1[0], m1[1], 1.0), ms) 
                                       for i, (m0, m1, ms) in enumerate(zip(mkpts0, mkpts1, m_scores))]
                        all_sat_kpts = [KeypointData(k[0], k[1], s) for k, s in zip(k0, s0)]
                    else:
                        f0, f1 = extractor({'image': img_s_t}), extractor({'image': img_d_t})
                        
                        # Проверка на пустые значения для LightGlue
                        if f0['keypoints'].shape[1] == 0 or f1['keypoints'].shape[1] == 0:
                            print(f"\r  [{frame_idx}] ⚠️ {img_name} | Точки не найдены! Пропуск...", end='')
                            model_frame_stats.append({"mAP": 0.0, "inliers": 0})
                            continue

                        k0, s0 = f0['keypoints'][0].cpu().numpy(), f0['keypoint_scores'][0].cpu().numpy()
                        res = matcher({'image0': f0, 'image1': f1})
                        idx = res['matches'][0].cpu().numpy()
                        m_scores = res['scores'][0].cpu().numpy() if 'scores' in res else [1.0]*len(idx)
                        raw_matches = [RawMatch(KeypointData(k0[i0][0], k0[i0][1], s0[i0]), 
                                                KeypointData(f1['keypoints'][0][i1][0].item(), f1['keypoints'][0][i1][1].item(), 1.0), 
                                                m_scores[i]) for i, (i0, i1) in enumerate(idx) if i0 != -1]
                        all_sat_kpts = [KeypointData(k[0], k[1], s) for k, s in zip(k0, s0)]

                inliers, _ = verifier.filter_matches(raw_matches)
                stats = metrics_calc.calculate_reliability_metrics(all_sat_kpts, inliers)
                model_frame_stats.append(stats)

                # Визуализация
                self.visualizer.save_analysis_plot(img_s, img_d, all_sat_kpts, raw_matches, inliers, model_name, img_name)
                print(f"\r  [{frame_idx}] Processed {img_name} | Inliers: {len(inliers)}", end='')

            all_frames_data[model_name] = model_frame_stats
            if model_frame_stats:
                final_summary[model_name] = {
                    "mAP": np.mean([f['mAP'] for f in model_frame_stats]),
                    "Inliers": np.mean([f['inliers'] for f in model_frame_stats])
                }

            if model_name == "xfeat": del model
            else: del extractor, matcher
            gc.collect(); torch.cuda.empty_cache()

        self.visualizer.save_summary_statistics(all_frames_data)
        return final_summary

# ==========================================
# 5. ТОЧКА ВХОДА
# ==========================================

if __name__ == "__main__":
    from DataProvider import DataProvider # Импорт вашего существующего класса
    
    DATASET_PATH = "./" # Укажите ваш путь
    # DATASET_PATH = "../Novorosia_dataset_sp_lg/data/401_080824_Novorossia/DCIM_1/"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loader = DataProvider(DATASET_PATH)
    verifier = GeometryVerifier(threshold=3.0)
    calculator = MetricsCalculator()
    benchmark = WaypointBenchmark(device=DEVICE, vis_scale=45)
    
    results = benchmark.execute(loader, verifier, calculator)
    
    print("\n\n" + "="*50)
    print(f"{'Algorithm':<15} | {'mAP':<10} | {'Avg Inliers'}")
    print("-" * 50)
    for model, res in results.items():
        print(f"{model.upper():<15} | {res['mAP']:<10.4f} | {res['Inliers']:.1f}")
    print("="*50)