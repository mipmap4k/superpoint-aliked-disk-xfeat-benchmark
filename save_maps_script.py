import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from DataProvider import DataProvider

def save_maps_for_dataset(dataset_path, output_path):
    """
    Проходит по всему датасету и сохраняет соответствующие куски карты.
    """
    print("Инициализация индекса датасета...")
    provider = DataProvider(dataset_path)
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_files = len(provider.data_pairs)
    print(f"Найдено изображений: {total_files}. Начало обработки...")

    generator = provider.generator(shuffle=False)
    
    success_count = 0
    error_count = 0

    for sat_img, drone_img, meta in tqdm(generator, total=total_files, unit="img"):
        try:
            filename = meta['file']
            
            save_name = filename
            
            # Если нужно сохранить оригинальное расширение или заменить на PNG (рекомендуется для карт без потери качества)
            save_name = os.path.splitext(save_name)[0] + ".png"
            
            save_full_path = output_dir / save_name

            # 3. Конвертация RGB -> BGR перед сохранением (так как cv2 пишет в BGR)
            if sat_img is not None and sat_img.size > 0:
                sat_img_bgr = cv2.cvtColor(sat_img, cv2.COLOR_RGB2BGR)
                
                # 4. Сохранение
                cv2.imwrite(str(save_full_path), sat_img_bgr)
                success_count += 1
            else:
                # Если вернулся пустой массив (ошибка кропа)
                error_count += 1
                
        except Exception as e:
            print(f"\nОшибка при сохранении {meta.get('file', 'unknown')}: {e}")
            error_count += 1

    print("\n" + "="*40)
    print(f"Готово!")
    print(f"Успешно сохранено карт: {success_count}")
    print(f"Ошибок/Пустых карт: {error_count}")
    print(f"Папка с картами: {output_dir.absolute()}")
    print("="*40)

if __name__ == "__main__":

    DATASET_ROOT = "./"
    
    OUTPUT_DIR = "./Generated_Maps"
    
    if os.path.exists(DATASET_ROOT):
        save_maps_for_dataset(DATASET_ROOT, OUTPUT_DIR)
    else:
        print(f"Папка {DATASET_ROOT} не найдена.")