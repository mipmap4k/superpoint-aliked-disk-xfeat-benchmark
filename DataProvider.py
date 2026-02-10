import glob
import math
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from rasterio.windows import Window


class DataProvider:
    def __init__(self, dataset_path, camera_specs=None):
        """
        Args:
            dataset_path (str): Путь к корневой папке.
            camera_specs (dict): Характеристики камеры для расчета масштаба.
                                 По умолчанию для Sony RX1R II.
        """
        self.dataset_path = pathlib.Path(dataset_path)

        # Характеристики камеры Sony RX1R II (Full Frame 35mm lens)
        # Если изображения кропнуты или уменьшены, нужно скорректировать sensor_width_mm / image_width_px
        self.camera_specs = (
            camera_specs
            if camera_specs
            else {
                "sensor_width_mm": 35.9,
                "focal_length_mm": 35.0,
                # Разрешение по умолчанию
                "image_width_px": 7952,
                "image_height_px": 5304,
            }
        )

        self.folders = ["DCIM_1", "DCIM_2"]
        self.data_pairs = self._index_dataset()

    def _apply_clahe(self, img):
        """Умное улучшение контраста"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # clipLimit=3.0 дает сильный эффект, хорошо для бетона
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Возвращаем псевдо-RGB (Models need 3 channels)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def _index_dataset(self):
        """Сканирует папки и создает индекс всех доступных пар изображение-метаданные."""
        pairs = []

        for folder_name in self.folders:
            dcim_path = self.dataset_path / folder_name
            if not dcim_path.exists():
                continue

            images_path = dcim_path / "100MSDCF"
            gcs_path = dcim_path / "GCS"
            map_path = gcs_path / "map.tif"

            if not map_path.exists():
                print(f"Warning: Map not found in {gcs_path}")
                continue

            # Поиск всех CSV файлов (обычно один на папку, но код универсален)
            csv_files = list(images_path.glob("*telemetry.csv"))

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    # Очистка имен файлов от пробелов, если есть
                    df["file"] = df["file"].astype(str).str.strip()

                    for _, row in df.iterrows():
                        img_name = row["file"]
                        img_full_path = images_path / img_name

                        if img_full_path.exists():
                            pairs.append(
                                {
                                    "drone_img_path": str(img_full_path),
                                    "map_path": str(map_path),
                                    "metadata": row.to_dict(),
                                }
                            )
                except Exception as e:
                    print(f"Error reading CSV {csv_file}: {e}")

        print(f"Dataset indexed: {len(pairs)} images found.")
        return pairs

    def _calculate_gsd(self, altitude, img_width):
        """
        Рассчитывает Ground Sample Distance (метры/пиксель) для изображения дрона.
        GSD = (Altitude * SensorWidth) / (FocalLength * ImageWidth)
        """
        if altitude <= 0:
            altitude = 50.0  # Fallback, если высота некорректна

        gsd = (altitude * self.camera_specs["sensor_width_mm"] / 1000) / (
            self.camera_specs["focal_length_mm"] / 1000 * img_width
        )
        return gsd

    def _get_satellite_patch(self, map_path, lat, lon, altitude, yaw, target_shape):
        """
        Вырезает, масштабирует и поворачивает участок карты.
        target_shape: (height, width) изображения дрона.
        """
        target_h, target_w = target_shape

        with rasterio.open(map_path) as src:
            # 1. Преобразование географических координат в пиксельные
            transformer = Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
            map_x, map_y = transformer.transform(lon, lat)

            # Получаем пиксельные координаты центра на исходной карте
            py, px = src.index(map_x, map_y)

            # 2. Расчет масштаба
            drone_gsd = self._calculate_gsd(
                altitude, target_w
            )  # метры/пиксель на фото дрона
            map_transform = src.transform
            map_res_x = map_transform[
                0
            ]  # метры/пиксель на карте (предполагаем квадратный пиксель)

            # Коэффициент масштабирования (сколько пикселей карты в одном пикселе дрона)
            # Если дрон снимает детальнее карты, scale < 1. Если карта детальнее, scale > 1.
            scale_factor = drone_gsd / abs(map_res_x)

            # 3. Вырезаем большой кусок карты с запасом (чтобы можно было повернуть без черных углов)
            # Диагональ целевого изображения в пикселях карты
            diag_len = math.sqrt(target_w**2 + target_h**2)
            # Размер окна в пикселях исходной карты, которое нужно вырезать
            crop_size_map_px = int(diag_len * scale_factor * 1.5)

            # Определяем окно чтения (Window)
            window = Window(
                px - crop_size_map_px // 2,
                py - crop_size_map_px // 2,
                crop_size_map_px,
                crop_size_map_px,
            )

            # Читаем данные (с обработкой границ карты)
            map_crop = src.read(window=window, boundless=True, fill_value=0)

            # rasterio возвращает (Channels, H, W), переводим в (H, W, Channels) для OpenCV
            map_crop = np.moveaxis(map_crop, 0, -1)
            # Если карта одноканальная (ч/б), делаем 3 канала для совместимости
            if map_crop.shape[2] == 1:
                map_crop = cv2.cvtColor(map_crop, cv2.COLOR_GRAY2RGB)
            else:
                map_crop = map_crop[:, :, :3]  # Берем RGB, откидываем Alpha если есть

        # 4. Поворот и масштабирование с помощью OpenCV
        # Центр вырезанного куска
        center = (map_crop.shape[1] // 2, map_crop.shape[0] // 2)

        # Матрица поворота.
        M = cv2.getRotationMatrix2D(center, yaw, 1.0)
        rotated_map = cv2.warpAffine(
            map_crop, M, (map_crop.shape[1], map_crop.shape[0])
        )

        # 5. Вырезаем центр нужного размера (с учетом масштаба)
        # Нам нужно получить картинку, которая соответствует target_w * scale_factor в пикселях карты
        final_width_map_px = int(target_w * scale_factor)
        final_height_map_px = int(target_h * scale_factor)

        start_x = center[0] - final_width_map_px // 2
        start_y = center[1] - final_height_map_px // 2

        # Кроп по центру повернутой карты
        final_crop = rotated_map[
            start_y : start_y + final_height_map_px,
            start_x : start_x + final_width_map_px,
        ]

        if final_crop.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 6. Финальный ресайз до размера изображения с дрона
        satellite_image = cv2.resize(
            final_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        return satellite_image

    def generator(self, shuffle=False):
        """
        Генератор данных.
        Yields:
            (satellite_image, drone_image, metadata_dict)
        """
        indices = np.arange(len(self.data_pairs))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            item = self.data_pairs[idx]

            # Чтение изображения дрона
            drone_img = cv2.imread(item["drone_img_path"])
            if drone_img is None:
                continue

            # Конвертация BGR (OpenCV) -> RGB
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
            h, w = drone_img.shape[:2]

            # Извлечение метаданных
            meta = item["metadata"]
            lat = meta["lat"]
            lon = meta["lon"]
            alt = meta["altBaro"]  # Или altBaro, в зависимости от точности
            yaw = meta["yaw"]

            # Генерация спутникового патча
            try:
                sat_img = self._get_satellite_patch(
                    item["map_path"], lat, lon, alt, yaw, (h, w)
                )
            except Exception as e:
                print(f"Error processing map for {item['drone_img_path']}: {e}")
                # Возвращаем черный квадрат в случае ошибки геометрии
                sat_img = np.zeros_like(drone_img)

            sat_img = self._apply_clahe(sat_img)
            drone_img = self._apply_clahe(drone_img)

            yield sat_img, drone_img, meta
