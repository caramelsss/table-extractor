# %% Import packages
from sklearn.cluster import AgglomerativeClustering
from pdf2image import convert_from_path
from ultralyticsplus import YOLO
from pytesseract import Output
from PIL import Image
import pandas as pd
import numpy as np
import pytesseract
import argparse


# %% Table extraction
class TableExtractor():
    
    def __init__(self, conf=0.25, iou=0.45, agnostic_nms=False, max_det=1000):
        self.model = YOLO('foduucom/table-detection-and-extraction')
        self.model.overrides['conf'] = conf  # NMS confidence threshold
        self.model.overrides['iou'] = iou  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = agnostic_nms  # NMS class-agnostic
        self.model.overrides['max_det'] = max_det  # maximum number of detections per image
        
    
    def extract_from_pdf(self, pdf_path, save_csv=True):
        assert pdf_path.endswith('.pdf'), 'Must be a pdf file'

        # Convert pdf to image
        pages = TableExtractor.pdf_to_image(pdf_path)
        
        # Detect tables in table
        tables = []
        for page in pages:
            page_tables = self.extract_from_image(page, save_crop=False)
            if page_tables:
                tables.extend(page_tables)
            
        # Parse table to csv format
        if not save_csv:
            dfs = []
        for i, table in enumerate(tables):
            try:
                table_df = TableExtractor.parse_image_to_df(table)
            except Exception as e:
                print(f'Error parsing into dataframe: {e}')
                table.save(f'{pdf_path.split(".")[0]}_{i}.png')
                continue
            if save_csv:
                table_df.to_csv(f'{pdf_path.split(".")[0]}_{i}.csv', index=False, encoding='utf-8-sig')
            else:
                dfs.append(table_df)
        
        return dfs if not save_csv else None


    def extract_from_image(self, img_src, save_crop=True):
        img = Image.open(img_src) if type(img_src)==str else img_src
        if save_crop:
            self.model.predict(img, save_crop=save_crop)
            
        else:
            width, height = img.size
            preds = self.model.predict(img)
            tables = []
            
            for box in preds[0].boxes:
                left, top, right, bottom = box.xyxy[0].tolist()
                left, top, right, bottom = (
                    max(0, left-10),
                    max(0, top-10),
                    min(right+10, width),
                    min(bottom+10, height)
                )
                tables.append(img.crop((left, top, right, bottom)))
                    
            return tables

    
    @staticmethod
    def pdf_to_image(pdf_path):
        pages = convert_from_path(pdf_path)
        return pages
    
    
    @staticmethod
    def parse_image_to_df(img_src, lang='eng', conf_thres=-1, col_dist_thres=20, row_dist_thres=20, min_size=2):
        # Extract text data
        img = Image.open(img_src) if type(img_src)==str else img_src
        results = pytesseract.image_to_data(
            img, 
            lang=lang, 
            output_type=Output.DICT, 
            config='--psm 12 --oem 1'
        )
        
        coords = []
        ocrText = []
        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            text = results["text"][i]
            conf = int(results["conf"][i])
            if conf > conf_thres:
                coords.append((x, y, w, h))
                ocrText.append(text)
                
        # Cluster to get columns
        xCoords = [(c[0], 0) for c in coords]
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="manhattan",
            linkage="complete",
            distance_threshold=col_dist_thres
        )
        clustering.fit(xCoords)
        sortedClusters = []
        for l in np.unique(clustering.labels_):
            idxs = np.where(clustering.labels_ == l)[0]
            if len(idxs) > min_size:
                avg = np.average([coords[i][0] for i in idxs])
                sortedClusters.append((l, avg))
        sortedClusters.sort(key=lambda x: x[1])
        df = pd.DataFrame()
        
        # Determnine text in cells
        for (l, _) in sortedClusters:
            idxs = np.where(clustering.labels_ == l)[0]
            yCoords = [coords[i][1] for i in idxs]
            sortedIdxs = idxs[np.argsort(yCoords)]
            cols = []
            text, (x, y, _, _) = ocrText[sortedIdxs[0]].strip(), coords[sortedIdxs[0]]
            cell_text = [(text, x, y)]
            for i in sortedIdxs[1:]:
                text = ocrText[i].strip()
                (x, y, _, _) = coords[i]
                if abs(cell_text[-1][2] - y) <= row_dist_thres:
                    cell_text.append((text, x, y))
                else:
                    cell_text.sort(key=lambda cell: cell[1])
                    cell_text = ' '.join([cell[0] for cell in cell_text])
                    cols.append(cell_text)
                    cell_text = [(text, x, y)]
            cell_text.sort(key=lambda cell: cell[1])
            cell_text = ' '.join([cell[0] for cell in cell_text])
            cols.append(cell_text)
            currentDF = pd.DataFrame({cols[0]: cols[1:]})
            df = pd.concat([df, currentDF], axis=1)
            
        df.fillna("", inplace=True)
        return df


# %% Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--agnostic_nms', type=bool, default=False)
    parser.add_argument('--max_det', type=int, default=1000)
    parser.add_argument('--save_csv', type=bool, default=True)
    
    args = parser.parse_args()
    te = TableExtractor(
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        max_det=args.max_det
    )
    te.extract_from_pdf(
        args.pdf_path,
        save_csv=args.save_csv
    )