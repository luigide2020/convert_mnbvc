import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import hashlib
from datetime import datetime
from datasets import Dataset
from PIL import Image
import io
import os

def crop_and_convert_to_bytes(image, bbox):
    # 裁剪图片
    left, top, width, height = bbox
    cropped_image = image.crop((left, top, left + width, top + height))
    
    # 将裁剪后的图片转换为字节
    img_byte_arr = io.BytesIO()
    cropped_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def convert_to_parquet(dataset, output_file):
    dataset_list = list(dataset)
    dataset_list.sort(key=lambda x: (x['doc_name'], x['page_no']))

    converted_data = []
    block_counter = {}

    for item in dataset_list:
        doc_key = item['doc_name']
        if doc_key not in block_counter:
            block_counter[doc_key] = 0

        current_time = datetime.now().strftime("%Y%m%d")

        extended_fields = {
            'doc_category': item['doc_category'],
            'collection': item['collection'],
            'page_no': item['page_no'],
            'width': item['width'],
            'height': item['height']
        }

        original_image = item['image']  # PIL Image对象

        for obj in item['objects']:
            block_id = block_counter[doc_key]

            # 裁剪图片并转换为字节
            image_content = crop_and_convert_to_bytes(original_image, obj['bbox'])

            category_id = obj['category_id']
            category_name = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
            block_type = category_name[category_id]
            extended_fields['bbox'] = obj['bbox']

            row = {
                '实体ID': item['doc_name'],
                '块ID': block_id,
                '时间': current_time,
                '扩展字段': json.dumps(extended_fields),
                '文本': obj['text'],
                '图片': image_content,
                'OCR文本': json.dumps(obj['cells']),  # 使用原始文本以及bbox作为OCR文本
                '音频': None,
                'STT文本': None,
                '其它块': None,
                '块类型': block_type,  
                '文件md5': hashlib.md5(item['doc_name'].encode()).hexdigest(),
                '页ID': item['page_no']
            }

            converted_data.append(row)
            block_counter[doc_key] += 1

    df = pd.DataFrame(converted_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
