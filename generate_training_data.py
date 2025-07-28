import sys
import json
import os
import re
from collections import Counter
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def extract_text_with_metadata(pdf_path):
    """
    Extracts text lines with font size, bold status, and position using PDFMiner.
    Returns a DataFrame with text, font_size, is_bold, page, x0, y0, x1, y1, line_height.
    Uses 1-based page indexing.
    """
    rows = []
    try:
        for page_num, layout in enumerate(extract_pages(pdf_path), start=1):  # 1-based indexing
            for element in layout:
                if not isinstance(element, LTTextContainer):
                    continue
                for text_line in element:
                    if not isinstance(text_line, LTTextLine):
                        continue
                    text = text_line.get_text().strip()
                    if not text:
                        continue
                    char_sizes = []
                    bold_chars = 0
                    total_chars = 0
                    for char in text_line:
                        if isinstance(char, LTChar):
                            char_sizes.append(char.size)
                            if hasattr(char, 'fontname') and any(keyword in char.fontname.lower() for keyword in ['bold', 'black', 'demi', 'extrab']):
                                bold_chars += 1
                            total_chars += 1
                    if not char_sizes:
                        continue
                    avg_font_size = round(np.mean(char_sizes), 1)
                    is_bold = (bold_chars / total_chars) > 0.5 if total_chars > 0 else False
                    rows.append({
                        'text': text,
                        'font_size': avg_font_size,
                        'is_bold': is_bold,
                        'page': page_num,
                        'x0': text_line.x0,
                        'y0': text_line.y0,
                        'x1': text_line.x1,
                        'y1': text_line.y1,
                        'line_height': text_line.y1 - text_line.y0
                    })
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if not df.empty:
        df['font_size'] = pd.to_numeric(df['font_size'])
        df['is_bold'] = df['is_bold'].astype(bool)
        df['page'] = df['page'].astype(int)
    return df

def merge_heading_lines(df, model=None):
    """
    Merges text lines that belong to the same heading using a trained model or heuristic rules.
    """
    if df.empty:
        return df
    df = df.sort_values(['page', 'y0', 'x0']).reset_index(drop=True)
    merged_rows = []
    i = 0
    while i < len(df):
        current = df.iloc[i]
        merged_text = current['text']
        merged_font_size = current['font_size']
        merged_is_bold = current['is_bold']
        merged_page = current['page']
        merged_x0 = current['x0']
        merged_y0 = current['y0']
        merged_x1 = current['x1']
        merged_y1 = current['y1']
        j = i + 1
        while j < len(df):
            next_row = df.iloc[j]
            if next_row['page'] != current['page']:
                break
            features = {
                'vertical_gap': next_row['y0'] - merged_y1,
                'horizontal_diff': abs(next_row['x0'] - merged_x0),
                'font_size_diff': abs(next_row['font_size'] - merged_font_size),
                'is_bold_match': int(current['is_bold'] == next_row['is_bold']),
                'is_numbered': int(bool(re.match(r"^\d+(?:\.\d+)*\.?$", merged_text.strip()))),
                'word_count': len(merged_text.split())
            }
            should_merge = False
            if model:
                X = pd.DataFrame([features])
                should_merge = model.predict(X)[0] == 1
            else:
                # Heuristic rules
                is_vertical_proximity = features['vertical_gap'] < current['line_height'] * 2.5 and features['vertical_gap'] > -2.0
                is_horizontal_alignment = features['horizontal_diff'] < 50 or (next_row['x0'] > merged_x0 and features['horizontal_diff'] < 60)
                is_font_size_match = features['font_size_diff'] < 2.0
                is_contextually_related = features['is_numbered'] or features['word_count'] < 20
                should_merge = is_vertical_proximity and is_horizontal_alignment and is_font_size_match and is_contextually_related
            if should_merge:
                merged_text += " " + next_row['text']
                merged_x1 = max(merged_x1, next_row['x1'])
                merged_y1 = max(merged_y1, next_row['y1'])
                merged_is_bold = merged_is_bold or next_row['is_bold']
                merged_font_size = max(merged_font_size, next_row['font_size'])
                j += 1
            else:
                break
        merged_text = re.sub(r'\s+', ' ', merged_text.strip())
        merged_rows.append({
            'text': merged_text,
            'font_size': merged_font_size,
            'is_bold': merged_is_bold,
            'page': merged_page,
            'x0': merged_x0,
            'y0': merged_y0,
            'x1': merged_x1,
            'y1': merged_y1,
            'line_height': current['line_height']
        })
        i = j
    return pd.DataFrame(merged_rows)

def identify_headings(df):
    """
    Identifies Title, H1, H2, H3 based on font size, bolding, numbering patterns, and hierarchy.
    Preserves the original logic from the provided code.
    """
    if df.empty:
        return "", []

    # Collect font sizes
    font_size_counts = Counter(df[df['is_bold']]['font_size'].round(1))
    relevant_sizes = sorted([s for s, count in font_size_counts.items() if s >= 9 and count >= 2], reverse=True)

    # Define thresholds
    title_size = relevant_sizes[0] if relevant_sizes else 16
    h1_size = relevant_sizes[1] if len(relevant_sizes) > 1 and relevant_sizes[1] < title_size * 0.95 else title_size * 0.9
    h2_size = relevant_sizes[2] if len(relevant_sizes) > 2 and relevant_sizes[2] < h1_size * 0.95 else h1_size * 0.9
    h3_size = relevant_sizes[3] if len(relevant_sizes) > 3 and relevant_sizes[3] < h2_size * 0.95 else h2_size * 0.9

    h1_size = max(h1_size, 14)
    h2_size = max(h2_size, 12)
    h3_size = max(h3_size, 10)

    # Title detection
    title = ""
    first_page = df[df['page'] == 1]
    title_rows = first_page[(first_page['is_bold']) & (first_page['font_size'] >= title_size * 0.9) & (first_page['y0'] < 300)]
    if not title_rows.empty:
        title = " ".join(title_rows['text']).strip()
        title = re.sub(r'\s+', ' ', title)

    # Heading detection
    outline = []
    seen_headings = set()
    current_h1 = None
    current_h2 = None

    for _, row in df.iterrows():
        text = row['text'].strip()
        size = row['font_size']
        is_bold = row['is_bold']
        page = row['page']
        y0 = row['y0']

        if not text or (title and text.lower() in title.lower()):
            continue

        # Heading classification
        potential_level = None
        numbered_heading = re.match(r"^(\d+(?:\.\d+)*)\s*(.+)", text)
        text_length = len(text.split())

        if is_bold and text_length < 20:
            if abs(size - h1_size) <= 1.5:
                potential_level = "H1"
            elif abs(size - h2_size) <= 1.5:
                potential_level = "H2"
            elif abs(size - h3_size) <= 1.5:
                potential_level = "H3"
        elif numbered_heading and text_length < 20:
            number = numbered_heading.group(1)
            dot_count = number.count('.')
            content = numbered_heading.group(2).strip()
            text = f"{number} {content}"
            if dot_count == 0 and abs(size - h1_size) <= 2.0:
                potential_level = "H1"
            elif dot_count == 1 and abs(size - h2_size) <= 2.0:
                potential_level = "H2"
            elif dot_count >= 2 and abs(size - h3_size) <= 2.0:
                potential_level = "H3"
        elif text_length < 10:
            common_headings = ["Table of Contents", "Acknowledgements", "Revision History", "References"]
            if any(h.lower() in text.lower() for h in common_headings) and abs(size - h1_size) <= 2.0:
                potential_level = "H1"

        if potential_level:
            heading_tuple = (text, page, potential_level)
            if heading_tuple not in seen_headings:
                heading = {"level": potential_level, "text": text, "page": page, "y0": y0}
                seen_headings.add(heading_tuple)

                # Hierarchy enforcement
                add_heading = False
                if potential_level == "H1":
                    add_heading = True
                    current_h1 = heading
                    current_h2 = None
                elif potential_level == "H2" and current_h1 and (
                    page > current_h1["page"] or (page == current_h1["page"] and y0 > current_h1["y0"])
                ):
                    add_heading = True
                    current_h2 = heading
                elif potential_level == "H3" and current_h2 and (
                    page > current_h2["page"] or (page == current_h2["page"] and y0 > current_h2["y0"])
                ):
                    add_heading = True
                elif potential_level == "H2" and not current_h1:
                    heading["level"] = "H1"
                    add_heading = True
                    current_h1 = heading
                elif potential_level == "H3" and current_h1 and not current_h2:
                    heading["level"] = "H2"
                    add_heading = True
                    current_h2 = heading

                if add_heading:
                    outline.append(heading)

    # Sort and finalize outline
    outline.sort(key=lambda x: (x["page"], x["y0"]))
    final_outline = [{"level": h["level"], "text": h["text"], "page": h["page"]} for h in outline]

    return title, final_outline

def process_pdf(input_pdf_path, output_json_path, model_path=None):
    """
    Processes a PDF and saves its outline as JSON.
    """
    try:
        print(f"Processing {os.path.basename(input_pdf_path)}...")
        df = extract_text_with_metadata(input_pdf_path)
        if df.empty:
            raise ValueError("No text extracted from PDF")
        model = None
        if model_path and os.path.exists(model_path):
            model = joblib.load(model_path)
        df = merge_heading_lines(df, model)
        title, outline = identify_headings(df)
        result = {
            "title": title or "Untitled",
            "outline": outline
        }
        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Output saved to {output_json_path}")
    except Exception as e:
        print(f"Error processing {input_pdf_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python main.py <input_pdf_path> <output_json_path> [model_path]")
        sys.exit(1)
    input_pdf_path = sys.argv[1]
    output_json_path = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) == 4 else None
    process_pdf(input_pdf_path, output_json_path, model_path)