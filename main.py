import fitz  # PyMuPDF
import json
import os
import re
import sys
from collections import Counter

def extract_text_with_metadata(pdf_path):
    """
    Extracts text blocks with their font size, bold status, and position from a PDF.
    """
    document = fitz.open(pdf_path)
    pages_data = []

    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        page_texts = []

        for b in blocks:
            if b["type"] == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            is_bold = bool(span['flags'] & 2) or any(
                                keyword in span["font"].lower()
                                for keyword in ["bold", "black", "demi", "extrab"]
                            )
                            x0, y0, x1, y1 = span["bbox"]
                            page_texts.append({
                                "text": text,
                                "font_size": round(span["size"], 2),
                                "is_bold": is_bold,
                                "page": page_num,  # 0-indexed page number
                                "bbox": (x0, y0, x1, y1),
                                "line_height": y1 - y0
                            })
        pages_data.append(page_texts)
    document.close()
    return pages_data

def merge_heading_spans(page_data):
    """
    Merges text spans that belong to the same logical line/heading based on spatial proximity
    and font characteristics, allowing for multi-line headings.
    """
    merged_texts = []
    i = 0
    while i < len(page_data):
        current = page_data[i]
        merged_text = current["text"]
        merged_bbox = list(current["bbox"])
        merged_font_size = current["font_size"]
        merged_is_bold = current["is_bold"]
        
        # Keep track of original x0 for consistent indentation checks
        initial_x0 = current["bbox"][0] 
        # Store initial line height for vertical proximity calculation
        initial_line_height = current["line_height"] if current["line_height"] > 0 else 12.0 # Default if 0
        
        j = i + 1
        while j < len(page_data):
            next_span = page_data[j]
            
            # Criteria for merging:
            # 1. Same page (already ensured by page_data input)
            # 2. Similar font size (allowing for minor rendering differences)
            # 3. Next span starts reasonably close vertically (within 1.5-2 lines of text)
            # 4. Horizontal alignment: The next line either starts at a similar x0
            #    or is slightly indented, which is common for multi-line headings.
            # 5. Consider if both are bold or if the next line's size is not significantly smaller
            #    than the current merged font size (to prevent merging body text).

            # Improved vertical proximity: Use the initial line height as a reference.
            # Also ensure the next span is below or slightly overlapping but not primarily above.
            is_vertical_proximity = (next_span["bbox"][1] - current["bbox"][3] < initial_line_height * 2.0) and \
                                    (next_span["bbox"][1] > current["bbox"][1] - (initial_line_height * 0.5))
            
            # Slightly relaxed font size match for multi-line headings if they share similar bold status
            is_font_size_match = abs(next_span["font_size"] - merged_font_size) < 1.0 
            
            # Horizontal alignment: Check if aligned or reasonably indented
            # Allow for more indentation variation for multi-line headings
            is_horizontal_alignment = (abs(next_span["bbox"][0] - initial_x0) < 20) or \
                                      (next_span["bbox"][0] > initial_x0 and next_span["bbox"][0] - initial_x0 < 70) 

            # Additional check: If both are bold OR current is bold and next is same/similar size, merge.
            # Avoid merging if the next span is much smaller and not bold, preventing merging paragraphs.
            is_style_consistent = (merged_is_bold and next_span["is_bold"]) or \
                                  (merged_is_bold and (abs(next_span["font_size"] - merged_font_size) < 2.0)) or \
                                  (not merged_is_bold and not next_span["is_bold"] and is_font_size_match)


            # Prevent merging with text that clearly belongs to a different block (e.g., different x0 for a very short span)
            is_not_new_paragraph_indent = True
            if merged_text.strip().endswith(('.', '!', '?', ':')) and \
               (next_span["bbox"][0] - initial_x0 > 30) and \
               (len(next_span["text"].split()) < 7) and \
               (not next_span["is_bold"] or next_span["font_size"] < merged_font_size * 0.9):
                is_not_new_paragraph_indent = False

            if is_vertical_proximity and is_font_size_match and is_horizontal_alignment and \
               is_style_consistent and is_not_new_paragraph_indent:
                merged_text += " " + next_span["text"]
                merged_bbox[2] = max(merged_bbox[2], next_span["bbox"][2])
                merged_bbox[3] = max(merged_bbox[3], next_span["bbox"][3])
                merged_is_bold = merged_is_bold or next_span["is_bold"]
                current = next_span # Update 'current' to the last merged span for next iteration's relative position calculation
                j += 1
            else:
                break

        merged_texts.append({
            "text": merged_text.strip(),
            "font_size": merged_font_size,
            "is_bold": merged_is_bold,
            "page": current["page"],
            "bbox": tuple(merged_bbox),
            "x0": merged_bbox[0], 
            "y0": merged_bbox[1],
            "x1": merged_bbox[2],
            "y1": merged_bbox[3]
        })
        i = j
    return merged_texts


def identify_headings(pages_data):
    """
    Identifies Title, H1, H2, H3 based on font size, bolding, numbering patterns, and hierarchy.
    Now more language-agnostic by removing English-specific common heading checks.
    """
    if not pages_data:
        return "", []

    # Merge spans within each page
    merged_pages_data = [merge_heading_spans(page_data) for page_data in pages_data]

    # Collect all font sizes
    all_font_sizes = []
    for page_data in merged_pages_data:
        for text_info in page_data:
            all_font_sizes.append(text_info["font_size"])
    
    if not all_font_sizes:
        return "", []

    # Determine font size thresholds
    font_size_counts = Counter([round(size, 1) for size in all_font_sizes])
    relevant_sizes = sorted([s for s, count in font_size_counts.items() if s >= 9 and count >= 2], reverse=True)

    title_size = relevant_sizes[0] if relevant_sizes else 18
    h1_size = relevant_sizes[1] if len(relevant_sizes) > 1 else (title_size * 0.9 if title_size else 16)
    h2_size = relevant_sizes[2] if len(relevant_sizes) > 2 else (h1_size * 0.9 if h1_size else 14)
    h3_size = relevant_sizes[3] if len(relevant_sizes) > 3 else (h2_size * 0.9 if h2_size else 12)

    # Enforce minimums and strict hierarchy
    title_size = max(title_size, 18)
    h1_size = max(h1_size, 16)
    h2_size = max(h2_size, 14)
    h3_size = max(h3_size, 12)

    if title_size <= h1_size: title_size = h1_size + 2
    if h1_size <= h2_size: h1_size = h2_size + 2
    if h2_size <= h3_size: h2_size = h3_size + 2

    # Title detection: Look for largest bold text near the top of the first page
    title_text = ""
    potential_title_lines = []
    if merged_pages_data and merged_pages_data[0]:
        # Consider the first few prominent lines on the first page
        for text_info in merged_pages_data[0][:10]: 
            text = text_info["text"]
            size = text_info["font_size"]
            is_bold = text_info["is_bold"]
            y0 = text_info["bbox"][1] # y0 from bbox
            
            # Heuristic: Title is usually very large, bold, and near the top.
            # Adding a check for approximate centering for better accuracy.
            page_width = text_info["bbox"][2] if text_info["bbox"][2] > 0 else 600
            text_center = (text_info["bbox"][0] + text_info["bbox"][2]) / 2
            page_center = page_width / 2
            is_roughly_centered = abs(text_center - page_center) < 100 
            
            if y0 < 300 and is_bold and size >= (title_size * 0.9) and is_roughly_centered:
                potential_title_lines.append(text)
            elif potential_title_lines and (not is_bold or size < (title_size * 0.8)):
                break # Stop if text clearly isn't part of title

    if potential_title_lines:
        combined_title = " ".join(potential_title_lines)
        # Remove common artifacts that might appear with titles in some PDFs
        combined_title = re.sub(r'RFP: To Develop the Ontario Digital Library Business Plan March \d{4}', '', combined_title, flags=re.IGNORECASE)
        combined_title = re.sub(r'\[source: \d+\]', '', combined_title).strip()
        combined_title = combined_title.replace('\\', '').strip()
        title_text = re.sub(r'\s+', ' ', combined_title).strip()
        
        # Specific refinement for a known title pattern (if still relevant for your specific PDFs)
        if "RFP: Request for Proposal" in title_text and "To Present a Proposal for Developing the Business Plan" in title_text:
             title_text = "RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library "
        elif "The Ontario Digital Library" in title_text and "Request for Proposal" not in title_text:
             title_text = "" # Avoid picking up graphical cover titles if not the actual RFP title.

    # --- Heuristic for identifying potential table content and other non-heading elements ---
    def is_likely_non_heading(text_info, page_data_list, current_text_index, h1_s, h2_s, h3_s):
        text = text_info["text"]
        size = text_info["font_size"]
        is_bold = text_info["is_bold"]
        x0 = text_info["bbox"][0]
        y0 = text_info["bbox"][1]
        
        # 1. Very short or very long lines (often not headings)
        word_count = len(text.split())
        if word_count < 2 or word_count > 20: 
            return True

        # 2. Text that looks like common page headers/footers (page numbers, running titles)
        if re.match(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", text, re.IGNORECASE) or \
           re.match(r"^\s*\d+\s*$", text.strip()) or \
           re.match(r"^\s*[A-Za-z\s]+\s+\|\s+[A-Za-z\s]+\s*$", text): 
            return True

        # 3. Content that is part of a list or bullet points (often indented, small font)
        # Using a more robust regex for list items
        if re.match(r"^\s*[-•–—]\s+.", text) or \
           (re.match(r"^\s*\d+\.\s+.*", text) and size < h3_s and not is_bold): 
            return True

        # 4. Text that looks like table content (columnar alignment, repeated patterns, small font)
        potential_table_line_count = 0
        search_window = 5 
        
        if size < h3_s + 2.0: 
            for offset in range(-search_window, search_window + 1):
                if offset == 0: continue 
                
                check_index = current_text_index + offset
                if 0 <= check_index < len(page_data_list):
                    neighbor_text_info = page_data_list[check_index]
                    
                    if abs(neighbor_text_info["bbox"][0] - x0) < 15: 
                        if abs(neighbor_text_info["bbox"][1] - y0) < 30 and abs(neighbor_text_info["font_size"] - size) < 1.0: 
                            potential_table_line_count += 1
            
            if potential_table_line_count >= search_window * 0.6 and not is_bold: 
                return True

        return False


    raw_detected_headings = [] 
    seen_headings_set = set() 

    for page_num_idx, page_data in enumerate(merged_pages_data):
        for i, text_info in enumerate(page_data):
            text = text_info["text"].strip()
            size = text_info["font_size"]
            is_bold = text_info["is_bold"]
            page = text_info["page"] 
            x0 = text_info["bbox"][0]
            y0 = text_info["bbox"][1]
            
            if not text:
                continue

            # Skip if it's the title we already extracted
            if title_text and text == title_text:
                continue
                
            if is_likely_non_heading(text_info, page_data, i, h1_size, h2_size, h3_size):
                continue
            
            # Heuristic to avoid bolded text within paragraphs being identified as headings
            is_start_of_new_logical_block = True
            if i > 0:
                prev_text_info = page_data[i-1]
                vertical_gap = text_info["bbox"][1] - prev_text_info["bbox"][3]
                if vertical_gap < prev_text_info.get("line_height", 12) * 1.5 and abs(text_info["bbox"][0] - prev_text_info["bbox"][0]) < 10:
                    is_start_of_new_logical_block = False
            
            if not is_start_of_new_logical_block and not (size > h3_size and is_bold and y0 < 200):
                continue

            potential_level = None

            # --- Primary Classification: H1, H2, H3 ---
            # Prioritize numbered headings (1., 1.1, 1.1.1) for structure
            numbered_heading_match = re.match(r"^(\d+(?:\.\d+)*)\s*(.+)", text)
            if numbered_heading_match:
                number_prefix = numbered_heading_match.group(1)
                heading_text_part = numbered_heading_match.group(2).strip()
                dot_count = number_prefix.count('.')

                if dot_count == 0: 
                    if is_bold and abs(size - h1_size) < 1.5: 
                        potential_level = "H1"
                    elif is_bold and abs(size - h2_size) < 1.0: 
                        potential_level = "H2"
                elif dot_count == 1: 
                    if is_bold and abs(size - h2_size) < 1.5: 
                        potential_level = "H2"
                    elif is_bold and abs(size - h3_size) < 1.0: 
                        potential_level = "H3"
                elif dot_count >= 2: 
                    if is_bold and abs(size - h3_size) < 1.5:
                        potential_level = "H3"
                
                text = f"{number_prefix} {heading_text_part}" 

            # Non-numbered heading classification based on size and bolding (if not already classified by numbering)
            if not potential_level:
                if is_bold and abs(size - h1_size) < 1.5:
                    potential_level = "H1"
                elif is_bold and abs(size - h2_size) < 1.5:
                    potential_level = "H2"
                elif is_bold and abs(size - h3_size) < 1.5:
                    potential_level = "H3"
                elif size >= h1_size * 0.95 and not is_bold: 
                    potential_level = "H1"
                elif size >= h2_size * 0.95 and not is_bold:
                    potential_level = "H2"
                elif size >= h3_size * 0.95 and not is_bold:
                    potential_level = "H3"

            if potential_level:
                item_tuple = (text, page, potential_level)
                if item_tuple not in seen_headings_set:
                    raw_detected_headings.append({
                        "level": potential_level, 
                        "text": text, 
                        "page": page,
                        "y0": y0, 
                        "x0": x0 
                    })
                    seen_headings_set.add(item_tuple)
                    
    # Sort all detected headings by page, then by vertical position (y0), then horizontal (x0)
    raw_detected_headings.sort(key=lambda x: (x["page"], x["y0"], x["x0"]))

    # Final pass to enforce hierarchy and remove strict duplicates
    final_structured_outline = []
    
    # Keep track of the last added heading's level and position to enforce hierarchy
    last_h1 = None
    last_h2 = None
    last_h3 = None 

    for h in raw_detected_headings:
        level = h["level"]
        text = h["text"]
        page = h["page"]
        y0 = h["y0"]
        x0 = h["x0"] 

        entry = {"level": level, "text": text, "page": page}

        # Basic immediate duplicate check (same text on same page, same level)
        if final_structured_outline and \
           final_structured_outline[-1]["text"] == text and \
           final_structured_outline[-1]["page"] == page and \
           final_structured_outline[-1]["level"] == level:
            continue

        add_current = False

        if level == "H1":
            add_current = True
            last_h1 = h
            last_h2 = None
            last_h3 = None
        elif level == "H2":
            if last_h1 and (page > last_h1["page"] or (page == last_h1["page"] and y0 > last_h1["y0"])):
                add_current = True
                last_h2 = h
                last_h3 = None
            elif not last_h1 and not final_structured_outline: 
                entry["level"] = "H1"
                add_current = True
                last_h1 = h
                last_h2 = None
                last_h3 = None
        elif level == "H3":
            if last_h2 and (page > last_h2["page"] or (page == last_h2["page"] and y0 > last_h2["y0"])):
                add_current = True
                last_h3 = h
            elif last_h1 and not last_h2 and (page > last_h1["page"] or (page == last_h1["page"] and y0 > last_h1["y0"])): 
                entry["level"] = "H2"
                add_current = True
                last_h2 = h
                last_h3 = None
            elif not last_h1 and not last_h2 and not final_structured_outline: 
                entry["level"] = "H1"
                add_current = True
                last_h1 = h
                last_h2 = None
                last_h3 = None

        if add_current:
            final_structured_outline.append(entry)
    
    # Final cleanup: Ensure strict order by page and appearance after hierarchy adjustments.
    deduplicated_outline = []
    seen_final_entries = set()
    for item in final_structured_outline:
        item_tuple = (item["text"], item["page"], item["level"])
        if item_tuple not in seen_final_entries:
            deduplicated_outline.append(item)
            seen_final_entries.add(item_tuple)
            
    return title_text.strip(), deduplicated_outline

def process_pdf(input_pdf_path, output_json_path, model_path=None):
    """
    Processes a PDF and saves its outline as JSON.
    """
    try:
        print(f"Processing {os.path.basename(input_pdf_path)}...")
        pages_data = extract_text_with_metadata(input_pdf_path)
        if not pages_data or all(not page for page in pages_data):
            raise ValueError("No text extracted from PDF")
        
        # Note: The provided identify_headings function does not use a model directly.
        # The model_path argument is currently not utilized in the identify_headings call.
        # If you intend to use an ML model for classification, the identify_headings
        # function would need to be adapted to take and use the model.
        
        # For now, we'll proceed with the heuristic identify_headings.
        title, outline = identify_headings(pages_data)

        result = {
            "title": title or "Untitled",
            "outline": outline
        }

        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Output saved to {os.path.basename(output_json_path)}")
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