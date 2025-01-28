import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import os
import textwrap
import re

def save_trajectory_to_pdf(trajectory, save_dir):
    """
    Save the trajectory to a pdf file with improved layout that maintains image aspect ratios
    =========================
    Arguments:
        - trajectory (list): The trajectory to save
        - save_dir (str): Directory to save the PDF files
    =========================
    """
    import matplotlib
    # 设置支持中文的字体
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    os.makedirs(save_dir, exist_ok=True)

    for data_idx, data in enumerate(trajectory):
        filename = os.path.join(save_dir, f"trajectory_data_{data_idx}.pdf")
        with PdfPages(filename) as pdf:
            n_steps = len(data['img_before_action'])
            
            for step in range(n_steps):
                # Calculate figure size based on image aspect ratio
                img_height, img_width = data['img_before_action'][step].shape[:2]
                aspect_ratio = img_width / img_height
                fig_width = 12  # Fixed width
                fig_height = (fig_width / aspect_ratio) * 3.5  # Scale height based on aspect ratio and space needs
                
                # Create figure without constrained_layout
                fig = plt.figure(figsize=(fig_width, fig_height))
                
                # Create grid with appropriate height ratios for images and text
                gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.3, 1.2], hspace=0.1)
                
                
                # Before Action Image
                ax_before = fig.add_subplot(gs[0])
                ax_before.imshow(data['img_before_action'][step])
                ax_before.set_title(f'Step {step+1} - Before Action', 
                                  fontsize=16, pad=10)
                ax_before.axis('off')
                ax_before.set_aspect('equal')  # Preserve aspect ratio
                
                # Text Response Section
                ax_text = fig.add_subplot(gs[1])
                # Clean and format the response text
                parsed_response = "<think>" + data['parsed_response'][step]['raw'].replace("<|im_end|>", "").replace("<|endoftext|>", "")
                # 先替换所有的 \n 为实际换行
                parsed_response = parsed_response.replace('\\n', '\n')
                # 然后在 </think> 和 <answer> 之间添加换行
                parsed_response = re.sub(r'(</think>)\s*(<answer>)', r'\1\n\2', parsed_response)
                
                # Wrap text with improved parameters
                wrapped_text = textwrap.fill(
                    parsed_response,
                    width=65,
                    initial_indent='  ',
                    subsequent_indent='    ',
                    break_long_words=True,
                    break_on_hyphens=True
                )
                
                # Display text with optimized parameters
                ax_text.text(
                    0.02, 0.97,
                    f"Model Response:\n{wrapped_text}",
                    transform=ax_text.transAxes,
                    ha='left',
                    va='top',
                    wrap=True,
                    fontsize=13,
                    linespacing=1.4,
                    bbox=dict(facecolor='#f8f8f8', alpha=0.9, pad=8),
                    family='DejaVu Sans'  # 确保使用支持中文的字体
                )
                ax_text.axis('off')
                
                # After Action Image
                ax_after = fig.add_subplot(gs[2])
                ax_after.imshow(data['img_after_action'][step])
                ax_after.set_title('After Action', fontsize=16, pad=10)
                ax_after.axis('off')
                ax_after.set_aspect('equal')  # Preserve aspect ratio
                
                # Save the page
                # 使用 bbox_layout='tight' 来自动调整布局
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.3)
                plt.close(fig)


def parse_llm_output(llm_output: str, strategy: str):
    """
    Parse the llm output
    =========================
    Arguments:
        - llm_output: the output of the llm
        - strategy: the strategy to parse the llm output
            - "formated": parse action and thought
            - "raw": parse the raw output
    =========================
    Returns (dict): parsed output
    """
    if strategy == "raw":
        return {'raw': llm_output}
    elif strategy == "formatted":
        if "<answer>" in llm_output:
            action = llm_output.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            action = ""
        if "<think>" in llm_output:
            thought = llm_output.split("<think>")[1].split("</think>")[0].strip()
        else:
            thought = ""
        return {
            'action': action,
            'thought': thought
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")