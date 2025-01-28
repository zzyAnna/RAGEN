import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import os
import textwrap

def save_trajectory_to_pdf(trajectory, save_dir):
    """
    Save the trajectory to a pdf file with improved text formatting
    =========================
    Arguments:
        - trajectory (list): The trajectory to save, each element is one data
            - trajectory[i] is a list of dict, each dict contains step data
        - save_dir (str): Directory to save the PDF files
    =========================
    """
    # Create output directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    for data_idx, data in enumerate(trajectory):
        filename = os.path.join(save_dir, f"trajectory_data_{data_idx}.pdf")
        with PdfPages(filename) as pdf:
            n_steps = len(data['img_before_action'])
            
            for step in range(n_steps):
                # Create figure with constrained layout and larger vertical size
                # 
                fig = plt.figure(figsize=(12, 24), constrained_layout=True)
                gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1.2])  # Adjusted ratios

                # Before Action Image
                ax_before = fig.add_subplot(gs[0])
                ax_before.imshow(data['img_before_action'][step])
                ax_before.set_title(f'Step {step+1} - Before Action', 
                                  fontsize=18, pad=12)
                ax_before.axis('off')

                # Text Response Section
                ax_text = fig.add_subplot(gs[1])
                parsed_response = str(data['parsed_response'][step])
                
                # Text formatting with automatic wrapping
                wrapped_text = textwrap.fill(
                    parsed_response, 
                    width=90,  # Characters per line
                    replace_whitespace=False,  # Preserve original whitespace
                    subsequent_indent='    '  # Indent for wrapped lines
                )
                
                # Display text with proper alignment
                ax_text.text(
                    0.02,  # Start from left edge
                    0.95,  # Start from top
                    f"Response:\n{wrapped_text}",
                    transform=ax_text.transAxes,
                    ha='left',
                    va='top',
                    wrap=True,
                    fontsize=14,  # Reduced font size
                    linespacing=1.3
                )
                ax_text.axis('off')

                # After Action Image
                ax_after = fig.add_subplot(gs[2])
                ax_after.imshow(data['img_after_action'][step])
                ax_after.set_title('After Action', fontsize=18, pad=12)
                ax_after.axis('off')

                # Save and cleanup
                pdf.savefig(fig)
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