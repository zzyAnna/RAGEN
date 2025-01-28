import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import os

def save_trajectory_to_pdf(trajectory, save_dir):
    """
    Save the trajectory to a pdf file
    =========================
    Arguments:
        - trajectory (list): the trajectory to save, each element is one data
            - trajectory[i] is a list of dict, each dict is for a data in the batch
        - save_dir (str): the directory to save the trajectory
    =========================
    """    
    for data_idx, data in enumerate(trajectory):
        # Create a separate PDF for each data
        filename = f"{save_dir}/trajectory_data_{data_idx}.pdf"
        with PdfPages(filename) as pdf:
            
            # Number of steps in this rollout
            n_steps = len(data['img_before_action'])
            
            for step in range(n_steps):
                # Create a new figure for each step
                fig = plt.figure(figsize=(10, 15))
                
                # Create a grid with three rows: top for before action, middle for text, bottom for after action
                gs = plt.GridSpec(3, 1, height_ratios=[1, 0.3, 1])

                # Plot before action image
                ax_before = fig.add_subplot(gs[0, 0])
                ax_before.imshow(data['img_before_action'][step])
                ax_before.set_title(f'Step {step + 1} - Before Action', fontsize=20)
                ax_before.axis('off')

                # Create text box in the middle
                ax_text = fig.add_subplot(gs[1, 0])
                parsed_response = data['parsed_response'][step]
                text_content = f"Parsed Response:\n{parsed_response}"
                ax_text.text(0.5, 0.5, text_content,
                            transform=ax_text.transAxes,
                            horizontalalignment='center',
                            verticalalignment='center',
                            wrap=True,
                            fontsize=24)
                ax_text.axis('off')

                # Plot after action image
                ax_after = fig.add_subplot(gs[2, 0])
                ax_after.imshow(data['img_after_action'][step])
                ax_after.set_title('After Action', fontsize=20)
                ax_after.axis('off')

                # Adjust layout and save the figure to the PDF
                plt.tight_layout()
                pdf.savefig(fig)  # Save the current figure to the PDF
                plt.close(fig)  # Close the figure to free up memory




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
    elif strategy == "formated":
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