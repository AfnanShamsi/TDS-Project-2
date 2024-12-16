import os
import csv
import requests
import sys
import json
import chardet
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from colorama import init, Fore

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

init(autoreset=True)
if len(sys.argv) != 2:
    print(f"{Fore.RED}Error Occurred -> Usage: uv run autolysis.py dataset.csv")
    sys.exit(1)

csv_filename = sys.argv[1]

def load_csv_data(file):
    try:
        print(f"{Fore.GREEN}PROCESS -> Reading CSV file")
        with open(file, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        with open(file, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            return [row for row in reader]
    except FileNotFoundError:
        print(f"{Fore.RED}Error Occurred -> {file} not found")
        sys.exit(1)

def analyze_data_with_llm(data):
    api_key = os.environ.get("AIPROXY_TOKEN")
    if not api_key:
        raise EnvironmentError("AIPROXY_TOKEN environment variable is not set.")

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f"""
                    Please analyze the following dataset and generate a narrative story based on the insights. The story should be structured, professional, and concise, fitting within a word limit of 1400-1800 words. The output should be formatted in Markdown (`README.md`), with clear sectioning and proper headings for readability. Throughout the narrative, you must include up to 2-3 image placeholders marked as `##IMG-MAIN##`. These placeholders should be inserted where relevant within the text, ensuring that the images enhance the story without interrupting the flow of analysis. The story should flow naturally with appropriate breaks for images, and each placeholder should correspond to the appropriate section of the analysis. The overall tone should reflect a data scientist's perspective, with precise, analytical, and evidence-based language throughout the narrative. Please ensure the content is well-structured and follows the conventions of a technical document.

                    STRICT WARNING: **For image placeholders, only insert `##IMG-MAIN##` and nothing else. Do not add any other text or words around the placeholder, only `##IMG-MAIN##`.**

                    Please ensure that each image is customized based on the dataset, including adjustments to the color palette, axes, titles, ticks, and any other relevant details. Do not repeat the same type of image. For each image placeholder, you must specify the following customization details for the image generation, using the format `##IMG-MAIN##[xlabel='X Label', ylabel='Y Label', title='Graph Title', color='Color Palette', xticks=range, yticks=range]`. These customizations are essential for generating distinct, accurate, and meaningful visualizations based on the data.

                    For each of the images, please provide the following information:
                    - **Type of plot:** Choose between **Bar plot**, **Line plot**, or **Box plot**.
                    - **Types of plotstyle:** Choose between **white** **dark** **whitegrid**.
                    - **Axes labels** (`xlabel` and `ylabel`): Define what the axes represent.
                    - **Title** (`title`): Provide a concise, descriptive title.
                    - **Color palette** (`color`): Choose one of the following color palettes: **'white'**, or **'Set1'**.

                    Example format for a placeholder:
                    ##IMG-MAIN##[type='Bar plot', typestyle='white', xlabel='Country', ylabel='Happiness Score', title='Happiness Score by Country', color='white']
                     ATTENTION -> YOU MUST FOLLOW THE ABOVE GIVEN EXAMPLE FORMAT (MUST)
                    **NOTE -> images must be minimum of 2 and maximum of 3**
                    **STRICT NOTE**: Ensure that the keys of the data match the given 'xlabel' and 'ylabel' exactly. For example, if the key in the is 'Year', it must be referenced as 'Year' (case-sensitive). This is a critical step; any mismatch will result in an error.
                    also dont use the "```markdown and closing ```"
                    Here's the dataset to analyze: {data}
                """
            }
        ],
        "temperature": 0.7
    }

    try:
        print(f"{Fore.GREEN}PROCESS -> Sending data for analysis this may take a few minutes")
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 429:
            error_msg = response.json().get("error", {}).get("message", "")
            if "Request too large for gpt-4o-mini" in error_msg:
                print(f"{Fore.RED}Error Occurred -> {error_msg}")
                print(f"{Fore.YELLOW}We are Really sorry that the given data too large for this API")
                sys.exit(1)

        if response.status_code != 200:
            error_msg = response.json().get('error', {}).get('message', response.json())
            print(f"{Fore.RED}Error Occurred -> {error_msg}")
            sys.exit(1)
            return None
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"{Fore.RED}Error Occurred -> {str(e)}")
        sys.exit(1)
        return None

def write_markdown_file(csv_file_path, content):
    print(f"{Fore.GREEN}PROCESS -> Writing Markdown file")
    directory_path = csv_file_path.rsplit('.', 1)[0]
    with open(os.path.join(directory_path, "README.md"), "w", encoding='utf-8') as f:
        f.write(content)


def send_ai(content_question):
    api_key = os.environ.get("AIPROXY_TOKEN")
    if not api_key:
        raise EnvironmentError("AIPROXY_TOKEN environment variable is not set.")
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": content_question
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        return f"Error: Failed to fetch AI response, status code: {response.status_code}"

def extract_image_details(file_path, csv_file_path):
    dataset_name = csv_file_path.rsplit('.', 1)[0]
    with open(file_path, 'r') as file:
        content = file.read()

    image_pattern = r"##IMG-MAIN##\[(.*?)\]"
    matches = re.findall(image_pattern, content)
    extracted_details = []
    try:
        df = pd.read_csv(csv_file_path)
        csv_columns = df.columns.tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    image_counter = 1
    for match in matches:
        details = {}
        attributes = match.split(', ')
        for attribute in attributes:
            key_value = attribute.split('=')
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = key_value[1].strip().strip("'")
                details[key] = value

        plot_type = details.get('type', 'N/A')
        typestyle = details.get('typestyle', 'N/A')
        x_label = details.get('xlabel', 'N/A')
        y_label = details.get('ylabel', 'N/A')
        x_label_old = details.get('xlabel', 'N/A')
        y_label_old = details.get('ylabel', 'N/A')
        title = details.get('title', 'N/A')
        color = details.get('color', 'N/A')

        has_error = False
        error_messages = []
        corrected_x_label = x_label
        corrected_y_label = y_label

        if x_label not in csv_columns:
            error_messages.append(f"xlabel '{x_label}'")
            has_error = True
            corrected_x_label = 'CORRECTEDXLABEL'
        if y_label not in csv_columns:
            error_messages.append(f"ylabel '{y_label}'")
            has_error = True
            corrected_y_label = 'CORRECTEDYLABEL'

        if has_error:
            print(f"The LLM has done a mistake in image {image_counter} data. Please wait, we are trying to correct it.")
            #print(f"At Text: ##IMG-MAIN##[{match}]")
            if len(error_messages) == 1:
                if 'xlabel' in error_messages[0]:
                    #print(f"Can you correct the xlabel by the below given bunch of keys please? I need your response only in this form ##IMG-MAIN##[type='{plot_type}', typestyle='{typestyle}', xlabel='{corrected_x_label}', ylabel='{y_label}', title='{title}', color='{color}'] , STRICT NOTE -> MUST GIVE RESPONSE ACCORRDING TO THE GIVEN CORRECTEION , I WANT NO OTHER TALK JUST GIVE THIS CORRECTED NOTHING MORE, HERE IS THE KEY DATA -> '{csv_columns}'")
                    print(f"We found a error in image {image_counter} at xlabel we are regenrating the response please wait")
                    xlab_d = send_ai(f"Can you correct the xlabel by the below given bunch of keys please? I need your response only in this form ##IMG-MAIN##[type='{plot_type}', typestyle='{typestyle}', xlabel='{corrected_x_label}', ylabel='{y_label}', title='{title}', color='{color}'] , STRICT NOTE -> MUST GIVE RESPONSE ACCORRDING TO THE GIVEN CORRECTEION , I WANT NO OTHER TALK JUST GIVE THIS CORRECTED NOTHING MORE, HERE IS THE KEY DATA -> '{csv_columns}'")
                    if xlab_d.startswith("##IMG-MAIN##["):
                        match = re.search(r"xlabel='(.*?)'", xlab_d)
                        if match:
                            xlabell = match.group(1)
                            if xlabell not in csv_columns:
                                 print(f"Image {image_counter} cant be genrated cause of AI-0 error")
                            else:
                                 x_label = xlabell;
                                 extracted_details.append({
                                     'type': plot_type,
                                     'typestyle': typestyle,
                                     'xlabel': x_label,
                                     'ylabel': y_label,
                                     'title': title,
                                     'color': color
                                 })
                        else:
                            print(f"Image {image_counter} cant be genrated cause of AI-1 error")
                    else:
                        print(f"Image {image_counter} cant be genrated cause of AI-2 error")
                else:
                    #print(f"Can you correct the ylabel by the below given bunch of keys please? I need your response only in this form ##IMG-MAIN##[type='{plot_type}', typestyle='{typestyle}', xlabel='{x_label}', ylabel='{corrected_y_label}', title='{title}', color='{color}'] , STRICT NOTE -> MUST GIVE RESPONSE ACCORRDING TO THE GIVEN CORRECTEION , I WANT NO OTHER TALK JUST GIVE THIS CORRECTED NOTHING MORE, HERE IS THE KEY DATA -> '{csv_columns}'")
                    print(f"We found a error in image {image_counter} at ylabel we are regenrating the response please wait")
                    ylab_d = send_ai(f"Can you correct the ylabel by the below given bunch of keys please? I need your response only in this form ##IMG-MAIN##[type='{plot_type}', typestyle='{typestyle}', xlabel='{x_label}', ylabel='{corrected_y_label}', title='{title}', color='{color}'] , STRICT NOTE -> MUST GIVE RESPONSE ACCORRDING TO THE GIVEN CORRECTEION , I WANT NO OTHER TALK JUST GIVE THIS CORRECTED NOTHING MORE, HERE IS THE KEY DATA -> '{csv_columns}'")
                    if ylab_d.startswith("##IMG-MAIN##["):
                        match = re.search(r"ylabel='(.*?)'", xlab_d)
                        if match:
                            ylabell = match.group(1)
                            if ylabell not in csv_columns:
                                 print(f"Image {image_counter} cant be genrated cause of AI-0 error")
                            else:
                                 y_label = ylabell;
                                 extracted_details.append({
                                     'type': plot_type,
                                     'typestyle': typestyle,
                                     'xlabel': x_label,
                                     'ylabel': y_label,
                                     'title': title,
                                     'color': color
                                 })
                        else:
                            print(f"Image {image_counter} cant be genrated cause of AI-1 error")
                    else:
                        print(f"Image {image_counter} cant be genrated cause of AI-2 error")
            else:
                #print(f"Can you correct the xlabel and ylabel by the below given bunch of keys please? I need your response only in this form ##IMG-MAIN##[type='{plot_type}', typestyle='{typestyle}', xlabel='{corrected_x_label}', ylabel='{corrected_y_label}', title='{title}', color='{color}'], STRICT NOTE -> MUST GIVE RESPONSE ACCORRDING TO THE GIVEN CORRECTEION , I WANT NO OTHER TALK JUST GIVE THIS CORRECTED NOTHING  MORE, HERE IS THE KEY DATA -> '{csv_columns}'")
                print(f"We found a error in image {image_counter} at xlabel and ylabel we are regenrating the response please wait")
                both_d = send_ai(f"Can you correct the xlabel and ylabel by the below given bunch of keys please? I need your response only in this form ##IMG-MAIN##[type='{plot_type}', typestyle='{typestyle}', xlabel='{corrected_x_label}', ylabel='{corrected_y_label}', title='{title}', color='{color}'], STRICT NOTE -> MUST GIVE RESPONSE ACCORRDING TO THE GIVEN CORRECTEION , I WANT NO OTHER TALK JUST GIVE THIS CORRECTED NOTHING  MORE, HERE IS THE KEY DATA -> '{csv_columns}'")
                if both_d.startswith("##IMG-MAIN##["):
                    match = re.search(r"ylabel='(.*?)'", xlab_d)
                    match2 = re.search(r"xlabel='(.*?)'", xlab_d)
                    if match:
                        ylabell = match.group(1)
                        if ylabell not in csv_columns:
                             print(f"Image {image_counter} cant be genrated cause of AI-0 error")
                        else:
                             y_label = ylabell;
                    else:
                        print(f"Image {image_counter} cant be genrated cause of AI-1 error")
                    if match2:
                        xlabell = match.group(1)
                        if xlabell not in csv_columns:
                             print(f"Image {image_counter} cant be genrated cause of AI-2 error")
                        else:
                             x_label = xlabell;
                             extracted_details.append({
                                 'type': plot_type,
                                 'typestyle': typestyle,
                                 'xlabel': x_label,
                                 'ylabel': y_label,
                                 'title': title,
                                 'color': color
                             })
                    else:
                        print(f"Image {image_counter} cant be genrated cause of AI error")
                else:
                    print(f"Image {image_counter} cant be genrated cause of AI error")
            formatted_title = title.replace(' ', '_')
            image_markdown = f"![{title}]({formatted_title}.png)"
            content = content.replace(f"##IMG-MAIN##[type='{plot_type}', typestyle='{typestyle}', xlabel='{x_label_old}', ylabel='{y_label_old}', title='{title}', color='{color}']", image_markdown)
            #print(content)
            image_counter += 1
            continue

        formatted_title = title.replace(' ', '_')
        image_markdown = f"![{title}]({formatted_title}.png)"
        content = content.replace(f"##IMG-MAIN##[type='{plot_type}', typestyle='{typestyle}', xlabel='{x_label_old}', ylabel='{y_label_old}', title='{title}', color='{color}']", image_markdown)
        #print(content)
        extracted_details.append({
            'type': plot_type,
            'typestyle': typestyle,
            'xlabel': x_label,
            'ylabel': y_label,
            'title': title,
            'color': color
        })

        image_counter += 1
        with open(file_path, 'w') as readme_file:
           readme_file.write(content)

    return extracted_details

def generate_plot(details, csv_file_path):
    dataset_name = csv_file_path.rsplit('.', 1)[0]
    df = pd.read_csv(csv_file_path)
    x_col = details['xlabel']
    y_col = details['ylabel']

    if x_col not in df.columns or y_col not in df.columns:
        print(f"Error: Columns '{x_col}' or '{y_col}' not found in the CSV file.")
        return

    plt.figure(figsize=(5.12, 5.12))
    sns.set_style(details['typestyle'])
    palette = details['color']
    plot_type = details['type']

    if plot_type == 'Bar plot':
        sns.barplot(x=x_col, y=y_col, data=df, palette=palette)
    elif plot_type == 'Line plot':
        sns.lineplot(x=x_col, y=y_col, data=df, palette=palette)
    elif plot_type == 'Box plot':
        sns.boxplot(data=df, y=y_col, color=palette)
    else:
        print(f"Unsupported plot type: {plot_type}")
        return

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(details['title'])
    directory_path = csv_file_path.rsplit('.', 1)[0]
    formatted_title = details['title'].replace(' ', '_')
    plot_filename = f"{formatted_title}.png"
    plt.savefig(os.path.join(directory_path, plot_filename), dpi=300)
    plt.close()

    print(f"{Fore.GREEN}PROCESS -> Plot saved as {plot_filename}")

def create_directory(directory_name):
    current_path = os.getcwd()
    new_directory_path = os.path.join(current_path, directory_name)
    if not os.path.exists(new_directory_path):
         os.mkdir(new_directory_path)

data = load_csv_data(csv_filename)
create_directory(csv_filename.rsplit('.', 1)[0])
analysis_result = analyze_data_with_llm(data)

if analysis_result:
    write_markdown_file(csv_filename, analysis_result)
directory_path = csv_filename.rsplit('.', 1)[0]
print(f"{Fore.GREEN}PROCESS -> Extracting Image Details")
image_details = extract_image_details(os.path.join(directory_path, "README.md"), csv_filename)

if image_details:
    df = pd.read_csv(csv_filename)
    for detail in image_details:
        print(f"{Fore.GREEN}PROCESS -> Genrating Images")
        generate_plot(detail, csv_filename)
else:
    print(f"{Fore.RED}PROCESS -> NO IMAGES GENRATED DURING THE PROCESS")

print(f"{Fore.GREEN}PROCESS -> Completed successfully Chcek Result At {directory_path}/README.md")
