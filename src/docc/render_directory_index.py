import os

def extract_docstring_or_comment(filepath):
    """
    Extracts the module-level docstring or, if not present, the first comment block from a Python file.
    Returns an empty string if neither is found or file does not exist.
    """
    if not os.path.exists(filepath):
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # extract docstring
    in_docstring = False
    docstring = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
                docstring.append(stripped.strip('"""').strip("'''"))
                if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    break
            else:
                break
        elif in_docstring:
            docstring.append(stripped)
        elif stripped.startswith('#'):
            docstring.append(stripped.lstrip('#').strip())
        elif stripped:
            break
    return ' '.join(docstring).strip()

def render_directory_index(path):
    # 1. Read __init__.py for the directory summary
    init_path = os.path.join(path, '__init__.py')
    summary = extract_docstring_or_comment(init_path) if os.path.exists(init_path) else ""

    # 2. List children, separating dirs and files
    children = os.listdir(path)
    dirs = sorted([c for c in children if os.path.isdir(os.path.join(path, c))])
    files = sorted([c for c in children if os.path.isfile(os.path.join(path, c)) and c != '__init__.py'])

    # 3. For each child, extract a snippet
    def get_snippet(child_path):
        if os.path.isdir(child_path):
            init_file = os.path.join(child_path, '__init__.py')
            return extract_docstring_or_comment(init_file) if os.path.exists(init_file) else ""
        elif child_path.endswith('.py'):
            return extract_docstring_or_comment(child_path)
        else:
            return ""
    
    # 4. Render the merged page
    html = "<h1>Directory: {}</h1>\n".format(os.path.basename(path))
    if summary:
        html += "<div class='summary'>{}</div>\n".format(summary)
    html += "<ul>\n"
    for d in dirs + files:
        snippet = get_snippet(os.path.join(path, d))
        html += "<li><a href='{}'>{}</a>: {}</li>\n".format(d, d, snippet)
    html += "</ul>\n"
    return html