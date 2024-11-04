from pathlib import Path
from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer

session = PydocMarkdown()

# Set the root folder
root_folder = Path(".")

# Initialize the Markdown renderer
renderer = MarkdownRenderer(render_module_header=False)

ignored_folders_prefix = ("_", "__", ".", "test", "colorlog", "cfg", "api")
# Find all relevant directories in the 'jaqpotpy' folder
dirs_rglob = [
    x
    for x in root_folder.glob("jaqpotpy/*/")
    if x.is_dir() and not any(x.name.startswith(p) for p in ignored_folders_prefix)
]

# Process each directory
for folder in dirs_rglob:
    print(f"Processing folder: {folder}")

    session.loaders[0].search_path = [folder]

    modules = session.load_modules()
    session.process(modules)
    markdown = session.renderer.render_to_string(modules)

    print(markdown)
    # Save the Markdown to a file
    output_file = Path(f"pydoc_docs/{folder.name}.md")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text(markdown)
