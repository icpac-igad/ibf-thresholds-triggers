# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Drought Triggers Training Document'
#copyright = '2025, Nishadh Kalladath, Eunice Koech, Anthony Mwanthi'
#author = 'Nishadh Kalladath, Eunice Koech, Anthony Mwanthi'
author ='-'
release = 'v0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Add these to the extensions list
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',  # Read the Docs theme
    # ... your other extensions
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'rinoh.frontend.sphinx'
]

# For using WeasyPrint with Sphinx
html_css_files = [
    'custom.css',  # Optional: for any custom styling
]
# Set the theme
html_theme = 'sphinx_rtd_theme'

# For PDF generation via LaTeX
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': '',
    'figure_align': 'htbp',
}
