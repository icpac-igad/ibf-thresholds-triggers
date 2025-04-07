# Converting RST Files to Sphinx HTML and PDF

To convert your RST files into Sphinx HTML and then PDF documentation, follow these steps:

## Step 1: Set Up Sphinx Project

If you haven't already set up a Sphinx project, create one:

```bash
# Create a docs directory
mkdir docs
cd docs

# Initialize Sphinx project
sphinx-quickstart
```

When running `sphinx-quickstart`, you'll be prompted with several questions:
- Separate source and build directories: Choose "y" (recommended)
- Project name: Enter your project name (e.g., "Drought Anticipatory Action System")
- Author name: Enter your name or organization
- Project release: Enter version number
- Project language: Enter language code (e.g., "en" for English)

## Step 2: Configure Sphinx

Edit the `conf.py` file in the source directory (usually `docs/source/`) to add necessary extensions:

```python
# Add these to the extensions list
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',  # Read the Docs theme
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
```

## Step 3: Organize Your RST Files

1. Place your RST files in the source directory (e.g., `docs/source/`).
2. Create an `index.rst` file that includes references to your other RST files:

```rst
Welcome to Drought Anticipatory Action System Documentation
===========================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   1
   2 
   spi_calculation
   forecast_verification
   visualization
   api_reference
```

Make sure the filenames in the toctree match your actual RST files (without the `.rst` extension).

## Step 4: Install Required Packages

Install the necessary packages for Sphinx and PDF generation:

```bash
pip install sphinx sphinx-rtd-theme sphinxcontrib-napoleon sphinx-autodoc-typehints
pip install sphinx-latex latexmk
```
## Step 5: Use sphinx-build directly to create a PDF

For PDF, use Sphinx's built-in PDF builders:

1. **Install rinohtype**:
   ```bash
   pip install rinohtype
   ```

2. **Add the extension to your Sphinx configuration**:
   Add to your `conf.py`:
   ```python
   extensions = [
       # ... your other extensions
       'rinoh.frontend.sphinx'
   ]
   ```

3. **Build the PDF directly**:
   ```bash
   cd docs
   sphinx-build -b rinoh source build/rinoh
   ```

4. The PDF will be in the `build/rinoh` directory.


