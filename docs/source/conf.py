import os
import sys
sys.path.insert(0, os.path.abspath('../../topoptlab'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'topoptlab'
copyright = '2025, Stefan Hiemer'
author = 'Stefan Hiemer'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#
extensions = ['myst_parser',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',]
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "deflist",
    "linkify",
    "attrs_inline"]

templates_path = ['_templates']




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html output
html_theme = 'sphinx_rtd_theme'
#html_theme_options = {'navigation_depth': 3}
html_static_path = ['_static']


