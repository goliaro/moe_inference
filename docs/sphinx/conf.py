import sphinx_rtd_theme
import os
import glob
import pypandoc

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MoE Inference'
copyright = '2022, the authors'
author = 'Daiyaan Arfeen, Zhihao Jia, Xupeng Miao, Gabriele Oliaro, Zeyu Wang, Rae Wong, Jidong Zhai (alphabetic order)'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'Python, C++'
source_suffix = ['.rst', '.md']

# sphinx_path = os.path.dirname(os.path.abspath(__file__))
# md_docs_path = os.path.join(sphinx_path, "..")
# generated_md_docs = os.path.join(sphinx_path, "_generated_rst")
# os.system(f"rm -rf {generated_md_docs}")
# os.mkdir(generated_md_docs)
# md_sources = glob.glob(os.path.join(md_docs_path, "*.md"))
# for m in md_sources:
# 	rst_filename = ".".join((os.path.basename(m).split(".")[:-1] + ["rst"]))
# 	rst_filepath = os.path.join(generated_md_docs, rst_filename)
# 	pypandoc.convert_file(m, 'rst', outputfile=rst_filepath)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
#output = pypandoc.convert('somefile.md', 'rst')