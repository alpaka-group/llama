# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DOXYGENDIR    = .

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	doxygen $(DOXYGENDIR)/Doxyfile > /dev/null
	# fixing doxygen bug which puts typedef in every "using" statement
	find doxygen/xml -type f -exec sed -i "s/typedef //g" {} \;
	# fixing breathe bug which is confused by members of templated classes
	find doxygen/xml -type f -exec sed -i "s/llama::Array&lt; T, T_dim &gt;:://g" {} \;
	find doxygen/xml -type f -exec sed -i "s/<definition>using llama::DatumCoord&lt; T_coords &gt;::/<definition>using llama::DatumCoord::/g" {} \;
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
